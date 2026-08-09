# ============================================================
# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
# ============================================================
import os
from contextlib import asynccontextmanager

import logfire
from dotenv import load_dotenv

load_dotenv()

logfire.configure(
    token=os.getenv("LOGFIRE_TOKEN"),
    service_name="enterprise-ingestion-service",
)

# Now safe to import app modules - logfire is already active
import time
import uuid
from typing import List

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from nemoguardrails.exceptions import LLMCallException
from prometheus_client import Counter, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field

from app.config import settings
from app.guardrails.rails import guard, initialize_rails
from app.health import router as health_router
from app.logging import set_request_id
from app.services.health.connection_checker import check_all_connections, log_connection_summary
from app.utils.streaming import format_sse, stream_agent
from app.agents.graph import build_graph

# Custom Prometheus metrics
RAG_REQUESTS_TOTAL = Counter(
    "rag_requests_total",
    "Total number of /query requests",
    ["status"],
)
RAG_REQUEST_DURATION = Histogram(
    "rag_request_duration_seconds",
    "Latency of /query requests in seconds",
)
GUARDRAILS_BLOCKS_TOTAL = Counter(
    "guardrails_blocks_total",
    "Number of requests blocked or allowed by guardrails",
    ["blocked"],
)

_security = HTTPBearer(auto_error=False)


# Rate Limiting Logic
def _init_rate_limiter():
    """Initialize rate limiting. Use Redis in production; fall back to in-memory storage locally."""
    from limits.storage import RedisStorage
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.extension import _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address

    try:
        storage = RedisStorage(settings.redis_url)
        # `storage.check()` returns False silently on some failures; ping the
        # underlying Redis client so we only use Redis when it is really reachable.
        if not storage.check() or not storage.storage.ping():
            raise ConnectionError("Redis did not respond to ping")
        app.state.limiter = Limiter(key_func=get_remote_address, storage_uri=settings.redis_url)
        app.state.rate_limiter_storage = "redis"
        logfire.info("🚦 Rate limiting initialized via Redis.")
    except Exception as e:
        app.state.limiter = Limiter(key_func=get_remote_address)
        app.state.rate_limiter_storage = "memory"
        logfire.warning(f"⚠️ Redis unavailable ({e}); using in-memory rate limiting.")

    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    return True


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(_security)):
    """
    Require a valid bearer token when RAG_API_KEY is configured.
    In development, omit RAG_API_KEY to disable authentication.
    """
    if not settings.RAG_API_KEY:
        # Development mode: no API key required.
        return None

    if not credentials or credentials.credentials != settings.RAG_API_KEY:
        logfire.warning("🔒 Unauthorized /query request: invalid or missing API key.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


def _get_limiter_rule(times: int, seconds: int) -> str:
    """Convert times/seconds into a slowapi limit string, e.g. '20/minute'."""
    if seconds % 60 == 0:
        return f"{times}/{seconds // 60}minute"
    if seconds % 3600 == 0:
        return f"{times}/{seconds // 3600}hour"
    return f"{times}/{seconds}second"


class _AppLimiter:
    """
    Thin wrapper around the Limiter instance that is initialized at startup.
    Allows routes to be decorated at import time while the real limiter
    (Redis-backed or in-memory) is configured in startup_event.
    """

    def limit(self, rule_or_callable):
        def decorator(func):
            import functools

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                limiter = getattr(app.state, "limiter", None)
                if limiter is None:
                    return func(*args, **kwargs)

                rule = rule_or_callable() if callable(rule_or_callable) else rule_or_callable
                # Build the slowapi wrapper at request time so the limiter
                # instance and storage backend are always current.
                return limiter.limit(rule)(func)(*args, **kwargs)

            return wrapper

        return decorator

app_limiter = _AppLimiter()

def rate_limit(times: int = None, seconds: int = None):
    """
    Decorator factory that applies slowapi rate limiting using the limiter
    initialized at startup. Falls back to a no-op if the limiter is missing.
    The rule is resolved at request time so settings can be overridden in tests.
    """

    def _resolve_rule() -> str:
        t = times or settings.RATE_LIMIT_PER_MINUTE
        s = seconds or 60
        return _get_limiter_rule(t, s)

    return app_limiter.limit(_resolve_rule)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup (DB connections, guardrails) and shutdown.
    """
    # Run external connection checks
    logfire.info("Running startup external connection checks...")
    results = await run_in_threadpool(check_all_connections)
    log_connection_summary(results)

    # Initialize guardrails
    initialize_rails()

    # Initialize Async Database Pool for LangGraph Memory
    # max_size=20 allows 20 concurrent customer queries to run simultaneously
    SUPABASE_URI = settings.SUPABASE_URI
    async with AsyncConnectionPool(
        conninfo=SUPABASE_URI,
        max_size=20,
        kwargs={
            "autocommit": True,
            "prepare_threshold": None,
            "row_factory": dict_row,  # <--- Essential for LangGraph
        },
    ) as pool:
        app.state.pool = pool

        checkpointer = AsyncPostgresSaver(pool)

        # Automatically create Supabase tables if they don't exist
        await checkpointer.setup()

        # Compile the graph with the async checkpointer
        app.state.rag_agent = build_graph(checkpointer=checkpointer)

        # The app serves traffic while inside this block
        yield

    # The pool automatically closes when the server shuts down.


# Initialize FastAPI
app = FastAPI(
    title="Agentic RAG API",
    description="An Enterprise Chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

_init_rate_limiter()

app.include_router(health_router)

# Expose Prometheus metrics at /metrics with default request instrumentation.
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

logfire.instrument_fastapi(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace ["*"] with your actual frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(LLMCallException)
async def llm_call_exception_handler(request: Request, exc: LLMCallException):
    logfire.warning(f"⚠️ LLM Rate Limit Hit: {exc}")
    return JSONResponse(
        status_code=429,
        content={
            "message": "The AI is currently experiencing high traffic due to API limits. Please try again in a few seconds."
        },
    )


class QueryRequest(BaseModel):
    q: str = Field(..., description="The message sent by the user.")
    thread_id: str = Field(..., description="Unique identifier tracking this specific conversation session.")


class QueryResponse(BaseModel):
    question: str = Field(..., description="The original query sent by the user.")
    answer: str = Field(..., description="The final answer generated by the RAG agent.")
    thought_process: List[str] = Field(default=[], description="The step-by-step execution plan taken by the agent.")
    status: str = Field(..., description="The final execution state of the graph.")
    sources: List[str] = Field(default=[], description="List of source documents used to ground the answer.")


@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live."}


@app.get("/graph")
def get_graph_image(request: Request, _api_key: str = Depends(verify_api_key)):
    """
    Returns the Mermaid image of the agent's workflow.
    """
    rag_agent = getattr(request.app.state, "rag_agent", None)

    if rag_agent is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent graph is still initializing. Please try again in a few seconds.",
        )
    try:
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}


@app.post("/query", response_model=QueryResponse)
@rate_limit()
async def query(
    request: Request,
    body: QueryRequest,
    _api_key: str = Depends(verify_api_key),
):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = body.q
    thread_id = body.thread_id
    request_id = str(uuid.uuid4())
    set_request_id(request_id)

    start = time.perf_counter()
    rag_agent = request.app.state.rag_agent

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph...",
        "final_answer": "",
    }

    # Configuration for Memory (Thread ID)
    config = {"configurable": {"thread_id": thread_id}}

    with logfire.span("🔍 /query", request_id=request_id, thread_id=thread_id):
        try:
            # Gate 1: NeMo Guardrails — blocks off-topic, jailbreaks, and handles dialog
            rail_fired, rail_response = await run_in_threadpool(guard, q)
            if rail_fired:
                GUARDRAILS_BLOCKS_TOTAL.labels(blocked="true").inc()
                RAG_REQUESTS_TOTAL.labels(status="blocked").inc()
                logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
                return {
                    "question": q,
                    "answer": rail_response,
                    "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                    "status": "Blocked by guardrails.",
                    "sources": [],
                }

            GUARDRAILS_BLOCKS_TOTAL.labels(blocked="false").inc()

            # Gate 2: LangGraph RAG pipeline
            # Run the graph asynchronously to preserve Logfire context variables
            final_output = await rag_agent.ainvoke(initial_state, config=config)
            RAG_REQUESTS_TOTAL.labels(status="success").inc()

            return {
                "question": q,
                "answer": final_output.get("final_answer"),
                "thought_process": final_output.get("plan"),
                "status": final_output.get("status"),
                "sources": final_output.get("documents", []),
            }

        except Exception as e:
            RAG_REQUESTS_TOTAL.labels(status="error").inc()
            logfire.error(f"❌ Backend Execution Failed: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "question": q,
                    "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
                    "thought_process": ["Error encountered during execution."],
                    "status": "error",
                    "sources": [],
                },
            )

        finally:
            # This ensures latency is tracked no matter what path the code took!
            RAG_REQUEST_DURATION.observe(time.perf_counter() - start)


@app.post("/stream")
@rate_limit()
async def stream_query(
    request: Request,
    body: QueryRequest,
    _api_key: str = Depends(verify_api_key),
):
    """
    Executes the LangGraph RAG flow and streams the output via SSE.
    """
    q = body.q
    thread_id = body.thread_id
    request_id = str(uuid.uuid4())
    set_request_id(request_id)

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph...",
        "final_answer": "",
    }

    config = {"configurable": {"thread_id": thread_id}}

    with logfire.span("🌊 /stream", request_id=request_id, thread_id=thread_id):
        # Gate 1: NeMo Guardrails
        rail_fired, rail_response = await run_in_threadpool(guard, q)
        if rail_fired:
            GUARDRAILS_BLOCKS_TOTAL.labels(blocked="true").inc()
            RAG_REQUESTS_TOTAL.labels(status="blocked").inc()
            logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")

            async def blocked_stream():
                yield format_sse("status", "Blocked by guardrails.")
                yield format_sse("token", rail_response)
                yield format_sse("end")

            return StreamingResponse(
                blocked_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        GUARDRAILS_BLOCKS_TOTAL.labels(blocked="false").inc()
        RAG_REQUESTS_TOTAL.labels(status="streamed").inc()

        # Gate 2: Pass into the standalone generator
        rag_agent = request.app.state.rag_agent

        return StreamingResponse(
            stream_agent(rag_agent, initial_state, config, thread_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )
