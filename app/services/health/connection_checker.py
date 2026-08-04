"""Standalone and startup-time connection health checks for your Enterprise RAG stack.

Run from the command line:
    source .venv/bin/activate
    python -m app.services.health.connection_checker
"""

from __future__ import annotations

import sys
from typing import Callable

import logfire
import requests
from psycopg_pool import ConnectionPool
from qdrant_client import QdrantClient

from app.config import settings
from app.gateway.client import portkey_client

if settings.LOGFIRE_TOKEN:
    logfire.configure(token=settings.LOGFIRE_TOKEN)
else:
    logfire.configure(send_to_logfire=False)

class ConnectionResult:
    """Result of a single connectivity check."""

    def __init__(self, name: str, healthy: bool, message: str = ""):
        self.name = name
        self.healthy = healthy
        self.message = message

    def to_dict(self) -> dict[str, object]:
        status = "ok" if self.healthy else "unavailable"
        if self.message:
            status = f"{status}: {self.message}"
        return {"status": status, "healthy": self.healthy, "message": self.message}


def _check_supabase_postgres() -> ConnectionResult:
    """Verify Supabase PostgreSQL is reachable and accepts queries."""
    if not settings.SUPABASE_URI:
        return ConnectionResult("postgres", False, "SUPABASE_URI not set")
    
    pool = None
    conn = None
    try:
        pool = ConnectionPool(
            conninfo=settings.SUPABASE_URI,
            min_size=1,
            max_size=2,
            open=True,
            timeout=5,
            check=ConnectionPool.check_connection,
        )
        conn = pool.getconn(timeout=5)
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return ConnectionResult("postgres", True, "Supabase PostgreSQL reachable")
    except Exception as e:
        logfire.warning(f"Postgres health check failed: {e}")
        return ConnectionResult("postgres", False, str(e))
    finally:
        if conn is not None and pool is not None:
            try:
                pool.putconn(conn)
            except Exception:
                pass
        if pool is not None:
            try:
                pool.close(timeout=5)
            except Exception:
                pass


def _check_qdrant() -> ConnectionResult:
    """Verify Qdrant cluster is reachable."""
    if not settings.QDRANT_URL:
        return ConnectionResult("qdrant", False, "QDRANT_URL not set")
    try:
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=5,
        )
        client.get_collections()
        return ConnectionResult("qdrant", True, "Qdrant reachable")
    except Exception as e:
        logfire.warning(f"Qdrant health check failed: {e}")
        return ConnectionResult("qdrant", False, str(e))

def _check_portkey_gateway() -> ConnectionResult:
    """Verify Portkey LLM gateway responds to a minimal completion."""
    try:
        resp = portkey_client.chat.completions.create(
            model=f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}",
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_completion_tokens=100,
            timeout=10,
        )
        if resp.choices and resp.choices[0].message.content is not None:
            return ConnectionResult("llm_gateway", True, "Portkey gateway reachable")
        raise RuntimeError("empty response")
    except Exception as e:
        logfire.warning(f"LLM gateway health check failed: {e}")
        return ConnectionResult("llm_gateway", False, str(e))

def _check_groq_llm() -> ConnectionResult:
    """Verify Groq API key and model connectivity."""
    if not settings.GROQ_API_KEY:
        return ConnectionResult("groq_llm", False, "GROQ_API_KEY not set")
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.GROQ_MODEL,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 5,
            },
            timeout=10,
        )
        response.raise_for_status()
        return ConnectionResult("groq_llm", True, f"Groq LLM ({settings.GROQ_MODEL}) reachable")
    except Exception as e:
        logfire.warning(f"Groq LLM health check failed: {e}")
        return ConnectionResult("groq_llm", False, str(e))


def _check_logfire() -> ConnectionResult:
    """Verify Logfire is configured with a token."""
    if not settings.LOGFIRE_TOKEN:
        return ConnectionResult("logfire", False, "LOGFIRE_TOKEN not set — spans dropped")
    return ConnectionResult("logfire", True, "Logfire configured")


def _check_langsmith() -> ConnectionResult:
    """Verify LangSmith API key is valid and the endpoint is reachable."""
    if not settings.LANGSMITH_API_KEY:
        return ConnectionResult("langsmith", False, "LANGSMITH_API_KEY not set — tracing disabled")
    try:
        response = requests.get(
            f"{settings.LANGSMITH_ENDPOINT}/ok",
            headers={"x-api-key": settings.LANGSMITH_API_KEY},
            timeout=5,
        )
        response.raise_for_status()
        return ConnectionResult("langsmith", True, f"LangSmith reachable (project: {settings.LANGSMITH_PROJECT})")
    except Exception as e:
        logfire.warning(f"LangSmith health check failed: {e}")
        return ConnectionResult("langsmith", False, str(e))


# Ordered list of all checks to run during startup and checks.
_CHECKERS: list[Callable[[], ConnectionResult]] = [
    _check_supabase_postgres,
    _check_qdrant,
    _check_portkey_gateway,
    _check_groq_llm,
    _check_logfire,
    _check_langsmith,
]


def check_all_connections() -> dict[str, ConnectionResult]:
    """Run all connection checks and return a map of service name to result."""
    results: dict[str, ConnectionResult] = {}
    for checker in _CHECKERS:
        result = checker()
        results[result.name] = result
    return results


def log_connection_summary(results: dict[str, ConnectionResult]) -> bool:
    """Log a human-readable summary. Returns True if all checks passed."""
    healthy = all(r.healthy for r in results.values())
    for name, result in results.items():
        icon = "✅" if result.healthy else "❌"
        logfire.info(f"{icon} {name}: {result.message or result.to_dict()['status']}")
    if healthy:
        logfire.info("🟢 All external connections healthy.")
    else:
        logfire.warning("🟠 Some external connections are unavailable.")
    return healthy


def _print_cli_report(results: dict[str, ConnectionResult]) -> int:
    """Print a CLI report and return an exit code."""
    healthy = True
    print("\nExternal Connection Health Report")
    print("=" * 50)
    for name, result in results.items():
        status = "OK" if result.healthy else "FAIL"
        print(f"{status:4} {name:20} {result.message}")
        if not result.healthy:
            healthy = False
    print("=" * 50)
    if healthy:
        print("All connections healthy.")
        return 0
    print("One or more connections failed.")
    return 1


if __name__ == "__main__":
    results = check_all_connections()
    log_connection_summary(results)
    sys.exit(_print_cli_report(results))