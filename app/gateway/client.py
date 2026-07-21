import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import settings


# Production gateway config:
#   - Fallback: primary @groq/llama-3.3-70b-versatile → @vertex-ai/gemini-3.5-flash on failure
#   - Cache: semantic mode (requires Portkey Enterprise — silently falls back to simple on free/starter)
#   - Retry: 2 attempts on rate limit / server error before triggering the fallback target

PORTKEY_CONFIG_ID = settings.PORTKEY_CONFIG_ID

PORTKEY_EMBEDDING_CONFIG_ID = settings.PORTKEY_EMBEDDING_CONFIG_ID

portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    config=settings.PORTKEY_CONFIG_ID
)


def get_langchain_llm(feature: str = "rag") -> ChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI — a drop-in for ChatGroq in LangChain nodes.

    Why ChatOpenAI and not ChatGroq:
      Portkey is a proxy. It exposes an OpenAI-compatible endpoint at PORTKEY_GATEWAY_URL.
      ChatGroq is hardwired to Groq's API and does not support routing through a proxy.
      ChatOpenAI supports base_url (points at Portkey) and default_headers (passes Portkey
      auth + config). The @rag/model-name format is Portkey-specific — Groq's own client
      does not understand it. You are still using Groq models; Portkey is just in the middle.
    """
    return ChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}",
        temperature=0,
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config=PORTKEY_CONFIG_ID,
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "production"
            }
        )
    )

def get_langchain_embeddings(feature: str = "rag-embeddings") -> OpenAIEmbeddings:
    """
    Returns a Portkey-backed OpenAIEmbeddings instance for vectorizing data.

    Like ChatOpenAI, OpenAIEmbeddings natively accepts a custom base_url and 
    default_headers, allowing Portkey to intercept the request and handle 
    automatic fallback routing between your Vertex AI and AI Studio virtual keys.
    """
    return OpenAIEmbeddings(
        api_key=settings.PORTKEY_API_KEY,  
        base_url=PORTKEY_GATEWAY_URL,
        model=f"@{settings.GEMINI_SLUG}/{settings.VERTEXAI_EMBEDDING_MODEL}",
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            config=PORTKEY_EMBEDDING_CONFIG_ID,  
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "production"
            }
        )
    )

def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"