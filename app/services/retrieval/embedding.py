import logfire
from portkey_ai import Portkey
from app.config import settings

BATCH_SIZE = 50

_portkey_chat_client = None
_portkey_ingestion_client = None
_EMBEDDING_DIM = None

def _init_chat_client():
    """Initializes the Portkey client lazily on first use."""
    global _portkey_chat_client
    if _portkey_chat_client is not None:
        return

    logfire.info("Connecting to Portkey Gateway for embeddings...")
    _portkey_chat_client = Portkey(
        api_key=settings.PORTKEY_API_KEY,
        config=settings.PORTKEY_CHAT_CONFIG_ID
    )

def _init_ingestion_client():
    """Initializes the Portkey ingestion client lazily on first use."""
    global _portkey_ingestion_client
    if _portkey_ingestion_client is not None:
        return

    logfire.info("Connecting to Portkey Gateway for ingestion...")
    _portkey_ingestion_client = Portkey(
        api_key=settings.PORTKEY_API_KEY,
        config=settings.PORTKEY_INGESTION_CONFIG_ID
    )

# ── Public helpers ─────────────────────────────────────────────────────────────

def get_embedding_dim() -> int:
    """
    Dynamically fetches the embedding dimension size.
    Caches the result so it only probes the API once per startup.
    """
    global _EMBEDDING_DIM
    if _EMBEDDING_DIM is None:
        logfire.info("Probing Portkey embedding model to verify vector dimension...")
        try:
            sample_vector = embed_query("Dimension probe")
            _EMBEDDING_DIM = len(sample_vector)
            logfire.info(f"✅ Detected active embedding dimension: {_EMBEDDING_DIM}")
        except Exception as e:
            logfire.error(f"❌ Failed to detect embedding dimension: {e}")
            raise
    return _EMBEDDING_DIM


# ── Public API (same signatures as before) ─────────────────────────────────────

def embed_query(query: str) -> list[float]:
    """Generates an embedding for a single search query."""
    _init_chat_client()
    try:
        response = _portkey_chat_client.embeddings.create(
            input=[query],
            model=settings.VERTEXAI_EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        logfire.error(f"❌ Portkey query embedding failed: {e}")
        raise

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generates embeddings for a list of strings sequentially"""
    _init_ingestion_client()
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    
    # Process in chunks of BATCH_SIZE
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        
        with logfire.span("Embed batch", start=i, size=len(batch)):
            try:
                # Portkey cleanly processes the batch array, inheriting 
                # all retries from your Config ID.
                response = _portkey_ingestion_client.embeddings.create(
                    input=batch,
                    model=settings.GEMINI_API_KEY
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logfire.error(f"❌ Portkey batch embedding failed at index {i}: {e}")
                raise
                
    return all_embeddings