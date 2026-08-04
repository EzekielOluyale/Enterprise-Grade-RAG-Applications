import time

import logfire
from portkey_ai import Portkey

from app.config import settings

BATCH_SIZE = 50

_portkey_client = None
_EMBEDDING_DIM = None

def _init_client():
    """Initializes the Portkey client lazily on first use."""
    global _portkey_client
    if _portkey_client is not None:
        return

    logfire.info("Connecting to Portkey Gateway for embeddings...")
    _portkey_client = Portkey(
        api_key=settings.PORTKEY_API_KEY,
        config=settings.PORTKEY_CHAT_CONFIG_ID
    )

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

def _embed_batch(batch: list[str]) -> list[list[float]]:
    """Sends each text in the batch safely to preserve 1-to-1 vector mapping."""
    _init_client()
    batch_embeddings = []
    for text in batch:
        for attempt in range(4):
            try:
                response = _portkey_client.embeddings.create(
                    input=[text],
                    model=settings.VERTEXAI_EMBEDDING_MODEL
                )
                batch_embeddings.append(response.data[0].embedding)
                break
            except Exception as e:
                err = str(e).lower()
                is_rate_limit = any(x in err for x in ("429", "rate", "quota", "resource_exhausted"))
                if is_rate_limit and attempt < 3:
                    wait = 2 ** attempt
                    logfire.warning(
                        f"Embedding rate limit hit — retrying in {wait}s "
                        f"(attempt {attempt + 1}/4)."
                    )
                    time.sleep(wait)
                else:
                    logfire.error(f"Embedding failed for text item: {e}")
                    raise
    return batch_embeddings

def embed_query(query: str) -> list[float]:
    """Generates an embedding for a single search query."""
    _init_client()
    try:
        response = _portkey_client.embeddings.create(
            input=[query],
            model=settings.VERTEXAI_EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        logfire.error(f"❌ Portkey query embedding failed: {e}")
        raise

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generates embeddings for a list of strings using Portkey gateway batching."""
    if not texts:
        return []

    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]

        with logfire.span("Embed batch", start=i, size=len(batch)):
            try:
                batch_embeddings = _embed_batch(batch)
                all_embeddings.extend(batch_embeddings)
                logfire.info(f"Batch index {i}: sent {len(batch)} texts, received {len(batch_embeddings)} vectors")
            except Exception as e:
                logfire.error(f"❌ Batch ingestion pipeline failed at index {i}: {e}")
                raise

    logfire.info(f"Total input texts: {len(texts)} | Total embeddings generated: {len(all_embeddings)}")
    return all_embeddings
