import time

import logfire
from flashrank import Ranker, RerankRequest
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

# Lazy initialization - Ranker is loaded on first use to ensure logfire.configure() has run
_ranker = None


class _FlashRankWrapper:
    def __init__(self):
        try:
            # We use a specific cache directory to avoid permission issues in production
            self.ranker = Ranker(cache_dir="/tmp/flashrank")
        except Exception:
            self.ranker = Ranker()

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[str]:
        # Hide the messy formatting inside this wrapper!
        passages = [{"id": i, "text": doc} for i, doc in enumerate(documents)]
        request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(request)

        reranked_docs = []
        for res in results[:top_n]:
            reranked_docs.append(res["text"])

        return reranked_docs


def _get_ranker() -> Ranker:
    """
    Initializes the FlashRank engine lazily.
    FlashRank uses a local ONNX model (ms-marco-MiniLM-L-6-v2) for ultra-fast reranking.
    """
    global _ranker
    if _ranker is None:
        logfire.info("🧠 Initializing FlashRank Model (TinyBERT) locally...")
        _ranker = _FlashRankWrapper()
    return _ranker


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    reraise=True,
    before_sleep=before_sleep_log(logfire, "warning"),
)
def _rerank(query: str, documents: list[str], top_n: int) -> list[str]:
    """Core FlashRank reranking with retry on transient failures."""
    ranker = _get_ranker()
    return ranker.rerank(query, documents, top_n)


def rerank_documents(query: str, documents: list[str], top_n: int = 5) -> list[str]:
    """
    Refines retrieval results by re-scoring documents against the query semantically.

    Why FlashRank?
    Standard vector search (Cosine Similarity) is fast but mathematically "fuzzy."
    FlashRank uses a Cross-Encoder approach which is much more precise but usually slow.
    FlashRank solves this by using highly optimized, quantized ONNX models locally.
    """
    if not documents:
        return []

    start_time = time.time()
    logfire.info(f"📡 [Reranker] Sending {len(documents)} docs to FlashRank Cross-Encoder...")

    try:
        reranked_docs = _rerank(query, documents, top_n)
        duration = time.time() - start_time
        logfire.info(f"✅ [Reranker] Done in {duration:.2f}s.")

        return reranked_docs

    except Exception as e:
        logfire.error(f"❌ [Reranker] Semantic Reranking Failed: {e}")
        # Fallback to the original Qdrant order to ensure the user still gets an answer
        return documents[:top_n]
