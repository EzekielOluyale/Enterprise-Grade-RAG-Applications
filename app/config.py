import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GORQ_API_KEY: str = os.getenv("GORQ_API_KEY")
    GROQ_MODEL='llama-3.3-70b-versatile'
    GROQ_FALLBACK_API_KEY: str = os.getenv("GROQ_FALLBACK_API_KEY")

    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY")
    QDRANT_URL: str = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_COLLECTION: str = "enterprise_rag"

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")

settings = Settings()