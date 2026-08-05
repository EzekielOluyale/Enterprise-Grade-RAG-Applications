import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings:
    # GEMINI EMBEDDINGS
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # VERTEX AI EMBEDDINGS
    GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    VERTEXAI_EMBEDDING_MODEL = os.getenv("VERTEXAI_EMBEDDING_MODEL")

    # VECTOR DB (QDRANT)
    QDRANT_URL = os.getenv("QDRANT_URL") or os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "enterprise_rag"

    # Production persistence (Supabase serverless Postgres) & cache (Upstash Redis)
    SUPABASE_URI = os.getenv("DATABASE_URL")
    REDIS_URL = os.getenv("REDIS_URL")

    # REASONING ENGINE (GROQ)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL")

    # LLM GATEWAY (PORTKEY)
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    PORTKEY_CONFIG_ID = os.getenv("PORTKEY_CONFIG_ID")
    PORTKEY_CHAT_CONFIG_ID = os.getenv("PORTKEY_CHAT_CONFIG_ID")

    # PORTKEY SLUGS
    GROQ_SLUG = "groq"
    GEMINI_SLUG = "gemini"
    VERTEXAI_SLUG = "vertex-ai"

    # API safety
    RAG_API_KEY = os.getenv("RAG_API_KEY")
    RATE_LIMIT_PER_MINUTE=20

    # OBSERVABILITY 
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

    # Pydantic Logfire Observability
    LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")

# Apply LangChain environment variables for automatic tracing
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGSMITH_TRACING", "true")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

settings = Settings()