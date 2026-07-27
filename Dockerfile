FROM python:3.11-slim-bookworm

# Pull uv binary from the official page - no pip install needed
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Environment Configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    XDG_CACHE_HOME=/app/.cache

# Patch OS-level CVEs, then install system deps required by torch and native packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Security: Create a non-privileged dedicated app user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/.cache && \
    chown -R appuser:appuser /app

# Fast Dependency Installation (The Cache Layer)
# Copy ONLY requirements.txt first. If you don't change this file, 
# Docker skips the install step on rebuilds and uses the cache.
COPY requirements.txt ./
RUN uv pip install --system --no-cache -r requirements.txt

# Copy only the app package — everything else (evals/, ui/, DATA/, DOCS/) stays out
COPY app/ ./app/

# Switch to the non-root user for security
USER appuser

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2", "--proxy-headers", "--forwarded-allow-ips", "*"]