FROM python:3.12-slim-bookworm

# Pull uv binary from the official image — no pip install needed.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Environment Configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    XDG_CACHE_HOME=/home/appuser/.cache

# Patch OS-level CVEs, then install system deps required by torch and native packages.
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Run as a non-root user for production hardening 
RUN useradd -m appuser

# Layer: install dependencies only (cached until requirements.txt changes).
COPY requirements.txt .

# Docker BuildKit cache mount instead of --no-cache
# Point to the PyTorch CPU index to avoid downloading 2.5GB of useless CUDA drivers
RUN --mount=type=cache,target=/home/appuser/.cache/uv \
    uv pip install --system \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Layer: copy source WITH proper ownership (saves disk space)
COPY --chown=appuser:appuser app/ ./app/

# Expose the port documented in the task definitions and health checks.
EXPOSE 8080

USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-graceful-shutdown", "5"]