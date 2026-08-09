"""Tests for rate limiting."""

from unittest.mock import patch

from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings
from app.main import app, verify_api_key


def test_rate_limit_blocks_excessive_requests():
    # Bypass API key auth so requests don't return 401 Unauthorized
    app.dependency_overrides[verify_api_key] = lambda: "test-user-authorized"

    """With a 1/minute limit, the second request within the window is rejected."""
    original_rate = settings.RATE_LIMIT_PER_MINUTE
    try:
        settings.RATE_LIMIT_PER_MINUTE = 1
        # Force an in-memory limiter for this test so state is isolated.
        app.state.limiter = Limiter(key_func=get_remote_address)

        with TestClient(app) as client:
            with patch("app.main.guard") as mock_guard:
                mock_guard.return_value = (True, "blocked")
                response1 = client.post("/query", json={"q": "hi", "thread_id": "test-1"})
                response2 = client.post("/query", json={"q": "hi again", "thread_id": "test-1"})

        assert response1.status_code == 200, f"Expected 200 on first request, got {response1.status_code}: {response1.text}"
        assert response2.status_code == 429, f"Expected 429 on rate limit, got {response2.status_code}: {response2.text}"

    finally:
        settings.RATE_LIMIT_PER_MINUTE = original_rate
        app.dependency_overrides.clear()