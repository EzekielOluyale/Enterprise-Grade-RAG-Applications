"""Tests for the Prometheus /metrics endpoint."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app, verify_api_key


def test_metrics_endpoint_exposes_rag_counters():
    # Bypass API key auth so requests don't return 401 Unauthorized
    app.dependency_overrides[verify_api_key] = lambda: "test-user-authorized"

    # Make a blocked /query request so no real Celery backend is needed.
    with TestClient(app) as client:
        with patch("app.main.guard") as mock_guard:
            mock_guard.return_value = (True, "blocked")
            response = client.post("/query", json={"q": "test", "thread_id": "test-thread-123"})

    # Clean up the override after the test request
    app.dependency_overrides.clear()

    assert response.status_code == 200

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    body = metrics_resp.text

    assert 'rag_requests_total{status="blocked"}' in body
    assert 'guardrails_blocks_total{blocked="true"}' in body
    assert "rag_request_duration_seconds" in body
