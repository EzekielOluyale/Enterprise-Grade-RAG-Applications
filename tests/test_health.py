"""Tests for health and readiness endpoints."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.health.connection_checker import ConnectionResult


def _ok(name: str) -> ConnectionResult:
    return ConnectionResult(name, True, "all good")


def _fail(name: str, message: str = "unavailable") -> ConnectionResult:
    return ConnectionResult(name, False, message)


def test_health_returns_ok():
    """Verify that the /health endpoint returns a 200 OK immediately."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready_when_all_dependencies_healthy():
    """Verify /ready returns 200 when all your specific services are up."""
    # Your specific stack from the connection checker
    results = {
        "postgres": _ok("postgres"),
        "qdrant": _ok("qdrant"),
        "llm_gateway": _ok("llm_gateway"),
        "groq_llm": _ok("groq_llm"),
        "logfire": _ok("logfire"),
        "langsmith": _ok("langsmith"),
    }
    
    # NOTE: Ensure the patch string matches exactly where your API router imports 
    # the check_all_connections function. 
    with patch("app.health.check_all_connections", return_value=results):
        client = TestClient(app)
        response = client.get("/ready")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    # Ensure all services report as ok
    assert all(v.startswith("ok") for v in data["checks"].values())


def test_ready_returns_503_when_any_dependency_fails():
    """Verify /ready returns 503 if ANY service (e.g., Qdrant) is down."""
    results = {
        "postgres": _ok("postgres"),
        "qdrant": _fail("qdrant", "qdrant down"),
        "llm_gateway": _ok("llm_gateway"),
        "groq_llm": _ok("groq_llm"),
        "logfire": _ok("logfire"),
        "langsmith": _ok("langsmith"),
    }
    
    with patch("app.health.check_all_connections", return_value=results):
        client = TestClient(app)
        response = client.get("/ready")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    # Verify the failure message is passed through
    assert data["checks"]["qdrant"].startswith("unavailable")