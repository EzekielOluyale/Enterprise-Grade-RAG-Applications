"""Tests for synchronous /query endpoint."""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


def test_query_returns_direct_answer():
    """/query should run the pipeline synchronously and return the answer."""
    mock_state = {
        "final_answer": "hello",
        "plan": ["Intent: Technical", "Search Term: hi"],
        "status": "Response generated.",
        "documents": ["doc1"],
    }
    mock_agent = AsyncMock()
    mock_agent.ainvoke.return_value = mock_state
    
    original_key = settings.RAG_API_KEY
    settings.RAG_API_KEY = "test-secret"

    try:
        with TestClient(app) as client:
            # Ensure rag_agent exists on app.state before TestClient uses it.
            app.state.rag_agent = mock_agent
            with patch("app.main.guard") as mock_guard:
                mock_guard.return_value = (False, "")
            
                # MUST pass the Authorization header to bypass the 401 error
                response = client.post(
                    "/query", 
                    json={"q": "hi", "thread_id": "t1"},
                    headers={"Authorization": "Bearer test-secret"}
                )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "hello"
        assert data["status"] == "Response generated."
        assert data["sources"] == ["doc1"]
        # Verify the async method was awaited
        mock_agent.ainvoke.assert_awaited_once()
        
    finally:
        settings.RAG_API_KEY = original_key

def test_query_blocks_guardrails():
    """/query should return a blocked response when guardrails fire."""
    original_key = settings.RAG_API_KEY
    settings.RAG_API_KEY = "test-secret"
    
    try:
        with TestClient(app) as client:
            with patch("app.main.guard") as mock_guard:
                mock_guard.return_value = (True, "Blocked message")
            
                # MUST pass the Authorization header
                response = client.post(
                    "/query", 
                    json={"q": "bad prompt", "thread_id": "t2"},
                    headers={"Authorization": "Bearer test-secret"}
                )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "Blocked by guardrails."
        assert data["answer"] == "Blocked message"
        
    finally:
        settings.RAG_API_KEY = original_key