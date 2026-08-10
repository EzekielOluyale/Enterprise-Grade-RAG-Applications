from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver
from prometheus_client import REGISTRY

# IMPORTANT: Update this import path to point to your actual FastAPI 'app' instance
from app.main import app


@pytest.fixture(autouse=True, scope="session")
def cleanup_prometheus_registry():
    """
    Automatically clear Prometheus metrics before tests run
    so we don't get 'Duplicated timeseries' errors.
    """
    collectors = list(set(REGISTRY._names_to_collectors.values()))

    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except KeyError:
            pass

    yield


@pytest.fixture(autouse=True)
def mock_postgres_and_checkpointer():
    """
    Prevents psycopg_pool from attempting real DB connections to 'dummy.db'
    and swaps LangGraph's PostgresSaver for MemorySaver.
    """
    mock_pool = MagicMock()
    mock_conn = AsyncMock()

    # Mock async context manager for connection acquisition
    mock_pool.connection.return_value.__aenter__.return_value = mock_conn
    mock_pool.connection.return_value.__aexit__.return_value = None

    # Instantiate LangGraph's MemorySaver for test execution
    memory_checkpointer = MemorySaver()
    # Mock the setup() method as async so it doesn't crash when `await checkpointer.setup()` is called
    memory_checkpointer.setup = AsyncMock()

    # Patch the connection pools and LangGraph's PostgresSaver globally
    # Note: If this fails to mock, you might need to patch "app.main.AsyncPostgresSaver" instead
    with (
        patch("psycopg_pool.AsyncConnectionPool", return_value=mock_pool),
        patch("psycopg_pool.ConnectionPool", return_value=mock_pool),
        patch("langgraph.checkpoint.postgres.aio.AsyncPostgresSaver", return_value=memory_checkpointer),
    ):
        yield


@pytest.fixture
def client():
    """
    FastAPI TestClient fixture. Uses the mocked pool on app.state.
    """
    mock_pool = MagicMock()
    mock_conn = AsyncMock()
    mock_pool.connection.return_value.__aenter__.return_value = mock_conn
    mock_pool.connection.return_value.__aexit__.return_value = None

    app.state.pool = mock_pool

    with TestClient(app) as test_client:
        yield test_client
