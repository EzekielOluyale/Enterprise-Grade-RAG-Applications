from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

# IMPORTANT: Update this import path to point to your actual FastAPI 'app'
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


@pytest.fixture
def client():
    """
    FastAPI TestClient fixture that completely bypasses the production lifespan.
    This safely injects mocked dependencies into app.state WITHOUT letting
    the real lifespan overwrite them.
    """

    # Create a dummy lifespan
    @asynccontextmanager
    async def test_lifespan(app):
        # Safely inject the mock pool here, during the startup phase
        app.state.pool = AsyncMock()
        # (If you need a mock checkpointer on state, add it here too)
        yield
        # Teardown phase (if needed)

    # Swap the production lifespan with our dummy lifespan
    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = test_lifespan

    try:
        # Yield the test client. It will now run the dummy lifespan above!
        with TestClient(app) as test_client:
            yield test_client
    finally:
        # Restore the original lifespan
        app.router.lifespan_context = original_lifespan
