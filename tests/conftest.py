import pytest
from unittest.mock import AsyncMock
from contextlib import asynccontextmanager
from prometheus_client import REGISTRY
from fastapi.testclient import TestClient

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

@pytest.fixture(autouse=True)
def bypass_lifespan():
    """
    AUTOUSE=TRUE is the secret here.
    This forces FastAPI to use the fake lifespan for EVERY single test automatically.
    Even if a test manually calls `with TestClient(app):`, it will use this safe lifespan.
    """
    @asynccontextmanager
    async def test_lifespan(app_instance):
        # Inject the mock pool safely
        app_instance.state.pool = AsyncMock()
        yield

    # Swap the production lifespan with our dummy lifespan
    original_lifespan = app.router.lifespan_context
    app.router.lifespan_context = test_lifespan
    
    yield
    
    # Restore the original lifespan after the test completes
    app.router.lifespan_context = original_lifespan

@pytest.fixture
def client():
    """
    FastAPI TestClient fixture. 
    Tests can use this, but even if they don't, `bypass_lifespan` protects them.
    """
    with TestClient(app) as test_client:
        yield test_client