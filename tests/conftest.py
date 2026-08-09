import pytest
from prometheus_client import REGISTRY


@pytest.fixture(autouse=True, scope="session")
def cleanup_prometheus_registry():
    """
    Automatically clear Prometheus metrics before tests run
    so we don't get 'Duplicated timeseries' errors.
    """
    # Unregister everything in the global registry before tests
    collectors = list(REGISTRY._names_to_collectors.values())
    for collector in collectors:
        REGISTRY.unregister(collector)

    yield
