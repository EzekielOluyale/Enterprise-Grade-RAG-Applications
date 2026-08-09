import pytest
from prometheus_client import REGISTRY


@pytest.fixture(autouse=True, scope="session")
def cleanup_prometheus_registry():
    """
    Automatically clear Prometheus metrics before tests run
    so we don't get 'Duplicated timeseries' errors.
    """
    # Use set() to get unique collectors and prevent duplicate unregistering
    collectors = list(set(REGISTRY._names_to_collectors.values()))

    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except KeyError:
            # Already unregistered, safe to ignore
            pass

    yield
