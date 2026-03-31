import pytest


@pytest.fixture(scope="session")
def browser_type_launch_args():
    """Pass extra args to the browser launch (e.g. slow_mo for easier debugging)."""
    return {"slow_mo": 200}  # 200 ms between actions; set to 0 for CI
