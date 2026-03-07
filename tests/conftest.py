import pytest


@pytest.fixture
def sample_input():
    return {
        "cpu_usage": 80,
        "ram_usage": 70,
        "disk_io": 50,
        "network_traffic": 100,
        "temperature": 60,
        "cpu_spike_count": 3,
        "ram_spike_count": 2,
        "uptime_hours": 200,
    }
