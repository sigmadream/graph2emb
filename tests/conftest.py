"""
Pytest configuration and fixtures.
"""
import sys
import pytest


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "windows: marks tests that may not work on Windows"
    )


@pytest.fixture(scope="session")
def is_windows():
    """Check if running on Windows."""
    return sys.platform == "win32"

