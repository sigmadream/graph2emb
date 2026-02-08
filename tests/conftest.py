"""
Pytest configuration and fixtures.
"""
import sys
import pytest
import networkx as nx


@pytest.fixture(scope="session")
def is_windows():
    """Check if running on Windows."""
    return sys.platform == "win32"


@pytest.fixture(scope="session")
def small_graph():
    """Small reusable graph for fast unit tests."""
    return nx.fast_gnp_random_graph(n=10, p=0.4, seed=42)


@pytest.fixture(scope="session")
def medium_graph():
    """Medium reusable graph for tests that need a bit more structure."""
    return nx.fast_gnp_random_graph(n=20, p=0.3, seed=42)


@pytest.fixture(scope="session")
def default_node2vec_params():
    """Default fast params for Node2Vec tests."""
    return {
        "dimensions": 8,
        "walk_length": 3,
        "num_walks": 1,
        "workers": 1,
        "quiet": True,
        "seed": 42,
    }
