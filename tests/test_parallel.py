"""
Tests for parallel processing functionality.
"""
import pytest
import networkx as nx
import numpy as np
from node2vec import Node2Vec


class TestParallelProcessing:
    """Test cases for parallel processing."""
    
    def test_parallel_probability_precomputation(self):
        """Test that parallel probability precomputation produces same results as sequential."""
        graph = nx.fast_gnp_random_graph(n=50, p=0.3, seed=42)
        
        # Sequential processing
        node2vec_seq = Node2Vec(graph, dimensions=16, walk_length=5, num_walks=2, workers=1, quiet=True)
        
        # Parallel processing
        node2vec_par = Node2Vec(graph, dimensions=16, walk_length=5, num_walks=2, workers=2, quiet=True)
        
        # Check that both produce walks
        assert len(node2vec_seq.walks) > 0
        assert len(node2vec_par.walks) > 0
        
        # Check that probabilities are computed
        assert len(node2vec_seq.d_graph) > 0
        assert len(node2vec_par.d_graph) > 0
        
        # Both should have same number of nodes with probabilities
        assert len(node2vec_seq.d_graph) == len(node2vec_par.d_graph)
    
    def test_parallel_walk_generation(self):
        """Test parallel walk generation."""
        graph = nx.fast_gnp_random_graph(n=40, p=0.4, seed=42)
        
        # Single worker
        node2vec_single = Node2Vec(graph, dimensions=16, walk_length=5, num_walks=4, workers=1, quiet=True)
        
        # Multiple workers
        node2vec_multi = Node2Vec(graph, dimensions=16, walk_length=5, num_walks=4, workers=2, quiet=True)
        
        # Both should produce same number of walks
        assert len(node2vec_single.walks) == len(node2vec_multi.walks)
        
        # Check walk structure
        for walk in node2vec_multi.walks:
            assert isinstance(walk, list)
            assert len(walk) <= 5
    
    def test_workers_parameter(self):
        """Test different worker counts."""
        graph = nx.fast_gnp_random_graph(n=30, p=0.4, seed=42)
        
        for workers in [1, 2]:
            node2vec = Node2Vec(
                graph, 
                dimensions=16, 
                walk_length=5, 
                num_walks=2, 
                workers=workers, 
                quiet=True
            )
            assert node2vec.workers == workers
            assert len(node2vec.walks) > 0
    
    def test_parallel_probability_structure(self):
        """Test that parallel processing maintains correct probability structure."""
        graph = nx.fast_gnp_random_graph(n=30, p=0.4, seed=42)
        
        node2vec = Node2Vec(graph, dimensions=16, walk_length=5, num_walks=2, workers=2, quiet=True)
        
        # Check probability structure
        for node, data in node2vec.d_graph.items():
            if node2vec.PROBABILITIES_KEY in data:
                for source, probs in data[node2vec.PROBABILITIES_KEY].items():
                    assert isinstance(probs, np.ndarray)
                    assert np.isclose(probs.sum(), 1.0, atol=1e-6)
                    assert np.all(probs >= 0)  # All probabilities should be non-negative
            
            if node2vec.FIRST_TRAVEL_KEY in data:
                first_travel = data[node2vec.FIRST_TRAVEL_KEY]
                assert isinstance(first_travel, np.ndarray)
                assert np.isclose(first_travel.sum(), 1.0, atol=1e-6)
                assert np.all(first_travel >= 0)
    
    def test_sequential_vs_parallel_consistency(self):
        """Test that sequential and parallel processing produce consistent results."""
        graph = nx.fast_gnp_random_graph(n=25, p=0.4, seed=42)
        
        # Sequential
        node2vec_seq = Node2Vec(graph, dimensions=16, walk_length=5, num_walks=2, workers=1, quiet=True, seed=123)
        
        # Parallel
        node2vec_par = Node2Vec(graph, dimensions=16, walk_length=5, num_walks=2, workers=2, quiet=True, seed=123)
        
        # Both should have same number of walks
        assert len(node2vec_seq.walks) == len(node2vec_par.walks)
        
        # Both should have same graph structure in d_graph
        assert len(node2vec_seq.d_graph) == len(node2vec_par.d_graph)
        
        # Check that probabilities are computed for same nodes
        seq_nodes = set(node2vec_seq.d_graph.keys())
        par_nodes = set(node2vec_par.d_graph.keys())
        assert seq_nodes == par_nodes

