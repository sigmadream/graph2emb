"""
Tests for Node2Vec basic functionality.
"""
import pytest
import networkx as nx
import numpy as np
from node2vec import Node2Vec


class TestNode2Vec:
    """Test cases for Node2Vec class."""
    
    def test_basic_initialization(self):
        """Test basic Node2Vec initialization."""
        graph = nx.fast_gnp_random_graph(n=50, p=0.3, seed=42)
        node2vec = Node2Vec(graph, dimensions=32, walk_length=10, num_walks=5, workers=1, quiet=True)
        
        assert node2vec.graph == graph
        assert node2vec.dimensions == 32
        assert node2vec.walk_length == 10
        assert node2vec.num_walks == 5
        assert node2vec.workers == 1
        assert len(node2vec.walks) > 0
    
    def test_walks_generation(self):
        """Test that walks are generated correctly."""
        graph = nx.fast_gnp_random_graph(n=30, p=0.4, seed=42)
        node2vec = Node2Vec(graph, dimensions=16, walk_length=5, num_walks=3, workers=1, quiet=True)
        
        # Check that walks are generated
        assert len(node2vec.walks) > 0
        
        # Check walk structure
        for walk in node2vec.walks:
            assert isinstance(walk, list)
            assert len(walk) <= 5  # walk_length
            assert len(walk) > 0
            
            # All nodes in walk should be strings
            for node in walk:
                assert isinstance(node, str)
    
    def test_probability_precomputation(self):
        """Test that probabilities are precomputed correctly."""
        graph = nx.fast_gnp_random_graph(n=20, p=0.5, seed=42)
        node2vec = Node2Vec(graph, dimensions=16, walk_length=5, num_walks=2, workers=1, quiet=True)
        
        # Check that d_graph has probabilities
        assert len(node2vec.d_graph) > 0
        
        # Check that probabilities are normalized (sum to 1)
        for node, data in node2vec.d_graph.items():
            if node2vec.PROBABILITIES_KEY in data:
                for source, probs in data[node2vec.PROBABILITIES_KEY].items():
                    assert isinstance(probs, np.ndarray)
                    # Probabilities should sum to approximately 1
                    assert np.isclose(probs.sum(), 1.0, atol=1e-6)
            
            if node2vec.FIRST_TRAVEL_KEY in data:
                first_travel = data[node2vec.FIRST_TRAVEL_KEY]
                assert isinstance(first_travel, np.ndarray)
                assert np.isclose(first_travel.sum(), 1.0, atol=1e-6)
    
    def test_fit_model(self):
        """Test fitting a Word2Vec model."""
        graph = nx.fast_gnp_random_graph(n=30, p=0.4, seed=42)
        node2vec = Node2Vec(graph, dimensions=16, walk_length=5, num_walks=3, workers=1, quiet=True)
        
        model = node2vec.fit(window=5, min_count=1, batch_words=4)
        
        assert model is not None
        assert model.wv.vector_size == 16
        
        # Check that embeddings exist for nodes
        nodes = [str(n) for n in graph.nodes()]
        for node in nodes[:5]:  # Check first 5 nodes
            if node in model.wv:
                assert len(model.wv[node]) == 16
    
    def test_custom_parameters(self):
        """Test Node2Vec with custom p and q parameters."""
        graph = nx.fast_gnp_random_graph(n=25, p=0.4, seed=42)
        node2vec = Node2Vec(
            graph, 
            dimensions=16, 
            walk_length=5, 
            num_walks=2, 
            p=0.5, 
            q=2.0, 
            workers=1, 
            quiet=True
        )
        
        assert node2vec.p == 0.5
        assert node2vec.q == 2.0
        assert len(node2vec.walks) > 0
    
    def test_sampling_strategy(self):
        """Test Node2Vec with custom sampling strategy."""
        graph = nx.fast_gnp_random_graph(n=20, p=0.5, seed=42)
        
        # Create sampling strategy for first node
        first_node = list(graph.nodes())[0]
        sampling_strategy = {
            first_node: {
                'p': 0.5,
                'q': 2.0,
                'num_walks': 3,
                'walk_length': 7
            }
        }
        
        node2vec = Node2Vec(
            graph,
            dimensions=16,
            walk_length=5,
            num_walks=2,
            sampling_strategy=sampling_strategy,
            workers=1,
            quiet=True
        )
        
        assert node2vec.sampling_strategy == sampling_strategy
        assert len(node2vec.walks) > 0
    
    def test_seed_reproducibility(self):
        """Test that seed produces reproducible results."""
        graph = nx.fast_gnp_random_graph(n=20, p=0.5, seed=42)
        
        # Create two Node2Vec instances with same seed
        node2vec1 = Node2Vec(graph, dimensions=16, walk_length=5, num_walks=2, seed=123, workers=1, quiet=True)
        node2vec2 = Node2Vec(graph, dimensions=16, walk_length=5, num_walks=2, seed=123, workers=1, quiet=True)
        
        # With workers=1, results should be reproducible
        assert len(node2vec1.walks) == len(node2vec2.walks)
    
    def test_string_nodes(self):
        """Test Node2Vec with string node names."""
        graph = nx.Graph()
        graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')])
        
        node2vec = Node2Vec(graph, dimensions=16, walk_length=5, num_walks=2, workers=1, quiet=True)
        
        assert len(node2vec.walks) > 0
        # All nodes in walks should be strings
        for walk in node2vec.walks:
            for node in walk:
                assert isinstance(node, str)
                assert node in ['A', 'B', 'C', 'D']
    
    def test_weighted_graph(self):
        """Test Node2Vec with weighted graph."""
        graph = nx.Graph()
        graph.add_weighted_edges_from([
            (0, 1, 2.0),
            (1, 2, 1.5),
            (2, 3, 3.0),
            (3, 0, 1.0)
        ])
        
        node2vec = Node2Vec(graph, dimensions=16, walk_length=5, num_walks=2, workers=1, quiet=True)
        
        assert len(node2vec.walks) > 0
    
    def test_empty_graph_error(self):
        """Test that empty graph raises appropriate error."""
        graph = nx.Graph()
        
        # Empty graph should still work, but generate no walks
        node2vec = Node2Vec(graph, dimensions=16, walk_length=5, num_walks=2, workers=1, quiet=True)
        assert len(node2vec.walks) == 0

