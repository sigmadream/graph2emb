"""
Tests for Graph2Vec basic functionality.
"""
import pytest
import networkx as nx
import numpy as np
from node2vec import Graph2Vec
from node2vec.utils import WeisfeilerLehmanHashing


class TestGraph2Vec:
    """Test cases for Graph2Vec class."""
    
    def test_basic_initialization(self):
        """Test basic Graph2Vec initialization."""
        graph2vec = Graph2Vec(
            wl_iterations=2,
            dimensions=64,
            workers=1,
            seed=42
        )
        
        assert graph2vec.wl_iterations == 2
        assert graph2vec.dimensions == 64
        assert graph2vec.workers == 1
        assert graph2vec.seed == 42
        assert graph2vec.model is None
        assert graph2vec._embedding is None
    
    def test_default_parameters(self):
        """Test Graph2Vec with default parameters."""
        graph2vec = Graph2Vec()
        
        assert graph2vec.wl_iterations == 2
        assert graph2vec.dimensions == 128
        assert graph2vec.workers == 4
        assert graph2vec.down_sampling == 0.0001
        assert graph2vec.epochs == 10
        assert graph2vec.learning_rate == 0.025
        assert graph2vec.min_count == 5
        assert graph2vec.seed == 42
        assert graph2vec.erase_base_features is False
        assert graph2vec.use_node_attribute is None
    
    def test_weisfeiler_lehman_hashing_degree(self):
        """Test WeisfeilerLehmanHashing with degree-based features."""
        graph = nx.fast_gnp_random_graph(n=10, p=0.5, seed=42)
        wl = WeisfeilerLehmanHashing(
            graph=graph,
            wl_iterations=2,
            use_node_attribute=None,
            erase_base_features=False
        )
        
        node_features = wl.get_node_features()
        graph_features = wl.get_graph_features()
        
        assert isinstance(node_features, dict)
        assert len(node_features) == graph.number_of_nodes()
        assert isinstance(graph_features, list)
        assert len(graph_features) > 0
        
        # Check that each node has features
        for node in graph.nodes():
            assert node in node_features
            assert len(node_features[node]) > 0
    
    def test_weisfeiler_lehman_hashing_node_attribute(self):
        """Test WeisfeilerLehmanHashing with node attributes."""
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
        
        # Add node attributes
        for node in graph.nodes():
            graph.nodes[node]['feature'] = node * 10
        
        wl = WeisfeilerLehmanHashing(
            graph=graph,
            wl_iterations=2,
            use_node_attribute='feature',
            erase_base_features=False
        )
        
        node_features = wl.get_node_features()
        graph_features = wl.get_graph_features()
        
        assert isinstance(node_features, dict)
        assert len(node_features) == graph.number_of_nodes()
        assert isinstance(graph_features, list)
        assert len(graph_features) > 0
    
    def test_weisfeiler_lehman_hashing_missing_attribute(self):
        """Test WeisfeilerLehmanHashing with missing node attributes."""
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2)])
        
        # Only add attribute to one node
        graph.nodes[0]['feature'] = 10
        
        with pytest.raises(ValueError, match="We expected for ALL graph nodes"):
            WeisfeilerLehmanHashing(
                graph=graph,
                wl_iterations=2,
                use_node_attribute='feature',
                erase_base_features=False
            )
    
    def test_weisfeiler_lehman_hashing_erase_base(self):
        """Test WeisfeilerLehmanHashing with erase_base_features."""
        graph = nx.fast_gnp_random_graph(n=10, p=0.5, seed=42)
        wl = WeisfeilerLehmanHashing(
            graph=graph,
            wl_iterations=2,
            use_node_attribute=None,
            erase_base_features=True
        )
        
        node_features = wl.get_node_features()
        # Base features should be erased, so each node should have wl_iterations features
        for node in graph.nodes():
            assert len(node_features[node]) == 2  # wl_iterations
    
    def test_fit_single_graph(self):
        """Test fitting Graph2Vec with a single graph."""
        graph = nx.fast_gnp_random_graph(n=10, p=0.5, seed=42)
        graph2vec = Graph2Vec(dimensions=32, workers=1, seed=42, min_count=1, epochs=2)
        graph2vec.fit([graph])
        
        embedding = graph2vec.get_embedding()
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, 32)
        assert graph2vec.model is not None
    
    def test_fit_multiple_graphs(self):
        """Test fitting Graph2Vec with multiple graphs."""
        graphs = [
            nx.fast_gnp_random_graph(n=8, p=0.5, seed=42),
            nx.fast_gnp_random_graph(n=10, p=0.5, seed=43),
            nx.fast_gnp_random_graph(n=8, p=0.5, seed=44),
        ]
        
        graph2vec = Graph2Vec(dimensions=32, workers=1, seed=42, min_count=1, epochs=2)
        graph2vec.fit(graphs)
        
        embedding = graph2vec.get_embedding()
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (3, 32)
        assert len(embedding) == len(graphs)
    
    def test_get_embedding_before_fit(self):
        """Test that get_embedding raises error before fit."""
        graph2vec = Graph2Vec()
        
        with pytest.raises(ValueError, match="Model has not been fitted"):
            graph2vec.get_embedding()
    
    def test_infer_before_fit(self):
        """Test that infer raises error before fit."""
        graph = nx.fast_gnp_random_graph(n=10, p=0.5, seed=42)
        graph2vec = Graph2Vec()
        
        with pytest.raises(ValueError, match="Model has not been fitted"):
            graph2vec.infer([graph])
    
    def test_infer_new_graphs(self):
        """Test inferring embeddings for new graphs."""
        # Train on some graphs
        train_graphs = [
            nx.fast_gnp_random_graph(n=10, p=0.5, seed=42),
            nx.fast_gnp_random_graph(n=10, p=0.5, seed=43),
        ]
        
        graph2vec = Graph2Vec(dimensions=32, workers=1, seed=42, min_count=1, epochs=2)
        graph2vec.fit(train_graphs)
        
        # Infer on new graphs
        test_graphs = [
            nx.fast_gnp_random_graph(n=8, p=0.5, seed=45),
        ]
        
        inferred_embedding = graph2vec.infer(test_graphs)
        
        assert isinstance(inferred_embedding, np.ndarray)
        assert inferred_embedding.shape == (1, 32)
    
    def test_different_wl_iterations(self):
        """Test Graph2Vec with different wl_iterations values."""
        graphs = [
            nx.fast_gnp_random_graph(n=10, p=0.5, seed=42),
            nx.fast_gnp_random_graph(n=10, p=0.5, seed=43),
        ]
        
        for wl_iter in [1, 2]:
            graph2vec = Graph2Vec(
                wl_iterations=wl_iter,
                dimensions=32,
                workers=1,
                seed=42,
                min_count=1,
                epochs=2
            )
            graph2vec.fit(graphs)
            embedding = graph2vec.get_embedding()
            assert embedding.shape == (2, 32)
    
    def test_use_node_attribute(self):
        """Test Graph2Vec with use_node_attribute parameter."""
        graphs = []
        for i in range(2):
            graph = nx.Graph()
            graph.add_edges_from([(0, 1), (1, 2), (2, 0)])
            for node in graph.nodes():
                graph.nodes[node]['attr'] = node * 10 + i
            graphs.append(graph)
        
        graph2vec = Graph2Vec(
            use_node_attribute='attr',
            dimensions=32,
            workers=1,
            seed=42,
            min_count=1,
            epochs=2
        )
        graph2vec.fit(graphs)
        
        embedding = graph2vec.get_embedding()
        assert embedding.shape == (2, 32)
    
    def test_seed_reproducibility(self):
        """Test that seed produces reproducible results."""
        graphs = [
            nx.fast_gnp_random_graph(n=8, p=0.5, seed=42),
            nx.fast_gnp_random_graph(n=8, p=0.5, seed=43),
        ]
        
        graph2vec1 = Graph2Vec(dimensions=32, workers=1, seed=123, min_count=1, epochs=3)
        graph2vec1.fit(graphs)
        embedding1 = graph2vec1.get_embedding()
        
        graph2vec2 = Graph2Vec(dimensions=32, workers=1, seed=123, min_count=1, epochs=3)
        graph2vec2.fit(graphs)
        embedding2 = graph2vec2.get_embedding()
        
        np.testing.assert_array_almost_equal(embedding1, embedding2)
    
    def test_erase_base_features(self):
        """Test Graph2Vec with erase_base_features parameter."""
        graphs = [
            nx.fast_gnp_random_graph(n=10, p=0.5, seed=42),
            nx.fast_gnp_random_graph(n=10, p=0.5, seed=43),
        ]
        
        graph2vec = Graph2Vec(
            erase_base_features=True,
            dimensions=32,
            workers=1,
            seed=42,
            min_count=1,
            epochs=2
        )
        graph2vec.fit(graphs)
        
        embedding = graph2vec.get_embedding()
        assert embedding.shape == (2, 32)
    
    def test_empty_graph_list(self):
        """Test Graph2Vec with empty graph list."""
        graph2vec = Graph2Vec(dimensions=32, workers=1, seed=42)
        graph2vec.fit([])
        
        embedding = graph2vec.get_embedding()
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (0, 32)
    
    def test_single_node_graph(self):
        """Test Graph2Vec with single node graphs."""
        graphs = [
            nx.Graph(),
            nx.Graph(),
        ]
        graphs[0].add_node(0)
        graphs[1].add_node(0)
        graphs[1].add_node(1)
        
        graph2vec = Graph2Vec(
            dimensions=32,
            min_count=1,  # Lower min_count for small graphs
            workers=1,
            seed=42
        )
        graph2vec.fit(graphs)
        
        embedding = graph2vec.get_embedding()
        assert embedding.shape == (2, 32)
    
    def test_mixed_size_graphs(self):
        """Test Graph2Vec with graphs of different sizes."""
        graphs = [
            nx.fast_gnp_random_graph(n=5, p=0.5, seed=42),
            nx.fast_gnp_random_graph(n=10, p=0.5, seed=43),
            nx.fast_gnp_random_graph(n=8, p=0.6, seed=44),
        ]
        
        graph2vec = Graph2Vec(
            dimensions=32,
            min_count=1,
            workers=1,
            seed=42,
            epochs=2
        )
        graph2vec.fit(graphs)
        
        embedding = graph2vec.get_embedding()
        assert embedding.shape == (3, 32)
    
    def test_string_node_names(self):
        """Test Graph2Vec with string node names."""
        graphs = []
        for i in range(2):
            graph = nx.Graph()
            graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])
            graphs.append(graph)
        
        graph2vec = Graph2Vec(
            dimensions=32,
            min_count=1,
            workers=1,
            seed=42,
            epochs=2
        )
        graph2vec.fit(graphs)
        
        embedding = graph2vec.get_embedding()
        assert embedding.shape == (2, 32)
    
    def test_custom_parameters(self):
        """Test Graph2Vec with various custom parameters."""
        graphs = [
            nx.fast_gnp_random_graph(n=10, p=0.5, seed=42),
            nx.fast_gnp_random_graph(n=10, p=0.5, seed=43),
        ]
        
        graph2vec = Graph2Vec(
            wl_iterations=2,
            dimensions=32,
            workers=1,
            down_sampling=0.001,
            epochs=2,
            learning_rate=0.01,
            min_count=1,
            seed=100,
            erase_base_features=True
        )
        graph2vec.fit(graphs)
        
        embedding = graph2vec.get_embedding()
        assert embedding.shape == (2, 32)
        assert graph2vec.wl_iterations == 2
        assert graph2vec.dimensions == 32
        assert graph2vec.epochs == 2

