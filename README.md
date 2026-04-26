# graph2emb

`graph2emb` is a lightweight Python package for learning graph embeddings. It combines Node2Vec-style random-walk node embeddings, edge embeddings, and Graph2Vec-style whole-graph embeddings in one project.

The implementation is inspired by:

- [Elior Cohen's node2vec](https://github.com/eliorc/node2vec)
- [graph2vec](https://github.com/benedekrozemberczki/graph2vec)

## Features

- Node2Vec: random-walk based node embeddings.
- Edge embeddings: Hadamard, average, weighted L1, and weighted L2 edge representations.
- Graph2Vec: Weisfeiler-Lehman hashing + Doc2Vec-based graph embeddings.
- Gensim-like APIs: small in-repo `Word2Vec`, `Doc2Vec`, and `KeyedVectors` implementations.
- uv-friendly workflow for development, testing, and running examples.

## Installation

From the project root:

```bash
uv add graph2emb
```

Or install the package in editable mode with pip:

```bash
pip install graph2emb
```

## Quick start

### Node embeddings with Node2Vec

```python
import networkx as nx
from graph2emb import Node2Vec

# Build a small graph.
graph = nx.fast_gnp_random_graph(n=100, p=0.3, seed=42)

# Precompute transition probabilities and generate random walks.
node2vec = Node2Vec(
    graph,
    dimensions=64,
    walk_length=30,
    num_walks=20,
    workers=1,
    seed=42,
    quiet=True,
)

# Train embeddings from the walks.
model = node2vec.fit(window=10, min_count=1, epochs=5)

# Node ids are stored as strings.
print(model.wv.most_similar("2", topn=5))
```

### Edge embeddings

```python
from graph2emb.edges import HadamardEmbedder, AverageEmbedder

# Reuse a trained Node2Vec model.
hadamard = HadamardEmbedder(model.wv, quiet=True)
average = AverageEmbedder(model.wv, quiet=True)

print(hadamard[("1", "2")])
print(average[("1", "2")])
```

### Graph embeddings with Graph2Vec

```python
import networkx as nx
from graph2emb import Graph2Vec

graphs = [
    nx.fast_gnp_random_graph(n=12, p=0.3, seed=1),
    nx.fast_gnp_random_graph(n=14, p=0.2, seed=2),
]

graph2vec = Graph2Vec(
    dimensions=32,
    wl_iterations=2,
    workers=1,
    min_count=1,
    epochs=3,
    seed=42,
)
graph2vec.fit(graphs)

embeddings = graph2vec.get_embedding()
print(embeddings.shape)  # (2, 32)
```

## Running the sample scripts

The `sample/` directory contains small runnable examples with intentionally small parameters.

```bash
# Node2Vec from an edge-list file
uv run python sample/node2vec_from_edgelist.py

# Node2Vec + edge embeddings
uv run python sample/node2vec_edge_embeddings.py

# Graph2Vec
uv run python sample/graph2vec_basic.py
```

Outputs are written to `sample/out/`.

## Notes

- Node labels are converted to strings in generated walks and learned `KeyedVectors`.
- Weighted Node2Vec graphs use the `weight` edge attribute by default. Override this with `weight_key="..."`.
- Edge weights, `p`, and `q` must be finite positive numbers.
- For `MultiGraph` inputs, parallel edge weights are validated and summed.

## Development

```bash
# Run tests
uv run pytest

# Run tests in parallel
uv run pytest -n auto

# Run coverage
uv run pytest --cov=graph2emb --cov-report=term-missing

# Build package artifacts
uv build
```

## License

MIT. See [LICENSE](LICENSE).
