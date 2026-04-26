"""
Word2Vec implementation using Skip-gram with Negative Sampling.
Compatible with gensim's Word2Vec API.
"""
import numpy as np
import random
from collections import defaultdict, Counter
from typing import List, Optional, Dict, Any
from joblib import Parallel, delayed
from .keyedvectors import KeyedVectors


class Word2Vec:
    """
    Word2Vec model using Skip-gram with Negative Sampling.
    Compatible with gensim's Word2Vec interface.
    """

    def __init__(
        self,
        sentences: Optional[List[List[str]]] = None,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 5,
        workers: int = 3,
        sg: int = 1,  # 1 for skip-gram, 0 for CBOW
        hs: int = 0,  # 0 for negative sampling, 1 for hierarchical softmax
        negative: int = 5,  # number of negative samples
        ns_exponent: float = 0.75,
        alpha: float = 0.025,
        min_alpha: float = 0.0001,
        seed: int = 1,
        max_vocab_size: Optional[int] = None,
        sample: float = 0.001,
        epochs: int = 5,
        **kwargs,
    ):
        """
        Initialize Word2Vec model.

        Args:
            sentences: List of tokenized sentences.
            vector_size: Dimensionality of word vectors.
            window: Maximum distance between current and predicted word.
            min_count: Minimum count of words to be included in vocabulary.
            workers: Number of worker threads.
            sg: Training algorithm: 1 for skip-gram, 0 for CBOW.
            hs: If 1, use hierarchical softmax; if 0, use negative sampling.
            negative: Number of negative samples (only for negative sampling).
            ns_exponent: Exponent for negative sampling distribution.
            alpha: Initial learning rate.
            min_alpha: Final learning rate.
            seed: Random seed.
            max_vocab_size: Maximum vocabulary size.
            sample: Threshold for downsampling frequent words.
            epochs: Number of training epochs.
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.hs = hs
        self.negative = negative if hs == 0 else 0
        self.ns_exponent = ns_exponent
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.seed = seed
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.epochs = epochs

        # Vocabulary
        self.wv: Optional[KeyedVectors] = None
        self.vocab: Dict[str, Dict[str, Any]] = {}
        self.index2word: List[str] = []
        self.word2index: Dict[str, int] = {}

        # Model weights
        self.syn1neg: Optional[np.ndarray] = None  # Negative sampling weights

        if sentences is not None:
            self.build_vocab(sentences)
            self.train(sentences)

    def build_vocab(self, sentences: List[List[str]], update: bool = False):
        """Build vocabulary from sentences."""
        if not update:
            self.vocab = {}
            self.index2word = []
            self.word2index = {}

        # Count word frequencies
        word_counts = Counter()
        total_words = 0
        for sentence in sentences:
            for word in sentence:
                word_counts[word] += 1
                total_words += 1

        # Filter by min_count
        filtered_counts = {word: count for word, count in word_counts.items() if count >= self.min_count}

        # Sort by frequency (descending)
        sorted_words = sorted(filtered_counts.items(), key=lambda x: -x[1])

        # Limit vocabulary size if needed
        if self.max_vocab_size:
            sorted_words = sorted_words[: self.max_vocab_size]

        # Build vocabulary
        for idx, (word, count) in enumerate(sorted_words):
            self.word2index[word] = idx
            self.index2word.append(word)

            # Compute subsampling probability
            prob = (
                (np.sqrt(count / (self.sample * total_words)) + 1) * (self.sample * total_words) / count if self.sample > 0 else 1.0
            )
            prob = min(prob, 1.0)

            self.vocab[word] = {"count": count, "index": idx, "sample_prob": prob}

        # Build negative sampling table
        if self.negative > 0 and len(self.vocab) > 0:
            self._build_negative_table()

    def _build_negative_table(self):
        """Build table for negative sampling."""
        # Power law distribution
        vocab_size = len(self.vocab)
        if vocab_size == 0:
            return

        # Compute unnormalized probabilities
        pow_counts = np.zeros(vocab_size, dtype=np.float64)
        for word, info in self.vocab.items():
            pow_counts[info["index"]] = info["count"] ** self.ns_exponent

        # Normalize
        total = pow_counts.sum()
        if total > 0 and np.isfinite(total):
            pow_counts = pow_counts / total
        else:
            pow_counts = np.full(vocab_size, 1.0 / vocab_size, dtype=np.float64)

        # Store the cumulative probability distribution instead of expanding it
        # into a fixed 1e8-entry table. This keeps memory proportional to the
        # vocabulary size while preserving the same sampling distribution.
        self.negative_table_size = vocab_size
        self.negative_table = np.cumsum(pow_counts, dtype=np.float64)
        self.negative_table[-1] = 1.0

    def _get_negative_samples(self, target_idx: int, num_samples: int) -> List[int]:
        """Get negative samples (excluding target)."""
        vocab_size = len(self.vocab)
        if num_samples <= 0 or vocab_size <= 1:
            return []

        if hasattr(self, "negative_table") and self.negative_table is not None:
            samples = []
            attempts = 0
            while len(samples) < num_samples and attempts < 10:
                draws = np.searchsorted(
                    self.negative_table,
                    np.random.random(size=max(num_samples, 4)),
                    side="right",
                )
                samples.extend(int(idx) for idx in draws if idx != target_idx and idx < vocab_size)
                attempts += 1

            if len(samples) >= num_samples:
                return samples[:num_samples]

            probabilities = np.diff(np.concatenate(([0.0], self.negative_table)))
        else:
            probabilities = np.ones(vocab_size, dtype=np.float64)

        probabilities[target_idx] = 0.0
        total = probabilities.sum()
        if total <= 0 or not np.isfinite(total):
            probabilities = np.ones(vocab_size, dtype=np.float64)
            probabilities[target_idx] = 0.0
            total = probabilities.sum()

        probabilities /= total
        return np.random.choice(vocab_size, size=num_samples, replace=True, p=probabilities).astype(int).tolist()

    def train(
        self,
        sentences: List[List[str]],
        total_examples: Optional[int] = None,
        total_words: Optional[int] = None,
        epochs: Optional[int] = None,
        start_alpha: Optional[float] = None,
        end_alpha: Optional[float] = None,
        word_count: Optional[int] = None,
        queue_factor: int = 2,
        report_delay: float = 1.0,
        compute_loss: bool = False,
        **kwargs,
    ):
        """Train the model."""
        if len(self.vocab) == 0:
            raise ValueError("Vocabulary is empty. Call build_vocab first.")

        epochs = epochs or self.epochs
        start_alpha = start_alpha or self.alpha
        end_alpha = end_alpha or self.min_alpha

        # Initialize weights
        vocab_size = len(self.vocab)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Initialize word vectors (uniform random in [-0.5/vector_size, 0.5/vector_size])
        word_vectors = np.random.uniform(-0.5 / self.vector_size, 0.5 / self.vector_size, (vocab_size, self.vector_size)).astype(
            np.float32
        )

        # Initialize negative sampling weights
        if self.negative > 0:
            self.syn1neg = np.zeros((vocab_size, self.vector_size), dtype=np.float32)

        # Count total words for learning rate scheduling
        if total_words is None:
            total_words = sum(len(s) for s in sentences) * epochs

        # Training
        word_count_actual = 0
        alpha = start_alpha

        for epoch in range(epochs):
            for sentence in sentences:
                # Filter sentence to vocabulary
                sentence_indices = []
                for word in sentence:
                    if word in self.vocab:
                        # Apply subsampling
                        if random.random() > self.vocab[word]["sample_prob"]:
                            continue
                        sentence_indices.append(self.vocab[word]["index"])

                if len(sentence_indices) < 2:
                    continue

                # Train on each word
                for pos, word_idx in enumerate(sentence_indices):
                    # Get context window
                    start = max(0, pos - self.window)
                    end = min(len(sentence_indices), pos + self.window + 1)
                    context_indices = [sentence_indices[i] for i in range(start, end) if i != pos]

                    if not context_indices:
                        continue

                    # Update for each context word
                    for context_idx in context_indices:
                        # Positive sample
                        self._train_pair(word_idx, context_idx, 1, alpha, word_vectors)

                        # Negative samples
                        if self.negative > 0:
                            neg_samples = self._get_negative_samples(word_idx, self.negative)
                            for neg_idx in neg_samples:
                                self._train_pair(word_idx, neg_idx, 0, alpha, word_vectors)

                    # Update learning rate
                    word_count_actual += 1
                    if total_words > 0:
                        progress = word_count_actual / total_words
                        alpha = start_alpha - (start_alpha - end_alpha) * progress
                        alpha = max(alpha, end_alpha)

        # Create KeyedVectors
        self.wv = KeyedVectors(self.vector_size)
        keys = self.index2word
        weights = [word_vectors[i] for i in range(vocab_size)]
        self.wv.add_vectors(keys, weights)

    def _train_pair(self, word_idx: int, context_idx: int, label: int, alpha: float, word_vectors: np.ndarray):
        """Train on a single word-context pair."""
        if self.negative == 0 and label == 0:
            return

        # Get vectors
        word_vec = word_vectors[word_idx]
        if label == 0:
            context_vec = self.syn1neg[context_idx]
        else:
            context_vec = word_vectors[context_idx]

        # Compute dot product
        dot = np.dot(word_vec, context_vec)

        # Compute gradient
        if label == 1:
            # Positive sample
            g = (1 - self._sigmoid(dot)) * alpha
        else:
            # Negative sample
            g = -self._sigmoid(dot) * alpha

        # Update vectors
        word_vectors[word_idx] += g * context_vec
        if label == 0:
            self.syn1neg[context_idx] += g * word_vec
        else:
            word_vectors[context_idx] += g * word_vec

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid function with clipping for numerical stability."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -250, 250)))

    def save(self, fname: str):
        """Save model (simplified - just save vectors)."""
        if self.wv is None:
            raise ValueError("Model not trained yet")
        self.wv.save_word2vec_format(fname, binary=False)

    @classmethod
    def load(cls, fname: str) -> "Word2Vec":
        """Load model (simplified - just load vectors)."""
        wv = KeyedVectors.load_word2vec_format(fname, binary=False)
        model = cls.__new__(cls)
        model.wv = wv
        model.vector_size = wv.vector_size
        model.index2word = wv.index_to_key
        model.word2index = {word: idx for idx, word in enumerate(model.index2word)}
        return model
