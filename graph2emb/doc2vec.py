"""
Doc2Vec implementation using Paragraph Vector algorithm.
Compatible with gensim's Doc2Vec API.
"""
import numpy as np
import random
from collections import Counter, defaultdict
from typing import List, Optional, Dict, Any, Union
from .keyedvectors import KeyedVectors


class TaggedDocument:
    """Document with tags (for Doc2Vec)."""

    def __init__(self, words: List[str], tags: List[str]):
        """
        Initialize TaggedDocument.

        Args:
            words: List of word tokens.
            tags: List of document tags.
        """
        self.words = words
        self.tags = tags


class Doc2VecKeyedVectors(KeyedVectors):
    """KeyedVectors for document embeddings."""

    pass


class Doc2Vec:
    """
    Doc2Vec model using Paragraph Vector algorithm.
    Compatible with gensim's Doc2Vec interface.
    """

    def __init__(
        self,
        documents: Optional[List[TaggedDocument]] = None,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 5,
        workers: int = 3,
        dm: int = 1,  # 1 for PV-DM, 0 for PV-DBOW
        dbow_words: int = 0,  # If 1, train word vectors in DBOW mode
        dm_mean: int = 0,  # If 1, use mean of context vectors
        dm_concat: int = 0,  # If 1, concatenate context vectors
        dm_tag_count: int = 1,  # Number of document tags per document
        negative: int = 5,
        ns_exponent: float = 0.75,
        alpha: float = 0.025,
        min_alpha: float = 0.0001,
        seed: int = 1,
        max_vocab_size: Optional[int] = None,
        sample: float = 0.001,
        epochs: int = 10,
        **kwargs,
    ):
        """
        Initialize Doc2Vec model.

        Args:
            documents: List of TaggedDocument objects.
            vector_size: Dimensionality of vectors.
            window: Maximum distance between current and predicted word.
            min_count: Minimum count of words to be included.
            workers: Number of worker threads.
            dm: Defines training algorithm: 1 for PV-DM, 0 for PV-DBOW.
            dbow_words: If 1, train word vectors in DBOW mode.
            dm_mean: If 1, use mean of context vectors.
            dm_concat: If 1, concatenate context vectors.
            dm_tag_count: Number of document tags per document.
            negative: Number of negative samples.
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
        self.dm = dm
        self.dbow_words = dbow_words
        self.dm_mean = dm_mean
        self.dm_concat = dm_concat
        self.dm_tag_count = dm_tag_count
        self.negative = negative
        self.ns_exponent = ns_exponent
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.seed = seed
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.epochs = epochs

        # Vocabulary
        self.wv: Optional[KeyedVectors] = None  # Word vectors
        self.dv: Optional[Doc2VecKeyedVectors] = None  # Document vectors
        self.vocab: Dict[str, Dict[str, Any]] = {}
        self.index2word: List[str] = []
        self.word2index: Dict[str, int] = {}
        self.doc2index: Dict[str, int] = {}
        self.index2doc: List[str] = []

        # Model weights
        self.syn1neg: Optional[np.ndarray] = None  # Negative sampling weights

        if documents is not None:
            self.build_vocab(documents)
            self.train(documents)

    def build_vocab(self, documents: List[TaggedDocument], update: bool = False):
        """Build vocabulary from documents."""
        if not update:
            self.vocab = {}
            self.index2word = []
            self.word2index = {}
            self.doc2index = {}
            self.index2doc = []

        # Count word frequencies
        word_counts = Counter()
        total_words = 0
        for doc in documents:
            for word in doc.words:
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
                (np.sqrt(count / (self.sample * total_words)) + 1) * (self.sample * total_words) / count
                if self.sample > 0
                else 1.0
            )
            prob = min(prob, 1.0)

            self.vocab[word] = {"count": count, "index": idx, "sample_prob": prob}

        # Build document index
        for doc in documents:
            for tag in doc.tags:
                if tag not in self.doc2index:
                    self.doc2index[tag] = len(self.index2doc)
                    self.index2doc.append(tag)

        # Build negative sampling table
        if self.negative > 0 and len(self.vocab) > 0:
            self._build_negative_table()

    def _build_negative_table(self):
        """Build table for negative sampling."""
        vocab_size = len(self.vocab)
        if vocab_size == 0:
            return

        # Compute unnormalized probabilities
        pow_counts = np.zeros(vocab_size, dtype=np.float64)
        for word, info in self.vocab.items():
            pow_counts[info["index"]] = info["count"] ** self.ns_exponent

        # Normalize
        total = pow_counts.sum()
        if total > 0:
            pow_counts = pow_counts / total

        # Build cumulative distribution for sampling
        self.negative_table_size = int(1e8)
        self.negative_table = np.zeros(self.negative_table_size, dtype=np.uint32)

        cumulative = 0.0
        idx = 0
        for i, prob in enumerate(pow_counts):
            cumulative += prob
            while idx < self.negative_table_size and idx / self.negative_table_size < cumulative:
                self.negative_table[idx] = i
                idx += 1

        while idx < self.negative_table_size:
            self.negative_table[idx] = vocab_size - 1
            idx += 1

    def _get_negative_samples(self, target_idx: int, num_samples: int) -> List[int]:
        """Get negative samples (excluding target)."""
        if not hasattr(self, "negative_table") or self.negative_table is None:
            # Fallback: random sampling from vocabulary
            vocab_size = len(self.vocab)
            samples = []
            while len(samples) < num_samples:
                neg_idx = random.randint(0, vocab_size - 1)
                if neg_idx != target_idx and neg_idx not in samples:
                    samples.append(neg_idx)
            return samples

        samples = []
        while len(samples) < num_samples:
            table_idx = random.randint(0, self.negative_table_size - 1)
            neg_idx = int(self.negative_table[table_idx])
            if neg_idx != target_idx and neg_idx not in samples:
                samples.append(neg_idx)
        return samples

    def train(
        self,
        documents: List[TaggedDocument],
        total_examples: Optional[int] = None,
        total_words: Optional[int] = None,
        epochs: Optional[int] = None,
        start_alpha: Optional[float] = None,
        end_alpha: Optional[float] = None,
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
        doc_size = len(self.index2doc)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Initialize word vectors
        word_vectors = np.random.uniform(-0.5 / self.vector_size, 0.5 / self.vector_size, (vocab_size, self.vector_size)).astype(
            np.float32
        )

        # Initialize document vectors
        doc_vectors = np.random.uniform(-0.5 / self.vector_size, 0.5 / self.vector_size, (doc_size, self.vector_size)).astype(
            np.float32
        )

        # Initialize negative sampling weights
        if self.negative > 0:
            self.syn1neg = np.zeros((vocab_size, self.vector_size), dtype=np.float32)

        # Count total words for learning rate scheduling
        if total_words is None:
            total_words = sum(len(doc.words) for doc in documents) * epochs

        # Training
        word_count_actual = 0
        alpha = start_alpha

        for epoch in range(epochs):
            # Shuffle documents
            shuffled_docs = documents.copy()
            random.shuffle(shuffled_docs)

            for doc in shuffled_docs:
                # Get document indices
                doc_indices = []
                for tag in doc.tags:
                    if tag in self.doc2index:
                        doc_indices.append(self.doc2index[tag])

                if not doc_indices:
                    continue

                # Filter words to vocabulary
                word_indices = []
                for word in doc.words:
                    if word in self.vocab:
                        if random.random() > self.vocab[word]["sample_prob"]:
                            continue
                        word_indices.append(self.vocab[word]["index"])

                if not word_indices:
                    continue

                # Train
                if self.dm == 1:
                    # PV-DM mode
                    self._train_dm(word_indices, doc_indices, alpha, word_vectors, doc_vectors)
                else:
                    # PV-DBOW mode
                    self._train_dbow(word_indices, doc_indices, alpha, word_vectors, doc_vectors)

                # Update learning rate
                word_count_actual += len(word_indices)
                if total_words > 0:
                    progress = word_count_actual / total_words
                    alpha = start_alpha - (start_alpha - end_alpha) * progress
                    alpha = max(alpha, end_alpha)

        # Create KeyedVectors
        self.wv = KeyedVectors(self.vector_size)
        word_keys = self.index2word
        word_weights = [word_vectors[i] for i in range(vocab_size)]
        self.wv.add_vectors(word_keys, word_weights)

        self.dv = Doc2VecKeyedVectors(self.vector_size)
        doc_keys = self.index2doc
        doc_weights = [doc_vectors[i] for i in range(doc_size)]
        self.dv.add_vectors(doc_keys, doc_weights)

    def _train_dm(self, word_indices: List[int], doc_indices: List[int], alpha: float, word_vectors: np.ndarray, doc_vectors: np.ndarray):
        """Train in PV-DM mode."""
        for pos, word_idx in enumerate(word_indices):
            # Get context window
            start = max(0, pos - self.window)
            end = min(len(word_indices), pos + self.window + 1)
            context_indices = [word_indices[i] for i in range(start, end) if i != pos]

            if not context_indices:
                continue

            # Build input vector (document + context words)
            if self.dm_mean:
                # Mean of document and context
                input_vec = doc_vectors[doc_indices[0]].copy()
                for ctx_idx in context_indices:
                    input_vec += word_vectors[ctx_idx]
                input_vec /= (1 + len(context_indices))
            elif self.dm_concat:
                # Concatenate (would need larger vector, simplified here)
                input_vec = doc_vectors[doc_indices[0]].copy()
                for ctx_idx in context_indices[: self.window]:
                    input_vec += word_vectors[ctx_idx]
            else:
                # Sum
                input_vec = doc_vectors[doc_indices[0]].copy()
                for ctx_idx in context_indices:
                    input_vec += word_vectors[ctx_idx]

            # Normalize
            norm = np.linalg.norm(input_vec)
            if norm > 0:
                input_vec = input_vec / norm

            # Train on target word
            self._train_pair_dm(word_idx, input_vec, 1, alpha, word_vectors)

            # Negative samples
            if self.negative > 0:
                neg_samples = self._get_negative_samples(word_idx, self.negative)
                for neg_idx in neg_samples:
                    self._train_pair_dm(neg_idx, input_vec, 0, alpha, word_vectors)

    def _train_dbow(self, word_indices: List[int], doc_indices: List[int], alpha: float, word_vectors: np.ndarray, doc_vectors: np.ndarray):
        """Train in PV-DBOW mode."""
        doc_vec = doc_vectors[doc_indices[0]]

        for word_idx in word_indices:
            # Train document -> word
            self._train_pair_dbow(word_idx, doc_vec, 1, alpha, word_vectors, doc_vectors, doc_indices[0])

            # Negative samples
            if self.negative > 0:
                neg_samples = self._get_negative_samples(word_idx, self.negative)
                for neg_idx in neg_samples:
                    self._train_pair_dbow(neg_idx, doc_vec, 0, alpha, word_vectors, doc_vectors, doc_indices[0])

    def _train_pair_dm(self, word_idx: int, input_vec: np.ndarray, label: int, alpha: float, word_vectors: np.ndarray):
        """Train PV-DM pair."""
        word_vec = word_vectors[word_idx]
        dot = np.dot(input_vec, word_vec)

        if label == 1:
            g = (1 - self._sigmoid(dot)) * alpha
        else:
            g = -self._sigmoid(dot) * alpha

        word_vectors[word_idx] += g * input_vec
        input_vec += g * word_vec  # Update input vector (backprop)

    def _train_pair_dbow(
        self,
        word_idx: int,
        doc_vec: np.ndarray,
        label: int,
        alpha: float,
        word_vectors: np.ndarray,
        doc_vectors: np.ndarray,
        doc_idx: int,
    ):
        """Train PV-DBOW pair."""
        word_vec = word_vectors[word_idx]
        dot = np.dot(doc_vec, word_vec)

        if label == 1:
            g = (1 - self._sigmoid(dot)) * alpha
        else:
            g = -self._sigmoid(dot) * alpha

        if self.dbow_words:
            word_vectors[word_idx] += g * doc_vec
        doc_vectors[doc_idx] += g * word_vec

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid function with clipping."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -250, 250)))

    def infer_vector(
        self,
        doc_words: List[str],
        alpha: Optional[float] = None,
        min_alpha: Optional[float] = None,
        epochs: Optional[int] = None,
    ) -> np.ndarray:
        """
        Infer vector for a new document.

        Args:
            doc_words: List of words in the document.
            alpha: Learning rate.
            min_alpha: Minimum learning rate.
            epochs: Number of training epochs.

        Returns:
            Document vector.
        """
        if self.dv is None:
            raise ValueError("Model not trained yet")

        alpha = alpha or self.alpha
        min_alpha = min_alpha or self.min_alpha
        epochs = epochs or self.epochs

        # Filter words to vocabulary
        word_indices = []
        for word in doc_words:
            if word in self.vocab:
                word_indices.append(self.vocab[word]["index"])

        if not word_indices:
            # Return random vector if no words in vocabulary
            return np.random.uniform(-0.5 / self.vector_size, 0.5 / self.vector_size, self.vector_size).astype(np.float32)

        # Initialize document vector
        doc_vec = np.random.uniform(-0.5 / self.vector_size, 0.5 / self.vector_size, self.vector_size).astype(np.float32)

        # Train document vector
        current_alpha = alpha
        for epoch in range(epochs):
            random.shuffle(word_indices)

            for word_idx in word_indices:
                if self.dm == 1:
                    # PV-DM: use context
                    # Simplified: just use the word
                    word_vec = self.wv[self.wv.index_to_key[word_idx]]
                    dot = np.dot(doc_vec, word_vec)
                    g = (1 - self._sigmoid(dot)) * current_alpha
                    doc_vec += g * word_vec
                else:
                    # PV-DBOW
                    word_vec = self.wv[self.wv.index_to_key[word_idx]]
                    dot = np.dot(doc_vec, word_vec)
                    g = (1 - self._sigmoid(dot)) * current_alpha
                    doc_vec += g * word_vec

            # Decay learning rate
            current_alpha = alpha - (alpha - min_alpha) * (epoch + 1) / epochs

        return doc_vec


