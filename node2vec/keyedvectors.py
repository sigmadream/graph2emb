"""
KeyedVectors implementation for storing and querying word embeddings.
Compatible with gensim's KeyedVectors API.
"""
import numpy as np
from typing import List, Tuple, Optional, Union
from collections import OrderedDict


class KeyedVectors:
    """
    Stores embeddings for keys (words/nodes) and provides similarity queries.
    Compatible with gensim's KeyedVectors interface.
    """
    
    def __init__(self, vector_size: int):
        """
        Initialize KeyedVectors.
        
        Args:
            vector_size: Dimensionality of the vectors.
        """
        self.vector_size = vector_size
        self.index_to_key: List[str] = []
        self.key_to_index: OrderedDict[str, int] = OrderedDict()
        self.vectors: Optional[np.ndarray] = None
        self._norm: Optional[np.ndarray] = None  # Cached L2 norms
    
    def add_vectors(self, keys: List[str], weights: List[np.ndarray]):
        """
        Add vectors for keys.
        
        Args:
            keys: List of key strings.
            weights: List of numpy arrays (vectors).
        """
        if len(keys) != len(weights):
            raise ValueError("keys and weights must have the same length")
        
        vectors_list = []
        for key, weight in zip(keys, weights):
            if key in self.key_to_index:
                # Update existing vector
                idx = self.key_to_index[key]
                vectors_list.append(weight)
            else:
                # Add new key
                self.key_to_index[key] = len(self.index_to_key)
                self.index_to_key.append(key)
                vectors_list.append(weight)
        
        # Update vectors array
        if vectors_list:
            new_vectors = np.array(vectors_list, dtype=np.float32)
            if self.vectors is None:
                self.vectors = new_vectors
            else:
                # Resize if needed
                if len(self.index_to_key) > len(self.vectors):
                    # Pad with zeros for new vectors
                    old_size = len(self.vectors)
                    new_size = len(self.index_to_key)
                    padded = np.zeros((new_size, self.vector_size), dtype=np.float32)
                    padded[:old_size] = self.vectors
                    self.vectors = padded
                # Update vectors
                for i, key in enumerate(keys):
                    idx = self.key_to_index[key]
                    self.vectors[idx] = new_vectors[i]
            
            # Reset cached norms
            self._norm = None
    
    def __getitem__(self, key: str) -> np.ndarray:
        """Get vector for a key."""
        if key not in self.key_to_index:
            raise KeyError(f"'{key}' not in vocabulary")
        idx = self.key_to_index[key]
        return self.vectors[idx].copy()
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self.key_to_index
    
    def __len__(self) -> int:
        """Return number of keys."""
        return len(self.index_to_key)
    
    def _ensure_norm(self):
        """Ensure L2 norms are computed and cached."""
        if self._norm is None and self.vectors is not None:
            self._norm = np.linalg.norm(self.vectors, axis=1)
            # Avoid division by zero
            self._norm[self._norm == 0] = 1.0
    
    def most_similar(self, positive: Optional[Union[str, List[str]]] = None,
                     negative: Optional[List[str]] = None,
                     topn: int = 10,
                     restrict_vocab: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Find the most similar keys.
        
        Args:
            positive: Key or list of keys to add.
            negative: List of keys to subtract.
            topn: Number of results to return.
            restrict_vocab: Optional limit on vocabulary size.
            
        Returns:
            List of (key, similarity) tuples.
        """
        if self.vectors is None or len(self.vectors) == 0:
            return []
        
        # Compute query vector
        if positive is None:
            raise ValueError("must specify at least one positive key")
        
        if isinstance(positive, str):
            positive = [positive]
        
        query_vec = np.zeros(self.vector_size, dtype=np.float32)
        
        # Add positive keys
        for key in positive:
            if key not in self.key_to_index:
                raise KeyError(f"'{key}' not in vocabulary")
            query_vec += self[key]
        
        # Subtract negative keys
        if negative:
            for key in negative:
                if key not in self.key_to_index:
                    raise KeyError(f"'{key}' not in vocabulary")
                query_vec -= self[key]
        
        # Normalize query vector
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        query_vec = query_vec / query_norm
        
        # Compute similarities
        self._ensure_norm()
        vocab_size = len(self.vectors)
        if restrict_vocab:
            vocab_size = min(vocab_size, restrict_vocab)
        
        # Exclude positive and negative keys from results
        exclude_indices = set()
        for key in positive:
            if key in self.key_to_index:
                exclude_indices.add(self.key_to_index[key])
        if negative:
            for key in negative:
                if key in self.key_to_index:
                    exclude_indices.add(self.key_to_index[key])
        
        # Compute cosine similarities
        similarities = np.dot(self.vectors[:vocab_size], query_vec) / self._norm[:vocab_size]
        
        # Set excluded keys to -inf
        for idx in exclude_indices:
            if idx < vocab_size:
                similarities[idx] = -np.inf
        
        # Get topn most similar
        top_indices = np.argsort(similarities)[::-1][:topn]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > -np.inf:
                results.append((self.index_to_key[idx], float(similarities[idx])))
        
        return results
    
    def save_word2vec_format(self, fname: str, binary: bool = False):
        """
        Save vectors in word2vec format.
        
        Args:
            fname: Filename to save to.
            binary: If True, save in binary format.
        """
        if self.vectors is None:
            raise ValueError("No vectors to save")
        
        with open(fname, 'wb' if binary else 'w', encoding='utf-8' if not binary else None) as f:
            # Write header
            header = f"{len(self.index_to_key)} {self.vector_size}\n"
            if binary:
                f.write(header.encode('utf-8'))
            else:
                f.write(header)
            
            # Write vectors
            for i, key in enumerate(self.index_to_key):
                if binary:
                    # Binary format: key + space + vector as binary
                    key_bytes = key.encode('utf-8') + b' '
                    f.write(key_bytes)
                    self.vectors[i].astype(np.float32).tofile(f)
                    f.write(b'\n')
                else:
                    # Text format: key + space + space-separated vector values
                    vector_str = ' '.join(str(x) for x in self.vectors[i])
                    f.write(f"{key} {vector_str}\n")
    
    @classmethod
    def load_word2vec_format(cls, fname: str, binary: bool = False,
                             encoding: str = 'utf-8', unicode_errors: str = 'strict',
                             limit: Optional[int] = None) -> 'KeyedVectors':
        """
        Load vectors from word2vec format.
        
        Args:
            fname: Filename to load from.
            binary: If True, load from binary format.
            encoding: Text encoding.
            unicode_errors: How to handle unicode errors.
            limit: Optional limit on number of vectors to load.
            
        Returns:
            KeyedVectors instance.
        """
        if binary:
            return cls._load_word2vec_binary(fname, encoding, unicode_errors, limit)
        else:
            return cls._load_word2vec_text(fname, encoding, unicode_errors, limit)
    
    @classmethod
    def _load_word2vec_text(cls, fname: str, encoding: str, unicode_errors: str,
                            limit: Optional[int]) -> 'KeyedVectors':
        """Load from text format."""
        with open(fname, 'r', encoding=encoding, errors=unicode_errors) as f:
            # Read header
            header = f.readline().strip().split()
            if len(header) != 2:
                raise ValueError("Invalid word2vec format: header must have 2 values")
            vocab_size, vector_size = int(header[0]), int(header[1])
            
            kv = cls(vector_size)
            keys = []
            vectors = []
            
            count = 0
            for line in f:
                if limit and count >= limit:
                    break
                
                parts = line.rstrip().split(' ')
                if len(parts) < vector_size + 1:
                    continue
                
                key = parts[0]
                vector = np.array([float(x) for x in parts[1:vector_size+1]], dtype=np.float32)
                
                keys.append(key)
                vectors.append(vector)
                count += 1
            
            if keys:
                kv.add_vectors(keys, vectors)
            
            return kv
    
    @classmethod
    def _load_word2vec_binary(cls, fname: str, encoding: str, unicode_errors: str,
                               limit: Optional[int]) -> 'KeyedVectors':
        """Load from binary format."""
        with open(fname, 'rb') as f:
            # Read header
            header = f.readline().decode(encoding, errors=unicode_errors).strip().split()
            if len(header) != 2:
                raise ValueError("Invalid word2vec format: header must have 2 values")
            vocab_size, vector_size = int(header[0]), int(header[1])
            
            kv = cls(vector_size)
            keys = []
            vectors = []
            
            count = 0
            while count < vocab_size:
                if limit and count >= limit:
                    break
                
                # Read key (until space)
                key_bytes = b''
                while True:
                    char = f.read(1)
                    if not char or char == b' ':
                        break
                    key_bytes += char
                
                if not key_bytes:
                    break
                
                key = key_bytes.decode(encoding, errors=unicode_errors)
                
                # Read vector
                vector = np.fromfile(f, dtype=np.float32, count=vector_size)
                if len(vector) != vector_size:
                    break
                
                keys.append(key)
                vectors.append(vector)
                count += 1
                
                # Skip newline
                f.read(1)
            
            if keys:
                kv.add_vectors(keys, vectors)
            
            return kv

