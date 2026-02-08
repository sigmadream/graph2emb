"""
Tests for KeyedVectors implementation.
"""
import os
import tempfile

import numpy as np
import pytest

from graph2emb.keyedvectors import KeyedVectors


class TestKeyedVectors:
    """Test cases for KeyedVectors class."""

    def test_init(self):
        kv = KeyedVectors(vector_size=16)
        assert kv.vector_size == 16
        assert len(kv) == 0
        assert kv.vectors is None

    def test_add_vectors(self):
        kv = KeyedVectors(vector_size=4)
        keys = ["a", "b", "c"]
        weights = [np.array([1, 2, 3, 4], dtype=np.float32) for _ in keys]
        kv.add_vectors(keys, weights)

        assert len(kv) == 3
        assert "a" in kv
        assert "z" not in kv
        np.testing.assert_array_equal(kv["a"], weights[0])

    def test_add_vectors_length_mismatch(self):
        kv = KeyedVectors(vector_size=4)
        with pytest.raises(ValueError, match="same length"):
            kv.add_vectors(["a"], [np.zeros(4), np.zeros(4)])

    def test_getitem_missing_key(self):
        kv = KeyedVectors(vector_size=4)
        kv.add_vectors(["a"], [np.zeros(4)])
        with pytest.raises(KeyError):
            kv["missing"]

    def test_getitem_returns_copy(self):
        kv = KeyedVectors(vector_size=4)
        vec = np.array([1, 2, 3, 4], dtype=np.float32)
        kv.add_vectors(["a"], [vec])

        result = kv["a"]
        result[0] = 999
        # 원본이 변경되지 않아야 함
        assert kv["a"][0] == 1.0

    def test_get_vector_norm(self):
        kv = KeyedVectors(vector_size=3)
        kv.add_vectors(["x"], [np.array([3, 0, 4], dtype=np.float32)])

        normed = kv.get_vector("x", norm=True)
        assert np.isclose(np.linalg.norm(normed), 1.0)

        raw = kv.get_vector("x", norm=False)
        np.testing.assert_array_equal(raw, [3, 0, 4])

    def test_most_similar(self):
        kv = KeyedVectors(vector_size=3)
        kv.add_vectors(
            ["a", "b", "c"],
            [
                np.array([1, 0, 0], dtype=np.float32),
                np.array([0.9, 0.1, 0], dtype=np.float32),
                np.array([0, 0, 1], dtype=np.float32),
            ],
        )

        results = kv.most_similar("a", topn=2)
        assert len(results) == 2
        # b가 a에 가장 가까워야 함
        assert results[0][0] == "b"

    def test_most_similar_with_negative(self):
        kv = KeyedVectors(vector_size=3)
        kv.add_vectors(
            ["a", "b", "c"],
            [
                np.array([1, 0, 0], dtype=np.float32),
                np.array([0.9, 0.1, 0], dtype=np.float32),
                np.array([0, 0, 1], dtype=np.float32),
            ],
        )

        results = kv.most_similar(positive=["a"], negative=["c"], topn=1)
        assert len(results) == 1
        assert results[0][0] == "b"

    def test_most_similar_no_positive(self):
        kv = KeyedVectors(vector_size=3)
        kv.add_vectors(["a"], [np.ones(3, dtype=np.float32)])
        with pytest.raises(ValueError, match="positive"):
            kv.most_similar(positive=None)

    def test_save_load_text_format(self, tmp_path):
        kv = KeyedVectors(vector_size=4)
        vecs = [np.random.rand(4).astype(np.float32) for _ in range(3)]
        kv.add_vectors(["foo", "bar", "baz"], vecs)

        fpath = str(tmp_path / "vectors.txt")
        kv.save_word2vec_format(fpath, binary=False)

        loaded = KeyedVectors.load_word2vec_format(fpath, binary=False)
        assert len(loaded) == 3
        for key, orig in zip(["foo", "bar", "baz"], vecs):
            np.testing.assert_array_almost_equal(loaded[key], orig, decimal=5)

    def test_save_load_binary_format(self, tmp_path):
        kv = KeyedVectors(vector_size=4)
        vecs = [np.random.rand(4).astype(np.float32) for _ in range(3)]
        kv.add_vectors(["foo", "bar", "baz"], vecs)

        fpath = str(tmp_path / "vectors.bin")
        kv.save_word2vec_format(fpath, binary=True)

        loaded = KeyedVectors.load_word2vec_format(fpath, binary=True)
        assert len(loaded) == 3
        for key, orig in zip(["foo", "bar", "baz"], vecs):
            np.testing.assert_array_almost_equal(loaded[key], orig, decimal=5)

    def test_update_existing_vector(self):
        kv = KeyedVectors(vector_size=2)
        kv.add_vectors(["a"], [np.array([1, 2], dtype=np.float32)])
        kv.add_vectors(["a"], [np.array([3, 4], dtype=np.float32)])

        assert len(kv) == 1
        np.testing.assert_array_equal(kv["a"], [3, 4])

    def test_restrict_vocab(self):
        kv = KeyedVectors(vector_size=2)
        kv.add_vectors(
            ["a", "b", "c"],
            [
                np.array([1, 0], dtype=np.float32),
                np.array([0.9, 0.1], dtype=np.float32),
                np.array([0, 1], dtype=np.float32),
            ],
        )
        results = kv.most_similar("a", topn=2, restrict_vocab=2)
        # restrict_vocab=2이면 a, b만 후보
        assert all(r[0] in ("b",) for r in results)
