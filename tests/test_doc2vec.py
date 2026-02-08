"""
Tests for Doc2Vec implementation.
"""
import numpy as np
import pytest

from graph2emb.doc2vec import Doc2Vec, TaggedDocument, Doc2VecKeyedVectors


class TestTaggedDocument:
    """Test cases for TaggedDocument."""

    def test_creation(self):
        doc = TaggedDocument(words=["a", "b", "c"], tags=["0"])
        assert doc.words == ["a", "b", "c"]
        assert doc.tags == ["0"]


class TestDoc2Vec:
    """Test cases for Doc2Vec class."""

    @pytest.fixture()
    def sample_documents(self):
        return [
            TaggedDocument(words=["cat", "sat", "mat", "cat", "mat"], tags=["0"]),
            TaggedDocument(words=["dog", "sat", "rug", "dog", "rug"], tags=["1"]),
            TaggedDocument(words=["cat", "chased", "dog", "cat", "dog"], tags=["2"]),
            TaggedDocument(words=["dog", "chased", "cat", "dog", "cat"], tags=["3"]),
        ] * 5

    def test_basic_training(self, sample_documents):
        model = Doc2Vec(sample_documents, vector_size=16, min_count=1, epochs=3, seed=42, workers=1)

        assert model.wv is not None
        assert model.dv is not None
        assert isinstance(model.dv, Doc2VecKeyedVectors)

    def test_document_vectors(self, sample_documents):
        model = Doc2Vec(sample_documents, vector_size=16, min_count=1, epochs=3, seed=42, workers=1)

        assert "0" in model.dv
        vec = model.dv["0"]
        assert vec.shape == (16,)

    def test_build_vocab(self, sample_documents):
        model = Doc2Vec(vector_size=16, min_count=1, seed=42)
        model.build_vocab(sample_documents)

        assert len(model.vocab) > 0
        assert len(model.index2doc) > 0

    def test_train_without_vocab_raises(self):
        model = Doc2Vec(vector_size=8, seed=42)
        with pytest.raises(ValueError, match="Vocabulary is empty"):
            model.train([TaggedDocument(words=["a"], tags=["0"])])

    def test_pv_dm_mode(self, sample_documents):
        model = Doc2Vec(sample_documents, vector_size=16, dm=1, min_count=1, epochs=2, seed=42, workers=1)
        assert model.dm == 1
        assert model.dv is not None

    def test_pv_dbow_mode(self, sample_documents):
        model = Doc2Vec(sample_documents, vector_size=16, dm=0, min_count=1, epochs=2, seed=42, workers=1)
        assert model.dm == 0
        assert model.dv is not None

    def test_infer_vector(self, sample_documents):
        model = Doc2Vec(sample_documents, vector_size=16, min_count=1, epochs=3, seed=42, workers=1)

        inferred = model.infer_vector(["cat", "sat", "mat"], epochs=3)
        assert inferred.shape == (16,)
        assert inferred.dtype == np.float32

    def test_infer_vector_before_train_raises(self):
        model = Doc2Vec(vector_size=8, seed=42)
        with pytest.raises(ValueError, match="not trained"):
            model.infer_vector(["a", "b"])

    def test_infer_vector_unknown_words(self, sample_documents):
        model = Doc2Vec(sample_documents, vector_size=16, min_count=1, epochs=2, seed=42, workers=1)

        # 어휘에 없는 단어로 추론 — 랜덤 벡터 반환
        inferred = model.infer_vector(["unknown_xyz", "nonexistent_abc"])
        assert inferred.shape == (16,)

    def test_seed_reproducibility(self, sample_documents):
        m1 = Doc2Vec(sample_documents, vector_size=16, min_count=1, epochs=2, seed=99, workers=1)
        m2 = Doc2Vec(sample_documents, vector_size=16, min_count=1, epochs=2, seed=99, workers=1)

        np.testing.assert_array_equal(m1.dv["0"], m2.dv["0"])

    def test_min_count_filtering(self):
        docs = [
            TaggedDocument(words=["rare", "common", "common"], tags=["0"]),
            TaggedDocument(words=["common", "common", "common"], tags=["1"]),
        ]
        model = Doc2Vec(docs, vector_size=8, min_count=3, epochs=1, seed=42, workers=1)

        assert "common" in model.vocab
        assert "rare" not in model.vocab
