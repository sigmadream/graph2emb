"""
Tests for Word2Vec implementation.
"""
import numpy as np
import pytest

from graph2emb.word2vec import Word2Vec


class TestWord2Vec:
    """Test cases for Word2Vec class."""

    @pytest.fixture()
    def sample_sentences(self):
        return [
            ["the", "cat", "sat", "on", "the", "mat"],
            ["the", "dog", "sat", "on", "the", "rug"],
            ["the", "cat", "chased", "the", "dog"],
            ["the", "dog", "chased", "the", "cat"],
            ["cat", "and", "dog", "are", "friends"],
        ] * 10  # 반복해서 min_count 충족

    def test_basic_training(self, sample_sentences):
        model = Word2Vec(sample_sentences, vector_size=16, window=3, min_count=1, epochs=3, seed=42)

        assert model.wv is not None
        assert model.wv.vector_size == 16
        assert "cat" in model.wv
        assert "dog" in model.wv

    def test_build_vocab(self, sample_sentences):
        model = Word2Vec(vector_size=16, min_count=1, seed=42)
        model.build_vocab(sample_sentences)

        assert len(model.vocab) > 0
        assert "the" in model.vocab
        assert model.vocab["the"]["count"] > 0

    def test_min_count_filtering(self):
        sentences = [["rare_word", "common", "common"], ["common", "common", "common"]]
        model = Word2Vec(sentences, vector_size=8, min_count=3, epochs=1, seed=42)

        assert "common" in model.wv
        assert "rare_word" not in model.wv

    def test_train_without_vocab_raises(self):
        model = Word2Vec(vector_size=8, seed=42)
        with pytest.raises(ValueError, match="Vocabulary is empty"):
            model.train([["a", "b"]])

    def test_vector_dimensions(self, sample_sentences):
        model = Word2Vec(sample_sentences, vector_size=32, min_count=1, epochs=1, seed=42)
        vec = model.wv["cat"]
        assert vec.shape == (32,)

    def test_most_similar(self, sample_sentences):
        model = Word2Vec(sample_sentences, vector_size=16, window=3, min_count=1, epochs=5, seed=42)
        results = model.wv.most_similar("cat", topn=3)

        assert len(results) <= 3
        # 결과는 (word, similarity) 튜플
        for word, sim in results:
            assert isinstance(word, str)
            assert isinstance(sim, float)

    def test_seed_reproducibility(self, sample_sentences):
        m1 = Word2Vec(sample_sentences, vector_size=16, min_count=1, epochs=2, seed=99)
        m2 = Word2Vec(sample_sentences, vector_size=16, min_count=1, epochs=2, seed=99)

        np.testing.assert_array_equal(m1.wv["cat"], m2.wv["cat"])

    def test_save_and_load(self, sample_sentences, tmp_path):
        model = Word2Vec(sample_sentences, vector_size=16, min_count=1, epochs=1, seed=42)

        fpath = str(tmp_path / "w2v_model.txt")
        model.save(fpath)

        loaded = Word2Vec.load(fpath)
        assert loaded.wv is not None
        assert len(loaded.wv) == len(model.wv)
        np.testing.assert_array_almost_equal(loaded.wv["cat"], model.wv["cat"], decimal=5)

    def test_save_before_train_raises(self):
        model = Word2Vec(vector_size=8, seed=42)
        with pytest.raises(ValueError, match="not trained"):
            model.save("dummy.txt")

    def test_skipgram_default(self, sample_sentences):
        model = Word2Vec(sample_sentences, vector_size=8, min_count=1, epochs=1, sg=1, seed=42)
        assert model.sg == 1
        assert model.wv is not None
