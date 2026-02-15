import numpy as np
import pytest
from cosine_similarity import cosine_similarity, ecludian_norm


class TestCosineSimilarity:
    def test_identical_nonzero_vectors_returns_one(self):
        """cosine_similarity returns 1 for identical non-zero vectors"""
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 3])
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_orthogonal_vectors_returns_zero(self):
        """cosine_similarity returns 0 for orthogonal vectors"""
        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_direction_vectors_returns_negative_one(self):
        """cosine_similarity returns -1 for vectors in opposite directions"""
        a = np.array([1, 2, 3])
        b = np.array([-1, -2, -3])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        """cosine_similarity returns 0 when one or both vectors are zero"""
        zero = np.array([0, 0, 0])
        nonzero = np.array([1, 2, 3])

        assert cosine_similarity(zero, nonzero) == 0.0
        assert cosine_similarity(nonzero, zero) == 0.0
        assert cosine_similarity(zero, zero) == 0.0


class TestEuclideanNorm:
    def test_euclidean_norm_calculation(self):
        """ecludian_norm calculates the correct Euclidean norm"""
        a = np.array([3, 4])
        assert ecludian_norm(a) == pytest.approx(5.0)

        b = np.array([1, 2, 2])
        assert ecludian_norm(b) == pytest.approx(3.0)

        c = np.array([0, 0, 0])
        assert ecludian_norm(c) == pytest.approx(0.0)
