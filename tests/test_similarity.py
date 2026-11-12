import numpy as np
import pytest

from paper_rag.retrieval.similarity import cosine_similarity


def test_cosine_similarity_basic():
    query = np.array([1.0, 0.0])

    documents = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ]
    )

    scores = cosine_similarity(query, documents)

    np.testing.assert_allclose(
        scores,
        [1.0, 0.0, -1.0],
        atol=1e-6,
    )


def test_cosine_similarity_output_shape():
    query = np.array([1.0, 0.0])

    documents = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )

    scores = cosine_similarity(query, documents)

    assert scores.shape == (3,)


def test_cosine_similarity_dimension_error():
    query = np.array([1.0, 0.0])

    documents = np.array(
        [
            [1.0, 0.0, 1.0],
        ]
    )

    with pytest.raises(ValueError):
        cosine_similarity(query, documents)


def test_cosine_similarity_zero_vector():
    query = np.array([0.0, 0.0])

    documents = np.array(
        [
            [1.0, 0.0],
        ]
    )

    with pytest.raises(ValueError):
        cosine_similarity(query, documents)


def test_cosine_similarity_invalid_shape():
    query = np.array([[1.0, 0.0]])

    documents = np.array(
        [
            [1.0, 0.0],
        ]
    )

    with pytest.raises(ValueError):
        cosine_similarity(query, documents)
