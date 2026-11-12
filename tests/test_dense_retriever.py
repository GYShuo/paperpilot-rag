from typing import ClassVar

import numpy as np
import pytest

from paper_rag.retrieval.dense_retriever import DenseRetriever
from paper_rag.schemas import SectionChunk


class FakeEmbedder:
    """A deterministic fake embedder for unit tests."""

    vectors: ClassVar[dict[str, list[float]]] = {
        "A": [1.0, 0.0],
        "B": [0.0, 1.0],
        "C": [-1.0, 0.0],
    }

    def encode_documents(
        self,
        texts,
        show_progress_bar=False,
    ):
        return np.array(
            [self.vectors[text] for text in texts],
            dtype=np.float32,
        )

    def encode_query(self, query):
        return np.array([1.0, 0.0], dtype=np.float32)


def make_chunk(text: str) -> SectionChunk:
    return SectionChunk(
        chunk_id=f"test:{text}",
        paper_id="test-paper",
        split="train",
        section_index=0,
        section_name="Test",
        text=text,
    )


def make_retriever():
    return DenseRetriever(FakeEmbedder())


def test_build_index():
    retriever = make_retriever()

    chunks = [
        make_chunk("A"),
        make_chunk("B"),
        make_chunk("C"),
    ]

    retriever.build_index(chunks)

    assert len(retriever.chunks) == 3
    assert retriever.embeddings.shape == (3, 2)


def test_search_ranking():
    retriever = make_retriever()

    chunks = [
        make_chunk("A"),
        make_chunk("B"),
        make_chunk("C"),
    ]

    retriever.build_index(chunks)

    results = retriever.search("test query", top_k=3)

    assert [result.chunk.text for result in results] == [
        "A",
        "B",
        "C",
    ]

    np.testing.assert_allclose(
        [result.score for result in results],
        [1.0, 0.0, -1.0],
        atol=1e-6,
    )


def test_search_rank_numbers():
    retriever = make_retriever()

    retriever.build_index(
        [
            make_chunk("A"),
            make_chunk("B"),
            make_chunk("C"),
        ]
    )

    results = retriever.search("test query", top_k=2)

    assert [result.rank for result in results] == [1, 2]


def test_top_k_larger_than_corpus():
    retriever = make_retriever()

    retriever.build_index(
        [
            make_chunk("A"),
            make_chunk("B"),
        ]
    )

    results = retriever.search("test query", top_k=10)

    assert len(results) == 2


def test_search_before_build_index():
    retriever = make_retriever()

    with pytest.raises(RuntimeError):
        retriever.search("test query")


def test_empty_index():
    retriever = make_retriever()

    with pytest.raises(ValueError):
        retriever.build_index([])


def test_invalid_top_k():
    retriever = make_retriever()

    retriever.build_index([make_chunk("A")])

    with pytest.raises(ValueError):
        retriever.search("test query", top_k=0)
