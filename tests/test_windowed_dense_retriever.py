from typing import ClassVar

import numpy as np
import pytest

from paper_rag.retrieval.windowed_dense_retriever import (
    WindowedDenseRetriever,
)
from paper_rag.retrieval.windowed_index_store import WindowedIndexStore
from paper_rag.schemas import EmbeddingWindow, SentenceTokenChunk


class FakeEmbedder:
    """Deterministic fake embedder."""

    dimension = 2

    vectors: ClassVar[dict[str, list[float]]] = {
        "chunk-a:embedding:0": [0.8, 0.6],
        "chunk-a:embedding:1": [0.995, 0.1],
        "chunk-b:embedding:0": [0.9, 0.4358899],
        "chunk-c:embedding:0": [-1.0, 0.0],
    }

    def __init__(self):
        self.query_vector = np.array(
            [1.0, 0.0],
            dtype=np.float32,
        )

    def encode_windows(
        self,
        windows,
        batch_size=32,
        show_progress_bar=False,
    ):
        return np.array(
            [self.vectors[window.window_id] for window in windows],
            dtype=np.float32,
        )

    def encode_query(self, query):
        return self.query_vector


def make_chunk(chunk_id, text):
    return SentenceTokenChunk(
        chunk_id=chunk_id,
        paper_id="paper-1",
        split="train",
        section_index=0,
        section_name="Introduction",
        text=text,
        token_count=6,
        sentence_count=2,
    )


def make_window(
    chunk,
    window_index,
    start_char,
    end_char,
    token_start,
    token_end,
):
    return EmbeddingWindow(
        window_id=f"{chunk.chunk_id}:embedding:{window_index}",
        parent_chunk_id=chunk.chunk_id,
        paper_id=chunk.paper_id,
        split=chunk.split,
        section_index=chunk.section_index,
        section_name=chunk.section_name,
        window_index=window_index,
        text=chunk.text[start_char:end_char],
        start_char=start_char,
        end_char=end_char,
        token_start=token_start,
        token_end=token_end,
        token_count=8,
    )


def make_data():
    chunk_a = make_chunk("chunk-a", "abcdef")
    chunk_b = make_chunk("chunk-b", "ghijkl")
    chunk_c = make_chunk("chunk-c", "mnopqr")

    chunks = [
        chunk_a,
        chunk_b,
        chunk_c,
    ]

    windows = [
        make_window(chunk_a, 0, 0, 4, 0, 4),
        make_window(chunk_a, 1, 2, 6, 2, 6),
        make_window(chunk_b, 0, 0, 6, 0, 6),
        make_window(chunk_c, 0, 0, 6, 0, 6),
    ]

    embeddings = np.array(
        [
            [0.8, 0.6],
            [0.995, 0.1],
            [0.9, 0.4358899],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )

    return chunks, windows, embeddings


def make_retriever():
    return WindowedDenseRetriever(FakeEmbedder())


def test_build_index():
    retriever = make_retriever()

    chunks, windows, _ = make_data()

    retriever.build_index(
        chunks,
        windows,
        show_progress_bar=False,
    )

    assert len(retriever.chunks) == 3
    assert len(retriever.windows) == 4
    assert retriever.embeddings.shape == (4, 2)


def test_load_index():
    retriever = make_retriever()

    chunks, windows, embeddings = make_data()

    retriever.load_index(
        chunks,
        windows,
        embeddings,
    )

    assert len(retriever.chunks) == 3
    assert len(retriever.windows) == 4

    np.testing.assert_array_equal(
        retriever.embeddings,
        embeddings,
    )


def test_search_before_build_index():
    retriever = make_retriever()

    with pytest.raises(RuntimeError):
        retriever.search("test query")


def test_search_ranking():
    retriever = make_retriever()

    chunks, windows, embeddings = make_data()

    retriever.load_index(chunks, windows, embeddings)

    results = retriever.search("test query", top_k=3)

    assert [result.chunk.chunk_id for result in results] == [
        "chunk-a",
        "chunk-b",
        "chunk-c",
    ]

    assert [result.rank for result in results] == [1, 2, 3]


def test_search_collapses_duplicate_parent_chunks():
    retriever = make_retriever()

    chunks, windows, embeddings = make_data()

    retriever.load_index(chunks, windows, embeddings)

    results = retriever.search("test query", top_k=10)

    parent_ids = [result.chunk.chunk_id for result in results]

    assert len(parent_ids) == len(set(parent_ids))
    assert len(results) == 3


def test_search_uses_best_matching_window():
    retriever = make_retriever()

    chunks, windows, embeddings = make_data()

    retriever.load_index(chunks, windows, embeddings)

    results = retriever.search("test query", top_k=3)

    best_result = results[0]

    assert best_result.chunk.chunk_id == "chunk-a"
    assert best_result.matched_window.window_id == "chunk-a:embedding:1"

    assert best_result.score == pytest.approx(
        0.995 / np.linalg.norm([0.995, 0.1]),
        abs=1e-6,
    )


def test_search_rank_numbers():
    retriever = make_retriever()

    chunks, windows, embeddings = make_data()

    retriever.load_index(chunks, windows, embeddings)

    results = retriever.search("test query", top_k=2)

    assert [result.rank for result in results] == [1, 2]


def test_top_k_larger_than_corpus():
    retriever = make_retriever()

    chunks, windows, embeddings = make_data()

    retriever.load_index(chunks, windows, embeddings)

    results = retriever.search("test query", top_k=100)

    assert len(results) == 3


def test_invalid_top_k():
    retriever = make_retriever()

    chunks, windows, embeddings = make_data()

    retriever.load_index(chunks, windows, embeddings)

    with pytest.raises(ValueError, match="top_k"):
        retriever.search("test query", top_k=0)


def test_empty_chunks():
    retriever = make_retriever()

    _, windows, embeddings = make_data()

    with pytest.raises(ValueError, match="empty chunks"):
        retriever.load_index([], windows, embeddings)


def test_empty_windows():
    retriever = make_retriever()

    chunks, _, _ = make_data()

    with pytest.raises(ValueError, match="empty windows"):
        retriever.load_index(
            chunks,
            [],
            np.empty((0, 2), dtype=np.float32),
        )


def test_mismatched_embedding_count():
    retriever = make_retriever()

    chunks, windows, embeddings = make_data()

    with pytest.raises(ValueError, match="do not match"):
        retriever.load_index(
            chunks,
            windows,
            embeddings[:2],
        )


def test_invalid_embedding_dimension():
    retriever = make_retriever()

    chunks, windows, _ = make_data()

    with pytest.raises(ValueError, match="dimension"):
        retriever.load_index(
            chunks,
            windows,
            np.ones((4, 3), dtype=np.float32),
        )


def test_non_finite_embeddings():
    retriever = make_retriever()

    chunks, windows, embeddings = make_data()

    embeddings[0, 0] = np.nan

    with pytest.raises(ValueError, match="invalid values"):
        retriever.load_index(chunks, windows, embeddings)


def test_duplicate_chunk_ids():
    retriever = make_retriever()

    chunks, windows, embeddings = make_data()

    with pytest.raises(ValueError, match="Duplicate chunk ID"):
        retriever.load_index(
            [chunks[0], chunks[0].model_copy(), *chunks[1:]],
            windows,
            embeddings,
        )


def test_duplicate_window_ids():
    retriever = make_retriever()

    chunks, windows, embeddings = make_data()

    windows[1] = windows[1].model_copy(update={"window_id": windows[0].window_id})

    with pytest.raises(ValueError, match="Duplicate window ID"):
        retriever.load_index(chunks, windows, embeddings)


def test_unknown_parent_chunk():
    retriever = make_retriever()

    chunks, windows, embeddings = make_data()

    windows[0] = windows[0].model_copy(update={"parent_chunk_id": "unknown"})

    with pytest.raises(ValueError, match="Unknown parent"):
        retriever.load_index(chunks, windows, embeddings)


def test_window_text_mismatch():
    retriever = make_retriever()

    chunks, windows, embeddings = make_data()

    windows[0] = windows[0].model_copy(update={"text": "wrong"})

    with pytest.raises(ValueError, match="text mismatch"):
        retriever.load_index(chunks, windows, embeddings)


def test_window_metadata_mismatch():
    retriever = make_retriever()

    chunks, windows, embeddings = make_data()

    windows[0] = windows[0].model_copy(update={"paper_id": "another-paper"})

    with pytest.raises(ValueError, match="Metadata mismatch"):
        retriever.load_index(chunks, windows, embeddings)


def test_token_coverage_gap():
    retriever = make_retriever()

    chunks, windows, embeddings = make_data()

    windows[1] = windows[1].model_copy(update={"token_start": 5})

    with pytest.raises(ValueError, match="coverage gap"):
        retriever.load_index(chunks, windows, embeddings)


def test_non_sequential_window_indices():
    retriever = make_retriever()

    chunks, windows, embeddings = make_data()

    windows[1] = windows[1].model_copy(update={"window_index": 3})

    with pytest.raises(ValueError, match="Non-sequential"):
        retriever.load_index(chunks, windows, embeddings)


def test_parent_without_windows():
    retriever = make_retriever()

    chunks, windows, embeddings = make_data()

    with pytest.raises(ValueError, match="at least one window"):
        retriever.load_index(
            chunks,
            windows[:-1],
            embeddings[:-1],
        )


def test_invalid_query_dimension():
    embedder = FakeEmbedder()
    embedder.query_vector = np.array([1.0, 0.0, 0.0])

    retriever = WindowedDenseRetriever(embedder)

    chunks, windows, embeddings = make_data()

    retriever.load_index(chunks, windows, embeddings)

    with pytest.raises(ValueError, match="dimensions"):
        retriever.search("test query")


def test_non_finite_query_embedding():
    embedder = FakeEmbedder()
    embedder.query_vector = np.array(
        [np.nan, 0.0],
        dtype=np.float32,
    )

    retriever = WindowedDenseRetriever(embedder)

    chunks, windows, embeddings = make_data()

    retriever.load_index(chunks, windows, embeddings)

    with pytest.raises(ValueError, match="invalid values"):
        retriever.search("test query")


def test_load_from_store(tmp_path):
    chunks, windows, embeddings = make_data()

    store = WindowedIndexStore(tmp_path / "windowed")

    store.save(
        chunks=chunks,
        windows=windows,
        embeddings=embeddings,
        model_name="fake-model",
        max_sequence_length=10,
        overlap_tokens=2,
        show_progress_bar=False,
    )

    retriever = make_retriever()

    manifest = retriever.load_from_store(
        store,
        expected_model_name="fake-model",
        show_progress_bar=False,
    )

    assert manifest["index_type"] == "windowed"
    assert len(retriever.chunks) == 3
    assert len(retriever.windows) == 4

    results = retriever.search("test query", top_k=3)

    assert len(results) == 3
    assert results[0].chunk.chunk_id == "chunk-a"


def test_build_index_invalid_batch_size():
    retriever = make_retriever()

    chunks, windows, _ = make_data()

    with pytest.raises(ValueError, match="batch_size"):
        retriever.build_index(
            chunks,
            windows,
            batch_size=0,
            show_progress_bar=False,
        )
