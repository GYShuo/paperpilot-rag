from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from paper_rag.retrieval.embedder import TextEmbedder
from paper_rag.retrieval.similarity import cosine_similarity
from paper_rag.retrieval.windowed_index_store import WindowedIndexStore
from paper_rag.schemas import EmbeddingWindow, SentenceTokenChunk


@dataclass(frozen=True)
class WindowedRetrievalResult:
    """A retrieved parent chunk with its best matching window."""

    rank: int
    score: float
    chunk: SentenceTokenChunk
    matched_window: EmbeddingWindow


class WindowedDenseRetriever:
    """Dense retriever with window-level search and parent-level results."""

    def __init__(self, embedder: TextEmbedder):
        self.embedder = embedder

        self.chunks: list[SentenceTokenChunk] = []
        self.windows: list[EmbeddingWindow] = []
        self.embeddings: np.ndarray | None = None

        self._chunks_by_id: dict[str, SentenceTokenChunk] = {}

    @staticmethod
    def _validate_chunks_and_windows(
        chunks: list[SentenceTokenChunk],
        windows: list[EmbeddingWindow],
    ) -> dict[str, SentenceTokenChunk]:
        """Validate parent-window mappings and return a parent lookup."""

        if not chunks:
            raise ValueError("Cannot build index from empty chunks.")

        if not windows:
            raise ValueError("Cannot build index from empty windows.")

        chunks_by_id = {}

        for chunk in chunks:
            if chunk.chunk_id in chunks_by_id:
                raise ValueError(f"Duplicate chunk ID: {chunk.chunk_id}")

            if not chunk.text.strip():
                raise ValueError(f"Empty parent chunk: {chunk.chunk_id}")

            chunks_by_id[chunk.chunk_id] = chunk

        window_ids = set()
        windows_by_parent: dict[str, list[EmbeddingWindow]] = {}

        for window in windows:
            if window.window_id in window_ids:
                raise ValueError(f"Duplicate window ID: {window.window_id}")

            window_ids.add(window.window_id)

            parent = chunks_by_id.get(window.parent_chunk_id)

            if parent is None:
                raise ValueError(f"Unknown parent chunk: {window.parent_chunk_id}")

            if (
                window.paper_id != parent.paper_id
                or window.split != parent.split
                or window.section_index != parent.section_index
                or window.section_name != parent.section_name
            ):
                raise ValueError(f"Metadata mismatch for window {window.window_id}.")

            if not 0 <= window.start_char < window.end_char <= len(parent.text):
                raise ValueError(f"Invalid character offsets for {window.window_id}.")

            if parent.text[window.start_char : window.end_char] != window.text:
                raise ValueError(f"Window text mismatch for {window.window_id}.")

            if (
                window.token_start < 0
                or window.token_end <= window.token_start
                or window.token_count <= 0
            ):
                raise ValueError(f"Invalid token metadata for {window.window_id}.")

            if not window.text.strip():
                raise ValueError(f"Empty window: {window.window_id}")

            windows_by_parent.setdefault(
                window.parent_chunk_id,
                [],
            ).append(window)

        if set(windows_by_parent) != set(chunks_by_id):
            raise ValueError("Every parent chunk must have at least one window.")

        for chunk_id, parent_windows in windows_by_parent.items():
            parent_windows.sort(key=lambda item: item.window_index)

            indices = [window.window_index for window in parent_windows]

            if indices != list(range(len(parent_windows))):
                raise ValueError(f"Non-sequential window indices for {chunk_id}.")

            first = parent_windows[0]

            if first.token_start != 0 or first.start_char != 0:
                raise ValueError(
                    f"Window coverage does not start at zero for {chunk_id}."
                )

            for previous, current in pairwise(parent_windows):
                if current.token_start > previous.token_end:
                    raise ValueError(f"Token coverage gap detected for {chunk_id}.")

                if current.token_end <= previous.token_end:
                    raise ValueError(
                        f"Windows do not make forward progress for {chunk_id}."
                    )

                if (
                    current.start_char < previous.start_char
                    or current.end_char <= previous.end_char
                ):
                    raise ValueError(f"Invalid character ordering for {chunk_id}.")

            if parent_windows[-1].end_char != len(chunks_by_id[chunk_id].text):
                raise ValueError(
                    f"Window coverage does not reach the end of {chunk_id}."
                )

        return chunks_by_id

    def _validate_embeddings(
        self,
        embeddings: np.ndarray,
        expected_count: int,
    ) -> np.ndarray:
        """Validate embedding matrix."""

        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if vectors.shape[0] != expected_count:
            raise ValueError("Window count and embedding count do not match.")

        if vectors.shape[1] == 0:
            raise ValueError("Embedding dimension cannot be zero.")

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain invalid values.")

        expected_dimension = getattr(self.embedder, "dimension", None)

        if expected_dimension is not None and vectors.shape[1] != expected_dimension:
            raise ValueError("Embedding dimension does not match model.")

        return vectors

    def build_index(
        self,
        chunks: list[SentenceTokenChunk],
        windows: list[EmbeddingWindow],
        batch_size: int = 32,
        show_progress_bar: bool = True,
    ) -> None:
        """Build an in-memory index from parent chunks and windows."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self._validate_chunks_and_windows(chunks, windows)

        encode_windows = getattr(
            self.embedder,
            "encode_windows",
            None,
        )

        if not callable(encode_windows):
            raise TypeError("Embedder must implement encode_windows().")

        embeddings = encode_windows(
            windows,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

        self.load_index(
            chunks=chunks,
            windows=windows,
            embeddings=embeddings,
        )

    def load_index(
        self,
        chunks: list[SentenceTokenChunk],
        windows: list[EmbeddingWindow],
        embeddings: np.ndarray,
    ) -> None:
        """Restore an existing in-memory windowed index."""

        chunks_by_id = self._validate_chunks_and_windows(
            chunks,
            windows,
        )

        vectors = self._validate_embeddings(
            embeddings,
            expected_count=len(windows),
        )

        # Update only after all validation succeeds.
        self.chunks = list(chunks)
        self.windows = list(windows)
        self.embeddings = vectors
        self._chunks_by_id = chunks_by_id

    def load_from_store(
        self,
        store: WindowedIndexStore,
        expected_model_name: str | None = None,
        show_progress_bar: bool = True,
    ) -> dict:
        """Load a persisted index directly from WindowedIndexStore."""

        chunks, windows, embeddings, manifest = store.load(
            expected_model_name=expected_model_name,
            show_progress_bar=show_progress_bar,
        )

        self.load_index(
            chunks=chunks,
            windows=windows,
            embeddings=embeddings,
        )

        return manifest

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[WindowedRetrievalResult]:
        """Search windows and return unique parent chunks.

        Each parent chunk is ranked by the maximum similarity
        among all its embedding windows.
        """

        if self.embeddings is None:
            raise RuntimeError("Please build or load the index before searching.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_embedding = np.asarray(
            self.embedder.encode_query(query),
            dtype=np.float32,
        )

        if query_embedding.ndim != 1:
            raise ValueError("Query embedding must be 1D.")

        if query_embedding.shape[0] != self.embeddings.shape[1]:
            raise ValueError("Query and document embedding dimensions differ.")

        if not np.isfinite(query_embedding).all():
            raise ValueError("Query embedding contains invalid values.")

        scores = cosine_similarity(
            query_embedding,
            self.embeddings,
        )

        # Sort all windows by descending similarity.
        # Stable sorting preserves the original order for ties.
        indices = np.argsort(
            -scores,
            kind="stable",
        )

        results: list[WindowedRetrievalResult] = []
        seen_parent_ids = set()

        for index in indices:
            window_index = int(index)
            window = self.windows[window_index]

            parent_id = window.parent_chunk_id

            # The first occurrence is the highest-scoring
            # window belonging to this parent chunk.
            if parent_id in seen_parent_ids:
                continue

            seen_parent_ids.add(parent_id)

            results.append(
                WindowedRetrievalResult(
                    rank=len(results) + 1,
                    score=float(scores[window_index]),
                    chunk=self._chunks_by_id[parent_id],
                    matched_window=window,
                )
            )

            if len(results) >= top_k:
                break

        return results
