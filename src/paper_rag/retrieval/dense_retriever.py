from dataclasses import dataclass

import numpy as np

from paper_rag.retrieval.embedder import TextEmbedder
from paper_rag.retrieval.similarity import cosine_similarity
from paper_rag.schemas import SectionChunk


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved chunk with its ranking score."""

    rank: int
    score: float
    chunk: SectionChunk


class DenseRetriever:
    """In-memory dense vector retriever."""

    def __init__(self, embedder: TextEmbedder):
        self.embedder = embedder

        self.chunks: list[SectionChunk] = []
        self.embeddings: np.ndarray | None = None

    def build_index(
        self,
        chunks: list[SectionChunk],
        show_progress_bar: bool = False,
    ) -> None:
        """Build an in-memory vector index from chunks."""

        if not chunks:
            raise ValueError("Cannot build index from empty chunks.")

        texts = [chunk.text for chunk in chunks]

        embeddings = self.embedder.encode_documents(
            texts,
            show_progress_bar=show_progress_bar,
        )

        if embeddings.ndim != 2:
            raise ValueError("Document embeddings must be 2D.")

        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of embeddings does not match chunks.")

        if embeddings.shape[1] == 0:
            raise ValueError("Embedding dimension cannot be zero.")

        # Update the index only after successful encoding.
        self.chunks = list(chunks)
        self.embeddings = embeddings

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve the top-k most similar chunks."""

        if self.embeddings is None:
            raise RuntimeError("Please build the index before searching.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_embedding = self.embedder.encode_query(query)

        scores = cosine_similarity(
            query_embedding,
            self.embeddings,
        )

        # Sort scores in descending order.
        indices = np.argsort(-scores, kind="stable")[:top_k]

        results = []

        for rank, index in enumerate(indices, start=1):
            results.append(
                RetrievalResult(
                    rank=rank,
                    score=float(scores[index]),
                    chunk=self.chunks[int(index)],
                )
            )

        return results

    def load_index(
        self,
        chunks: list[SectionChunk],
        embeddings: np.ndarray,
    ) -> None:
        """Restore an existing in-memory index."""

        vectors = np.asarray(embeddings, dtype=np.float32)

        if not chunks:
            raise ValueError("Cannot load an empty index.")

        if vectors.ndim != 2:
            raise ValueError("Embeddings must be 2D.")

        if vectors.shape[0] != len(chunks):
            raise ValueError("Chunk count and embedding count do not match.")

        if vectors.shape[1] == 0:
            raise ValueError("Embedding dimension cannot be zero.")

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain invalid values.")

        expected_dimension = getattr(self.embedder, "dimension", None)

        if expected_dimension is not None and vectors.shape[1] != expected_dimension:
            raise ValueError("Embedding dimension does not match model.")

        self.chunks = list(chunks)
        self.embeddings = vectors
