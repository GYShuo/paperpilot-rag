import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class TextEmbedder:
    """Encode documents and queries into normalized vectors."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

        self.dimension = self.model.get_embedding_dimension()

        if self.dimension is None:
            raise RuntimeError("Cannot determine embedding dimension.")

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode multiple document texts."""

        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        if any(not text.strip() for text in texts):
            raise ValueError("Document texts cannot be empty.")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )

        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single user query."""

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(embedding, dtype=np.float32)
