import numpy as np
from sentence_transformers import SentenceTransformer

from paper_rag.schemas import EmbeddingWindow

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

        self.dimension = int(self.dimension)

    @property
    def max_seq_length(self) -> int:
        """Return the embedding model's maximum input length."""

        return int(self.model.max_seq_length)

    def count(self, text: str) -> int:
        """Count tokens using the embedding model's tokenizer.

        Special tokens are included in the returned count.
        Truncation is disabled to obtain the full token count.
        """

        if not isinstance(text, str):
            raise TypeError("Text must be a string.")

        encoded = self.model.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )

        return len(encoded["input_ids"])

    def _validate_embeddings(
        self,
        embeddings: np.ndarray,
        expected_count: int,
    ) -> np.ndarray:
        """Validate and normalize the embedding array representation."""

        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        expected_shape = (expected_count, self.dimension)

        if vectors.shape != expected_shape:
            raise ValueError(
                f"Embedding shape mismatch: expected {expected_shape}, "
                f"got {vectors.shape}."
            )

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain invalid values.")

        return vectors

    def encode_documents(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Encode multiple document texts."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Document texts must be non-empty strings.")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )

        return self._validate_embeddings(
            embeddings,
            expected_count=len(texts),
        )

    def encode_windows(
        self,
        windows: list[EmbeddingWindow],
        batch_size: int = 32,
        show_progress_bar: bool = True,
    ) -> np.ndarray:
        """Encode embedding windows without exceeding model input length.

        Each window is validated against the current model tokenizer
        before encoding.

        The returned embedding rows preserve the input window order.
        """

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if not windows:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        max_length = self.max_seq_length

        texts = []

        for window in windows:
            if not window.text.strip():
                raise ValueError(f"Embedding window {window.window_id} is empty.")

            actual_count = self.count(window.text)

            if actual_count > max_length:
                raise ValueError(
                    f"Embedding window {window.window_id} exceeds "
                    f"model maximum length: {actual_count} > {max_length}."
                )

            if actual_count != window.token_count:
                raise ValueError(
                    f"Token count mismatch for window {window.window_id}: "
                    f"stored={window.token_count}, actual={actual_count}. "
                    "Check whether the same tokenizer was used."
                )

            texts.append(window.text)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )

        return self._validate_embeddings(
            embeddings,
            expected_count=len(windows),
        )

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single user query."""

        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        vector = np.asarray(embedding, dtype=np.float32)

        if vector.shape != (self.dimension,):
            raise ValueError(
                f"Query embedding shape mismatch: expected "
                f"({self.dimension},), got {vector.shape}."
            )

        if not np.isfinite(vector).all():
            raise ValueError("Query embedding contains invalid values.")

        return vector
