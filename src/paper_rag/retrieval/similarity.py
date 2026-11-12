import numpy as np


def cosine_similarity(
    query_vector: np.ndarray,
    document_vectors: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between query and documents."""

    query = np.asarray(query_vector, dtype=np.float32)
    documents = np.asarray(document_vectors, dtype=np.float32)

    if query.ndim != 1:
        raise ValueError("Query vector must be 1D.")

    if documents.ndim != 2:
        raise ValueError("Document vectors must be 2D.")

    if query.shape[0] != documents.shape[1]:
        raise ValueError("Embedding dimensions do not match.")

    query_norm = np.linalg.norm(query)
    document_norms = np.linalg.norm(documents, axis=1)

    if query_norm == 0 or np.any(document_norms == 0):
        raise ValueError("Zero vectors are not supported.")

    scores = documents @ query / (document_norms * query_norm)

    return scores
