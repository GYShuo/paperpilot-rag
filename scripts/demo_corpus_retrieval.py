from pathlib import Path

from paper_rag.retrieval.dense_retriever import DenseRetriever
from paper_rag.retrieval.embedder import DEFAULT_MODEL_NAME, TextEmbedder
from paper_rag.retrieval.index_store import IndexStore

INDEX_DIR = Path("data/indexes/qasper_train_dense")

DEVICE = "cpu"
TOP_K = 10

QUERY = "What is the seed lexicon?"


def main():
    # Step 1: Initialize the embedding model.
    print("\n========== Step 1: Loading Model ==========")

    embedder = TextEmbedder(
        model_name=DEFAULT_MODEL_NAME,
        device=DEVICE,
    )

    print("Model:", embedder.model_name)
    print("Dimension:", embedder.dimension)

    # Step 2: Load the full corpus index.
    print("\n========== Step 2: Loading Full Corpus Index ==========")

    store = IndexStore(INDEX_DIR)

    if not store.exists():
        raise FileNotFoundError(f"Corpus index not found: {INDEX_DIR}")

    chunks, embeddings, manifest = store.load(
        expected_model_name=embedder.model_name,
    )

    print("Loaded chunks:", len(chunks))
    print("Embedding shape:", embeddings.shape)
    print("Model:", manifest["model_name"])

    # Step 3: Restore the retriever.
    print("\n========== Step 3: Initializing Retriever ==========")

    retriever = DenseRetriever(embedder)

    retriever.load_index(
        chunks=chunks,
        embeddings=embeddings,
    )

    print("Retriever initialized successfully.")

    # Step 4: Search across the entire corpus.
    print("\n========== Step 4: Dense Retrieval ==========")

    results = retriever.search(
        query=QUERY,
        top_k=TOP_K,
    )

    print("Query:", QUERY)
    print("Top-K:", TOP_K)
    print("Results:", len(results))

    # Step 5: Display retrieved chunks.
    print("\n========== Retrieval Results ==========")

    for result in results:
        chunk = result.chunk

        print(f"\nRank: {result.rank}")
        print(f"Score: {result.score:.4f}")
        print(f"Paper ID: {chunk.paper_id}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Section: {chunk.section_name}")
        print(f"Text: {chunk.text[:500]}")

    print("\n========== Retrieval Completed ==========")


if __name__ == "__main__":
    main()
