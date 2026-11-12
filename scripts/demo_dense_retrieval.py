import json
from pathlib import Path

from paper_rag.data.section_chunker import chunk_paper_by_section
from paper_rag.retrieval.dense_retriever import DenseRetriever
from paper_rag.retrieval.embedder import (
    DEFAULT_MODEL_NAME,
    TextEmbedder,
)
from paper_rag.retrieval.index_store import IndexStore
from paper_rag.schemas import PaperDocument

PAPER_PATH = Path("data/processed/train_papers.jsonl")
INDEX_DIR = Path("data/indexes/dense")


def load_paper_chunks():
    """Load and chunk the first paper."""

    with PAPER_PATH.open("r", encoding="utf-8") as file:
        record = json.loads(file.readline())

    paper = PaperDocument.model_validate(record)

    chunks = chunk_paper_by_section(
        paper,
        chunk_size=500,
    )

    print("Paper:", paper.title)
    print("Number of chunks:", len(chunks))

    return chunks


def main():
    # 1. Initialize embedding model.
    embedder = TextEmbedder(
        model_name=DEFAULT_MODEL_NAME,
        device="cpu",
    )

    # 2. Initialize retriever and index store.
    retriever = DenseRetriever(embedder)
    store = IndexStore(INDEX_DIR)

    # 3. Load existing index or build a new one.
    if store.exists():
        print("\nExisting index found. Loading...")

        chunks, embeddings, _manifest = store.load(
            expected_model_name=embedder.model_name,
        )

        retriever.load_index(chunks, embeddings)

        print("Index loaded successfully.")
        print("Number of chunks:", len(chunks))
        print("Embedding shape:", embeddings.shape)

    else:
        print("\nNo existing index found. Building...")

        chunks = load_paper_chunks()

        retriever.build_index(
            chunks,
            show_progress_bar=True,
        )

        store.save(
            chunks=retriever.chunks,
            embeddings=retriever.embeddings,
            model_name=embedder.model_name,
        )

        print("Index saved successfully.")
        print("Embedding shape:", retriever.embeddings.shape)

    # 4. Search.
    query = "What is the seed lexicon?"

    results = retriever.search(
        query,
        top_k=10,
    )

    print("\nQuery:", query)
    print("\nTop-10 Results:")

    for result in results:
        print(f"\nRank {result.rank}")
        print(f"Score: {result.score:.4f}")
        print(f"Chunk ID: {result.chunk.chunk_id}")
        print(f"Section: {result.chunk.section_name}")
        print(f"Text: {result.chunk.text[:500]}")


if __name__ == "__main__":
    main()
