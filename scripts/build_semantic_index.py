import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from paper_rag.data.section_chunker import chunk_paper_by_section
from paper_rag.retrieval.embedder import DEFAULT_MODEL_NAME, TextEmbedder
from paper_rag.retrieval.index_store import IndexStore
from paper_rag.schemas import PaperDocument, SectionChunk

PAPER_PATH = Path("data/processed/train_papers.jsonl")
INDEX_DIR = Path("data/indexes/qasper_train_semantic_v1")

CHUNK_SIZE = 500
BATCH_SIZE = 32
DEVICE = "cpu"


def load_corpus_chunks() -> tuple[list[SectionChunk], int, list[str]]:
    """Load all training papers and generate section-aware chunks."""

    chunks: list[SectionChunk] = []
    paper_count = 0
    empty_paper_ids: list[str] = []

    with PAPER_PATH.open("r", encoding="utf-8") as file:
        total_records = sum(1 for _ in file)
        file.seek(0)

        for line in tqdm(
            file,
            total=total_records,
            desc="Chunking papers",
        ):
            if not line.strip():
                continue

            record = json.loads(line)
            paper = PaperDocument.model_validate(record)

            paper_chunks = chunk_paper_by_section(
                paper,
                chunk_size=CHUNK_SIZE,
                strategy="semantic",
                min_tail_size=100,
            )

            paper_count += 1

            if not paper_chunks:
                empty_paper_ids.append(paper.paper_id)

            chunks.extend(paper_chunks)

    if not chunks:
        raise ValueError("No valid chunks found in corpus.")

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs found.")

    return chunks, paper_count, empty_paper_ids


def build_embeddings(
    chunks: list[SectionChunk],
    embedder: TextEmbedder,
) -> np.ndarray:
    """Encode all chunks into normalized embedding vectors."""

    texts = [chunk.text for chunk in chunks]

    embeddings = embedder.encode_documents(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
    )

    expected_shape = (len(chunks), embedder.dimension)

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}, "
            f"expected: {expected_shape}"
        )

    if embeddings.dtype != np.float32:
        raise ValueError("Embeddings must use float32.")

    return embeddings


def main():
    if not PAPER_PATH.is_file():
        raise FileNotFoundError(f"Paper file not found: {PAPER_PATH}")

    # Avoid accidentally overwriting an existing index.
    if INDEX_DIR.exists():
        raise FileExistsError(
            f"Index directory already exists: {INDEX_DIR}. "
            "Inspect it before rebuilding."
        )

    # Step 1: Load and chunk the corpus.
    print("\n========== Step 1: Chunking ==========")

    chunks, paper_count, empty_paper_ids = load_corpus_chunks()

    print("Papers processed:", paper_count)
    print("Papers without chunks:", len(empty_paper_ids))
    print("Empty paper IDs:", empty_paper_ids)
    print("Total chunks:", len(chunks))

    # Step 2: Initialize embedding model.
    print("\n========== Step 2: Loading Model ==========")

    embedder = TextEmbedder(
        model_name=DEFAULT_MODEL_NAME,
        device=DEVICE,
    )

    print("Model:", embedder.model_name)
    print("Embedding dimension:", embedder.dimension)
    print("Batch size:", BATCH_SIZE)
    print("Device:", DEVICE)

    # Step 3: Generate embeddings.
    print("\n========== Step 3: Embedding ==========")

    embeddings = build_embeddings(chunks, embedder)

    print("Embedding shape:", embeddings.shape)
    print("Embedding dtype:", embeddings.dtype)
    print(
        "Embedding memory:",
        round(embeddings.nbytes / (1024**2), 2),
        "MiB",
    )

    # Step 4: Save the index.
    print("\n========== Step 4: Saving Index ==========")

    store = IndexStore(INDEX_DIR)

    store.save(
        chunks=chunks,
        embeddings=embeddings,
        model_name=embedder.model_name,
    )

    print("Index saved to:", INDEX_DIR)
    print("Index files:")
    print(" - embeddings.npy")
    print(" - chunks.jsonl")
    print(" - manifest.json")

    print("\n========== Build Completed ==========")


if __name__ == "__main__":
    main()
