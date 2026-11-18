import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from paper_rag.data.section_chunker import chunk_paper_by_section
from paper_rag.retrieval.embedder import DEFAULT_MODEL_NAME, TextEmbedder
from paper_rag.retrieval.index_store import IndexStore
from paper_rag.schemas import PaperDocument, SectionChunk


def load_chunks(
    paper_path: Path,
    strategy: str,
    chunk_size: int,
    min_tail_size: int,
) -> tuple[list[SectionChunk], int, list[str]]:
    """Generate chunks using the selected strategy."""

    chunks = []
    paper_count = 0
    empty_papers = []

    with paper_path.open("r", encoding="utf-8") as file:
        for line in tqdm(file, desc="Chunking papers"):
            if not line.strip():
                continue

            paper = PaperDocument.model_validate_json(line)
            paper_count += 1

            paper_chunks = chunk_paper_by_section(
                paper,
                chunk_size=chunk_size,
                strategy=strategy,
                min_tail_size=min_tail_size,
            )

            if not paper_chunks:
                empty_papers.append(paper.paper_id)

            chunks.extend(paper_chunks)

    if not chunks:
        raise ValueError("No chunks generated.")

    ids = [chunk.chunk_id for chunk in chunks]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    return chunks, paper_count, empty_papers


def main():
    parser = argparse.ArgumentParser(
        description="Build a persistent paper retrieval index."
    )

    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        required=True,
    )
    parser.add_argument(
        "--strategy",
        choices=("fixed", "semantic"),
        required=True,
    )
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--min-tail-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()

    if args.chunk_size <= 0 or args.batch_size <= 0:
        parser.error("chunk-size and batch-size must be positive")

    if not 0 <= args.min_tail_size <= args.chunk_size:
        parser.error("min-tail-size must be within chunk-size")

    paper_path = Path(f"data/processed/{args.split}_papers.jsonl")

    suffix = "dense" if args.strategy == "fixed" else "semantic_v1"

    index_dir = Path(f"data/indexes/qasper_{args.split}_{suffix}")

    if not paper_path.is_file():
        raise FileNotFoundError(paper_path)

    if index_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing index: {index_dir}")

    print("\n========== Index Configuration ==========")
    print("Split:", args.split)
    print("Strategy:", args.strategy)
    print("Chunk size:", args.chunk_size)
    print("Index:", index_dir)

    chunks, paper_count, empty_papers = load_chunks(
        paper_path=paper_path,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        min_tail_size=args.min_tail_size,
    )

    print("\nPapers:", paper_count)
    print("Chunks:", len(chunks))
    print("Papers without chunks:", empty_papers)

    embedder = TextEmbedder(
        model_name=DEFAULT_MODEL_NAME,
        device=args.device,
    )

    embeddings = embedder.encode_documents(
        [chunk.text for chunk in chunks],
        batch_size=args.batch_size,
        show_progress_bar=True,
    )

    expected = (len(chunks), embedder.dimension)
    if embeddings.shape != expected:
        raise ValueError(f"Embedding shape mismatch: {embeddings.shape}")

    if embeddings.dtype != np.float32:
        raise ValueError("Expected float32 embeddings")

    store = IndexStore(index_dir)
    store.save(
        chunks=chunks,
        embeddings=embeddings,
        model_name=embedder.model_name,
    )

    # Validate the files after saving.
    restored_chunks, restored_vectors, manifest = store.load(
        expected_model_name=embedder.model_name
    )

    assert len(restored_chunks) == len(chunks)
    assert restored_vectors.shape == expected

    print("\n========== Index Built ==========")
    print("Chunk count:", len(restored_chunks))
    print("Embedding shape:", restored_vectors.shape)
    print("Manifest:", json.dumps(manifest, indent=2))
    print("Index saved:", index_dir)


if __name__ == "__main__":
    main()
