import argparse
import json
from pathlib import Path

from tqdm import tqdm

from paper_rag.pipeline.windowed_index_builder import (
    build_windowed_index,
)
from paper_rag.retrieval.embedder import (
    DEFAULT_MODEL_NAME,
    TextEmbedder,
)
from paper_rag.retrieval.windowed_index_store import WindowedIndexStore
from paper_rag.schemas import PaperDocument


def load_papers_jsonl(
    input_path: Path,
    limit: int | None = None,
) -> list[PaperDocument]:
    """Load normalized PaperDocument records from JSONL."""

    papers = []

    with input_path.open("r", encoding="utf-8") as file:
        total_lines = sum(1 for _ in file)
        file.seek(0)

        total = min(total_lines, limit) if limit is not None else total_lines

        iterator = tqdm(
            file,
            total=total,
            desc="Loading papers",
        )

        for line in iterator:
            if not line.strip():
                continue

            paper = PaperDocument.model_validate_json(line)
            papers.append(paper)

            if limit is not None and len(papers) >= limit:
                break

    if not papers:
        raise ValueError(f"No papers found in {input_path}")

    return papers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a sentence-aware windowed paper retrieval index."
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input normalized PaperDocument JSONL file.",
    )

    parser.add_argument(
        "--index-dir",
        type=Path,
        required=True,
        help="Output index directory.",
    )

    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--language",
        default="en",
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of papers for smoke testing.",
    )

    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    if args.max_chunk_tokens <= 0:
        parser.error("--max-chunk-tokens must be positive.")

    if args.overlap_tokens < 0:
        parser.error("--overlap-tokens cannot be negative.")

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")

    if args.index_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing index: {args.index_dir}")

    print("\n========== Windowed Index Configuration ==========")
    print("Input:", args.input)
    print("Output:", args.index_dir)
    print("Language:", args.language)
    print("Maximum parent chunk tokens:", args.max_chunk_tokens)
    print("Embedding window overlap:", args.overlap_tokens)
    print("Batch size:", args.batch_size)
    print("Device:", args.device)

    # Load normalized papers.
    papers = load_papers_jsonl(
        input_path=args.input,
        limit=args.limit,
    )

    print("\nLoaded papers:", len(papers))

    # Initialize embedding model.
    print("\n========== Loading Embedding Model ==========")

    embedder = TextEmbedder(
        model_name=DEFAULT_MODEL_NAME,
        device=args.device,
    )

    print("Model:", embedder.model_name)
    print("Dimension:", embedder.dimension)
    print("Maximum sequence length:", embedder.max_seq_length)

    print(
        "\nNote: This QASPER baseline uses the embedding model "
        "tokenizer for parent chunk token counting. "
        "The builder supports injecting another TokenCounter."
    )

    # Build and persist index.
    store = WindowedIndexStore(args.index_dir)

    report = build_windowed_index(
        papers=papers,
        embedder=embedder,
        store=store,
        chunk_token_counter=embedder,
        max_chunk_tokens=args.max_chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        batch_size=args.batch_size,
        language=args.language,
        show_progress_bar=True,
        source_label=str(args.input),
    )

    print("\n========== Final Summary ==========")

    print("Papers:", report.paper_count)
    print("Chunks:", report.chunk_count)
    print("Windows:", report.window_count)
    print("Oversized chunks:", len(report.oversize_chunk_ids))
    print("Empty papers:", report.empty_paper_ids)
    print("Embedding shape:", report.embedding_shape)
    print("Index:", report.index_dir)
    print("Build config:", report.build_config_path)

    print(
        "\nManifest:",
        json.dumps(
            report.manifest,
            ensure_ascii=False,
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
