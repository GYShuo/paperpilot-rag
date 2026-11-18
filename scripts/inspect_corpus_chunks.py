import json
import statistics
from pathlib import Path

from tqdm import tqdm

from paper_rag.data.section_chunker import chunk_paper_by_section
from paper_rag.schemas import PaperDocument

PAPER_PATH = Path("data/processed/train_papers.jsonl")
CHUNK_SIZE = 500


def main():
    paper_count = 0
    empty_paper_count = 0

    total_chunks = 0
    chunk_lengths = []
    chunks_per_paper = []

    with PAPER_PATH.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    print(f"Total records: {len(lines)}")

    for line in tqdm(lines, desc="Processing papers"):
        if not line.strip():
            continue

        record = json.loads(line)
        paper = PaperDocument.model_validate(record)

        chunks = chunk_paper_by_section(
            paper,
            chunk_size=CHUNK_SIZE,
        )

        paper_count += 1

        if not chunks:
            empty_paper_count += 1

        total_chunks += len(chunks)
        chunks_per_paper.append(len(chunks))

        chunk_lengths.extend(len(chunk.text) for chunk in chunks)

    print("\n========== Corpus Statistics ==========")

    print("Papers processed:", paper_count)
    print("Papers without chunks:", empty_paper_count)
    print("Total chunks:", total_chunks)

    if not chunk_lengths:
        print("No chunks generated.")
        return

    print("\n========== Chunk Length Statistics ==========")

    print("Chunk size:", CHUNK_SIZE)
    print("Average length:", round(statistics.mean(chunk_lengths), 2))
    print("Median length:", statistics.median(chunk_lengths))
    print("Minimum length:", min(chunk_lengths))
    print("Maximum length:", max(chunk_lengths))

    short_chunks = sum(length < 100 for length in chunk_lengths)

    print("Chunks shorter than 100:", short_chunks)
    print(
        "Short chunk ratio:",
        round(short_chunks / total_chunks * 100, 2),
        "%",
    )

    print("\n========== Chunks Per Paper ==========")

    print(
        "Average chunks per paper:",
        round(statistics.mean(chunks_per_paper), 2),
    )
    print("Maximum chunks per paper:", max(chunks_per_paper))
    print("Minimum chunks per paper:", min(chunks_per_paper))


if __name__ == "__main__":
    main()
