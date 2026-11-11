import json
from pathlib import Path

from paper_rag.data.chunker import chunk_paper
from paper_rag.data.section_chunker import chunk_paper_by_section
from paper_rag.schemas import PaperDocument

PAPER_PATH = Path("data/processed/train_papers.jsonl")


def main():
    with PAPER_PATH.open("r", encoding="utf-8") as f:
        record = json.loads(f.readline())

    paper = PaperDocument.model_validate(record)

    chunk_size = 500

    # Existing paragraph-based chunker
    baseline_chunks = chunk_paper(
        paper,
        chunk_size=chunk_size,
        chunk_overlap=0,
    )

    # New section-aware chunker
    section_chunks = chunk_paper_by_section(
        paper,
        chunk_size=chunk_size,
    )

    print(f"Paper: {paper.title}")
    print(f"Chunk size: {chunk_size}")

    for name, chunks in [
        ("Paragraph-based", baseline_chunks),
        ("Section-aware", section_chunks),
    ]:
        lengths = [len(c.text) for c in chunks]

        print(f"\n{name}")
        print("Number of chunks:", len(chunks))
        print("Average length:", round(sum(lengths) / len(lengths), 2))
        print("Min length:", min(lengths))
        print("Max length:", max(lengths))

    print("\nFirst three section-aware chunks:")

    for chunk in section_chunks[:3]:
        print("\nChunk ID:", chunk.chunk_id)
        print("Section:", chunk.section_name)
        print("Source spans:", len(chunk.spans))
        print("Text:", chunk.text[:250])


if __name__ == "__main__":
    main()
