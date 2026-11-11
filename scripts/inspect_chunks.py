import json
from pathlib import Path

from paper_rag.data.chunker import chunk_paper
from paper_rag.schemas import PaperDocument

PAPER_PATH = Path("data/processed/train_papers.jsonl")


def load_first_paper() -> PaperDocument:
    with PAPER_PATH.open("r", encoding="utf-8") as f:
        record = json.loads(f.readline())

    return PaperDocument.model_validate(record)


def main():
    paper = load_first_paper()

    print(f"Paper: {paper.title}")
    print(f"Sections: {len(paper.sections)}")

    configs = [
        (300, 0),
        (500, 0),
        (500, 100),
        (800, 100),
    ]

    for chunk_size, overlap in configs:
        chunks = chunk_paper(
            paper,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )

        lengths = [len(chunk.text) for chunk in chunks]
        avg_length = sum(lengths) / len(lengths) if lengths else 0

        print("\n------------------------------")
        print(f"Chunk size: {chunk_size}")
        print(f"Overlap: {overlap}")
        print(f"Total chunks: {len(chunks)}")
        print(f"Average length: {avg_length:.2f}")

        if chunks:
            first = chunks[0]

            print(f"First chunk ID: {first.chunk_id}")
            print(f"Section: {first.section_name}")
            print(f"Text: {first.text[:200]}")


if __name__ == "__main__":
    main()
