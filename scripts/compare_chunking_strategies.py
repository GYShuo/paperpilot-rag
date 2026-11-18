import json
import statistics
from pathlib import Path

from tqdm import tqdm

from paper_rag.data.section_chunker import chunk_paper_by_section
from paper_rag.schemas import PaperDocument

DATA_PATH = Path("data/processed/train_papers.jsonl")
CHUNK_SIZE = 500
SHORT_THRESHOLD = 100


def evaluate_strategy(
    papers: list[PaperDocument],
    strategy: str,
) -> dict:
    lengths = []
    paper_without_chunks = []
    invalid_chunks = 0
    alignment_errors = 0
    duplicate_ids = 0
    broken_word_boundaries = 0
    seen_ids = set()

    for paper in tqdm(papers, desc=f"Evaluating {strategy}"):
        chunks = chunk_paper_by_section(
            paper,
            chunk_size=CHUNK_SIZE,
            strategy=strategy,
            min_tail_size=SHORT_THRESHOLD,
        )

        if not chunks:
            paper_without_chunks.append(paper.paper_id)

        sections = {section.section_index: section for section in paper.sections}

        for chunk in chunks:
            lengths.append(len(chunk.text))

            if chunk.chunk_id in seen_ids:
                duplicate_ids += 1
            seen_ids.add(chunk.chunk_id)

            if not any(char.isalnum() for char in chunk.text):
                invalid_chunks += 1

            section = sections[chunk.section_index]
            paragraphs = {p.paragraph_index: p for p in section.paragraphs}

            for span in chunk.spans:
                source = paragraphs[span.paragraph_index].text

                original_piece = source[span.start_char : span.end_char]
                chunk_piece = chunk.text[span.chunk_start : span.chunk_end]

                if original_piece != chunk_piece:
                    alignment_errors += 1

                # Count cuts occurring inside a word.
                if (
                    0 < span.start_char < len(source)
                    and source[span.start_char - 1].isalnum()
                    and source[span.start_char].isalnum()
                ):
                    broken_word_boundaries += 1

                if (
                    0 < span.end_char < len(source)
                    and source[span.end_char - 1].isalnum()
                    and source[span.end_char].isalnum()
                ):
                    broken_word_boundaries += 1

    return {
        "strategy": strategy,
        "papers": len(papers),
        "empty_paper_ids": paper_without_chunks,
        "total_chunks": len(lengths),
        "average_length": statistics.mean(lengths),
        "median_length": statistics.median(lengths),
        "minimum_length": min(lengths),
        "maximum_length": max(lengths),
        "short_chunks": sum(length < SHORT_THRESHOLD for length in lengths),
        "short_ratio": sum(length < SHORT_THRESHOLD for length in lengths)
        / len(lengths)
        * 100,
        "invalid_chunks": invalid_chunks,
        "alignment_errors": alignment_errors,
        "duplicate_ids": duplicate_ids,
        "broken_word_boundaries": broken_word_boundaries,
    }


def main():
    papers = []

    with DATA_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                papers.append(PaperDocument.model_validate(json.loads(line)))

    print("Loaded papers:", len(papers))

    results = [
        evaluate_strategy(papers, "fixed"),
        evaluate_strategy(papers, "semantic"),
    ]

    print("\n========== V0 / V1 Comparison ==========")

    for result in results:
        print(f"\nStrategy: {result['strategy']}")

        for key, value in result.items():
            if key != "strategy":
                print(f"{key}: {value}")

    v0, v1 = results

    print("\n========== Differences (V1 - V0) ==========")

    for key in (
        "total_chunks",
        "average_length",
        "median_length",
        "short_chunks",
        "short_ratio",
        "broken_word_boundaries",
    ):
        difference = v1[key] - v0[key]
        print(f"{key}: {difference:+.2f}")

    # Hard correctness checks.
    for result in results:
        assert result["alignment_errors"] == 0
        assert result["duplicate_ids"] == 0
        assert result["invalid_chunks"] == 0
        assert result["maximum_length"] <= CHUNK_SIZE

    print("\nAll structural checks passed!")


if __name__ == "__main__":
    main()
