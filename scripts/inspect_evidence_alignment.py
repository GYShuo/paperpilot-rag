from collections import Counter, defaultdict
from pathlib import Path

from paper_rag.schemas import PaperDocument, QAExample, SectionChunk

PAPER_PATH = Path("data/processed/train_papers.jsonl")
QA_PATH = Path("data/processed/train_qa.jsonl")

INDEX_DIRS = {
    "V0": Path("data/indexes/qasper_train_dense"),
    "V1": Path("data/indexes/qasper_train_semantic_v1"),
}


def load_chunks(path: Path) -> dict:
    """Group indexed chunks by paper ID."""
    grouped = defaultdict(list)

    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunk = SectionChunk.model_validate_json(line)
                grouped[chunk.paper_id].append(chunk)

    return grouped


def locate_evidence(paper: PaperDocument, evidence: str) -> list:
    """Find every exact occurrence of evidence in body paragraphs."""
    locations = []

    for section in paper.sections:
        for paragraph in section.paragraphs:
            text = paragraph.text
            start = 0

            while True:
                pos = text.find(evidence, start)

                if pos == -1:
                    break

                locations.append(
                    (
                        section.section_index,
                        paragraph.paragraph_index,
                        pos,
                        pos + len(evidence),
                    )
                )

                start = pos + 1

    return locations


def check_coverage(chunks: list[SectionChunk], location: tuple):
    """Check how indexed chunks cover one evidence occurrence."""
    section_index, paragraph_index, start, end = location

    intervals = []

    for chunk in chunks:
        if chunk.section_index != section_index:
            continue

        for span in chunk.spans:
            if span.paragraph_index != paragraph_index:
                continue

            left = max(start, span.start_char)
            right = min(end, span.end_char)

            if left < right:
                intervals.append((left, right))

    # Full evidence within a single chunk span.
    single_chunk_full = any(left == start and right == end for left, right in intervals)

    # Merge overlapping covered source intervals.
    intervals.sort()
    merged = []

    for left, right in intervals:
        if merged and left <= merged[-1][1]:
            merged[-1] = (
                merged[-1][0],
                max(merged[-1][1], right),
            )
        else:
            merged.append((left, right))

    covered_chars = sum(right - left for left, right in merged)

    return single_chunk_full, covered_chars / (end - start)


def main():
    papers = {}

    with PAPER_PATH.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                paper = PaperDocument.model_validate_json(line)
                papers[paper.paper_id] = paper

    indexes = {
        name: load_chunks(directory / "chunks.jsonl")
        for name, directory in INDEX_DIRS.items()
    }

    stats = Counter()

    with QA_PATH.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            qa = QAExample.model_validate_json(line)
            paper = papers[qa.paper_id]

            for answer in qa.answers:
                for item in answer.evidence:
                    evidence = item.strip()

                    if not evidence:
                        continue

                    locations = locate_evidence(paper, evidence)

                    if not locations:
                        continue

                    stats["body_evidence"] += 1

                    if len(locations) > 1:
                        stats["ambiguous_locations"] += 1

                    for name, index in indexes.items():
                        chunks = index.get(qa.paper_id, [])

                        coverage_results = [
                            check_coverage(chunks, loc) for loc in locations
                        ]

                        if any(full for full, _ in coverage_results):
                            stats[f"{name}_single_chunk_full"] += 1

                        if any(ratio >= 1.0 - 1e-9 for _, ratio in coverage_results):
                            stats[f"{name}_union_full"] += 1

    print("\n========== Evidence Alignment ==========")
    print("Body evidence:", stats["body_evidence"])
    print("Evidence with multiple locations:", stats["ambiguous_locations"])

    for name in INDEX_DIRS:
        print(f"\n--- {name} ---")
        print(
            "Fully contained in one chunk:",
            stats[f"{name}_single_chunk_full"],
        )
        print(
            "Fully covered by chunk union:",
            stats[f"{name}_union_full"],
        )

    assert stats["body_evidence"] == 3679

    print("\nAlignment inspection completed.")


if __name__ == "__main__":
    main()
