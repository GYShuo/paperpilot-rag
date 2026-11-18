from collections import Counter, defaultdict
from pathlib import Path

from paper_rag.schemas import PaperDocument, QAExample

PAPER_PATH = Path("data/processed/train_papers.jsonl")
QA_PATH = Path("data/processed/train_qa.jsonl")

MAX_SAMPLES = 3


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace without changing other characters."""
    return " ".join(text.split())


def classify_evidence(
    evidence: str,
    paragraphs: list[str],
    captions: list[str],
) -> str:
    """Classify evidence by its matching characteristics."""

    # Category 1: exact substring in a body paragraph.
    if any(evidence in paragraph for paragraph in paragraphs):
        return "exact_body_match"

    # Category 2: explicit figure/table evidence marker.
    if evidence.startswith("FLOAT SELECTED:"):
        return "float_selected"

    # Category 3: whitespace-only differences.
    normalized = normalize_whitespace(evidence)
    if any(normalized in normalize_whitespace(paragraph) for paragraph in paragraphs):
        return "whitespace_difference"

    # Category 4: evidence matches a figure/table caption.
    if any(
        caption and (evidence in caption or caption in evidence)
        for caption in captions
        if len(caption) >= 12
    ):
        return "caption_related"

    # Category 5: other unmatched evidence.
    return "other_unmatched"


def main():
    papers = {}

    with PAPER_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                paper = PaperDocument.model_validate_json(line)
                papers[paper.paper_id] = paper

    counts = Counter()
    samples = defaultdict(list)

    total = 0

    with QA_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            qa = QAExample.model_validate_json(line)
            paper = papers.get(qa.paper_id)

            if paper is None:
                continue

            paragraphs = [
                p.text for section in paper.sections for p in section.paragraphs
            ]

            captions = [
                item.caption for item in paper.figures_and_tables if item.caption
            ]

            for answer in qa.answers:
                for item in answer.evidence:
                    evidence = item.strip()

                    if not evidence:
                        continue

                    total += 1

                    category = classify_evidence(
                        evidence,
                        paragraphs,
                        captions,
                    )

                    counts[category] += 1

                    if len(samples[category]) < MAX_SAMPLES:
                        samples[category].append(
                            {
                                "paper_id": qa.paper_id,
                                "question": qa.question,
                                "evidence": evidence[:350],
                            }
                        )

    print("\n========== Evidence Classification ==========")
    print("Total evidence:", total)

    for category, count in counts.most_common():
        ratio = 100 * count / total
        print(f"{category}: {count} ({ratio:.2f}%)")

    print("\n========== Unmatched Samples ==========")

    for category in (
        "float_selected",
        "whitespace_difference",
        "caption_related",
        "other_unmatched",
    ):
        print(f"\n--- {category} ---")

        for example in samples[category]:
            print("Paper:", example["paper_id"])
            print("Question:", example["question"])
            print("Evidence:", repr(example["evidence"]))
            print()

    assert sum(counts.values()) == total
    print("\nClassification completed.")


if __name__ == "__main__":
    main()
