from pathlib import Path

from paper_rag.schemas import PaperDocument, QAExample

PAPER_PATH = Path("data/processed/train_papers.jsonl")
QA_PATH = Path("data/processed/train_qa.jsonl")

MAX_EXAMPLES = 5


def main():
    papers = {}

    with PAPER_PATH.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            paper = PaperDocument.model_validate_json(line)
            papers[paper.paper_id] = paper

    total_questions = 0
    total_annotations = 0
    annotations_with_evidence = 0
    total_evidence = 0
    exact_match_evidence = 0
    missing_papers = 0
    examples = []

    with QA_PATH.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            qa = QAExample.model_validate_json(line)
            total_questions += 1

            paper = papers.get(qa.paper_id)

            if paper is None:
                missing_papers += 1
                continue

            paragraphs = [
                paragraph.text
                for section in paper.sections
                for paragraph in section.paragraphs
            ]

            for answer in qa.answers:
                total_annotations += 1

                evidence_list = [
                    item.strip() for item in answer.evidence if item.strip()
                ]

                if evidence_list:
                    annotations_with_evidence += 1

                for evidence in evidence_list:
                    total_evidence += 1

                    matched = any(evidence in paragraph for paragraph in paragraphs)

                    if matched:
                        exact_match_evidence += 1

                    if len(examples) < MAX_EXAMPLES:
                        examples.append(
                            {
                                "question": qa.question,
                                "paper_id": qa.paper_id,
                                "evidence": evidence[:400],
                                "exact_match": matched,
                            }
                        )

    print("\n========== QASPER Evidence Statistics ==========")
    print("Papers:", len(papers))
    print("Questions:", total_questions)
    print("Annotations:", total_annotations)
    print("Annotations with evidence:", annotations_with_evidence)
    print("Evidence items:", total_evidence)
    print("Questions with missing paper:", missing_papers)

    print("\n========== Exact Match ==========")
    print("Matched evidence:", exact_match_evidence)
    print("Unmatched evidence:", total_evidence - exact_match_evidence)

    if total_evidence:
        print(
            "Exact match ratio:",
            round(exact_match_evidence / total_evidence * 100, 2),
            "%",
        )

    print("\n========== Sample Evidence ==========")

    for index, example in enumerate(examples, start=1):
        print(f"\nExample {index}")
        print("Paper:", example["paper_id"])
        print("Question:", example["question"])
        print("Evidence:", repr(example["evidence"]))
        print("Exact match:", example["exact_match"])


if __name__ == "__main__":
    main()
