from collections import defaultdict

from paper_rag.schemas import PaperDocument, QAExample, SectionChunk

PAPER_PATH = "data/processed/train_papers.jsonl"
QA_PATH = "data/processed/train_qa.jsonl"
CHUNK_PATH = "data/indexes/qasper_train_dense/chunks.jsonl"


def merge_intervals(intervals):
    intervals.sort()
    merged = []

    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (
                merged[-1][0],
                max(merged[-1][1], end),
            )
        else:
            merged.append((start, end))

    return merged


def fully_covered(chunks, location):
    section_idx, paragraph_idx, start, end = location
    intervals = []

    for chunk in chunks:
        if chunk.section_index != section_idx:
            continue

        for span in chunk.spans:
            if span.paragraph_index != paragraph_idx:
                continue

            left = max(start, span.start_char)
            right = min(end, span.end_char)

            if left < right:
                intervals.append((left, right))

    merged = merge_intervals(intervals)

    return len(merged) == 1 and merged[0][0] <= start and merged[0][1] >= end


def locate(paper, evidence):
    locations = []

    for section in paper.sections:
        for paragraph in section.paragraphs:
            offset = 0

            while True:
                pos = paragraph.text.find(evidence, offset)
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
                offset = pos + 1

    return locations


def main():
    papers = {}

    with open(PAPER_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = PaperDocument.model_validate_json(line)
                papers[p.paper_id] = p

    chunks_by_paper = defaultdict(list)

    with open(CHUNK_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = SectionChunk.model_validate_json(line)
                chunks_by_paper[c.paper_id].append(c)

    missing = []

    with open(QA_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            qa = QAExample.model_validate_json(line)
            paper = papers[qa.paper_id]
            paper_chunks = chunks_by_paper[qa.paper_id]

            for answer in qa.answers:
                for item in answer.evidence:
                    evidence = item.strip()
                    if not evidence:
                        continue

                    locations = locate(paper, evidence)
                    if not locations:
                        continue

                    if not any(fully_covered(paper_chunks, loc) for loc in locations):
                        missing.append((qa, evidence, locations))

    print("========== V0 Missing Evidence ==========")
    print("Total:", len(missing))

    for i, (qa, evidence, locations) in enumerate(missing, 1):
        print(f"\n--- Case {i} ---")
        print("Paper:", qa.paper_id)
        print("Question:", qa.question)
        print("Evidence:", repr(evidence[:250]))
        print("Evidence length:", len(evidence))
        print("Locations:", locations)

    assert len(missing) == 11


if __name__ == "__main__":
    main()
