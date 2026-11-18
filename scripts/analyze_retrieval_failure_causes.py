import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from paper_rag.evaluation.retrieval_evaluator import (
    chunk_evidence_coverage,
    locate_body_evidence,
)
from paper_rag.retrieval.embedder import TextEmbedder
from paper_rag.retrieval.index_store import IndexStore
from paper_rag.schemas import PaperDocument, QAExample

RESULT_PATH = Path("storage/metadata/retrieval_eval_train_v0_v1.json")
PAPER_PATH = Path("data/processed/train_papers.jsonl")
QA_PATH = Path("data/processed/train_qa.jsonl")

INDEX_DIRS = {
    "V0": Path("data/indexes/qasper_train_dense"),
    "V1": Path("data/indexes/qasper_train_semantic_v1"),
}

THRESHOLD = 0.5
TOP_K = 10


def load_evidence_locations(target_keys):
    """Locate body evidence for selected questions."""

    papers = {}

    with PAPER_PATH.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                paper = PaperDocument.model_validate_json(line)
                papers[paper.paper_id] = paper

    locations_by_question = defaultdict(set)

    with QA_PATH.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            qa = QAExample.model_validate_json(line)
            key = (qa.paper_id, qa.question_id)

            if key not in target_keys:
                continue

            for answer in qa.answers:
                for item in answer.evidence:
                    evidence = item.strip()

                    if evidence:
                        locations_by_question[key].update(
                            locate_body_evidence(papers[qa.paper_id], evidence)
                        )

    return locations_by_question


def load_index_by_paper(directory, model_name):
    """Load existing vectors and group them by paper."""

    chunks, vectors, _ = IndexStore(directory).load(expected_model_name=model_name)

    grouped_indices = defaultdict(list)

    for index, chunk in enumerate(chunks):
        grouped_indices[chunk.paper_id].append(index)

    return {
        paper_id: (
            [chunks[i] for i in indices],
            vectors[np.asarray(indices)],
        )
        for paper_id, indices in grouped_indices.items()
    }


def inspect_query(query_vector, chunks, vectors, locations):
    """Find the first qualifying evidence rank in the entire paper."""

    scores = vectors @ query_vector
    order = np.argsort(-scores, kind="stable")

    best_coverage = 0.0
    first_rank = None

    for rank, index in enumerate(order, 1):
        chunk = chunks[int(index)]

        coverage = max(
            (chunk_evidence_coverage(chunk, location) for location in locations),
            default=0.0,
        )

        best_coverage = max(best_coverage, coverage)

        if first_rank is None and coverage >= THRESHOLD:
            first_rank = rank

    return first_rank, best_coverage


def main():
    with RESULT_PATH.open(encoding="utf-8") as file:
        result = json.load(file)

    selected = []

    for row in result["per_question"]:
        r0 = row["V0_first_hit_rank"]
        r1 = row["V1_first_hit_rank"]

        if r0 is not None and r1 is None:
            category = "v0_only_hit"
        elif r0 is None and r1 is not None:
            category = "v1_only_hit"
        else:
            continue

        selected.append((row, category))

    print("Selected questions:", len(selected))

    target_keys = {(row["paper_id"], row["question_id"]) for row, _ in selected}

    locations_by_question = load_evidence_locations(target_keys)

    embedder = TextEmbedder(device="cpu")

    indexes = {
        name: load_index_by_paper(path, embedder.model_name)
        for name, path in INDEX_DIRS.items()
    }

    texts = [row["question"] for row, _ in selected]

    query_vectors = embedder.encode_documents(
        texts,
        batch_size=32,
        show_progress_bar=True,
    )

    counts = Counter()
    examples = defaultdict(list)

    for (row, category), query_vector in tqdm(
        zip(selected, query_vectors, strict=True),
        total=len(selected),
        desc="Analyzing failures",
    ):
        key = (row["paper_id"], row["question_id"])
        locations = locations_by_question[key]

        assert locations, f"Missing evidence locations: {key}"

        if category == "v0_only_hit":
            losing_version = "V1"
        else:
            losing_version = "V0"

        chunks, vectors = indexes[losing_version][row["paper_id"]]

        rank, best_coverage = inspect_query(
            query_vector,
            chunks,
            vectors,
            locations,
        )

        if rank is None:
            cause = "coverage_failure"
        elif rank > TOP_K:
            cause = "ranking_degradation"
        else:
            cause = "unexpected"

        label = f"{category}/{cause}"
        counts[label] += 1

        if len(examples[label]) < 5:
            examples[label].append(
                {
                    "paper_id": row["paper_id"],
                    "question": row["question"],
                    "losing_version": losing_version,
                    "full_rank": rank,
                    "best_coverage": round(best_coverage, 4),
                }
            )

    print("\n========== Failure Cause Statistics ==========")

    for category, count in counts.most_common():
        print(f"{category}: {count}")

    print("\n========== Representative Examples ==========")

    for category, records in examples.items():
        print(f"\n--- {category} ---")

        for record in records:
            print(json.dumps(record, ensure_ascii=False, indent=2))

    assert sum(counts.values()) == len(selected)

    print("\nFailure analysis completed.")


if __name__ == "__main__":
    main()
