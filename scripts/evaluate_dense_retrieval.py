import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from paper_rag.evaluation.retrieval_evaluator import (
    first_hit_rank,
    locate_body_evidence,
    rank_chunks,
    summarize_ranks,
)
from paper_rag.retrieval.embedder import DEFAULT_MODEL_NAME, TextEmbedder
from paper_rag.retrieval.index_store import IndexStore
from paper_rag.schemas import PaperDocument, QAExample

PAPER_PATH = Path("data/processed/train_papers.jsonl")
QA_PATH = Path("data/processed/train_qa.jsonl")

INDEX_DIRS = {
    "V0": Path("data/indexes/qasper_train_dense"),
    "V1": Path("data/indexes/qasper_train_semantic_v1"),
}

OUTPUT_PATH = Path("storage/metadata/retrieval_eval_train_v0_v1.json")

TOP_K = 10
THRESHOLD = 0.5
BATCH_SIZE = 32


def load_papers() -> dict:
    """Load normalized paper documents."""
    papers = {}

    with PAPER_PATH.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                paper = PaperDocument.model_validate_json(line)
                papers[paper.paper_id] = paper

    return papers


def prepare_questions(papers: dict) -> tuple[list[dict], dict]:
    """Aggregate annotations and locate body evidence per question."""
    questions = {}
    counters = {
        "original_qa_records": 0,
        "missing_paper": 0,
        "without_body_evidence": 0,
    }

    with QA_PATH.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            qa = QAExample.model_validate_json(line)
            counters["original_qa_records"] += 1

            paper = papers.get(qa.paper_id)
            if paper is None:
                counters["missing_paper"] += 1
                continue

            # Include paper ID to avoid cross-paper ID collisions.
            key = (qa.paper_id, qa.question_id)

            if key not in questions:
                questions[key] = {
                    "paper_id": qa.paper_id,
                    "question_id": qa.question_id,
                    "question": qa.question,
                    "locations": set(),
                }

            for answer in qa.answers:
                for item in answer.evidence:
                    evidence = item.strip()
                    if not evidence:
                        continue

                    locations = locate_body_evidence(paper, evidence)
                    questions[key]["locations"].update(locations)

    prepared = []

    for question in questions.values():
        if not question["locations"]:
            counters["without_body_evidence"] += 1
            continue

        question["locations"] = sorted(
            question["locations"],
            key=lambda x: (
                x.section_index,
                x.paragraph_index,
                x.start_char,
                x.end_char,
            ),
        )
        prepared.append(question)

    counters["unique_questions"] = len(questions)
    counters["evaluable_questions"] = len(prepared)

    return prepared, counters


def load_grouped_index(index_dir: Path, model_name: str) -> dict:
    """Load index and group vectors by paper once."""
    store = IndexStore(index_dir)

    chunks, embeddings, manifest = store.load(expected_model_name=model_name)

    indices_by_paper = defaultdict(list)

    for index, chunk in enumerate(chunks):
        indices_by_paper[chunk.paper_id].append(index)

    grouped = {}

    for paper_id, indices in indices_by_paper.items():
        grouped[paper_id] = (
            [chunks[i] for i in indices],
            embeddings[np.asarray(indices, dtype=np.int64)],
        )

    print(
        f"Loaded {index_dir.name}: {len(chunks)} chunks, {len(grouped)} indexed papers"
    )

    assert len(chunks) == manifest["num_chunks"]

    return grouped


def main():
    print("\n========== Step 1: Loading Evaluation Data ==========")

    papers = load_papers()
    questions, counters = prepare_questions(papers)

    for key, value in counters.items():
        print(f"{key}: {value}")

    if not questions:
        raise ValueError("No evaluable questions found.")

    print("\n========== Step 2: Loading Model ==========")

    embedder = TextEmbedder(
        model_name=DEFAULT_MODEL_NAME,
        device="cpu",
    )

    print("\n========== Step 3: Loading V0 / V1 Indexes ==========")

    indexes = {
        name: load_grouped_index(path, embedder.model_name)
        for name, path in INDEX_DIRS.items()
    }

    # Ensure all evaluation papers exist in BOTH indexes.
    eligible = []
    excluded_missing_index = 0

    for question in questions:
        paper_id = question["paper_id"]

        if all(paper_id in index for index in indexes.values()):
            eligible.append(question)
        else:
            excluded_missing_index += 1

    print("Questions with both indexes:", len(eligible))
    print("Excluded (missing index):", excluded_missing_index)

    if not eligible:
        raise ValueError("No questions with both indexes.")

    print("\n========== Step 4: Encoding Queries ==========")

    query_texts = [item["question"] for item in eligible]

    # Both strategies use exactly the same query embeddings.
    query_embeddings = embedder.encode_documents(
        query_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
    )

    assert query_embeddings.shape == (
        len(eligible),
        embedder.dimension,
    )

    print("Query embeddings:", query_embeddings.shape)

    print("\n========== Step 5: Evaluating Retrieval ==========")

    ranks = {name: [] for name in indexes}
    per_question = []

    for question, query_vector in tqdm(
        zip(eligible, query_embeddings, strict=True),
        total=len(eligible),
        desc="Evaluating questions",
    ):
        paper_id = question["paper_id"]

        row = {
            "paper_id": paper_id,
            "question_id": question["question_id"],
            "question": question["question"],
            "evidence_locations": len(question["locations"]),
        }

        for name, grouped in indexes.items():
            chunks, vectors = grouped[paper_id]

            ranked_chunks = rank_chunks(
                query_embedding=query_vector,
                chunks=chunks,
                embeddings=vectors,
                top_k=TOP_K,
            )

            rank = first_hit_rank(
                ranked_chunks,
                question["locations"],
                threshold=THRESHOLD,
            )

            ranks[name].append(rank)
            row[f"{name}_first_hit_rank"] = rank

        per_question.append(row)

    print("\n========== Step 6: Metrics ==========")

    metrics = {}

    for name, rank_list in ranks.items():
        result = summarize_ranks(rank_list, ks=(1, 5, 10))
        metrics[name] = result

        print(f"\n--- {name} ---")

        for metric_name, value in result.items():
            print(f"{metric_name}: {value:.4f}")

    print("\n========== V1 - V0 ==========")

    for key in metrics["V0"]:
        delta = metrics["V1"][key] - metrics["V0"][key]
        print(f"{key}: {delta:+.4f}")

    v0_wins = 0
    v1_wins = 0
    ties = 0

    for row in per_question:
        # Unhit questions are ranked after all Top-K results.
        r0 = row["V0_first_hit_rank"] or TOP_K + 1
        r1 = row["V1_first_hit_rank"] or TOP_K + 1

        if r0 < r1:
            v0_wins += 1
        elif r1 < r0:
            v1_wins += 1
        else:
            ties += 1

    print("\n========== Paired Comparison ==========")
    print("V0 better:", v0_wins)
    print("V1 better:", v1_wins)
    print("Tie:", ties)

    output = {
        "experiment": "qasper_train_v0_vs_v1",
        "evaluation_scope": "within_paper",
        "model_name": embedder.model_name,
        "top_k": TOP_K,
        "evidence_coverage_threshold": THRESHOLD,
        "question_statistics": {
            **counters,
            "excluded_missing_index": excluded_missing_index,
            "final_evaluable_questions": len(eligible),
        },
        "metrics": metrics,
        "paired_comparison": {
            "V0_better": v0_wins,
            "V1_better": v1_wins,
            "tie": ties,
        },
        "per_question": per_question,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print("\nSaved results to:", OUTPUT_PATH)
    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
