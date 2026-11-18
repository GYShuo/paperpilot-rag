import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from paper_rag.evaluation.retrieval_evaluator import (
    chunk_evidence_coverage,
    first_union_hit_rank,
    locate_body_evidence,
    summarize_ranks,
)
from paper_rag.retrieval.embedder import TextEmbedder
from paper_rag.retrieval.index_store import IndexStore
from paper_rag.schemas import PaperDocument, QAExample

THRESHOLDS = (0.25, 0.5, 0.75)
TOP_K = 10
MODES = ("single", "union")
VERSIONS = ("V0", "V1")


def get_index_dirs(split: str) -> dict[str, Path]:
    return {
        "V0": Path(f"data/indexes/qasper_{split}_dense"),
        "V1": Path(f"data/indexes/qasper_{split}_semantic_v1"),
    }


def load_questions(split: str):
    paper_path = Path(f"data/processed/{split}_papers.jsonl")
    qa_path = Path(f"data/processed/{split}_qa.jsonl")

    papers = {}

    with paper_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                paper = PaperDocument.model_validate_json(line)
                papers[paper.paper_id] = paper

    questions = {}

    with qa_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            qa = QAExample.model_validate_json(line)
            key = (qa.paper_id, qa.question_id)

            if key not in questions:
                questions[key] = {
                    "paper_id": qa.paper_id,
                    "question_id": qa.question_id,
                    "question": qa.question,
                    "locations": set(),
                }

            for answer in qa.answers:
                for evidence in answer.evidence:
                    evidence = evidence.strip()

                    if evidence:
                        locations = locate_body_evidence(
                            papers[qa.paper_id],
                            evidence,
                        )
                        questions[key]["locations"].update(locations)

    prepared = [
        {
            **item,
            "locations": list(item["locations"]),
        }
        for item in questions.values()
        if item["locations"]
    ]

    stats = {
        "total_questions": len(questions),
        "body_evidence_questions": len(prepared),
        "excluded_without_body_evidence": (len(questions) - len(prepared)),
        "total_papers": len(papers),
    }

    return prepared, stats


def load_index(directory: Path, model_name: str):
    chunks, vectors, _ = IndexStore(directory).load(expected_model_name=model_name)

    grouped_indices = defaultdict(list)

    for index, chunk in enumerate(chunks):
        grouped_indices[chunk.paper_id].append(index)

    grouped = {}

    for paper_id, indices in grouped_indices.items():
        grouped[paper_id] = (
            [chunks[i] for i in indices],
            vectors[np.asarray(indices, dtype=np.int64)],
        )

    print(f"Loaded {directory.name}: {len(chunks)} chunks, {len(grouped)} papers")

    return grouped


def evaluate(questions, query_vectors, indexes):
    ranks = {
        mode: {
            version: {threshold: [] for threshold in THRESHOLDS} for version in VERSIONS
        }
        for mode in MODES
    }

    for item, query_vector in tqdm(
        zip(questions, query_vectors, strict=True),
        total=len(questions),
        desc="Evaluating retrieval",
    ):
        for version in VERSIONS:
            chunks, vectors = indexes[version][item["paper_id"]]

            scores = vectors @ query_vector
            order = np.argsort(-scores, kind="stable")[:TOP_K]

            ranked_chunks = [chunks[int(index)] for index in order]

            coverages = [
                max(
                    (
                        chunk_evidence_coverage(
                            chunk,
                            location,
                        )
                        for location in item["locations"]
                    ),
                    default=0.0,
                )
                for chunk in ranked_chunks
            ]

            for threshold in THRESHOLDS:
                single_rank = next(
                    (
                        rank
                        for rank, coverage in enumerate(
                            coverages,
                            start=1,
                        )
                        if coverage >= threshold
                    ),
                    None,
                )

                union_rank = first_union_hit_rank(
                    ranked_chunks,
                    item["locations"],
                    threshold=threshold,
                )

                ranks["single"][version][threshold].append(single_rank)
                ranks["union"][version][threshold].append(union_rank)

    results = {}

    for mode in MODES:
        results[mode] = {}

        print(f"\n========== Mode: {mode.upper()} ==========")

        for threshold in THRESHOLDS:
            key = str(threshold)
            results[mode][key] = {}

            print(f"\n--- Threshold: {threshold:.0%} ---")

            for version in VERSIONS:
                metrics = summarize_ranks(
                    ranks[mode][version][threshold],
                    ks=(1, 5, 10),
                )

                results[mode][key][version] = metrics

                print(
                    f"{version}: "
                    f"Hit@1={metrics['hit@1']:.4f}, "
                    f"Hit@5={metrics['hit@5']:.4f}, "
                    f"Hit@10={metrics['hit@10']:.4f}, "
                    f"MRR@10={metrics['mrr@10']:.4f}"
                )

            v0 = results[mode][key]["V0"]
            v1 = results[mode][key]["V1"]

            print(
                "V1 - V0:",
                f"Hit@10={v1['hit@10'] - v0['hit@10']:+.4f},",
                f"MRR@10={v1['mrr@10'] - v0['mrr@10']:+.4f}",
            )

    # Mathematical invariant: joint coverage cannot
    # perform worse than individual-chunk coverage.
    for threshold in THRESHOLDS:
        key = str(threshold)

        for version in VERSIONS:
            single = results["single"][key][version]
            union = results["union"][key][version]

            for metric in ("hit@1", "hit@5", "hit@10", "mrr@10"):
                assert union[metric] >= single[metric] - 1e-9

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate QASPER retrieval coverage.")
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        required=True,
    )

    args = parser.parse_args()
    split = args.split

    print("\n========== Configuration ==========")
    print("Dataset:", split)
    print("Thresholds:", THRESHOLDS)
    print("Top K:", TOP_K)

    questions, stats = load_questions(split)

    print("\n========== Question Statistics ==========")
    for key, value in stats.items():
        print(f"{key}: {value}")

    embedder = TextEmbedder(device="cpu")
    index_dirs = get_index_dirs(split)

    indexes = {
        version: load_index(
            path,
            embedder.model_name,
        )
        for version, path in index_dirs.items()
    }

    eligible = [
        question
        for question in questions
        if all(question["paper_id"] in indexes[version] for version in VERSIONS)
    ]

    excluded_missing_index = len(questions) - len(eligible)

    print("\n========== Evaluation Set ==========")
    print("Eligible questions:", len(eligible))
    print("Excluded missing index:", excluded_missing_index)

    if not eligible:
        raise ValueError("No evaluable questions.")

    query_vectors = embedder.encode_documents(
        [item["question"] for item in eligible],
        batch_size=32,
        show_progress_bar=True,
    )

    print("Query embeddings:", query_vectors.shape)

    results = evaluate(
        eligible,
        query_vectors,
        indexes,
    )

    output = {
        "split": split,
        "model_name": embedder.model_name,
        "retrieval_scope": "within_paper",
        "top_k": TOP_K,
        "thresholds": list(THRESHOLDS),
        "question_statistics": {
            **stats,
            "excluded_missing_index": excluded_missing_index,
            "final_evaluable_questions": len(eligible),
        },
        "results": results,
    }

    output_path = Path(f"storage/metadata/coverage_comparison_{split}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:", output_path)
    print("Evaluation completed.")


if __name__ == "__main__":
    main()
