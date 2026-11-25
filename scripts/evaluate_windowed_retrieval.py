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
from paper_rag.retrieval.windowed_index_store import WindowedIndexStore
from paper_rag.schemas import PaperDocument, QAExample

THRESHOLDS = (0.25, 0.5, 0.75)
TOP_K = 10
MODES = ("single", "union")
VERSIONS = ("V0", "V1", "V2")


def get_index_dirs(split: str) -> dict[str, Path]:
    """Return index directories for all three versions."""

    return {
        "V0": Path(f"data/indexes/qasper_{split}_dense"),
        "V1": Path(f"data/indexes/qasper_{split}_semantic_v1"),
        "V2": Path(f"data/indexes/qasper_{split}_windowed_v2"),
    }


def load_questions(split: str):
    """Load papers and prepare QA examples with body evidence."""

    paper_path = Path(f"data/processed/{split}_papers.jsonl")
    qa_path = Path(f"data/processed/{split}_qa.jsonl")

    papers = {}

    with paper_path.open(encoding="utf-8") as file:
        for line in tqdm(file, desc="Loading papers"):
            if not line.strip():
                continue

            paper = PaperDocument.model_validate_json(line)
            papers[paper.paper_id] = paper

    questions = {}
    missing_paper_records = 0

    with qa_path.open(encoding="utf-8") as file:
        for line in tqdm(file, desc="Loading QA"):
            if not line.strip():
                continue

            qa = QAExample.model_validate_json(line)

            if qa.paper_id not in papers:
                missing_paper_records += 1
                continue

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

                    if not evidence:
                        continue

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
        "missing_paper_records": missing_paper_records,
        "total_papers": len(papers),
    }

    return prepared, stats


def load_dense_index(directory: Path, model_name: str):
    """Load and group a conventional dense index by paper."""

    if not directory.is_dir():
        raise FileNotFoundError(f"Dense index directory does not exist: {directory}")

    chunks, vectors, _ = IndexStore(directory).load(expected_model_name=model_name)

    grouped_indices = defaultdict(list)

    for index, chunk in enumerate(chunks):
        grouped_indices[chunk.paper_id].append(index)

    grouped = {}

    for paper_id, indices in grouped_indices.items():
        indices = np.asarray(indices, dtype=np.int64)

        grouped[paper_id] = (
            [chunks[int(i)] for i in indices],
            vectors[indices],
        )

    print(f"Loaded {directory.name}: {len(chunks)} chunks, {len(grouped)} papers")

    return grouped


def load_windowed_index(directory: Path, model_name: str):
    """Load and group a windowed index by paper."""

    if not directory.is_dir():
        raise FileNotFoundError(f"Windowed index directory does not exist: {directory}")

    chunks, windows, vectors, manifest = WindowedIndexStore(directory).load(
        expected_model_name=model_name
    )

    chunks_by_paper = defaultdict(list)
    windows_by_paper = defaultdict(list)

    for chunk in chunks:
        chunks_by_paper[chunk.paper_id].append(chunk)

    for index, window in enumerate(windows):
        windows_by_paper[window.paper_id].append(index)

    grouped = {}

    for paper_id, paper_chunks in chunks_by_paper.items():
        indices = np.asarray(
            windows_by_paper[paper_id],
            dtype=np.int64,
        )

        chunks_by_id = {chunk.chunk_id: chunk for chunk in paper_chunks}

        grouped[paper_id] = {
            "chunks": paper_chunks,
            "chunks_by_id": chunks_by_id,
            "windows": [windows[int(i)] for i in indices],
            "embeddings": vectors[indices],
        }

    print(
        f"Loaded {directory.name}: "
        f"{len(chunks)} parent chunks, "
        f"{len(windows)} windows, "
        f"{len(grouped)} papers"
    )

    return grouped, manifest


def rank_windowed_chunks(
    query_vector: np.ndarray,
    index: dict,
    top_k: int,
):
    """Rank windows and return unique parent chunks.

    The ranking logic matches WindowedDenseRetriever.search:
    each parent receives the score of its highest-scoring window.
    """

    windows = index["windows"]
    embeddings = index["embeddings"]
    chunks_by_id = index["chunks_by_id"]

    scores = embeddings @ query_vector

    order = np.argsort(
        -scores,
        kind="stable",
    )

    ranked_chunks = []
    seen_parent_ids = set()

    for index_in_windows in order:
        window_index = int(index_in_windows)
        window = windows[window_index]

        parent_id = window.parent_chunk_id

        if parent_id in seen_parent_ids:
            continue

        chunk = chunks_by_id.get(parent_id)

        if chunk is None:
            raise ValueError(f"Unknown parent chunk: {parent_id}")

        seen_parent_ids.add(parent_id)
        ranked_chunks.append(chunk)

        if len(ranked_chunks) >= top_k:
            break

    return ranked_chunks


def evaluate(questions, query_vectors, indexes, top_k=TOP_K):
    """Evaluate all versions using identical questions and metrics."""

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
        paper_id = item["paper_id"]

        for version in VERSIONS:
            if version == "V2":
                paper_index = indexes["V2"][paper_id]

                ranked_chunks = rank_windowed_chunks(
                    query_vector,
                    paper_index,
                    top_k,
                )

            else:
                chunks, vectors = indexes[version][paper_id]

                scores = vectors @ query_vector

                order = np.argsort(
                    -scores,
                    kind="stable",
                )[:top_k]

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
            v2 = results[mode][key]["V2"]

            print("\nV2 - V0:")
            print(
                f"Hit@10={v2['hit@10'] - v0['hit@10']:+.4f}, "
                f"MRR@10={v2['mrr@10'] - v0['mrr@10']:+.4f}"
            )

            print("V2 - V1:")
            print(
                f"Hit@10={v2['hit@10'] - v1['hit@10']:+.4f}, "
                f"MRR@10={v2['mrr@10'] - v1['mrr@10']:+.4f}"
            )

    # Mathematical invariant:
    # UNION retrieval cannot perform worse than SINGLE retrieval.
    for threshold in THRESHOLDS:
        key = str(threshold)

        for version in VERSIONS:
            single = results["single"][key][version]
            union = results["union"][key][version]

            for metric in (
                "hit@1",
                "hit@5",
                "hit@10",
                "mrr@10",
            ):
                assert union[metric] >= single[metric] - 1e-9, (
                    f"UNION invariant failed: {version}, {threshold}, {metric}"
                )

    # Paired comparison: smaller rank is better.
    paired = {}

    for mode in MODES:
        paired[mode] = {}

        for threshold in THRESHOLDS:
            key = str(threshold)
            paired[mode][key] = {}

            v2_ranks = ranks[mode]["V2"][threshold]

            for baseline in ("V0", "V1"):
                baseline_ranks = ranks[mode][baseline][threshold]

                v2_better = 0
                baseline_better = 0
                tie = 0

                for v2_rank, base_rank in zip(
                    v2_ranks,
                    baseline_ranks,
                    strict=True,
                ):
                    v2_value = float("inf") if v2_rank is None else v2_rank

                    base_value = float("inf") if base_rank is None else base_rank

                    if v2_value < base_value:
                        v2_better += 1
                    elif v2_value > base_value:
                        baseline_better += 1
                    else:
                        tie += 1

                paired[mode][key][baseline] = {
                    "V2_better": v2_better,
                    "baseline_better": baseline_better,
                    "tie": tie,
                }

                print(f"\nPaired {mode.upper()} {threshold:.0%}: V2 vs {baseline}")
                print("V2 better:", v2_better)
                print("Baseline better:", baseline_better)
                print("Tie:", tie)

    return results, paired


def main():
    parser = argparse.ArgumentParser(description="Compare V0, V1 and V2 retrieval.")

    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="train",
    )

    args = parser.parse_args()
    split = args.split

    print("\n========== Configuration ==========")
    print("Dataset:", split)
    print("Versions:", VERSIONS)
    print("Thresholds:", THRESHOLDS)
    print("Top K:", TOP_K)

    questions, stats = load_questions(split)

    print("\n========== Question Statistics ==========")

    for key, value in stats.items():
        print(f"{key}: {value}")

    embedder = TextEmbedder(device="cpu")

    index_dirs = get_index_dirs(split)

    indexes = {
        "V0": load_dense_index(
            index_dirs["V0"],
            embedder.model_name,
        ),
        "V1": load_dense_index(
            index_dirs["V1"],
            embedder.model_name,
        ),
    }

    indexes["V2"], v2_manifest = load_windowed_index(
        index_dirs["V2"],
        embedder.model_name,
    )

    print("\n========== V2 Index Configuration ==========")
    print(
        json.dumps(
            v2_manifest,
            ensure_ascii=False,
            indent=2,
        )
    )

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

    results, paired = evaluate(
        eligible,
        query_vectors,
        indexes,
        top_k=TOP_K,
    )

    output = {
        "split": split,
        "model_name": embedder.model_name,
        "retrieval_scope": "within_paper",
        "top_k": TOP_K,
        "thresholds": list(THRESHOLDS),
        "versions": list(VERSIONS),
        "question_statistics": {
            **stats,
            "excluded_missing_index": excluded_missing_index,
            "final_evaluable_questions": len(eligible),
        },
        "v2_index_manifest": v2_manifest,
        "results": results,
        "paired_comparisons": paired,
    }

    output_path = Path(f"storage/metadata/retrieval_eval_{split}_v0_v1_v2.json")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n========== Evaluation Completed ==========")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
