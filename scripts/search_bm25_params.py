"""Grid-search BM25 k1 and b on the frozen Dense V2 Validation set."""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from statistics import mean

from paper_rag.retrieval.bm25_retriever import BM25Retriever

# ============================================================
# Experiment configuration
# ============================================================

K1_VALUES = (
    0.50,
    0.75,
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
    2.25,
    2.50,
    2.75,
    3.00,
)

B_VALUES = (
    0.00,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.90,
    1.00,
)

GRID_OUTPUT_PATH = Path("storage/metadata/day7_bm25_parameter_grid_validation.json")

BEST_OUTPUT_PATH = Path("storage/metadata/day7_bm25_v2_best_eval_validation.json")


# ============================================================
# Load existing BM25 evaluation helpers
# ============================================================


def load_script_module(module_name: str, script_path: Path):
    """Load an existing evaluation script without executing main."""

    existing = sys.modules.get(module_name)

    if existing is not None:
        return existing

    if not script_path.is_file():
        raise FileNotFoundError(f"Evaluation script not found: {script_path}")

    spec = importlib.util.spec_from_file_location(
        module_name,
        script_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load evaluation script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return module


SCRIPT_DIRECTORY = Path(__file__).resolve().parent

EVALUATION = load_script_module(
    "_paperpilot_bm25_grid_helpers",
    SCRIPT_DIRECTORY / "evaluate_bm25_retrieval.py",
)

DENSE = EVALUATION.DENSE

SPLIT = EVALUATION.SPLIT
CURRENT_K1 = EVALUATION.DEFAULT_K1
CURRENT_B = EVALUATION.DEFAULT_B


# ============================================================
# Parameter grid
# ============================================================


def build_parameter_grid(
    k1_values: tuple[float, ...] = K1_VALUES,
    b_values: tuple[float, ...] = B_VALUES,
    current_k1: float = CURRENT_K1,
    current_b: float = CURRENT_B,
) -> list[tuple[float, float]]:
    """Build a validated parameter grid including the current baseline."""

    if not k1_values or not b_values:
        raise ValueError("Parameter search ranges cannot be empty.")

    candidates_k1 = sorted(set(k1_values) | {current_k1})
    candidates_b = sorted(set(b_values) | {current_b})

    combinations = [(k1, b) for k1 in candidates_k1 for b in candidates_b]

    # Validate every parameter combination before starting the experiment.
    for k1, b in combinations:
        BM25Retriever(k1=k1, b=b)

    return combinations


def apply_parameters(
    retriever: BM25Retriever,
    k1: float,
    b: float,
) -> None:
    """Change scoring parameters without rebuilding the inverted index."""

    validated = BM25Retriever(k1=k1, b=b)

    retriever.k1 = validated.k1
    retriever.b = validated.b


# ============================================================
# Ranking-only evaluation
# ============================================================


def evaluate_ranking_only(
    questions: list[dict],
    retriever: BM25Retriever,
    thresholds: tuple[float, ...],
) -> dict:
    """Evaluate BM25 ranking without repeating context tokenization."""

    if not questions:
        raise ValueError("No evaluation questions.")

    ranking_records = []

    for index, item in enumerate(questions, start=1):
        ranked_chunks = EVALUATION.rank_bm25_question(
            retriever=retriever,
            question=item["question"],
            paper_id=item["paper_id"],
        )

        raw_ranking = DENSE.evaluate_ranked_chunks(
            ranked_chunks=ranked_chunks,
            locations=item["locations"],
            thresholds=thresholds,
        )

        ranking_records.append(raw_ranking)

        if index % 100 == 0 or index == len(questions):
            print(f"    Evaluated questions: {index}/{len(questions)}")

    return DENSE.summarize_ranking_results(
        question_results=ranking_records,
        thresholds=thresholds,
    )


# ============================================================
# Parameter selection
# ============================================================


def calculate_objective(
    ranking: dict,
    thresholds: tuple[float, ...],
) -> dict:
    """Calculate macro-averaged UNION ranking metrics."""

    union_metrics = ranking["union"]

    mrr_values = [union_metrics[str(threshold)]["mrr@10"] for threshold in thresholds]

    hit10_values = [union_metrics[str(threshold)]["hit@10"] for threshold in thresholds]

    hit1_values = [union_metrics[str(threshold)]["hit@1"] for threshold in thresholds]

    return {
        "mean_union_mrr@10": mean(mrr_values),
        "mean_union_hit@10": mean(hit10_values),
        "mean_union_hit@1": mean(hit1_values),
    }


def selection_key(
    record: dict,
    current_k1: float,
    current_b: float,
) -> tuple:
    """Return the deterministic parameter selection priority."""

    objective = record["objective"]

    is_current = math.isclose(record["k1"], current_k1, abs_tol=1e-12) and math.isclose(
        record["b"], current_b, abs_tol=1e-12
    )

    distance = abs(record["k1"] - current_k1) + abs(record["b"] - current_b)

    return (
        round(objective["mean_union_mrr@10"], 12),
        round(objective["mean_union_hit@10"], 12),
        round(objective["mean_union_hit@1"], 12),
        is_current,
        -distance,
    )


def select_best_parameters(
    grid_results: list[dict],
    current_k1: float,
    current_b: float,
) -> dict:
    """Select the best parameter combination."""

    if not grid_results:
        raise ValueError("Parameter search produced no results.")

    return max(
        grid_results,
        key=lambda item: selection_key(
            item,
            current_k1,
            current_b,
        ),
    )


# ============================================================
# Output
# ============================================================


def save_json_exclusive(
    path: Path,
    data: dict,
) -> None:
    """Save JSON without overwriting an existing experiment."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("x", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


# ============================================================
# Main
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid search BM25 k1 and b on Validation."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    parser.add_argument(
        "--grid-output",
        type=Path,
        default=GRID_OUTPUT_PATH,
    )

    parser.add_argument(
        "--best-output",
        type=Path,
        default=BEST_OUTPUT_PATH,
    )

    args = parser.parse_args()

    if args.grid_output.resolve() == args.best_output.resolve():
        raise ValueError("Grid and best output paths must be different.")

    EVALUATION.ensure_output_available(args.grid_output)
    EVALUATION.ensure_output_available(args.best_output)

    print("\n========== Day 7: BM25 Parameter Search ==========")
    print("Split:", SPLIT)
    print("Current k1:", CURRENT_K1)
    print("Current b:", CURRENT_B)
    print("Grid output:", args.grid_output)
    print("Best output:", args.best_output)

    # --------------------------------------------------------
    # Step 1: Load frozen Dense baseline.
    # --------------------------------------------------------

    print("\n========== Step 1: Loading Dense Baseline ==========")

    dense_baseline = EVALUATION.load_dense_baseline(EVALUATION.DENSE_BASELINE_PATH)

    budgets = tuple(dense_baseline["context_budget_evaluation"]["budgets"])

    thresholds = tuple(dense_baseline["context_budget_evaluation"]["thresholds"])

    embedding_model = dense_baseline["embedding_model"]

    # --------------------------------------------------------
    # Step 2: Load and align evaluation questions.
    # --------------------------------------------------------

    print("\n========== Step 2: Loading Evaluation Questions ==========")

    questions, question_stats = DENSE.load_questions(SPLIT)

    for key, value in question_stats.items():
        print(f"{key}: {value}")

    if not questions:
        raise ValueError("No evaluation questions.")

    # --------------------------------------------------------
    # Step 3: Load frozen V2 Parent Chunk index.
    # --------------------------------------------------------

    print("\n========== Step 3: Loading V2 Parent Chunks ==========")

    indexes, v2_manifest, v2_build_config = DENSE.load_windowed_index(
        directory=EVALUATION.INDEX_DIRECTORY,
        model_name=embedding_model,
    )

    if dense_baseline.get("v2_index_manifest") != v2_manifest:
        raise ValueError("Loaded V2 index differs from the Dense baseline manifest.")

    if dense_baseline.get("v2_index_build_config") != v2_build_config:
        raise ValueError("Loaded V2 build configuration differs from Dense baseline.")

    eligible = [item for item in questions if item["paper_id"] in indexes]

    excluded_missing_index = len(questions) - len(eligible)

    if not eligible:
        raise ValueError("No evaluable questions with V2 indexes.")

    EVALUATION.validate_question_alignment(
        questions=eligible,
        dense_records=dense_baseline["per_question"],
    )

    print("Eligible questions:", len(eligible))
    print("Excluded missing index:", excluded_missing_index)
    print("Question alignment: PASS")

    # --------------------------------------------------------
    # Step 4: Build BM25 inverted index once.
    # --------------------------------------------------------

    print("\n========== Step 4: Building BM25 Index ==========")

    chunks = EVALUATION.collect_v2_parent_chunks(
        indexes=indexes,
        expected_split=SPLIT,
    )

    retriever = BM25Retriever(
        k1=CURRENT_K1,
        b=CURRENT_B,
    )

    retriever.build_index(chunks)

    if retriever.num_documents != v2_manifest["num_chunks"]:
        raise ValueError("BM25 document count does not match V2 manifest.")

    print("Indexed chunks:", retriever.num_documents)
    print("Vocabulary size:", retriever.vocabulary_size)
    print("Average document length:", retriever.average_document_length)

    # --------------------------------------------------------
    # Step 5: Generate parameter combinations.
    # --------------------------------------------------------

    print("\n========== Step 5: Preparing Parameter Grid ==========")

    combinations = build_parameter_grid()

    print("k1 candidates:", sorted({item[0] for item in combinations}))
    print("b candidates:", sorted({item[1] for item in combinations}))
    print("Total combinations:", len(combinations))

    # --------------------------------------------------------
    # Step 6: Run ranking-only grid search.
    # --------------------------------------------------------

    print("\n========== Step 6: Running BM25 Grid Search ==========")

    grid_results = []

    for grid_index, (k1, b) in enumerate(combinations, start=1):
        print(f"\n[{grid_index}/{len(combinations)}] k1={k1:.2f}, b={b:.2f}")

        apply_parameters(
            retriever=retriever,
            k1=k1,
            b=b,
        )

        ranking = evaluate_ranking_only(
            questions=eligible,
            retriever=retriever,
            thresholds=thresholds,
        )

        objective = calculate_objective(
            ranking=ranking,
            thresholds=thresholds,
        )

        grid_results.append(
            {
                "k1": k1,
                "b": b,
                "is_current": (
                    math.isclose(k1, CURRENT_K1, abs_tol=1e-12)
                    and math.isclose(b, CURRENT_B, abs_tol=1e-12)
                ),
                "objective": objective,
                "ranking": ranking,
            }
        )

        print(
            "Mean UNION MRR@10:",
            f"{objective['mean_union_mrr@10']:.6f}",
        )

    # --------------------------------------------------------
    # Step 7: Select best parameters.
    # --------------------------------------------------------

    print("\n========== Step 7: Selecting Best Parameters ==========")

    best = select_best_parameters(
        grid_results=grid_results,
        current_k1=CURRENT_K1,
        current_b=CURRENT_B,
    )

    current = next(item for item in grid_results if item["is_current"])

    print("\nCurrent parameters:")
    print(f"k1={CURRENT_K1:.2f}, b={CURRENT_B:.2f}")

    print("\nBest parameters:")
    print(f"k1={best['k1']:.2f}, b={best['b']:.2f}")

    print("\nCurrent objective:")
    print(current["objective"])

    print("\nBest objective:")
    print(best["objective"])

    # --------------------------------------------------------
    # Step 8: Load context tokenizer.
    # --------------------------------------------------------

    print("\n========== Step 8: Loading Context Tokenizer ==========")

    embedder = EVALUATION.TextEmbedder(
        device=args.device,
    )

    if embedder.model_name != embedding_model:
        raise ValueError("Tokenizer model does not match Dense baseline.")

    tokenizer = embedder.model.tokenizer

    separator_tokens = DENSE.count_tokens(
        tokenizer,
        DENSE.CHUNK_SEPARATOR,
    )

    print("Tokenizer:", embedder.model_name)
    print("Separator tokens:", separator_tokens)

    # --------------------------------------------------------
    # Step 9: Full context evaluation for current and best.
    # --------------------------------------------------------

    print("\n========== Step 9: Context Budget Evaluation ==========")

    parameter_sets = [
        ("current", CURRENT_K1, CURRENT_B),
    ]

    if not math.isclose(best["k1"], CURRENT_K1, abs_tol=1e-12) or not math.isclose(
        best["b"], CURRENT_B, abs_tol=1e-12
    ):
        parameter_sets.append(("best", best["k1"], best["b"]))

    full_evaluations = {}

    for label, k1, b in parameter_sets:
        print(f"\nEvaluating {label}: k1={k1:.2f}, b={b:.2f}")

        apply_parameters(
            retriever=retriever,
            k1=k1,
            b=b,
        )

        full_evaluations[label] = EVALUATION.evaluate_bm25(
            questions=eligible,
            retriever=retriever,
            tokenizer=tokenizer,
            budgets=budgets,
            thresholds=thresholds,
            separator_tokens=separator_tokens,
        )

    current_full = full_evaluations["current"]
    best_full = full_evaluations.get("best", current_full)

    # --------------------------------------------------------
    # Step 10: Construct best BM25 evaluation output.
    # --------------------------------------------------------

    best_output = {
        "schema_version": 1,
        "experiment": "day7_bm25_v2_best_eval",
        "split": SPLIT,
        "retrieval_method": "bm25_v2",
        "retrieval_scope": "within_paper",
        "bm25": {
            "k1": best["k1"],
            "b": best["b"],
            "idf_scope": "global_validation_parent_chunks",
            "idf_formula": ("log(1 + (N - df + 0.5) / (df + 0.5))"),
            "tokenization": (
                "Unicode word and number extraction, casefolded, "
                "without stopword removal or stemming."
            ),
            "candidate_policy": (
                "Return only positive-score lexical matches "
                "within the corresponding paper."
            ),
        },
        "parameter_selection": {
            "current_k1": CURRENT_K1,
            "current_b": CURRENT_B,
            "best_k1": best["k1"],
            "best_b": best["b"],
            "objective": best["objective"],
        },
        "ranking": best_full["ranking"],
        "context_budget_evaluation": {
            "tokenizer": embedder.model_name,
            "tokenizer_is_proxy": True,
            "special_tokens": False,
            "truncation": False,
            "context_separator": DENSE.CHUNK_SEPARATOR,
            "budgets": list(budgets),
            "thresholds": list(thresholds),
            "packing_policy": (
                "Same Dense V2 greedy ranking-order packing of "
                "complete parent chunks. Non-fitting chunks are skipped; "
                "later smaller chunks may still be selected. "
                "No chunk is truncated or split. "
                "The final assembled context is re-tokenized."
            ),
            "metrics": best_full["budgets"],
        },
        "question_statistics": {
            **question_stats,
            "excluded_missing_index": excluded_missing_index,
            "final_evaluable_questions": len(eligible),
            "questions_without_lexical_match": (
                best_full["questions_without_lexical_match"]
            ),
            "mean_matched_chunks": best_full["mean_matched_chunks"],
        },
        "v2_index_manifest": v2_manifest,
        "v2_index_build_config": v2_build_config,
        "dense_baseline": str(EVALUATION.DENSE_BASELINE_PATH),
        "bm25_index_statistics": {
            "num_documents": retriever.num_documents,
            "vocabulary_size": retriever.vocabulary_size,
            "average_document_length": retriever.average_document_length,
        },
        "token_cache_entries": best_full["token_cache_entries"],
        "per_question": best_full["per_question"],
    }

    # --------------------------------------------------------
    # Step 11: Construct grid search report.
    # --------------------------------------------------------

    sorted_results = sorted(
        grid_results,
        key=lambda item: selection_key(
            item,
            CURRENT_K1,
            CURRENT_B,
        ),
        reverse=True,
    )

    grid_output = {
        "schema_version": 1,
        "experiment": "day7_bm25_parameter_grid_search",
        "split": SPLIT,
        "retrieval_method": "bm25_v2",
        "retrieval_scope": "within_paper",
        "parameter_search": {
            "k1_values": sorted({item[0] for item in combinations}),
            "b_values": sorted({item[1] for item in combinations}),
            "total_combinations": len(combinations),
        },
        "selection_criteria": {
            "primary": "mean_union_mrr@10",
            "secondary": "mean_union_hit@10",
            "tertiary": "mean_union_hit@1",
            "thresholds": list(thresholds),
            "tie_policy": (
                "Prefer current parameters; otherwise prefer "
                "the closest combination to current parameters."
            ),
        },
        "current_parameters": {
            "k1": CURRENT_K1,
            "b": CURRENT_B,
            "objective": current["objective"],
        },
        "best_parameters": {
            "k1": best["k1"],
            "b": best["b"],
            "is_current": best["is_current"],
            "objective": best["objective"],
        },
        "context_budget_comparison": {
            "current": current_full["budgets"],
            "best": best_full["budgets"],
        },
        "question_statistics": {
            **question_stats,
            "excluded_missing_index": excluded_missing_index,
            "final_evaluable_questions": len(eligible),
        },
        "v2_index_manifest": v2_manifest,
        "grid_results": sorted_results,
    }

    # --------------------------------------------------------
    # Step 12: Save without overwriting previous experiments.
    # --------------------------------------------------------

    print("\n========== Step 12: Saving Results ==========")

    save_json_exclusive(
        args.grid_output,
        grid_output,
    )

    save_json_exclusive(
        args.best_output,
        best_output,
    )

    print("\n========== BM25 Parameter Search Completed ==========")
    print("Grid results:", args.grid_output)
    print("Best evaluation:", args.best_output)
    print("Best k1:", best["k1"])
    print("Best b:", best["b"])
    print("Best is current:", best["is_current"])


if __name__ == "__main__":
    main()
