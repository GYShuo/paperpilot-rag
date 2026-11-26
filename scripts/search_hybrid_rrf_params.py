"""Search RRF parameters using frozen Dense V2 and optimized BM25 V2 rankings."""

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from paper_rag.retrieval.bm25_retriever import BM25Retriever
from paper_rag.retrieval.hybrid_retriever import HybridRetriever

# ============================================================
# Experiment configuration
# ============================================================

SPLIT = "validation"

DENSE_BASELINE_PATH = Path("storage/metadata/day6_dense_v2_eval_validation.json")
HYBRID_BASELINE_PATH = Path("storage/metadata/day6_hybrid_rrf_v2_eval_validation.json")
BM25_OPTIMIZED_PATH = Path("storage/metadata/day7_bm25_v2_best_eval_validation.json")

OUTPUT_PATH = Path("storage/metadata/day7_rrf_rank_grid_validation_bm25_optimized.json")

INDEX_DIRECTORY = Path("data/indexes/qasper_validation_windowed_v2")

RRF_K_VALUES = (10, 20, 40, 60, 100, 200)
DENSE_WEIGHT = 1.0
BM25_WEIGHTS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)

TOP_K = 10


# ============================================================
# Reuse existing evaluation modules
# ============================================================


def load_script_module(module_name: str, script_path: Path):
    """Load an evaluation script without executing its main function."""

    existing = sys.modules.get(module_name)

    if existing is not None:
        return existing

    if not script_path.is_file():
        raise FileNotFoundError(f"Evaluation helper script not found: {script_path}")

    spec = importlib.util.spec_from_file_location(
        module_name,
        script_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load evaluation helper: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return module


SCRIPT_DIRECTORY = Path(__file__).resolve().parent

BM25_EVALUATION = load_script_module(
    "_paperpilot_day7_bm25_helpers",
    SCRIPT_DIRECTORY / "evaluate_bm25_retrieval.py",
)

# Use the exact same Dense helper module as BM25 evaluation.
DENSE = BM25_EVALUATION.DENSE


# ============================================================
# Data structures
# ============================================================


@dataclass(frozen=True)
class FusionConfig:
    """One RRF parameter combination."""

    rrf_k: int
    dense_weight: float
    bm25_weight: float


# ============================================================
# Parameter grid
# ============================================================


def build_parameter_grid() -> list[FusionConfig]:
    """Generate the complete RRF parameter grid."""

    return [
        FusionConfig(
            rrf_k=rrf_k,
            dense_weight=DENSE_WEIGHT,
            bm25_weight=bm25_weight,
        )
        for rrf_k in RRF_K_VALUES
        for bm25_weight in BM25_WEIGHTS
    ]


def config_id(config: FusionConfig) -> str:
    """Create a stable identifier for one parameter combination."""

    weight = f"{config.bm25_weight:g}".replace(".", "p")
    return f"k{config.rrf_k}_wb{weight}"


def fuse_rankings(
    dense_results: list,
    bm25_results: list,
    config: FusionConfig,
):
    """Apply the existing HybridRetriever with supplied parameters."""

    retriever = HybridRetriever(
        rrf_k=config.rrf_k,
        dense_weight=config.dense_weight,
        bm25_weight=config.bm25_weight,
    )

    return retriever.fuse(
        dense_results=dense_results,
        bm25_results=bm25_results,
    )


# ============================================================
# Load and validate baselines
# ============================================================


def load_hybrid_baseline(path: Path) -> dict:
    """Load the frozen Day 6 Hybrid result as a Dense ranking reference."""

    if not path.is_file():
        raise FileNotFoundError(f"Hybrid V2 baseline not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise TypeError("Hybrid baseline must be a JSON object.")

    expected_metadata = {
        "experiment": "day6_hybrid_rrf_v2_baseline",
        "split": SPLIT,
        "retrieval_method": "hybrid_rrf_v2",
        "retrieval_scope": "within_paper",
    }

    for key, expected in expected_metadata.items():
        if data.get(key) != expected:
            raise ValueError(
                f"Invalid Hybrid baseline metadata: {key}="
                f"{data.get(key)!r}, expected {expected!r}."
            )

    fusion = data.get("fusion")

    if not isinstance(fusion, dict):
        raise TypeError("Hybrid fusion configuration must be an object.")

    if fusion.get("algorithm") != "reciprocal_rank_fusion":
        raise ValueError("Unexpected fusion algorithm.")

    if fusion.get("rrf_k") != 60:
        raise ValueError("Expected frozen Day 6 RRF k=60.")

    if fusion.get("dense_weight") != 1.0:
        raise ValueError("Expected frozen Dense weight=1.0.")

    if fusion.get("bm25_weight") != 1.0:
        raise ValueError("Expected frozen BM25 weight=1.0.")

    records = data.get("per_question")

    if not isinstance(records, list) or not records:
        raise ValueError("Hybrid baseline has no per-question records.")

    return data


def load_bm25_optimized_result(path: Path) -> dict:
    """Load and validate the optimized Day 7 BM25 evaluation."""

    if not path.is_file():
        raise FileNotFoundError(f"Optimized BM25 result not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise TypeError("Optimized BM25 result must be a JSON object.")

    expected_metadata = {
        "experiment": "day7_bm25_v2_best_eval",
        "split": SPLIT,
        "retrieval_method": "bm25_v2",
        "retrieval_scope": "within_paper",
    }

    for key, expected in expected_metadata.items():
        if data.get(key) != expected:
            raise ValueError(
                f"Invalid optimized BM25 metadata: {key}="
                f"{data.get(key)!r}, expected {expected!r}."
            )

    bm25 = data.get("bm25")

    if not isinstance(bm25, dict):
        raise TypeError("Optimized BM25 configuration must be an object.")

    k1 = bm25.get("k1")
    b = bm25.get("b")

    if isinstance(k1, bool) or not isinstance(k1, (int, float)):
        raise TypeError("Optimized BM25 k1 must be numeric.")

    if isinstance(b, bool) or not isinstance(b, (int, float)):
        raise TypeError("Optimized BM25 b must be numeric.")

    # Reuse the production constructor for parameter validation.
    BM25Retriever(k1=k1, b=b)

    if bm25.get("idf_scope") != "global_validation_parent_chunks":
        raise ValueError("Optimized BM25 IDF scope is incorrect.")

    selection = data.get("parameter_selection")

    if not isinstance(selection, dict):
        raise TypeError("BM25 parameter selection must be an object.")

    if not math.isclose(
        float(selection.get("best_k1", -1)),
        float(k1),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("BM25 best_k1 does not match its configuration.")

    if not math.isclose(
        float(selection.get("best_b", -1)),
        float(b),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("BM25 best_b does not match its configuration.")

    records = data.get("per_question")

    if not isinstance(records, list) or not records:
        raise ValueError("Optimized BM25 result has no per-question records.")

    statistics = data.get("question_statistics")

    if not isinstance(statistics, dict):
        raise TypeError("BM25 question statistics must be an object.")

    if statistics.get("final_evaluable_questions") != len(records):
        raise ValueError("Optimized BM25 question count mismatch.")

    index_manifest = data.get("v2_index_manifest")
    build_config = data.get("v2_index_build_config")

    if not isinstance(index_manifest, dict):
        raise TypeError("BM25 V2 manifest must be an object.")

    if not isinstance(build_config, dict):
        raise TypeError("BM25 V2 build configuration must be an object.")

    index_statistics = data.get("bm25_index_statistics")

    if not isinstance(index_statistics, dict):
        raise TypeError("BM25 index statistics must be an object.")

    context_config = data.get("context_budget_evaluation")

    if not isinstance(context_config, dict):
        raise TypeError("BM25 context budget configuration must be an object.")

    return data


def validate_shared_configuration(
    dense_baseline: dict,
    hybrid_baseline: dict,
    bm25_result: dict,
) -> None:
    """Ensure all experiments share identical evaluation settings."""

    dense_config = dense_baseline["context_budget_evaluation"]
    hybrid_config = hybrid_baseline["context_budget_evaluation"]
    bm25_config = bm25_result["context_budget_evaluation"]

    shared_fields = (
        "budgets",
        "thresholds",
        "tokenizer",
        "tokenizer_is_proxy",
        "special_tokens",
        "truncation",
        "context_separator",
    )

    for name, config in (
        ("Hybrid", hybrid_config),
        ("Optimized BM25", bm25_config),
    ):
        for key in shared_fields:
            if dense_config.get(key) != config.get(key):
                raise ValueError(f"Dense and {name} evaluation settings differ: {key}.")

    if dense_baseline["embedding_model"] != dense_config["tokenizer"]:
        raise ValueError("Dense embedding model and tokenizer differ.")


def validate_index_compatibility(
    dense_baseline: dict,
    hybrid_baseline: dict,
    bm25_result: dict,
    manifest: dict,
    build_config: dict,
) -> None:
    """Verify all baselines refer to the loaded V2 index."""

    for name, baseline in (
        ("Dense", dense_baseline),
        ("Day 6 Hybrid", hybrid_baseline),
        ("Optimized BM25", bm25_result),
    ):
        if baseline.get("v2_index_manifest") != manifest:
            raise ValueError(f"{name} baseline manifest mismatch.")

        if baseline.get("v2_index_build_config") != build_config:
            raise ValueError(f"{name} baseline build configuration mismatch.")


def validate_bm25_index_statistics(
    baseline: dict,
    retriever: BM25Retriever,
) -> None:
    """Verify the rebuilt BM25 index against the optimized result."""

    expected = baseline["bm25_index_statistics"]

    for key, actual in (
        ("num_documents", retriever.num_documents),
        ("vocabulary_size", retriever.vocabulary_size),
    ):
        expected_value = expected.get(key)

        if not isinstance(expected_value, int) or isinstance(expected_value, bool):
            raise TypeError(f"BM25 index statistic {key} must be an integer.")

        if expected_value != actual:
            raise ValueError(f"BM25 index statistic mismatch: {key}.")

    expected_average = expected.get("average_document_length")

    if isinstance(expected_average, bool) or not isinstance(
        expected_average,
        (int, float),
    ):
        raise TypeError("Average document length must be numeric.")

    if not math.isfinite(expected_average):
        raise ValueError("Average document length must be finite.")

    if not math.isclose(
        retriever.average_document_length,
        expected_average,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("BM25 average document length mismatch.")


# ============================================================
# Validation helpers
# ============================================================


def assert_nested_close(
    actual,
    expected,
    path: str = "metrics",
) -> None:
    """Compare nested metric structures with numeric tolerance."""

    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"Metric structure mismatch at {path}.")

        for key in expected:
            assert_nested_close(
                actual[key],
                expected[key],
                f"{path}.{key}",
            )
        return

    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"Metric list mismatch at {path}.")

        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            assert_nested_close(
                actual_item,
                expected_item,
                f"{path}[{index}]",
            )
        return

    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            raise TypeError(f"Metric type mismatch at {path}.")

        if not math.isclose(
            actual,
            expected,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Metric mismatch at {path}: {actual} != {expected}.")
        return

    if actual != expected:
        raise ValueError(f"Metric mismatch at {path}.")


def assert_top_10_matches(
    ranked_chunks: list,
    expected_top_10: list,
    source_name: str,
) -> None:
    """Verify the top-10 ranking and scores against a frozen result."""

    if not isinstance(expected_top_10, list):
        raise TypeError(f"{source_name} top_10 must be a list.")

    expected_count = min(TOP_K, len(ranked_chunks))

    if len(expected_top_10) != expected_count:
        raise ValueError(f"{source_name} top-10 candidate count mismatch.")

    for actual, expected in zip(
        ranked_chunks[:TOP_K],
        expected_top_10,
        strict=True,
    ):
        if actual.rank != expected.get("rank"):
            raise ValueError(f"{source_name} top-10 rank mismatch.")

        if actual.chunk.chunk_id != expected.get("chunk_id"):
            raise ValueError(f"{source_name} top-10 chunk ID mismatch.")

        expected_score = expected.get("score")

        if isinstance(expected_score, bool) or not isinstance(
            expected_score,
            (int, float),
        ):
            raise TypeError(f"{source_name} top-10 score must be numeric.")

        if not math.isfinite(expected_score):
            raise ValueError(f"{source_name} top-10 score must be finite.")

        if not math.isclose(
            actual.score,
            expected_score,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{source_name} top-10 score mismatch.")


# ============================================================
# Reconstruct frozen Dense rankings
# ============================================================


def prepare_question_records(
    questions: list[dict],
    dense_records: list[dict],
    hybrid_records: list[dict],
    indexes: dict,
    thresholds: tuple[float, ...],
) -> list[dict]:
    """Reconstruct complete frozen Dense rankings from Day 6 Hybrid records.

    The Day 6 Hybrid result stores every Dense candidate and its original
    Dense rank and score. Old BM25 ranks and scores are deliberately ignored.
    """

    if len(questions) != len(dense_records):
        raise ValueError("Question and Dense baseline counts differ.")

    if len(questions) != len(hybrid_records):
        raise ValueError("Question and Hybrid baseline counts differ.")

    prepared = []

    for question, dense_record, hybrid_record in zip(
        questions,
        dense_records,
        hybrid_records,
        strict=True,
    ):
        paper_id = question["paper_id"]

        for key in ("paper_id", "question_id", "question"):
            if question[key] != dense_record[key]:
                raise ValueError(f"Dense question alignment mismatch: {key}.")

            if question[key] != hybrid_record[key]:
                raise ValueError(f"Hybrid question alignment mismatch: {key}.")

        if paper_id not in indexes:
            raise KeyError(f"Missing V2 index for paper {paper_id}.")

        chunk_map = indexes[paper_id]["chunks_by_id"]
        candidates = hybrid_record.get("ranked_candidates")

        if not isinstance(candidates, list) or not candidates:
            raise ValueError(
                f"Missing complete Dense candidates for "
                f"{paper_id}/{question['question_id']}."
            )

        candidate_ids = [item.get("chunk_id") for item in candidates]

        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("Hybrid baseline contains duplicate chunk IDs.")

        if set(candidate_ids) != set(chunk_map):
            raise ValueError(
                f"Hybrid candidate set differs from V2 index for {paper_id}."
            )

        if hybrid_record.get("dense_candidate_count") != len(candidates):
            raise ValueError("Dense candidate count mismatch.")

        if hybrid_record.get("hybrid_candidate_count") != len(candidates):
            raise ValueError("Hybrid candidate count mismatch.")

        saved_hybrid_ranks = [item.get("rank") for item in candidates]

        if saved_hybrid_ranks != list(range(1, len(candidates) + 1)):
            raise ValueError("Saved Hybrid ranks are not consecutive.")

        dense_results = []
        dense_ranks = []

        for item in candidates:
            dense_rank = item.get("dense_rank")
            dense_score = item.get("dense_score")

            if (
                not isinstance(dense_rank, int)
                or isinstance(dense_rank, bool)
                or dense_rank <= 0
            ):
                raise TypeError("Invalid Dense rank in cached candidates.")

            if isinstance(dense_score, bool) or not isinstance(
                dense_score,
                (int, float),
            ):
                raise TypeError("Invalid Dense score in cached candidates.")

            if not math.isfinite(dense_score):
                raise ValueError("Dense score must be finite.")

            chunk = chunk_map[item["chunk_id"]]

            if chunk.paper_id != paper_id or chunk.split != SPLIT:
                raise ValueError("Cached Dense candidate belongs to wrong paper.")

            dense_ranks.append(dense_rank)

            dense_results.append(
                DENSE.RankedChunk(
                    rank=dense_rank,
                    score=float(dense_score),
                    chunk=chunk,
                )
            )

        dense_results.sort(key=lambda item: item.rank)

        if sorted(dense_ranks) != list(range(1, len(candidates) + 1)):
            raise ValueError("Dense ranks do not cover all Parent Chunks.")

        # Verify the cached complete Dense ranking against the frozen
        # Day 6 Dense top-10 results and raw evidence-ranking metrics.
        assert_top_10_matches(
            ranked_chunks=dense_results,
            expected_top_10=dense_record.get("top_10"),
            source_name="Dense",
        )

        actual_raw_ranking = DENSE.evaluate_ranked_chunks(
            ranked_chunks=dense_results,
            locations=question["locations"],
            thresholds=thresholds,
        )

        assert_nested_close(
            actual_raw_ranking,
            dense_record.get("raw_ranking"),
            path="dense.raw_ranking",
        )

        prepared.append(
            {
                "question": question,
                "dense_results": dense_results,
            }
        )

    return prepared


# ============================================================
# Recompute optimized BM25 rankings
# ============================================================


def attach_optimized_bm25_rankings(
    prepared: list[dict],
    retriever: BM25Retriever,
    optimized_records: list[dict],
    thresholds: tuple[float, ...],
) -> None:
    """Recompute BM25 rankings and verify them against Day 7 BM25 results."""

    if len(prepared) != len(optimized_records):
        raise ValueError("Optimized BM25 question count mismatch.")

    for item, baseline_record in zip(
        prepared,
        optimized_records,
        strict=True,
    ):
        question = item["question"]

        for key in ("paper_id", "question_id", "question"):
            if question[key] != baseline_record[key]:
                raise ValueError(f"Optimized BM25 question alignment mismatch: {key}.")

        bm25_results = BM25_EVALUATION.rank_bm25_question(
            retriever=retriever,
            question=question["question"],
            paper_id=question["paper_id"],
        )

        if baseline_record.get("matched_chunk_count") != len(bm25_results):
            raise ValueError("Optimized BM25 matched chunk count mismatch.")

        assert_top_10_matches(
            ranked_chunks=bm25_results,
            expected_top_10=baseline_record.get("top_10"),
            source_name="Optimized BM25",
        )

        actual_raw_ranking = DENSE.evaluate_ranked_chunks(
            ranked_chunks=bm25_results,
            locations=question["locations"],
            thresholds=thresholds,
        )

        assert_nested_close(
            actual_raw_ranking,
            baseline_record.get("raw_ranking"),
            path="optimized_bm25.raw_ranking",
        )

        dense_ids = {result.chunk.chunk_id for result in item["dense_results"]}

        bm25_ids = {result.chunk.chunk_id for result in bm25_results}

        if not bm25_ids.issubset(dense_ids):
            raise ValueError("BM25 candidates are missing from Dense candidates.")

        item["bm25_results"] = bm25_results


# ============================================================
# Ranking metrics
# ============================================================


def calculate_objective(
    summary: dict,
    thresholds: tuple[float, ...],
) -> dict:
    """Calculate the multi-threshold ranking selection objective."""

    if not thresholds:
        raise ValueError("Evidence thresholds cannot be empty.")

    union_metrics = [summary["union"][str(threshold)] for threshold in thresholds]

    primary_threshold = 0.5 if 0.5 in thresholds else thresholds[0]
    primary_metrics = summary["union"][str(primary_threshold)]

    return {
        "mean_union_mrr_at_10": mean(item["mrr@10"] for item in union_metrics),
        "mean_union_hit_at_1": mean(item["hit@1"] for item in union_metrics),
        "primary_threshold": primary_threshold,
        "primary_union_mrr_at_10": primary_metrics["mrr@10"],
        "primary_union_hit_at_1": primary_metrics["hit@1"],
    }


def evaluate_config(
    prepared: list[dict],
    config: FusionConfig,
    thresholds: tuple[float, ...],
) -> dict:
    """Evaluate one RRF configuration on the complete question set."""

    ranking_records = []

    for item in prepared:
        fused = fuse_rankings(
            item["dense_results"],
            item["bm25_results"],
            config,
        )

        ranked_chunks = [
            DENSE.RankedChunk(
                rank=result.rank,
                score=result.score,
                chunk=result.chunk,
            )
            for result in fused
        ]

        dense_ids = {result.chunk.chunk_id for result in item["dense_results"]}

        hybrid_ids = {result.chunk.chunk_id for result in ranked_chunks}

        if hybrid_ids != dense_ids or len(ranked_chunks) != len(dense_ids):
            raise ValueError("Hybrid candidate set differs from Dense candidates.")

        ranking_records.append(
            DENSE.evaluate_ranked_chunks(
                ranked_chunks=ranked_chunks,
                locations=item["question"]["locations"],
                thresholds=thresholds,
            )
        )

    summary = DENSE.summarize_ranking_results(
        question_results=ranking_records,
        thresholds=thresholds,
    )

    return {
        "config_id": config_id(config),
        "config": asdict(config),
        "objective": calculate_objective(summary, thresholds),
        "ranking": summary,
    }


# ============================================================
# Output protection
# ============================================================


def save_json_exclusive(
    path: Path,
    data: dict,
) -> None:
    """Save JSON without overwriting existing experiment results."""

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
        description=(
            "Search RRF parameters using frozen Dense V2 and "
            "optimized BM25 V2 rankings."
        )
    )

    parser.add_argument(
        "--bm25-result",
        type=Path,
        default=BM25_OPTIMIZED_PATH,
        help="Optimized Day 7 BM25 evaluation JSON.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="New RRF grid search output JSON.",
    )

    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {args.output}")

    configs = build_parameter_grid()

    print("\n========== Day 7: RRF Search with Optimized BM25 ==========")
    print("Dataset:", SPLIT)
    print("Dense weight:", DENSE_WEIGHT)
    print("RRF k values:", RRF_K_VALUES)
    print("BM25 weights:", BM25_WEIGHTS)
    print("Total configurations:", len(configs))
    print("BM25 optimized result:", args.bm25_result)
    print("Output:", args.output)

    # --------------------------------------------------------
    # Step 1: Load frozen Dense, Hybrid and optimized BM25 results.
    # --------------------------------------------------------

    print("\n========== Step 1: Loading Experiment Results ==========")

    dense_baseline = BM25_EVALUATION.load_dense_baseline(DENSE_BASELINE_PATH)

    hybrid_baseline = load_hybrid_baseline(HYBRID_BASELINE_PATH)

    bm25_result = load_bm25_optimized_result(args.bm25_result)

    validate_shared_configuration(
        dense_baseline=dense_baseline,
        hybrid_baseline=hybrid_baseline,
        bm25_result=bm25_result,
    )

    budgets = tuple(dense_baseline["context_budget_evaluation"]["budgets"])

    thresholds = tuple(dense_baseline["context_budget_evaluation"]["thresholds"])

    embedding_model = dense_baseline["embedding_model"]
    bm25_config = bm25_result["bm25"]

    print("Dense baseline:", DENSE_BASELINE_PATH)
    print("Dense reference:", HYBRID_BASELINE_PATH)
    print("Optimized BM25:", args.bm25_result)
    print("BM25 k1:", bm25_config["k1"])
    print("BM25 b:", bm25_config["b"])
    print("Evidence thresholds:", thresholds)
    print("Context budgets:", budgets)

    # --------------------------------------------------------
    # Step 2: Load evaluation questions and V2 index.
    # --------------------------------------------------------

    print("\n========== Step 2: Loading Evaluation Questions ==========")

    questions, question_stats = DENSE.load_questions(SPLIT)

    for key, value in question_stats.items():
        print(f"{key}: {value}")

    if not questions:
        raise ValueError("No evaluation questions.")

    print("\n========== Step 3: Loading V2 Index ==========")

    indexes, manifest, build_config = DENSE.load_windowed_index(
        directory=INDEX_DIRECTORY,
        model_name=embedding_model,
    )

    eligible = [item for item in questions if item["paper_id"] in indexes]

    excluded_missing_index = len(questions) - len(eligible)

    if not eligible:
        raise ValueError("No evaluable questions with V2 indexes.")

    # --------------------------------------------------------
    # Step 4: Validate question and index alignment.
    # --------------------------------------------------------

    print("\n========== Step 4: Verifying Experiment Alignment ==========")

    BM25_EVALUATION.validate_question_alignment(
        eligible,
        dense_baseline["per_question"],
    )

    BM25_EVALUATION.validate_question_alignment(
        eligible,
        hybrid_baseline["per_question"],
    )

    BM25_EVALUATION.validate_question_alignment(
        eligible,
        bm25_result["per_question"],
    )

    validate_index_compatibility(
        dense_baseline=dense_baseline,
        hybrid_baseline=hybrid_baseline,
        bm25_result=bm25_result,
        manifest=manifest,
        build_config=build_config,
    )

    print("Question IDs and order: PASS")
    print("Question text: PASS")
    print("V2 index manifest and build config: PASS")
    print("Eligible questions:", len(eligible))
    print("Excluded missing index:", excluded_missing_index)

    # --------------------------------------------------------
    # Step 5: Restore and verify frozen Dense rankings.
    # --------------------------------------------------------

    print("\n========== Step 5: Restoring Frozen Dense Rankings ==========")

    prepared = prepare_question_records(
        questions=eligible,
        dense_records=dense_baseline["per_question"],
        hybrid_records=hybrid_baseline["per_question"],
        indexes=indexes,
        thresholds=thresholds,
    )

    print("Dense questions:", len(prepared))
    print("Dense top-10 alignment: PASS")
    print("Dense raw ranking metrics: PASS")

    # --------------------------------------------------------
    # Step 6: Build BM25 index using optimized parameters.
    # --------------------------------------------------------

    print("\n========== Step 6: Building Optimized BM25 Index ==========")

    chunks = BM25_EVALUATION.collect_v2_parent_chunks(
        indexes=indexes,
        expected_split=SPLIT,
    )

    bm25_retriever = BM25Retriever(
        k1=bm25_config["k1"],
        b=bm25_config["b"],
    )

    bm25_retriever.build_index(chunks)

    if bm25_retriever.num_documents != manifest["num_chunks"]:
        raise ValueError("BM25 document count does not match V2 manifest.")

    if bm25_retriever.split != SPLIT:
        raise ValueError("BM25 index split does not match Validation.")

    validate_bm25_index_statistics(
        baseline=bm25_result,
        retriever=bm25_retriever,
    )

    print("Indexed Parent Chunks:", bm25_retriever.num_documents)
    print("Vocabulary Size:", bm25_retriever.vocabulary_size)
    print("Average Document Length:", bm25_retriever.average_document_length)
    print("BM25 index statistics: PASS")

    # --------------------------------------------------------
    # Step 7: Recompute and verify optimized BM25 rankings.
    # --------------------------------------------------------

    print("\n========== Step 7: Recomputing Optimized BM25 Rankings ==========")

    attach_optimized_bm25_rankings(
        prepared=prepared,
        retriever=bm25_retriever,
        optimized_records=bm25_result["per_question"],
        thresholds=thresholds,
    )

    print("BM25 questions:", len(prepared))
    print("BM25 top-10 alignment: PASS")
    print("BM25 raw ranking metrics: PASS")

    # --------------------------------------------------------
    # Step 8: Run the RRF parameter grid.
    # --------------------------------------------------------

    print("\n========== Step 8: Evaluating RRF Parameter Grid ==========")

    results = []

    for index, config in enumerate(configs, start=1):
        result = evaluate_config(
            prepared=prepared,
            config=config,
            thresholds=thresholds,
        )

        results.append(result)

        objective = result["objective"]

        print(
            f"[{index:02d}/{len(configs)}] "
            f"k={config.rrf_k:<3} "
            f"BM25 weight={config.bm25_weight:<4} "
            f"Mean MRR@10={objective['mean_union_mrr_at_10']:.4f} "
            f"Mean Hit@1={objective['mean_union_hit_at_1']:.4f}"
        )

    # --------------------------------------------------------
    # Step 9: Sort and report results.
    # --------------------------------------------------------

    results.sort(
        key=lambda item: (
            -item["objective"]["mean_union_mrr_at_10"],
            -item["objective"]["mean_union_hit_at_1"],
            -item["objective"]["primary_union_mrr_at_10"],
            item["config"]["rrf_k"],
            item["config"]["bm25_weight"],
        )
    )

    print("\n========== Top 10 RRF Configurations ==========")

    for rank, result in enumerate(results[:10], start=1):
        config = result["config"]
        objective = result["objective"]

        print(
            f"{rank:02d}. "
            f"k={config['rrf_k']:<3} "
            f"BM25 weight={config['bm25_weight']:<4} "
            f"Mean MRR@10={objective['mean_union_mrr_at_10']:.4f} "
            f"Mean Hit@1={objective['mean_union_hit_at_1']:.4f}"
        )

    # The original Day 6 fusion weights are retained as a reference
    # configuration, but now evaluated with optimized BM25 rankings.
    reference_config = FusionConfig(
        rrf_k=60,
        dense_weight=1.0,
        bm25_weight=1.0,
    )

    reference_result = next(
        result
        for result in results
        if result["config_id"] == config_id(reference_config)
    )

    best_result = results[0]

    # --------------------------------------------------------
    # Step 10: Construct experiment output.
    # --------------------------------------------------------

    questions_without_bm25 = sum(not item["bm25_results"] for item in prepared)

    mean_dense_candidates = mean(len(item["dense_results"]) for item in prepared)

    mean_bm25_candidates = mean(len(item["bm25_results"]) for item in prepared)

    output = {
        "schema_version": 1,
        "experiment": "day7_rrf_rank_grid_search_optimized_bm25",
        "split": SPLIT,
        "question_statistics": {
            **question_stats,
            "excluded_missing_index": excluded_missing_index,
            "final_evaluable_questions": len(prepared),
            "questions_without_bm25_candidates": questions_without_bm25,
            "mean_dense_candidates": mean_dense_candidates,
            "mean_bm25_candidates": mean_bm25_candidates,
        },
        "bm25": {
            "k1": bm25_config["k1"],
            "b": bm25_config["b"],
            "source": str(args.bm25_result),
            "parameter_selection": bm25_result["parameter_selection"],
            "index_statistics": {
                "num_documents": bm25_retriever.num_documents,
                "vocabulary_size": bm25_retriever.vocabulary_size,
                "average_document_length": (bm25_retriever.average_document_length),
            },
        },
        "parameter_grid": {
            "rrf_k_values": list(RRF_K_VALUES),
            "dense_weight": DENSE_WEIGHT,
            "bm25_weights": list(BM25_WEIGHTS),
            "total_configurations": len(configs),
        },
        "objective_definition": {
            "primary": ("Mean UNION MRR@10 across all Evidence Thresholds."),
            "secondary": ("Mean UNION Hit@1 across all Evidence Thresholds."),
            "tertiary": ("UNION MRR@10 at the 0.5 Evidence Threshold."),
            "selection_note": (
                "Ranking-only parameter search. Context Budget "
                "evaluation is a separate subsequent stage."
            ),
        },
        "ranking_source_validation": {
            "dense_source": (
                "Complete Dense ranks and scores reconstructed from "
                "the frozen Day 6 Hybrid candidate records."
            ),
            "dense_top_10_matches_day6": True,
            "dense_raw_ranking_matches_day6": True,
            "bm25_source": (
                "Recomputed from V2 Parent Chunks using the optimized "
                "Day 7 BM25 parameters."
            ),
            "bm25_top_10_matches_day7": True,
            "bm25_raw_ranking_matches_day7": True,
        },
        "reference_configuration": {
            "description": (
                "Original Day 6 RRF parameters evaluated with optimized "
                "Day 7 BM25 rankings."
            ),
            "config": asdict(reference_config),
            "objective": reference_result["objective"],
            "ranking": reference_result["ranking"],
        },
        "best_configuration": {
            "config": best_result["config"],
            "objective": best_result["objective"],
        },
        "ranking_results": results,
        "v2_index_manifest": manifest,
        "v2_index_build_config": build_config,
        "baseline_files": {
            "dense": str(DENSE_BASELINE_PATH),
            "day6_hybrid_dense_ranking_reference": str(HYBRID_BASELINE_PATH),
            "optimized_bm25": str(args.bm25_result),
        },
    }

    # --------------------------------------------------------
    # Step 11: Save without overwriting previous experiments.
    # --------------------------------------------------------

    print("\n========== Step 11: Saving Results ==========")

    save_json_exclusive(
        args.output,
        output,
    )

    print("\n========== Optimized BM25 RRF Search Completed ==========")
    print("Configurations:", len(results))
    print("Evaluated questions:", len(prepared))
    print("Dense ranking validation: PASS")
    print("Optimized BM25 ranking validation: PASS")
    print("Best configuration:", best_result["config"])
    print("Best objective:", best_result["objective"])
    print("Results saved to:", args.output)


if __name__ == "__main__":
    main()
