"""Evaluate frozen Dense, optimized BM25 and selected Hybrid V2 context budgets."""

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from paper_rag.retrieval.bm25_retriever import BM25Retriever
from paper_rag.retrieval.embedder import TextEmbedder
from paper_rag.retrieval.hybrid_retriever import HybridRetriever

SPLIT = "validation"
DENSE_BASELINE_PATH = Path("storage/metadata/day6_dense_v2_eval_validation.json")
HYBRID_BASELINE_PATH = Path("storage/metadata/day6_hybrid_rrf_v2_eval_validation.json")
BM25_OPTIMIZED_PATH = Path("storage/metadata/day7_bm25_v2_best_eval_validation.json")
RRF_SEARCH_PATH = Path(
    "storage/metadata/day7_rrf_rank_grid_validation_bm25_optimized.json"
)
INDEX_DIRECTORY = Path("data/indexes/qasper_validation_windowed_v2")
OUTPUT_PATH = Path("storage/metadata/day7_context_budget_eval_validation.json")

DEFAULT_BUDGETS = (1024, 2048, 4096)
DEFAULT_THRESHOLDS = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class HybridConfig:
    """One selected RRF configuration for context-budget evaluation."""

    method_id: str
    label: str
    rrf_k: int
    dense_weight: float
    bm25_weight: float


HYBRID_CONFIGS = (
    HybridConfig("hybrid_k60_w1", "Hybrid A: best MRR", 60, 1.0, 1.0),
    HybridConfig("hybrid_k100_w1", "Hybrid B: stable k=100", 100, 1.0, 1.0),
    HybridConfig("hybrid_k10_w075", "Hybrid C: best Hit@1 candidate", 10, 1.0, 0.75),
    HybridConfig("hybrid_k20_w1", "Hybrid D: top-five MRR candidate", 20, 1.0, 1.0),
)


def load_script_module(module_name: str, script_path: Path):
    """Load an existing evaluation script without executing its main function."""

    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    if not script_path.is_file():
        raise FileNotFoundError(f"Evaluation helper script not found: {script_path}")

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load evaluation helper script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
RRF_SEARCH = load_script_module(
    "_paperpilot_optimized_rrf_helpers",
    SCRIPT_DIRECTORY / "search_hybrid_rrf_params.py",
)
DENSE = RRF_SEARCH.DENSE
BM25_EVALUATION = RRF_SEARCH.BM25_EVALUATION


def load_json_object(path: Path, description: str) -> dict:
    """Load a JSON object from an existing experiment result."""

    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{description} must be a JSON object.")
    return data


def same_hybrid_config(actual: dict, expected: HybridConfig) -> bool:
    """Compare a saved RRF configuration with a selected configuration."""

    if not isinstance(actual, dict):
        return False
    if actual.get("rrf_k") != expected.rrf_k:
        return False
    for key, expected_value in (
        ("dense_weight", expected.dense_weight),
        ("bm25_weight", expected.bm25_weight),
    ):
        value = actual.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if not math.isclose(value, expected_value, rel_tol=0.0, abs_tol=1e-12):
            return False
    return True


def validate_rrf_search_result(path: Path, bm25_config: dict) -> dict:
    """Verify the selected RRF configurations came from the completed 48-grid run."""

    data = load_json_object(path, "Optimized-BM25 RRF grid result")
    if data.get("experiment") != "day7_rrf_rank_grid_search_optimized_bm25":
        raise ValueError("Unexpected optimized-BM25 RRF experiment metadata.")
    if data.get("split") != SPLIT:
        raise ValueError("RRF grid result does not use Validation.")

    saved_bm25 = data.get("bm25")
    if not isinstance(saved_bm25, dict):
        raise TypeError("RRF grid BM25 metadata must be an object.")
    for key in ("k1", "b"):
        value = saved_bm25.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"RRF grid BM25 {key} must be numeric.")
        if not math.isclose(
            value,
            bm25_config[key],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"RRF grid and optimized BM25 {key} differ.")

    parameter_grid = data.get("parameter_grid")
    results = data.get("ranking_results")
    best = data.get("best_configuration")
    if not isinstance(parameter_grid, dict):
        raise TypeError("RRF parameter grid metadata must be an object.")
    if parameter_grid.get("total_configurations") != 48:
        raise ValueError("RRF grid result does not contain 48 configurations.")
    if not isinstance(results, list) or len(results) != 48:
        raise ValueError("RRF grid result must contain 48 ranking records.")
    if not isinstance(best, dict) or not isinstance(best.get("config"), dict):
        raise TypeError("RRF best configuration is missing or invalid.")

    expected_best = {"rrf_k": 60, "dense_weight": 1.0, "bm25_weight": 1.0}
    if best["config"] != expected_best:
        raise ValueError("RRF best configuration differs from the completed search.")

    for config in HYBRID_CONFIGS:
        if not any(
            same_hybrid_config(result.get("config"), config) for result in results
        ):
            raise ValueError(
                f"Selected RRF configuration is absent: {config.method_id}."
            )
    return data


def validate_rrf_index_compatibility(
    rrf_result: dict,
    manifest: dict,
    build_config: dict,
) -> None:
    """Ensure the RRF search used the same frozen V2 index."""

    if rrf_result.get("v2_index_manifest") != manifest:
        raise ValueError("RRF search V2 manifest differs from the loaded index.")
    if rrf_result.get("v2_index_build_config") != build_config:
        raise ValueError("RRF search V2 build config differs from the loaded index.")


def build_method_rankings(prepared: list[dict]) -> dict[str, list[list]]:
    """Build Dense-only, BM25-only and selected Hybrid ranking lists."""

    method_rankings = {
        "dense_v2": [item["dense_results"] for item in prepared],
        "bm25_v2_optimized": [item["bm25_results"] for item in prepared],
    }

    for config in HYBRID_CONFIGS:
        retriever = HybridRetriever(
            rrf_k=config.rrf_k,
            dense_weight=config.dense_weight,
            bm25_weight=config.bm25_weight,
        )
        fused_rankings = []
        for item in prepared:
            dense_results = item["dense_results"]
            bm25_results = item["bm25_results"]
            fused = retriever.fuse(
                dense_results=dense_results,
                bm25_results=bm25_results,
            )
            dense_ids = {result.chunk.chunk_id for result in dense_results}
            fused_ids = {result.chunk.chunk_id for result in fused}
            if fused_ids != dense_ids or len(fused) != len(dense_results):
                raise ValueError("Hybrid candidate set differs from Dense candidates.")
            fused_rankings.append(
                [
                    DENSE.RankedChunk(
                        rank=result.rank,
                        score=result.score,
                        chunk=result.chunk,
                    )
                    for result in fused
                ]
            )
        method_rankings[config.method_id] = fused_rankings

    return method_rankings


def evaluate_method(
    method_id: str,
    questions: list[dict],
    ranked_lists: list[list],
    tokenizer,
    budgets: tuple[int, ...],
    thresholds: tuple[float, ...],
    token_cache: dict,
    separator_tokens: int,
) -> dict:
    """Evaluate raw ranking and packed contexts for one retrieval method."""

    if len(questions) != len(ranked_lists):
        raise ValueError(f"{method_id} question and ranking counts differ.")

    ranking_records = []
    budget_records = {budget: [] for budget in budgets}
    per_question = []
    candidate_counts = []

    for question, ranked_chunks in zip(questions, ranked_lists, strict=True):
        raw_ranking = DENSE.evaluate_ranked_chunks(
            ranked_chunks=ranked_chunks,
            locations=question["locations"],
            thresholds=thresholds,
        )
        ranking_records.append(raw_ranking)
        candidate_counts.append(len(ranked_chunks))

        question_result = {
            "paper_id": question["paper_id"],
            "question_id": question["question_id"],
            "question": question["question"],
            "candidate_count": len(ranked_chunks),
            "top_10": [
                {
                    "rank": item.rank,
                    "chunk_id": item.chunk.chunk_id,
                    "score": item.score,
                }
                for item in ranked_chunks[:10]
            ],
            "raw_ranking": raw_ranking,
            "context_by_budget": {},
        }

        for budget in budgets:
            packed = DENSE.pack_ranked_chunks(
                ranked_chunks=ranked_chunks,
                tokenizer=tokenizer,
                budget=budget,
                version="V2",
                token_cache=token_cache,
                separator_tokens=separator_tokens,
            )
            context_result = DENSE.evaluate_packed_context(
                packed=packed,
                locations=question["locations"],
                thresholds=thresholds,
            )
            budget_records[budget].append(context_result)
            question_result["context_by_budget"][str(budget)] = {
                **context_result,
                "selected_chunk_ids": [item.chunk.chunk_id for item in packed.selected],
                "selected_retrieval_ranks": [item.rank for item in packed.selected],
            }

        per_question.append(question_result)

    return {
        "ranking": DENSE.summarize_ranking_results(
            question_results=ranking_records,
            thresholds=thresholds,
        ),
        "budgets": {
            str(budget): DENSE.summarize_budget_results(
                question_results=budget_records[budget],
                thresholds=thresholds,
            )
            for budget in budgets
        },
        "candidate_statistics": {
            "mean_candidates": mean(candidate_counts),
            "min_candidates": min(candidate_counts),
            "max_candidates": max(candidate_counts),
            "questions_without_candidates": sum(
                count == 0 for count in candidate_counts
            ),
        },
        "per_question": per_question,
    }


def ensure_output_available(path: Path) -> None:
    """Refuse to overwrite an existing experiment result."""

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen Dense V2, optimized BM25 V2 and selected Hybrid "
            "RRF configurations under fixed context budgets."
        )
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    ensure_output_available(args.output)

    print(
        "\n========== Day 7: Optimized Retrieval Context Budget Evaluation =========="
    )
    print("Dataset:", SPLIT)
    print("Budgets:", DEFAULT_BUDGETS)
    print("Evidence thresholds:", DEFAULT_THRESHOLDS)
    print("Hybrid configurations:", len(HYBRID_CONFIGS))
    print("Output:", args.output)

    print("\n========== Step 1: Loading Frozen Experiment Results ==========")
    dense_baseline = BM25_EVALUATION.load_dense_baseline(DENSE_BASELINE_PATH)
    hybrid_baseline = RRF_SEARCH.load_hybrid_baseline(HYBRID_BASELINE_PATH)
    bm25_result = RRF_SEARCH.load_bm25_optimized_result(BM25_OPTIMIZED_PATH)
    rrf_result = validate_rrf_search_result(RRF_SEARCH_PATH, bm25_result["bm25"])
    RRF_SEARCH.validate_shared_configuration(
        dense_baseline=dense_baseline,
        hybrid_baseline=hybrid_baseline,
        bm25_result=bm25_result,
    )

    context_config = dense_baseline["context_budget_evaluation"]
    budgets = tuple(context_config["budgets"])
    thresholds = tuple(context_config["thresholds"])
    tokenizer_name = context_config["tokenizer"]
    if budgets != DEFAULT_BUDGETS or thresholds != DEFAULT_THRESHOLDS:
        raise ValueError(
            "Frozen baselines do not use the expected budgets and thresholds."
        )
    if dense_baseline["embedding_model"] != tokenizer_name:
        raise ValueError("Dense embedding model and context tokenizer differ.")

    print("Dense baseline:", DENSE_BASELINE_PATH)
    print("Day 6 Hybrid reference:", HYBRID_BASELINE_PATH)
    print("Optimized BM25 result:", BM25_OPTIMIZED_PATH)
    print("Optimized RRF search:", RRF_SEARCH_PATH)
    print("BM25 k1:", bm25_result["bm25"]["k1"])
    print("BM25 b:", bm25_result["bm25"]["b"])

    print("\n========== Step 2: Loading Evaluation Questions ==========")
    questions, question_statistics = DENSE.load_questions(SPLIT)
    for key, value in question_statistics.items():
        print(f"{key}: {value}")

    print("\n========== Step 3: Loading V2 Index ==========")
    indexes, manifest, build_config = DENSE.load_windowed_index(
        directory=INDEX_DIRECTORY,
        model_name=tokenizer_name,
    )
    eligible = [item for item in questions if item["paper_id"] in indexes]
    excluded_missing_index = len(questions) - len(eligible)
    if not eligible:
        raise ValueError("No evaluable questions with V2 indexes.")

    RRF_SEARCH.validate_index_compatibility(
        dense_baseline=dense_baseline,
        hybrid_baseline=hybrid_baseline,
        bm25_result=bm25_result,
        manifest=manifest,
        build_config=build_config,
    )
    validate_rrf_index_compatibility(rrf_result, manifest, build_config)
    print("V2 manifest and build config: PASS")
    print("Eligible questions:", len(eligible))
    print("Excluded missing index:", excluded_missing_index)

    print("\n========== Step 4: Restoring Frozen Dense Rankings ==========")
    prepared = RRF_SEARCH.prepare_question_records(
        questions=eligible,
        dense_records=dense_baseline["per_question"],
        hybrid_records=hybrid_baseline["per_question"],
        indexes=indexes,
        thresholds=thresholds,
    )
    print("Dense questions:", len(prepared))
    print("Dense Top-10 and raw ranking metrics: PASS")

    print("\n========== Step 5: Rebuilding Optimized BM25 Rankings ==========")
    chunks = BM25_EVALUATION.collect_v2_parent_chunks(
        indexes=indexes,
        expected_split=SPLIT,
    )
    bm25_retriever = BM25Retriever(
        k1=bm25_result["bm25"]["k1"],
        b=bm25_result["bm25"]["b"],
    )
    bm25_retriever.build_index(chunks)
    if bm25_retriever.num_documents != manifest["num_chunks"]:
        raise ValueError("BM25 document count differs from the V2 manifest.")
    RRF_SEARCH.validate_bm25_index_statistics(bm25_result, bm25_retriever)
    RRF_SEARCH.attach_optimized_bm25_rankings(
        prepared=prepared,
        retriever=bm25_retriever,
        optimized_records=bm25_result["per_question"],
        thresholds=thresholds,
    )
    print("BM25 k1:", bm25_retriever.k1)
    print("BM25 b:", bm25_retriever.b)
    print("Indexed Parent Chunks:", bm25_retriever.num_documents)
    print("Vocabulary Size:", bm25_retriever.vocabulary_size)
    print("BM25 Top-10 and raw ranking metrics: PASS")

    print("\n========== Step 6: Loading Context Tokenizer ==========")
    embedder = TextEmbedder(model_name=tokenizer_name, device=args.device)
    if embedder.model_name != tokenizer_name:
        raise ValueError("Loaded tokenizer does not match the frozen baselines.")
    tokenizer = embedder.model.tokenizer
    separator_tokens = DENSE.count_tokens(tokenizer, DENSE.CHUNK_SEPARATOR)
    print("Tokenizer:", embedder.model_name)
    print("Separator tokens:", separator_tokens)
    print("Query embeddings: not recomputed")
    print("Truncation: disabled")

    print("\n========== Step 7: Building Retrieval Method Rankings ==========")
    method_rankings = build_method_rankings(prepared)
    print("Methods:", len(method_rankings))
    print("Dense-only: PASS")
    print("Optimized BM25-only: PASS")
    print("Four selected Hybrid configurations: PASS")

    print("\n========== Step 8: Evaluating Context Budgets ==========")
    token_cache: dict[tuple[str, str], int] = {}
    method_results = {}
    for method_id, ranked_lists in method_rankings.items():
        print("Evaluating:", method_id)
        method_results[method_id] = evaluate_method(
            method_id=method_id,
            questions=eligible,
            ranked_lists=ranked_lists,
            tokenizer=tokenizer,
            budgets=budgets,
            thresholds=thresholds,
            token_cache=token_cache,
            separator_tokens=separator_tokens,
        )

    print("\n========== Step 9: Context Budget Summary ==========")
    report_threshold = 0.5 if 0.5 in thresholds else thresholds[0]
    threshold_key = str(report_threshold)
    for method_id, result in method_results.items():
        print(f"\n--- {method_id} ---")
        raw_union = result["ranking"]["union"][threshold_key]
        print(f"Raw UNION MRR@10: {raw_union['mrr@10']:.4f}")
        for budget in budgets:
            metrics = result["budgets"][str(budget)]
            union = metrics["retrieval"]["union"][threshold_key]
            context_hit = metrics["retrieval"]["context_hit_rate"][threshold_key][
                "union"
            ]
            print(
                f"Budget {budget}: "
                f"Hit@10={union['hit@10']:.4f}, "
                f"MRR@10={union['mrr@10']:.4f}, "
                f"ContextHit={context_hit:.4f}, "
                f"MeanTokens={metrics['mean_context_tokens']:.1f}, "
                f"MeanChunks={metrics['mean_selected_chunks']:.2f}, "
                f"Utilization={metrics['mean_budget_utilization']:.4f}"
            )

    output = {
        "schema_version": 1,
        "experiment": "day7_optimized_retrieval_context_budget_eval",
        "split": SPLIT,
        "retrieval_scope": "within_paper",
        "question_statistics": {
            **question_statistics,
            "excluded_missing_index": excluded_missing_index,
            "final_evaluable_questions": len(eligible),
        },
        "context_budget_evaluation": {
            "tokenizer": tokenizer_name,
            "tokenizer_is_proxy": True,
            "special_tokens": False,
            "truncation": False,
            "context_separator": DENSE.CHUNK_SEPARATOR,
            "separator_tokens": separator_tokens,
            "budgets": list(budgets),
            "thresholds": list(thresholds),
            "packing_policy": (
                "Greedy retrieval-order packing of complete V2 Parent Chunks; "
                "oversized candidates are skipped and later candidates are checked."
            ),
        },
        "bm25": {
            "k1": bm25_retriever.k1,
            "b": bm25_retriever.b,
            "idf_scope": bm25_result["bm25"].get("idf_scope"),
            "source_result": str(BM25_OPTIMIZED_PATH),
            "index_statistics": {
                "num_documents": bm25_retriever.num_documents,
                "vocabulary_size": bm25_retriever.vocabulary_size,
                "average_document_length": bm25_retriever.average_document_length,
            },
        },
        "hybrid_configurations": [asdict(config) for config in HYBRID_CONFIGS],
        "method_results": method_results,
        "v2_index_manifest": manifest,
        "v2_index_build_config": build_config,
        "baseline_files": {
            "dense": str(DENSE_BASELINE_PATH),
            "day6_hybrid_dense_ranking_source": str(HYBRID_BASELINE_PATH),
            "optimized_bm25": str(BM25_OPTIMIZED_PATH),
            "optimized_bm25_rrf_grid": str(RRF_SEARCH_PATH),
        },
        "token_cache_entries": len(token_cache),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2, allow_nan=False)

    print("\n========== Context Budget Evaluation Completed ==========")
    print("Methods:", len(method_results))
    print("Evaluated questions:", len(eligible))
    print("Token cache entries:", len(token_cache))
    print("Results saved to:", args.output)


if __name__ == "__main__":
    main()
