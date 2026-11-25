"""Evaluate RRF Hybrid V2 retrieval against frozen Dense and BM25 baselines."""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from statistics import mean

from paper_rag.retrieval.bm25_retriever import BM25Retriever
from paper_rag.retrieval.embedder import TextEmbedder
from paper_rag.retrieval.hybrid_retriever import HybridRetriever

SPLIT = "validation"
RRF_K = 60
DENSE_WEIGHT = 1.0
BM25_WEIGHT = 1.0

DENSE_BASELINE_PATH = Path("storage/metadata/day6_dense_v2_eval_validation.json")
BM25_BASELINE_PATH = Path("storage/metadata/day6_bm25_v2_eval_validation.json")
INDEX_DIRECTORY = Path("data/indexes/qasper_validation_windowed_v2")
HYBRID_OUTPUT_PATH = Path("storage/metadata/day6_hybrid_rrf_v2_eval_validation.json")


def load_script_module(module_name: str, script_path: Path):
    """Import an existing evaluation script without running its main function."""

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

DENSE = load_script_module(
    "_paperpilot_dense_eval_helpers",
    SCRIPT_DIRECTORY / "evaluate_retrieval_token_budget.py",
)

BM25_EVALUATION = load_script_module(
    "_paperpilot_bm25_eval_helpers",
    SCRIPT_DIRECTORY / "evaluate_bm25_retrieval.py",
)


def ensure_output_available(path: Path) -> None:
    """Refuse to overwrite an existing experiment result."""

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {path}")


def load_bm25_baseline(path: Path) -> dict:
    """Load and validate the frozen BM25 V2 baseline."""

    if not path.is_file():
        raise FileNotFoundError(f"BM25 V2 baseline not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("BM25 baseline must be a JSON object.")

    expected_metadata = {
        "experiment": "day6_bm25_v2_baseline",
        "split": SPLIT,
        "retrieval_method": "bm25_v2",
        "retrieval_scope": "within_paper",
    }

    for key, expected in expected_metadata.items():
        if data.get(key) != expected:
            raise ValueError(
                f"Invalid BM25 baseline metadata: {key}={data.get(key)!r}, "
                f"expected {expected!r}."
            )

    bm25_config = data.get("bm25")
    if not isinstance(bm25_config, dict):
        raise TypeError("BM25 baseline configuration must be a JSON object.")

    k1 = bm25_config.get("k1")
    b = bm25_config.get("b")

    if isinstance(k1, bool) or not isinstance(k1, (int, float)):
        raise TypeError("BM25 baseline k1 must be numeric.")

    if isinstance(b, bool) or not isinstance(b, (int, float)):
        raise TypeError("BM25 baseline b must be numeric.")

    if not math.isfinite(k1) or k1 <= 0:
        raise ValueError("BM25 baseline k1 must be positive and finite.")

    if not math.isfinite(b) or not 0 <= b <= 1:
        raise ValueError("BM25 baseline b must be between 0 and 1.")

    if bm25_config.get("idf_scope") != "global_validation_parent_chunks":
        raise ValueError("BM25 baseline IDF scope is not global Validation.")

    context_config = data.get("context_budget_evaluation")
    if not isinstance(context_config, dict):
        raise TypeError("BM25 context budget configuration must be a JSON object.")

    budgets = context_config.get("budgets")
    thresholds = context_config.get("thresholds")

    if not isinstance(budgets, list):
        raise TypeError("BM25 baseline budgets must be a list.")

    if not budgets:
        raise ValueError("BM25 baseline budgets are empty.")

    if not isinstance(thresholds, list):
        raise TypeError("BM25 baseline thresholds must be a list.")

    if not thresholds:
        raise ValueError("BM25 baseline thresholds are empty.")

    if not isinstance(context_config.get("tokenizer"), str):
        raise TypeError("BM25 baseline tokenizer must be a string.")

    records = data.get("per_question")
    if not isinstance(records, list):
        raise TypeError("BM25 baseline per-question records must be a list.")

    if not records:
        raise ValueError("BM25 baseline per-question records are empty.")

    statistics = data.get("question_statistics")
    if not isinstance(statistics, dict):
        raise TypeError("BM25 question statistics must be a JSON object.")

    if statistics.get("final_evaluable_questions") != len(records):
        raise ValueError("BM25 baseline question count does not match its records.")

    return data


def validate_baseline_compatibility(
    dense_baseline: dict,
    bm25_baseline: dict,
) -> None:
    """Ensure both frozen baselines use identical evaluation settings."""

    dense_config = dense_baseline["context_budget_evaluation"]
    bm25_config = bm25_baseline["context_budget_evaluation"]

    shared_settings = (
        "budgets",
        "thresholds",
        "tokenizer",
        "tokenizer_is_proxy",
        "special_tokens",
        "truncation",
        "context_separator",
    )

    for key in shared_settings:
        if dense_config[key] != bm25_config[key]:
            raise ValueError(f"Dense and BM25 baseline {key} differ.")

    for name, config in (
        ("Dense", dense_config),
        ("BM25", bm25_config),
    ):
        policy = config.get("packing_policy")

        if not isinstance(policy, str) or not policy.strip():
            raise ValueError(f"{name} baseline has invalid packing policy.")

    if dense_baseline["embedding_model"] != dense_config["tokenizer"]:
        raise ValueError("Dense embedding model and context tokenizer differ.")

    BM25_EVALUATION.validate_question_alignment(
        dense_baseline["per_question"],
        bm25_baseline["per_question"],
    )


def validate_index_compatibility(
    dense_baseline: dict,
    bm25_baseline: dict,
    manifest: dict,
    build_config: dict,
) -> None:
    """Ensure both baselines were produced from the currently loaded V2 index."""

    for name, baseline in (
        ("Dense", dense_baseline),
        ("BM25", bm25_baseline),
    ):
        if baseline.get("v2_index_manifest") != manifest:
            raise ValueError(
                f"{name} baseline V2 manifest differs from the loaded index."
            )

        if baseline.get("v2_index_build_config") != build_config:
            raise ValueError(f"{name} baseline V2 build configuration differs.")


def validate_bm25_index_statistics(
    baseline: dict,
    retriever: BM25Retriever,
) -> None:
    """Ensure the rebuilt BM25 index matches the frozen baseline statistics."""

    expected = baseline.get("bm25_index_statistics")
    if not isinstance(expected, dict):
        raise TypeError("BM25 baseline index statistics must be a JSON object.")

    if expected.get("num_documents") != retriever.num_documents:
        raise ValueError("Rebuilt BM25 document count differs from its baseline.")

    if expected.get("vocabulary_size") != retriever.vocabulary_size:
        raise ValueError("Rebuilt BM25 vocabulary size differs from its baseline.")

    expected_average = expected.get("average_document_length")

    if isinstance(expected_average, bool) or not isinstance(
        expected_average,
        (int, float),
    ):
        raise TypeError("BM25 baseline average document length must be numeric.")

    if not math.isfinite(expected_average):
        raise ValueError("BM25 baseline average document length must be finite.")

    if not math.isclose(
        retriever.average_document_length,
        expected_average,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "Rebuilt BM25 average document length differs from its baseline."
        )


def evaluate_hybrid(
    questions: list[dict],
    query_vectors,
    indexes: dict,
    bm25_retriever: BM25Retriever,
    hybrid_retriever: HybridRetriever,
    tokenizer,
    budgets: tuple[int, ...],
    thresholds: tuple[float, ...],
    separator_tokens: int,
) -> dict:
    """Evaluate RRF rankings and packed contexts using Dense V2 helpers."""

    if not questions:
        raise ValueError("No evaluation questions.")

    if len(questions) != len(query_vectors):
        raise ValueError("Question count and query embedding count do not match.")

    budget_records = {budget: [] for budget in budgets}
    ranking_records = []
    per_question = []
    token_cache = {}

    for item, query_vector in zip(questions, query_vectors, strict=True):
        paper_id = item["paper_id"]

        if paper_id not in indexes:
            raise KeyError(f"Missing V2 index for paper {paper_id}.")

        # 1. Complete Dense V2 ranking.
        dense_results = DENSE.rank_windowed_chunks(
            query_vector,
            indexes[paper_id],
        )

        # 2. Partial BM25 ranking.
        bm25_results = BM25_EVALUATION.rank_bm25_question(
            retriever=bm25_retriever,
            question=item["question"],
            paper_id=paper_id,
        )

        # 3. RRF fusion.
        hybrid_results = hybrid_retriever.fuse(
            dense_results=dense_results,
            bm25_results=bm25_results,
        )

        dense_ids = {result.chunk.chunk_id for result in dense_results}
        hybrid_ids = {result.chunk.chunk_id for result in hybrid_results}

        if hybrid_ids != dense_ids or len(hybrid_results) != len(dense_results):
            raise ValueError("Hybrid candidate set does not match Dense candidates.")

        # Convert to the exact RankedChunk type used by Dense evaluation.
        ranked_chunks = [
            DENSE.RankedChunk(
                rank=result.rank,
                score=result.score,
                chunk=result.chunk,
            )
            for result in hybrid_results
        ]

        raw_ranking = DENSE.evaluate_ranked_chunks(
            ranked_chunks=ranked_chunks,
            locations=item["locations"],
            thresholds=thresholds,
        )
        ranking_records.append(raw_ranking)

        ranked_candidates = [
            {
                "rank": result.rank,
                "chunk_id": result.chunk.chunk_id,
                "rrf_score": result.score,
                "dense_rank": result.dense_rank,
                "bm25_rank": result.bm25_rank,
                "dense_score": result.dense_score,
                "bm25_score": result.bm25_score,
                "dense_contribution": result.dense_contribution,
                "bm25_contribution": result.bm25_contribution,
            }
            for result in hybrid_results
        ]

        question_result = {
            "paper_id": paper_id,
            "question_id": item["question_id"],
            "question": item["question"],
            "evidence_location_count": len(item["locations"]),
            "dense_candidate_count": len(dense_results),
            "bm25_candidate_count": len(bm25_results),
            "hybrid_candidate_count": len(hybrid_results),
            "top_10": ranked_candidates[:10],
            "ranked_candidates": ranked_candidates,
            "raw_ranking": raw_ranking,
            "context_by_budget": {},
        }

        # 4. Reuse the same Context Packing and Evidence Evaluation.
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
                locations=item["locations"],
                thresholds=thresholds,
            )

            budget_records[budget].append(context_result)

            question_result["context_by_budget"][str(budget)] = {
                **context_result,
                "selected_chunk_ids": [
                    candidate.chunk.chunk_id for candidate in packed.selected
                ],
                "selected_retrieval_ranks": [
                    candidate.rank for candidate in packed.selected
                ],
            }

        per_question.append(question_result)

    ranking_summary = DENSE.summarize_ranking_results(
        question_results=ranking_records,
        thresholds=thresholds,
    )

    budget_summary = {
        str(budget): DENSE.summarize_budget_results(
            question_results=budget_records[budget],
            thresholds=thresholds,
        )
        for budget in budgets
    }

    dense_counts = [item["dense_candidate_count"] for item in per_question]
    bm25_counts = [item["bm25_candidate_count"] for item in per_question]
    hybrid_counts = [item["hybrid_candidate_count"] for item in per_question]

    return {
        "ranking": ranking_summary,
        "budgets": budget_summary,
        "per_question": per_question,
        "token_cache_entries": len(token_cache),
        "questions_without_lexical_match": sum(count == 0 for count in bm25_counts),
        "mean_dense_candidates": mean(dense_counts),
        "mean_bm25_candidates": mean(bm25_counts),
        "mean_hybrid_candidates": mean(hybrid_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RRF Hybrid V2 on the frozen Validation question set."
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, default=HYBRID_OUTPUT_PATH)
    args = parser.parse_args()

    ensure_output_available(args.output)

    print("\n========== Day 6: Hybrid RRF V2 Evaluation ==========")
    print("Dataset:", SPLIT)
    print("RRF k:", RRF_K)
    print("Dense weight:", DENSE_WEIGHT)
    print("BM25 weight:", BM25_WEIGHT)
    print("Output:", args.output)

    print("\n========== Step 1: Loading Frozen Baselines ==========")

    dense_baseline = BM25_EVALUATION.load_dense_baseline(DENSE_BASELINE_PATH)
    bm25_baseline = load_bm25_baseline(BM25_BASELINE_PATH)

    validate_baseline_compatibility(
        dense_baseline,
        bm25_baseline,
    )

    budgets = tuple(dense_baseline["context_budget_evaluation"]["budgets"])
    thresholds = tuple(dense_baseline["context_budget_evaluation"]["thresholds"])
    embedding_model = dense_baseline["embedding_model"]
    bm25_config = bm25_baseline["bm25"]

    print("Dense baseline:", DENSE_BASELINE_PATH)
    print("BM25 baseline:", BM25_BASELINE_PATH)
    print("Reference questions:", len(dense_baseline["per_question"]))
    print("Context budgets:", budgets)
    print("Evidence thresholds:", thresholds)
    print("Embedding model:", embedding_model)

    print("\n========== Step 2: Loading Evaluation Questions ==========")

    questions, question_stats = DENSE.load_questions(SPLIT)

    for key, value in question_stats.items():
        print(f"{key}: {value}")

    if not questions:
        raise ValueError("No questions with body evidence.")

    print("\n========== Step 3: Loading V2 Index ==========")

    indexes, v2_manifest, v2_build_config = DENSE.load_windowed_index(
        directory=INDEX_DIRECTORY,
        model_name=embedding_model,
    )

    eligible = [item for item in questions if item["paper_id"] in indexes]
    excluded_missing_index = len(questions) - len(eligible)

    if not eligible:
        raise ValueError("No evaluable questions with V2 indexes.")

    print("Eligible questions:", len(eligible))
    print("Excluded missing V2 index:", excluded_missing_index)

    validate_index_compatibility(
        dense_baseline=dense_baseline,
        bm25_baseline=bm25_baseline,
        manifest=v2_manifest,
        build_config=v2_build_config,
    )
    print("V2 index manifest and build config: PASS")

    print("\n========== Step 4: Verifying Question Alignment ==========")

    BM25_EVALUATION.validate_question_alignment(
        eligible,
        dense_baseline["per_question"],
    )
    BM25_EVALUATION.validate_question_alignment(
        eligible,
        bm25_baseline["per_question"],
    )

    print("Question IDs and order: PASS")
    print("Question text: PASS")

    print("\n========== Step 5: Building BM25 Index ==========")

    chunks = BM25_EVALUATION.collect_v2_parent_chunks(
        indexes=indexes,
        expected_split=SPLIT,
    )

    bm25_retriever = BM25Retriever(
        k1=bm25_config["k1"],
        b=bm25_config["b"],
    )
    bm25_retriever.build_index(chunks)

    if bm25_retriever.num_documents != v2_manifest["num_chunks"]:
        raise ValueError("BM25 document count does not match V2 manifest.")

    validate_bm25_index_statistics(
        bm25_baseline,
        bm25_retriever,
    )

    print("Indexed Parent Chunks:", bm25_retriever.num_documents)
    print("Vocabulary Size:", bm25_retriever.vocabulary_size)
    print("BM25 baseline index statistics: PASS")

    print("\n========== Step 6: Loading Embedding Model ==========")

    embedder = TextEmbedder(
        model_name=embedding_model,
        device=args.device,
    )

    if embedder.model_name != embedding_model:
        raise ValueError("Embedding model does not match Dense baseline.")

    if bm25_baseline["context_budget_evaluation"]["tokenizer"] != (embedder.model_name):
        raise ValueError("Tokenizer does not match BM25 baseline.")

    tokenizer = embedder.model.tokenizer
    separator_tokens = DENSE.count_tokens(
        tokenizer,
        DENSE.CHUNK_SEPARATOR,
    )

    query_vectors = embedder.encode_documents(
        [item["question"] for item in eligible],
        batch_size=32,
        show_progress_bar=True,
    )

    if query_vectors.shape != (len(eligible), embedder.dimension):
        raise ValueError("Query embedding shape mismatch.")

    print("Query embeddings:", query_vectors.shape)
    print("Separator tokens:", separator_tokens)
    print(
        "NOTE: The embedding tokenizer remains a proxy for the future generation model."
    )

    print("\n========== Step 7: Initializing RRF ==========")

    hybrid_retriever = HybridRetriever(
        rrf_k=RRF_K,
        dense_weight=DENSE_WEIGHT,
        bm25_weight=BM25_WEIGHT,
    )

    print("\n========== Step 8: Evaluating Hybrid ==========")

    results = evaluate_hybrid(
        questions=eligible,
        query_vectors=query_vectors,
        indexes=indexes,
        bm25_retriever=bm25_retriever,
        hybrid_retriever=hybrid_retriever,
        tokenizer=tokenizer,
        budgets=budgets,
        thresholds=thresholds,
        separator_tokens=separator_tokens,
    )

    report_threshold = 0.5 if 0.5 in thresholds else thresholds[0]
    threshold_key = str(report_threshold)

    print("\n========== Raw Ranking Metrics ==========")

    for mode in ("single", "union"):
        metrics = results["ranking"][mode][threshold_key]
        print(f"\n--- {mode.upper()} / {report_threshold:.0%} ---")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

    print("\n========== Context Budget Metrics ==========")

    for budget in budgets:
        metrics = results["budgets"][str(budget)]
        union_metrics = metrics["retrieval"]["union"][threshold_key]
        context_hit = metrics["retrieval"]["context_hit_rate"][threshold_key]["union"]

        print(f"\n--- Budget: {budget} ---")
        print(f"Hit@10: {union_metrics['hit@10']:.4f}")
        print(f"MRR@10: {union_metrics['mrr@10']:.4f}")
        print(f"Context Hit Rate: {context_hit:.4f}")
        print("Mean Context Tokens:", f"{metrics['mean_context_tokens']:.1f}")
        print("Mean Selected Chunks:", f"{metrics['mean_selected_chunks']:.2f}")

    print("\n========== Candidate Statistics ==========")
    print(
        "Questions without BM25 lexical match:",
        results["questions_without_lexical_match"],
    )
    print("Mean Dense candidates:", f"{results['mean_dense_candidates']:.2f}")
    print("Mean BM25 candidates:", f"{results['mean_bm25_candidates']:.2f}")
    print("Mean Hybrid candidates:", f"{results['mean_hybrid_candidates']:.2f}")

    output = {
        "schema_version": 1,
        "experiment": "day6_hybrid_rrf_v2_baseline",
        "split": SPLIT,
        "retrieval_method": "hybrid_rrf_v2",
        "retrieval_scope": "within_paper",
        "embedding_model": embedder.model_name,
        "embedding_dimension": embedder.dimension,
        "fusion": {
            "algorithm": "reciprocal_rank_fusion",
            "rrf_k": RRF_K,
            "dense_weight": DENSE_WEIGHT,
            "bm25_weight": BM25_WEIGHT,
            "candidate_policy": (
                "All Dense V2 parent chunks; BM25 contributes when "
                "a lexical match exists."
            ),
            "missing_bm25_rank_contribution": 0.0,
            "tie_breaking": "descending RRF score, then Dense rank, then chunk ID",
        },
        "bm25": bm25_config,
        "ranking": {
            "ranking_strategy": (
                "Fuse complete Dense V2 and partial BM25 V2 "
                "parent-chunk rankings with RRF."
            ),
            "metrics": results["ranking"],
        },
        "context_budget_evaluation": {
            "tokenizer": embedder.model_name,
            "tokenizer_is_proxy": True,
            "special_tokens": False,
            "truncation": False,
            "context_separator": DENSE.CHUNK_SEPARATOR,
            "budgets": list(budgets),
            "thresholds": list(thresholds),
            "packing_policy": (
                "Same Dense V2 greedy ranking-order packing of complete parent chunks."
            ),
            "metrics": results["budgets"],
        },
        "question_statistics": {
            **question_stats,
            "excluded_missing_index": excluded_missing_index,
            "final_evaluable_questions": len(eligible),
            "questions_without_lexical_match": (
                results["questions_without_lexical_match"]
            ),
            "mean_dense_candidates": results["mean_dense_candidates"],
            "mean_bm25_candidates": results["mean_bm25_candidates"],
            "mean_hybrid_candidates": results["mean_hybrid_candidates"],
        },
        "v2_index_manifest": v2_manifest,
        "v2_index_build_config": v2_build_config,
        "baselines": {
            "dense": str(DENSE_BASELINE_PATH),
            "bm25": str(BM25_BASELINE_PATH),
        },
        "bm25_index_statistics": {
            "num_documents": bm25_retriever.num_documents,
            "vocabulary_size": bm25_retriever.vocabulary_size,
            "average_document_length": bm25_retriever.average_document_length,
        },
        "token_cache_entries": results["token_cache_entries"],
        "per_question": results["per_question"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("x", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print("\n========== Hybrid Evaluation Completed ==========")
    print("Results saved to:", args.output)
    print("Evaluated questions:", len(eligible))
    print("Token cache entries:", results["token_cache_entries"])


if __name__ == "__main__":
    main()
