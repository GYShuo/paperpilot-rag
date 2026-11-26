"""Evaluate BM25 V2 retrieval against the frozen Dense V2 baseline."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from statistics import mean

from paper_rag.retrieval.bm25_retriever import BM25Retriever
from paper_rag.retrieval.embedder import TextEmbedder
from paper_rag.schemas import SentenceTokenChunk

# ============================================================
# Experiment configuration
# ============================================================

SPLIT = "validation"

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75

DENSE_BASELINE_PATH = Path("storage/metadata/day6_dense_v2_eval_validation.json")

BM25_OUTPUT_PATH = Path("storage/metadata/day6_bm25_v2_eval_validation.json")

INDEX_DIRECTORY = Path("data/indexes/qasper_validation_windowed_v2")


# ============================================================
# Reuse the verified Dense evaluation implementation
# ============================================================


def load_dense_helpers():
    """Load the existing Dense evaluation module without executing main."""

    module_name = "_paperpilot_dense_eval_helpers"

    existing = sys.modules.get(module_name)

    if existing is not None:
        return existing

    script_path = Path(__file__).resolve().parent / "evaluate_retrieval_token_budget.py"

    if not script_path.is_file():
        raise FileNotFoundError(f"Dense evaluation script not found: {script_path}")

    spec = importlib.util.spec_from_file_location(
        module_name,
        script_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Dense evaluation helpers: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    required = (
        "RankedChunk",
        "load_questions",
        "load_windowed_index",
        "count_tokens",
        "pack_ranked_chunks",
        "evaluate_ranked_chunks",
        "evaluate_packed_context",
        "summarize_ranking_results",
        "summarize_budget_results",
        "CHUNK_SEPARATOR",
    )

    missing = [name for name in required if not hasattr(module, name)]

    if missing:
        raise RuntimeError(f"Dense evaluation module is missing helpers: {missing}")

    return module


DENSE = load_dense_helpers()

RankedChunk = DENSE.RankedChunk


# ============================================================
# Dense baseline validation
# ============================================================


def load_dense_baseline(path: Path) -> dict:
    """Load and validate the frozen Dense V2 experiment."""

    if not path.is_file():
        raise FileNotFoundError(f"Dense V2 baseline not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise TypeError("Dense baseline must be a JSON object.")

    expected_metadata = {
        "experiment": "day6_dense_v2_baseline",
        "split": SPLIT,
        "retrieval_method": "dense_v2",
        "retrieval_scope": "within_paper",
    }

    for key, expected in expected_metadata.items():
        actual = data.get(key)

        if actual != expected:
            raise ValueError(
                f"Invalid Dense baseline metadata: "
                f"{key}={actual!r}, expected {expected!r}"
            )

    evaluation_config = data.get("context_budget_evaluation")

    if not isinstance(evaluation_config, dict):
        raise TypeError("Dense baseline has no valid context budget configuration.")

    budgets = evaluation_config.get("budgets")
    thresholds = evaluation_config.get("thresholds")

    if not isinstance(budgets, list) or not budgets:
        raise ValueError("Dense baseline budgets are missing.")

    if not isinstance(thresholds, list) or not thresholds:
        raise ValueError("Dense baseline thresholds are missing.")

    if any(not isinstance(value, int) or value <= 0 for value in budgets):
        raise ValueError("Dense baseline contains invalid budgets.")

    if len(set(budgets)) != len(budgets):
        raise ValueError("Dense baseline budgets contain duplicates.")

    if any(
        not isinstance(value, (int, float)) or not 0 < value <= 1
        for value in thresholds
    ):
        raise ValueError("Dense baseline contains invalid thresholds.")

    if len(set(thresholds)) != len(thresholds):
        raise ValueError("Dense baseline thresholds contain duplicates.")

    records = data.get("per_question")

    if not isinstance(records, list) or not records:
        raise ValueError("Dense baseline has no per-question records.")

    statistics = data.get("question_statistics", {})

    if statistics.get("final_evaluable_questions") != len(records):
        raise ValueError("Dense baseline question count does not match its records.")

    return data


def validate_question_alignment(
    questions: list[dict],
    dense_records: list[dict],
) -> None:
    """Require exactly the same questions and ordering as Dense V2."""

    current_keys = [(item["paper_id"], item["question_id"]) for item in questions]

    dense_keys = [(item["paper_id"], item["question_id"]) for item in dense_records]

    if len(current_keys) != len(set(current_keys)):
        raise ValueError("Current evaluation set contains duplicate question IDs.")

    if len(dense_keys) != len(set(dense_keys)):
        raise ValueError("Dense baseline contains duplicate question IDs.")

    if current_keys != dense_keys:
        raise ValueError(
            "BM25 and Dense question sets or ordering differ. "
            "Refusing to perform an unfair comparison."
        )

    for current, baseline in zip(
        questions,
        dense_records,
        strict=True,
    ):
        if current["question"] != baseline["question"]:
            raise ValueError(
                "Question text mismatch for "
                f"{current['paper_id']} / {current['question_id']}."
            )


# ============================================================
# V2 Parent Chunk collection
# ============================================================


def collect_v2_parent_chunks(
    indexes: dict,
    expected_split: str = SPLIT,
) -> list[SentenceTokenChunk]:
    """Collect all V2 parent chunks from the per-paper index."""

    if not indexes:
        raise ValueError("V2 index contains no papers.")

    chunks = []

    for paper_id, paper_index in indexes.items():
        paper_chunks = paper_index.get("chunks")

        if not isinstance(paper_chunks, list):
            raise TypeError(f"Invalid parent chunk collection for {paper_id}.")

        if not paper_chunks:
            raise ValueError(f"Paper {paper_id} has no parent chunks.")

        for chunk in paper_chunks:
            if not isinstance(chunk, SentenceTokenChunk):
                raise TypeError(f"Invalid parent chunk type in paper {paper_id}.")

            if chunk.paper_id != paper_id:
                raise ValueError(f"Paper ID mismatch for chunk {chunk.chunk_id}.")

            if chunk.split != expected_split:
                raise ValueError(
                    f"Unexpected split for chunk {chunk.chunk_id}: {chunk.split}"
                )

            chunks.append(chunk)

    return chunks


# ============================================================
# BM25 ranking
# ============================================================


def rank_bm25_question(
    retriever: BM25Retriever,
    question: str,
    paper_id: str,
) -> list:
    """Convert BM25 results to the shared RankedChunk structure."""

    results = retriever.search(
        query=question,
        paper_id=paper_id,
    )

    ranked_chunks = [
        RankedChunk(
            rank=item.rank,
            score=item.score,
            chunk=item.chunk,
        )
        for item in results
    ]

    expected_ranks = list(range(1, len(ranked_chunks) + 1))

    actual_ranks = [item.rank for item in ranked_chunks]

    if actual_ranks != expected_ranks:
        raise ValueError(f"BM25 returned invalid ranks for paper {paper_id}.")

    if any(item.score <= 0 for item in ranked_chunks):
        raise ValueError("BM25 returned a non-positive retrieval score.")

    if any(item.chunk.paper_id != paper_id for item in ranked_chunks):
        raise ValueError("BM25 returned a chunk from another paper.")

    return ranked_chunks


# ============================================================
# BM25 evaluation
# ============================================================


def evaluate_bm25(
    questions: list[dict],
    retriever: BM25Retriever,
    tokenizer,
    budgets: tuple[int, ...],
    thresholds: tuple[float, ...],
    separator_tokens: int,
) -> dict:
    """Evaluate BM25 ranking and context budget performance."""

    if not questions:
        raise ValueError("No evaluation questions.")

    budget_records = {budget: [] for budget in budgets}

    ranking_records = []
    per_question = []

    token_cache = {}

    for item in questions:
        ranked_chunks = rank_bm25_question(
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

        question_result = {
            "paper_id": item["paper_id"],
            "question_id": item["question_id"],
            "question": item["question"],
            "evidence_location_count": len(item["locations"]),
            "matched_chunk_count": len(ranked_chunks),
            "top_10": [
                {
                    "rank": candidate.rank,
                    "chunk_id": candidate.chunk.chunk_id,
                    "score": candidate.score,
                }
                for candidate in ranked_chunks[:10]
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

    matched_counts = [item["matched_chunk_count"] for item in per_question]

    return {
        "ranking": ranking_summary,
        "budgets": budget_summary,
        "per_question": per_question,
        "token_cache_entries": len(token_cache),
        "questions_without_lexical_match": sum(count == 0 for count in matched_counts),
        "mean_matched_chunks": mean(matched_counts),
    }


# ============================================================
# Output protection
# ============================================================


def ensure_output_available(path: Path) -> None:
    """Refuse to overwrite any existing experimental result."""

    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {path}")


# ============================================================
# Main
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Evaluate BM25 V2 on the frozen Dense V2 Validation question set.")
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used to initialize the shared context tokenizer.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=BM25_OUTPUT_PATH,
    )

    args = parser.parse_args()

    output_path = args.output

    ensure_output_available(output_path)

    print("\n========== Day 6: BM25 V2 Evaluation ==========")
    print("Dataset:", SPLIT)
    print("Retrieval method: BM25 V2")
    print("BM25 k1:", DEFAULT_K1)
    print("BM25 b:", DEFAULT_B)
    print("Output:", output_path)

    # --------------------------------------------------------
    # Step 1: Load frozen Dense baseline.
    # --------------------------------------------------------

    print("\n========== Step 1: Loading Dense Baseline ==========")

    dense_baseline = load_dense_baseline(DENSE_BASELINE_PATH)

    budgets = tuple(dense_baseline["context_budget_evaluation"]["budgets"])

    thresholds = tuple(dense_baseline["context_budget_evaluation"]["thresholds"])

    embedding_model = dense_baseline["embedding_model"]

    print("Dense baseline:", DENSE_BASELINE_PATH)
    print("Reference questions:", len(dense_baseline["per_question"]))
    print("Context budgets:", budgets)
    print("Evidence thresholds:", thresholds)
    print("Context tokenizer:", embedding_model)

    # --------------------------------------------------------
    # Step 2: Load the exact same evaluation questions.
    # --------------------------------------------------------

    print("\n========== Step 2: Loading Evaluation Data ==========")

    questions, question_stats = DENSE.load_questions(SPLIT)

    for key, value in question_stats.items():
        print(f"{key}: {value}")

    if not questions:
        raise ValueError("No questions with body evidence.")

    # --------------------------------------------------------
    # Step 3: Load existing V2 index.
    # --------------------------------------------------------

    print("\n========== Step 3: Loading V2 Parent Chunks ==========")

    indexes, v2_manifest, v2_build_config = DENSE.load_windowed_index(
        directory=INDEX_DIRECTORY,
        model_name=embedding_model,
    )

    eligible = [item for item in questions if item["paper_id"] in indexes]

    excluded_missing_index = len(questions) - len(eligible)

    print("Eligible questions:", len(eligible))
    print("Excluded missing V2 index:", excluded_missing_index)

    if not eligible:
        raise ValueError("No evaluable questions with V2 indexes.")

    # --------------------------------------------------------
    # Step 4: Verify exact alignment with Dense.
    # --------------------------------------------------------

    print("\n========== Step 4: Verifying Question Alignment ==========")

    validate_question_alignment(
        questions=eligible,
        dense_records=dense_baseline["per_question"],
    )

    print("Question ID alignment: PASS")
    print("Question text alignment: PASS")
    print("Evaluation questions:", len(eligible))

    # --------------------------------------------------------
    # Step 5: Build BM25 index.
    # --------------------------------------------------------

    print("\n========== Step 5: Building BM25 Index ==========")

    chunks = collect_v2_parent_chunks(
        indexes=indexes,
        expected_split=SPLIT,
    )

    retriever = BM25Retriever(
        k1=DEFAULT_K1,
        b=DEFAULT_B,
    )

    retriever.build_index(chunks)

    if retriever.num_documents != v2_manifest["num_chunks"]:
        raise ValueError("BM25 document count does not match V2 manifest.")

    if retriever.split != SPLIT:
        raise ValueError("BM25 index split does not match Validation.")

    print("Indexed Parent Chunks:", retriever.num_documents)
    print("Vocabulary Size:", retriever.vocabulary_size)
    print(
        "Average Document Length:",
        f"{retriever.average_document_length:.2f}",
    )

    # --------------------------------------------------------
    # Step 6: Load the same context tokenizer as Dense.
    # --------------------------------------------------------

    print("\n========== Step 6: Loading Context Tokenizer ==========")

    embedder = TextEmbedder(
        device=args.device,
    )

    if embedder.model_name != embedding_model:
        raise ValueError(
            "Context tokenizer model does not match Dense baseline: "
            f"{embedder.model_name} != {embedding_model}"
        )

    tokenizer = embedder.model.tokenizer

    separator_tokens = DENSE.count_tokens(
        tokenizer,
        DENSE.CHUNK_SEPARATOR,
    )

    print("Tokenizer:", embedder.model_name)
    print("Separator Tokens:", separator_tokens)

    print(
        "NOTE: This tokenizer is a proxy shared with Dense V2; "
        "the final generation model has not been selected."
    )

    # --------------------------------------------------------
    # Step 7: Run BM25 evaluation.
    # --------------------------------------------------------

    print("\n========== Step 7: Evaluating BM25 ==========")

    results = evaluate_bm25(
        questions=eligible,
        retriever=retriever,
        tokenizer=tokenizer,
        budgets=budgets,
        thresholds=thresholds,
        separator_tokens=separator_tokens,
    )

    # --------------------------------------------------------
    # Step 8: Print metrics.
    # --------------------------------------------------------

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
        print(
            "Mean Context Tokens:",
            f"{metrics['mean_context_tokens']:.1f}",
        )
        print(
            "Mean Selected Chunks:",
            f"{metrics['mean_selected_chunks']:.2f}",
        )

    print("\n========== BM25 Candidate Statistics ==========")
    print(
        "Questions without lexical match:",
        results["questions_without_lexical_match"],
    )
    print(
        "Mean matched chunks:",
        f"{results['mean_matched_chunks']:.2f}",
    )

    # --------------------------------------------------------
    # Step 9: Construct result.
    # --------------------------------------------------------

    output = {
        "schema_version": 1,
        "experiment": "day6_bm25_v2_baseline",
        "split": SPLIT,
        "retrieval_method": "bm25_v2",
        "retrieval_scope": "within_paper",
        "bm25": {
            "k1": DEFAULT_K1,
            "b": DEFAULT_B,
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
        "ranking": {
            "ranking_strategy": (
                "BM25 over V2 Parent Chunks with global Validation IDF "
                "and paper-restricted candidate retrieval."
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
                "Same Dense V2 greedy ranking-order packing of "
                "complete parent chunks. Non-fitting chunks are skipped; "
                "later smaller chunks may still be selected. "
                "No chunk is truncated or split. "
                "The final assembled context is re-tokenized."
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
            "mean_matched_chunks": results["mean_matched_chunks"],
        },
        "v2_index_manifest": v2_manifest,
        "v2_index_build_config": v2_build_config,
        "dense_baseline": str(DENSE_BASELINE_PATH),
        "bm25_index_statistics": {
            "num_documents": retriever.num_documents,
            "vocabulary_size": retriever.vocabulary_size,
            "average_document_length": (retriever.average_document_length),
        },
        "token_cache_entries": results["token_cache_entries"],
        "per_question": results["per_question"],
    }

    # --------------------------------------------------------
    # Step 10: Save without overwriting existing results.
    # --------------------------------------------------------

    print("\n========== Step 10: Saving BM25 Results ==========")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Exclusive creation prevents accidental overwriting.
    with output_path.open("x", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print("\n========== BM25 Evaluation Completed ==========")
    print("Results saved to:", output_path)
    print("Evaluated questions:", len(eligible))
    print("Token cache entries:", results["token_cache_entries"])


if __name__ == "__main__":
    main()
