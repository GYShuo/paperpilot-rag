import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

import numpy as np
from tqdm import tqdm

from paper_rag.evaluation.retrieval_evaluator import (
    EvidenceLocation,
    chunk_evidence_coverage,
    first_hit_rank,
    first_union_hit_rank,
    locate_body_evidence,
    summarize_ranks,
    union_evidence_coverage,
)
from paper_rag.retrieval.embedder import TextEmbedder
from paper_rag.retrieval.windowed_index_store import WindowedIndexStore
from paper_rag.schemas import (
    PaperDocument,
    QAExample,
    SectionChunk,
    SentenceTokenChunk,
)

# ============================================================
# Configuration
# ============================================================

DEFAULT_BUDGETS = (1024, 2048, 4096)
DEFAULT_THRESHOLDS = (0.25, 0.5, 0.75)

CHUNK_SEPARATOR = "\n\n"
TOP_K = 10

RetrievableChunk = SectionChunk | SentenceTokenChunk


# ============================================================
# Data structures
# ============================================================


@dataclass(frozen=True)
class RankedChunk:
    """A parent chunk with its retrieval rank and score."""

    rank: int
    score: float
    chunk: RetrievableChunk


@dataclass(frozen=True)
class PackedContext:
    """A context containing complete parent chunks."""

    selected: tuple[RankedChunk, ...]
    text: str
    token_count: int
    budget: int
    skipped_candidates: int
    oversized_candidates: int
    selected_oversize_sentences: int
    candidate_count: int


# ============================================================
# Token counting
# ============================================================


def count_tokens(tokenizer, text: str) -> int:
    """Count tokens without special tokens or truncation."""

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    )

    input_ids = encoded["input_ids"]

    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()

    if not isinstance(input_ids, list):
        input_ids = list(input_ids)

    if input_ids and isinstance(input_ids[0], (list, tuple)):
        raise ValueError("Expected a single unbatched tokenizer input.")

    return len(input_ids)


def build_context_text(ranked_chunks: list[RankedChunk]) -> str:
    """Join complete parent chunks in retrieval order."""

    return CHUNK_SEPARATOR.join(item.chunk.text for item in ranked_chunks)


# ============================================================
# Context packing
# ============================================================


def pack_ranked_chunks(
    ranked_chunks: list[RankedChunk],
    tokenizer,
    budget: int,
    version: str,
    token_cache: dict[tuple[str, str], int],
    separator_tokens: int,
) -> PackedContext:
    """Greedily pack complete chunks into a fixed token budget.

    Rules:
    1. Preserve retrieval ranking order.
    2. Never truncate or split a parent chunk.
    3. Skip a chunk that cannot fit.
    4. Continue checking subsequent smaller chunks.
    5. Re-tokenize the final context to verify the hard budget.

    Token counts are cached by version and chunk ID.
    """

    if budget <= 0:
        raise ValueError("Token budget must be positive.")

    if separator_tokens < 0:
        raise ValueError("Separator token count cannot be negative.")

    selected = []
    estimated_tokens = 0
    oversized_candidates = 0

    for candidate in ranked_chunks:
        chunk = candidate.chunk

        if not chunk.text.strip():
            raise ValueError(f"Empty chunk encountered: {chunk.chunk_id}")

        cache_key = (version, chunk.chunk_id)

        if cache_key not in token_cache:
            token_cache[cache_key] = count_tokens(
                tokenizer,
                chunk.text,
            )

        chunk_tokens = token_cache[cache_key]

        if chunk_tokens < 0:
            raise ValueError("Token count cannot be negative.")

        if chunk_tokens > budget:
            oversized_candidates += 1
            continue

        separator_cost = separator_tokens if selected else 0
        additional_tokens = chunk_tokens + separator_cost

        if estimated_tokens + additional_tokens <= budget:
            selected.append(candidate)
            estimated_tokens += additional_tokens

    context_text = build_context_text(selected)

    actual_tokens = count_tokens(tokenizer, context_text) if context_text else 0

    # Verify the actual assembled context.
    # Tokenization at text boundaries may not be additive.
    while selected and actual_tokens > budget:
        selected.pop()

        context_text = build_context_text(selected)

        actual_tokens = count_tokens(tokenizer, context_text) if context_text else 0

    if actual_tokens > budget:
        raise RuntimeError("Final context exceeds the token budget.")

    selected_oversize_sentences = sum(
        bool(getattr(item.chunk, "oversize_sentence", False)) for item in selected
    )

    return PackedContext(
        selected=tuple(selected),
        text=context_text,
        token_count=actual_tokens,
        budget=budget,
        skipped_candidates=len(ranked_chunks) - len(selected),
        oversized_candidates=oversized_candidates,
        selected_oversize_sentences=selected_oversize_sentences,
        candidate_count=len(ranked_chunks),
    )


# ============================================================
# Raw ranking evaluation
# ============================================================


def evaluate_ranked_chunks(
    ranked_chunks: list[RankedChunk],
    locations: list[EvidenceLocation],
    thresholds: tuple[float, ...],
) -> dict:
    """Evaluate raw retrieval ranking before context packing.

    SINGLE:
        The first individual parent chunk that reaches the
        evidence coverage threshold.

    UNION:
        The first prefix of retrieved parent chunks whose
        combined coverage reaches the threshold.
    """

    if not locations:
        raise ValueError("Evidence locations cannot be empty.")

    chunks = [item.chunk for item in ranked_chunks]

    single_ranks = {}
    union_ranks = {}

    for threshold in thresholds:
        if not 0 < threshold <= 1:
            raise ValueError("Threshold must be in (0, 1].")

        key = str(threshold)

        single_ranks[key] = first_hit_rank(
            chunks,
            locations,
            threshold=threshold,
        )

        union_ranks[key] = first_union_hit_rank(
            chunks,
            locations,
            threshold=threshold,
        )

    return {
        "single_ranks": single_ranks,
        "union_ranks": union_ranks,
    }


def summarize_ranking_results(
    question_results: list[dict],
    thresholds: tuple[float, ...],
) -> dict:
    """Aggregate raw ranking metrics across questions."""

    if not question_results:
        raise ValueError("No question results to summarize.")

    summary = {}

    for mode in ("single", "union"):
        summary[mode] = {}

        rank_key = f"{mode}_ranks"

        for threshold in thresholds:
            key = str(threshold)

            ranks = [item[rank_key][key] for item in question_results]

            summary[mode][key] = summarize_ranks(
                ranks,
                ks=(1, 5, 10),
            )

    return summary


# ============================================================
# Context evidence evaluation
# ============================================================


def evaluate_packed_context(
    packed: PackedContext,
    locations: list[EvidenceLocation],
    thresholds: tuple[float, ...],
) -> dict:
    """Evaluate evidence coverage in the final packed context."""

    if not locations:
        raise ValueError("Evidence locations cannot be empty.")

    chunks = [item.chunk for item in packed.selected]

    best_single_coverage = max(
        (
            chunk_evidence_coverage(chunk, location)
            for chunk in chunks
            for location in locations
        ),
        default=0.0,
    )

    best_union_coverage = max(
        (union_evidence_coverage(chunks, location) for location in locations),
        default=0.0,
    )

    single_ranks = {}
    union_ranks = {}

    for threshold in thresholds:
        if not 0 < threshold <= 1:
            raise ValueError("Threshold must be in (0, 1].")

        key = str(threshold)

        single_rank = next(
            (
                rank
                for rank, item in enumerate(
                    packed.selected,
                    start=1,
                )
                if any(
                    chunk_evidence_coverage(
                        item.chunk,
                        location,
                    )
                    >= threshold
                    for location in locations
                )
            ),
            None,
        )

        union_rank = first_union_hit_rank(
            chunks,
            locations,
            threshold=threshold,
        )

        single_ranks[key] = single_rank
        union_ranks[key] = union_rank

    return {
        "context_tokens": packed.token_count,
        "budget": packed.budget,
        "budget_utilization": packed.token_count / packed.budget,
        "selected_chunk_count": len(packed.selected),
        "candidate_count": packed.candidate_count,
        "skipped_candidates": packed.skipped_candidates,
        "oversized_candidates": packed.oversized_candidates,
        "selected_oversize_sentences": (packed.selected_oversize_sentences),
        "best_single_evidence_coverage": best_single_coverage,
        "best_union_evidence_coverage": best_union_coverage,
        "single_ranks": single_ranks,
        "union_ranks": union_ranks,
    }


# ============================================================
# Context budget metrics
# ============================================================


def summarize_budget_results(
    question_results: list[dict],
    thresholds: tuple[float, ...],
) -> dict:
    """Aggregate context and evidence metrics."""

    if not question_results:
        raise ValueError("No question results to summarize.")

    single_metrics = {}
    union_metrics = {}
    context_hit_rates = {}

    for threshold in thresholds:
        key = str(threshold)

        single_ranks = [item["single_ranks"][key] for item in question_results]

        union_ranks = [item["union_ranks"][key] for item in question_results]

        single_metrics[key] = summarize_ranks(
            single_ranks,
            ks=(1, 5, 10),
        )

        union_metrics[key] = summarize_ranks(
            union_ranks,
            ks=(1, 5, 10),
        )

        context_hit_rates[key] = {
            "single": (
                sum(rank is not None for rank in single_ranks) / len(single_ranks)
            ),
            "union": (sum(rank is not None for rank in union_ranks) / len(union_ranks)),
        }

    context_tokens = [item["context_tokens"] for item in question_results]

    return {
        "questions": len(question_results),
        "mean_context_tokens": mean(context_tokens),
        "median_context_tokens": median(context_tokens),
        "mean_budget_utilization": mean(
            item["budget_utilization"] for item in question_results
        ),
        "mean_selected_chunks": mean(
            item["selected_chunk_count"] for item in question_results
        ),
        "mean_skipped_candidates": mean(
            item["skipped_candidates"] for item in question_results
        ),
        "mean_oversized_candidates": mean(
            item["oversized_candidates"] for item in question_results
        ),
        "mean_selected_oversize_sentences": mean(
            item["selected_oversize_sentences"] for item in question_results
        ),
        "questions_with_empty_context": sum(
            item["context_tokens"] == 0 for item in question_results
        ),
        "mean_best_single_evidence_coverage": mean(
            item["best_single_evidence_coverage"] for item in question_results
        ),
        "mean_best_union_evidence_coverage": mean(
            item["best_union_evidence_coverage"] for item in question_results
        ),
        "retrieval": {
            "single": single_metrics,
            "union": union_metrics,
            "context_hit_rate": context_hit_rates,
        },
    }


# ============================================================
# QASPER loading
# ============================================================


def load_questions(split: str) -> tuple[list[dict], dict]:
    """Load QA and locate exact evidence occurrences in paper bodies."""

    paper_path = Path(f"data/processed/{split}_papers.jsonl")
    qa_path = Path(f"data/processed/{split}_qa.jsonl")

    if not paper_path.is_file():
        raise FileNotFoundError(paper_path)

    if not qa_path.is_file():
        raise FileNotFoundError(qa_path)

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

    prepared = []

    for item in questions.values():
        if not item["locations"]:
            continue

        item["locations"] = sorted(
            item["locations"],
            key=lambda location: (
                location.section_index,
                location.paragraph_index,
                location.start_char,
                location.end_char,
            ),
        )

        prepared.append(item)

    stats = {
        "total_questions": len(questions),
        "body_evidence_questions": len(prepared),
        "excluded_without_body_evidence": (len(questions) - len(prepared)),
        "missing_paper_records": missing_paper_records,
        "total_papers": len(papers),
    }

    return prepared, stats


# ============================================================
# V2 index loading
# ============================================================


def get_v2_index_dir(split: str) -> Path:
    """Return the only active index directory for this experiment."""

    return Path(f"data/indexes/qasper_{split}_windowed_v2")


def load_windowed_index(
    directory: Path,
    model_name: str,
) -> tuple[dict, dict, dict]:
    """Load V2 Parent Chunks, Windows and embeddings by paper."""

    if not directory.is_dir():
        raise FileNotFoundError(f"V2 index directory does not exist: {directory}")

    store = WindowedIndexStore(directory)

    chunks, windows, embeddings, manifest = store.load(
        expected_model_name=model_name,
    )

    if len(chunks) != manifest["num_chunks"]:
        raise ValueError("V2 parent chunk count mismatch.")

    if embeddings.shape[0] != len(windows):
        raise ValueError("V2 window and embedding count mismatch.")

    chunks_by_paper = defaultdict(list)
    windows_by_paper = defaultdict(list)

    for chunk in chunks:
        chunks_by_paper[chunk.paper_id].append(chunk)

    for index, window in enumerate(windows):
        windows_by_paper[window.paper_id].append(index)

    grouped = {}

    for paper_id, paper_chunks in chunks_by_paper.items():
        indices = windows_by_paper.get(paper_id, [])

        if not indices:
            raise ValueError(f"Paper {paper_id} has parent chunks but no windows.")

        index_array = np.asarray(indices, dtype=np.int64)

        paper_windows = [windows[int(index)] for index in index_array]

        paper_embeddings = embeddings[index_array]

        if paper_embeddings.shape[0] != len(paper_windows):
            raise ValueError(f"Window embedding mismatch for paper {paper_id}.")

        grouped[paper_id] = {
            "chunks": paper_chunks,
            "chunks_by_id": {chunk.chunk_id: chunk for chunk in paper_chunks},
            "windows": paper_windows,
            "embeddings": paper_embeddings,
        }

    config_path = directory / "build_config.json"

    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    build_config = json.loads(config_path.read_text(encoding="utf-8"))

    if not isinstance(build_config, dict):
        raise TypeError("V2 build configuration must be a JSON object.")

    print(
        f"Loaded V2 index: {len(chunks)} parent chunks, "
        f"{len(windows)} windows, {len(grouped)} papers"
    )

    return grouped, manifest, build_config


# ============================================================
# Dense V2 ranking
# ============================================================


def rank_windowed_chunks(
    query_vector: np.ndarray,
    index: dict,
) -> list[RankedChunk]:
    """Rank V2 Parent Chunks using their best matching Window.

    All Parent Chunks in the current paper are ranked.
    Multiple matching windows from the same parent are collapsed.
    """

    query = np.asarray(query_vector, dtype=np.float32)

    if query.ndim != 1:
        raise ValueError("Query embedding must be one-dimensional.")

    windows = index["windows"]
    embeddings = index["embeddings"]
    chunks_by_id = index["chunks_by_id"]

    if embeddings.ndim != 2:
        raise ValueError("Window embeddings must be a 2D array.")

    if embeddings.shape != (len(windows), len(query)):
        raise ValueError("Window embedding dimensions do not match.")

    scores = embeddings @ query

    if not np.isfinite(scores).all():
        raise ValueError("Dense retrieval produced invalid scores.")

    order = np.argsort(
        -scores,
        kind="stable",
    )

    ranked_chunks = []
    seen_parent_ids = set()

    for position in order:
        window_index = int(position)
        window = windows[window_index]
        parent_id = window.parent_chunk_id

        if parent_id in seen_parent_ids:
            continue

        chunk = chunks_by_id.get(parent_id)

        if chunk is None:
            raise ValueError(f"Unknown parent chunk: {parent_id}")

        seen_parent_ids.add(parent_id)

        ranked_chunks.append(
            RankedChunk(
                rank=len(ranked_chunks) + 1,
                score=float(scores[window_index]),
                chunk=chunk,
            )
        )

    if len(ranked_chunks) != len(chunks_by_id):
        raise ValueError("Not all V2 parent chunks received a retrieval rank.")

    return ranked_chunks


# ============================================================
# V2 Dense baseline evaluation
# ============================================================


def evaluate_v2_dense(
    questions: list[dict],
    query_vectors: np.ndarray,
    indexes: dict,
    tokenizer,
    budgets: tuple[int, ...],
    thresholds: tuple[float, ...],
    token_cache: dict,
    separator_tokens: int,
) -> dict:
    """Evaluate V2 Dense ranking and context budget performance."""

    if len(questions) != len(query_vectors):
        raise ValueError("Question count and query embedding count do not match.")

    budget_records = {budget: [] for budget in budgets}

    ranking_records = []

    per_question = []

    iterator = zip(
        questions,
        query_vectors,
        strict=True,
    )

    for item, query_vector in tqdm(
        iterator,
        total=len(questions),
        desc="Evaluating V2 Dense",
    ):
        paper_id = item["paper_id"]

        if paper_id not in indexes:
            raise KeyError(f"Missing V2 index for paper {paper_id}.")

        paper_index = indexes[paper_id]

        ranked_chunks = rank_windowed_chunks(
            query_vector,
            paper_index,
        )

        raw_ranking = evaluate_ranked_chunks(
            ranked_chunks,
            item["locations"],
            thresholds,
        )

        ranking_records.append(raw_ranking)

        question_result = {
            "paper_id": paper_id,
            "question_id": item["question_id"],
            "question": item["question"],
            "evidence_location_count": len(item["locations"]),
            "top_10": [
                {
                    "rank": candidate.rank,
                    "chunk_id": candidate.chunk.chunk_id,
                    "score": candidate.score,
                }
                for candidate in ranked_chunks[:TOP_K]
            ],
            "raw_ranking": raw_ranking,
            "context_by_budget": {},
        }

        for budget in budgets:
            packed = pack_ranked_chunks(
                ranked_chunks=ranked_chunks,
                tokenizer=tokenizer,
                budget=budget,
                version="V2",
                token_cache=token_cache,
                separator_tokens=separator_tokens,
            )

            context_result = evaluate_packed_context(
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

    ranking_summary = summarize_ranking_results(
        question_results=ranking_records,
        thresholds=thresholds,
    )

    budget_summary = {
        str(budget): summarize_budget_results(
            question_results=budget_records[budget],
            thresholds=thresholds,
        )
        for budget in budgets
    }

    return {
        "ranking": ranking_summary,
        "budgets": budget_summary,
        "per_question": per_question,
        "token_cache_entries": len(token_cache),
    }


# ============================================================
# Main
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the V2 Dense retrieval baseline only. "
            "V0 and V1 are not loaded or compared."
        )
    )

    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        required=True,
    )

    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=list(DEFAULT_BUDGETS),
    )

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    budgets = tuple(args.budgets)
    thresholds = tuple(args.thresholds)

    if not budgets or any(budget <= 0 for budget in budgets):
        parser.error("All token budgets must be positive.")

    if len(set(budgets)) != len(budgets):
        parser.error("Token budgets must be unique.")

    if not thresholds or any(not 0 < threshold <= 1 for threshold in thresholds):
        parser.error("All thresholds must be in (0, 1].")

    if len(set(thresholds)) != len(thresholds):
        parser.error("Thresholds must be unique.")

    split = args.split

    output_path = args.output or Path(
        f"storage/metadata/day6_dense_v2_eval_{split}.json"
    )

    # Never overwrite existing experimental results.
    if output_path.exists():
        parser.error(
            f"Refusing to overwrite existing result: {output_path}. "
            "Choose a new --output path."
        )

    print("\n========== Day 6: V2 Dense Baseline ==========")
    print("Dataset:", split)
    print("Retrieval method: Dense V2")
    print("Context budgets:", budgets)
    print("Evidence thresholds:", thresholds)
    print("Ranking scope: all Parent Chunks within the same paper")
    print("Output:", output_path)

    # --------------------------------------------------------
    # Step 1: Load evaluation questions.
    # --------------------------------------------------------

    print("\n========== Step 1: Loading Evaluation Data ==========")

    questions, question_stats = load_questions(split)

    for key, value in question_stats.items():
        print(f"{key}: {value}")

    if not questions:
        raise ValueError("No questions with body evidence.")

    # --------------------------------------------------------
    # Step 2: Load embedding model and tokenizer.
    # --------------------------------------------------------

    print("\n========== Step 2: Loading Embedding Model ==========")

    embedder = TextEmbedder(
        device=args.device,
    )

    tokenizer = embedder.model.tokenizer

    separator_tokens = count_tokens(
        tokenizer,
        CHUNK_SEPARATOR,
    )

    print("Embedding model:", embedder.model_name)
    print("Embedding dimension:", embedder.dimension)
    print("Maximum sequence length:", embedder.max_seq_length)
    print("Context tokenizer:", embedder.model_name)
    print("Separator tokens:", separator_tokens)

    print(
        "WARNING: The embedding tokenizer is used as a proxy "
        "because the final generation model has not been selected."
    )

    # --------------------------------------------------------
    # Step 3: Load V2 index only.
    # --------------------------------------------------------

    print("\n========== Step 3: Loading V2 Index ==========")

    index_dir = get_v2_index_dir(split)

    indexes, v2_manifest, v2_build_config = load_windowed_index(
        directory=index_dir,
        model_name=embedder.model_name,
    )

    print("\nV2 Manifest:")
    print(
        json.dumps(
            v2_manifest,
            ensure_ascii=False,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # Step 4: Build the evaluation question set.
    # --------------------------------------------------------

    eligible = [item for item in questions if item["paper_id"] in indexes]

    excluded_missing_index = len(questions) - len(eligible)

    print("\n========== Step 4: Evaluation Set ==========")
    print("Eligible questions:", len(eligible))
    print("Excluded missing V2 index:", excluded_missing_index)

    if not eligible:
        raise ValueError("No evaluable questions with V2 indexes.")

    # --------------------------------------------------------
    # Step 5: Encode queries once.
    # --------------------------------------------------------

    print("\n========== Step 5: Encoding Queries ==========")

    query_vectors = embedder.encode_documents(
        [item["question"] for item in eligible],
        batch_size=32,
        show_progress_bar=True,
    )

    if query_vectors.shape != (
        len(eligible),
        embedder.dimension,
    ):
        raise ValueError("Query embedding shape mismatch.")

    print("Query embeddings:", query_vectors.shape)

    # --------------------------------------------------------
    # Step 6: Evaluate raw ranking and context budgets.
    # --------------------------------------------------------

    print("\n========== Step 6: V2 Dense Evaluation ==========")

    token_cache: dict[tuple[str, str], int] = {}

    results = evaluate_v2_dense(
        questions=eligible,
        query_vectors=query_vectors,
        indexes=indexes,
        tokenizer=tokenizer,
        budgets=budgets,
        thresholds=thresholds,
        token_cache=token_cache,
        separator_tokens=separator_tokens,
    )

    # --------------------------------------------------------
    # Step 7: Print ranking results.
    # --------------------------------------------------------

    report_threshold = 0.5 if 0.5 in thresholds else thresholds[0]

    threshold_key = str(report_threshold)

    print("\n========== Raw Ranking Metrics ==========")

    for mode in ("single", "union"):
        metrics = results["ranking"][mode][threshold_key]

        print(f"\n--- {mode.upper()} / {report_threshold:.0%} ---")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

    # --------------------------------------------------------
    # Step 8: Print context budget results.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Step 9: Save new results.
    # --------------------------------------------------------

    output = {
        "schema_version": 1,
        "experiment": "day6_dense_v2_baseline",
        "split": split,
        "retrieval_method": "dense_v2",
        "retrieval_scope": "within_paper",
        "embedding_model": embedder.model_name,
        "embedding_dimension": embedder.dimension,
        "ranking": {
            "ranking_strategy": (
                "Rank embedding windows by cosine similarity, "
                "then assign each parent chunk its best window score."
            ),
            "parent_deduplication": True,
            "ranking_scope": ("All parent chunks within the corresponding paper."),
            "metrics": results["ranking"],
        },
        "context_budget_evaluation": {
            "tokenizer": embedder.model_name,
            "tokenizer_is_proxy": True,
            "special_tokens": False,
            "truncation": False,
            "context_separator": CHUNK_SEPARATOR,
            "budgets": list(budgets),
            "thresholds": list(thresholds),
            "packing_policy": (
                "Greedy ranking-order packing of complete parent chunks. "
                "Non-fitting chunks are skipped; later smaller chunks "
                "may still be selected. No chunk is truncated or split. "
                "The final assembled context is re-tokenized."
            ),
            "metrics": results["budgets"],
        },
        "question_statistics": {
            **question_stats,
            "excluded_missing_index": excluded_missing_index,
            "final_evaluable_questions": len(eligible),
        },
        "v2_index_manifest": v2_manifest,
        "v2_index_build_config": v2_build_config,
        "token_cache_entries": results["token_cache_entries"],
        "per_question": results["per_question"],
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("\n========== Evaluation Completed ==========")
    print("Results saved to:", output_path)
    print("Token cache entries:", results["token_cache_entries"])
    print("Evaluated questions:", len(eligible))


if __name__ == "__main__":
    main()
