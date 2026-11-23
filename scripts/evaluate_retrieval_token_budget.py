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
    first_union_hit_rank,
    locate_body_evidence,
    summarize_ranks,
    union_evidence_coverage,
)
from paper_rag.retrieval.embedder import TextEmbedder
from paper_rag.retrieval.index_store import IndexStore
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

VERSIONS = ("V0", "V1", "V2")

VERSION_ROLES = {
    "V0": "diagnostic_only",
    "V1": "diagnostic_only",
    "V2": "only_valid_sentence_integrity_candidate",
}

CHUNK_SEPARATOR = "\n\n"

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


def build_context_text(
    ranked_chunks: list[RankedChunk],
) -> str:
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
    5. Re-tokenize the final assembled context to guarantee
       that its actual token count does not exceed the budget.

    Individual chunk token counts are cached across questions
    and budgets to avoid repeated tokenization.
    """

    if budget <= 0:
        raise ValueError("Token budget must be positive.")

    if separator_tokens < 0:
        raise ValueError("Separator token count cannot be negative.")

    selected = []
    estimated_tokens = 0
    oversized_candidates = 0

    # First pass: greedy packing using cached chunk token counts.
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

        # An individual chunk cannot fit into this budget.
        if chunk_tokens > budget:
            oversized_candidates += 1
            continue

        separator_cost = separator_tokens if selected else 0

        additional_tokens = chunk_tokens + separator_cost

        if estimated_tokens + additional_tokens <= budget:
            selected.append(candidate)
            estimated_tokens += additional_tokens

    # Second pass: verify the actual assembled context.
    # Tokenization across text boundaries is not assumed to be
    # perfectly additive.
    context_text = build_context_text(selected)

    actual_tokens = count_tokens(tokenizer, context_text) if context_text else 0

    # If the exact count exceeds the budget, remove complete
    # trailing chunks until the hard limit is satisfied.
    while selected and actual_tokens > budget:
        selected.pop()

        context_text = build_context_text(selected)

        actual_tokens = count_tokens(tokenizer, context_text) if context_text else 0

    if actual_tokens > budget:
        raise RuntimeError("Final context exceeds the token budget.")

    selected_oversize_sentences = sum(
        bool(
            getattr(
                item.chunk,
                "oversize_sentence",
                False,
            )
        )
        for item in selected
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
# Evidence evaluation
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

    # Maximum coverage achieved by any individual selected chunk.
    best_single_coverage = max(
        (
            chunk_evidence_coverage(chunk, location)
            for chunk in chunks
            for location in locations
        ),
        default=0.0,
    )

    # Maximum coverage achieved by the union of all selected chunks.
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

        # First individually sufficient chunk.
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

        # First prefix of selected chunks whose union
        # reaches the evidence coverage threshold.
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
# Metrics
# ============================================================


def summarize_budget_results(
    question_results: list[dict],
    thresholds: tuple[float, ...],
) -> dict:
    """Aggregate context and retrieval metrics."""

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

        # Evidence appears anywhere in the complete selected context.
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
# Index loading
# ============================================================


def get_index_dirs(split: str) -> dict[str, Path]:
    """Return index directories for V0, V1 and V2."""

    return {
        "V0": Path(f"data/indexes/qasper_{split}_dense"),
        "V1": Path(f"data/indexes/qasper_{split}_semantic_v1"),
        "V2": Path(f"data/indexes/qasper_{split}_windowed_v2"),
    }


def load_dense_index(
    directory: Path,
    model_name: str,
) -> dict:
    """Load a dense index and group chunks by paper."""

    if not directory.is_dir():
        raise FileNotFoundError(f"Index directory does not exist: {directory}")

    chunks, embeddings, manifest = IndexStore(directory).load(
        expected_model_name=model_name,
    )

    if len(chunks) != manifest["num_chunks"]:
        raise ValueError("Dense index chunk count mismatch.")

    indices_by_paper = defaultdict(list)

    for index, chunk in enumerate(chunks):
        indices_by_paper[chunk.paper_id].append(index)

    grouped = {}

    for paper_id, indices in indices_by_paper.items():
        index_array = np.asarray(
            indices,
            dtype=np.int64,
        )

        grouped[paper_id] = {
            "chunks": [chunks[int(index)] for index in index_array],
            "embeddings": embeddings[index_array],
        }

    print(f"Loaded {directory.name}: {len(chunks)} chunks, {len(grouped)} papers")

    return grouped


def load_windowed_index(
    directory: Path,
    model_name: str,
) -> tuple[dict, dict]:
    """Load V2 parent chunks, windows and window embeddings."""

    if not directory.is_dir():
        raise FileNotFoundError(f"Index directory does not exist: {directory}")

    chunks, windows, embeddings, manifest = WindowedIndexStore(directory).load(
        expected_model_name=model_name,
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

        grouped[paper_id] = {
            "chunks": paper_chunks,
            "chunks_by_id": {chunk.chunk_id: chunk for chunk in paper_chunks},
            "windows": [windows[int(index)] for index in indices],
            "embeddings": embeddings[indices],
        }

    print(
        f"Loaded {directory.name}: "
        f"{len(chunks)} parent chunks, "
        f"{len(windows)} windows, "
        f"{len(grouped)} papers"
    )

    return grouped, manifest


# ============================================================
# Retrieval ranking
# ============================================================


def rank_dense_chunks(
    query_vector: np.ndarray,
    index: dict,
) -> list[RankedChunk]:
    """Rank all parent chunks in one paper using dense similarity."""

    chunks = index["chunks"]
    embeddings = index["embeddings"]

    if embeddings.ndim != 2:
        raise ValueError("Dense embeddings must be a 2D array.")

    if embeddings.shape != (len(chunks), len(query_vector)):
        raise ValueError("Dense embedding dimensions do not match.")

    scores = embeddings @ query_vector

    order = np.argsort(
        -scores,
        kind="stable",
    )

    return [
        RankedChunk(
            rank=rank,
            score=float(scores[int(position)]),
            chunk=chunks[int(position)],
        )
        for rank, position in enumerate(order, start=1)
    ]


def rank_windowed_chunks(
    query_vector: np.ndarray,
    index: dict,
) -> list[RankedChunk]:
    """Rank V2 parent chunks by their best matching embedding window."""

    windows = index["windows"]
    embeddings = index["embeddings"]
    chunks_by_id = index["chunks_by_id"]

    if embeddings.ndim != 2:
        raise ValueError("Window embeddings must be a 2D array.")

    if embeddings.shape != (len(windows), len(query_vector)):
        raise ValueError("Window embedding dimensions do not match.")

    scores = embeddings @ query_vector

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
# Version evaluation
# ============================================================


def evaluate_version(
    version: str,
    questions: list[dict],
    query_vectors: np.ndarray,
    index: dict,
    tokenizer,
    budgets: tuple[int, ...],
    thresholds: tuple[float, ...],
    token_cache: dict[tuple[str, str], int],
    separator_tokens: int,
) -> dict:
    """Evaluate one retrieval version under multiple token budgets."""

    records = {budget: [] for budget in budgets}

    failure_examples = {budget: [] for budget in budgets}

    iterator = zip(
        questions,
        query_vectors,
        strict=True,
    )

    for item, query_vector in tqdm(
        iterator,
        total=len(questions),
        desc=f"Evaluating {version}",
    ):
        paper_id = item["paper_id"]
        paper_index = index[paper_id]

        if version == "V2":
            ranked_chunks = rank_windowed_chunks(
                query_vector,
                paper_index,
            )
        else:
            ranked_chunks = rank_dense_chunks(
                query_vector,
                paper_index,
            )

        for budget in budgets:
            packed = pack_ranked_chunks(
                ranked_chunks=ranked_chunks,
                tokenizer=tokenizer,
                budget=budget,
                version=version,
                token_cache=token_cache,
                separator_tokens=separator_tokens,
            )

            result = evaluate_packed_context(
                packed=packed,
                locations=item["locations"],
                thresholds=thresholds,
            )

            records[budget].append(result)

            # Collect V2 cases where the complete context still
            # fails to cover at least 50% of any gold occurrence.
            if version == "V2" and result["best_union_evidence_coverage"] < 0.5:
                failure_examples[budget].append(
                    {
                        "paper_id": paper_id,
                        "question_id": item["question_id"],
                        "question": item["question"],
                        "context_tokens": result["context_tokens"],
                        "selected_chunk_count": (result["selected_chunk_count"]),
                        "best_single_evidence_coverage": (
                            result["best_single_evidence_coverage"]
                        ),
                        "best_union_evidence_coverage": (
                            result["best_union_evidence_coverage"]
                        ),
                        "oversized_candidates": (result["oversized_candidates"]),
                        "selected_chunk_ids": [
                            candidate.chunk.chunk_id for candidate in packed.selected
                        ],
                    }
                )

    summaries = {}

    for budget in budgets:
        summaries[str(budget)] = summarize_budget_results(
            question_results=records[budget],
            thresholds=thresholds,
        )

    output = {
        "role": VERSION_ROLES[version],
        "budgets": summaries,
    }

    if version == "V2":
        output["failure_examples_below_50pct_union"] = {}

        for budget in budgets:
            examples = sorted(
                failure_examples[budget],
                key=lambda item: (
                    item["best_union_evidence_coverage"],
                    item["best_single_evidence_coverage"],
                ),
            )

            output["failure_examples_below_50pct_union"][str(budget)] = examples[:20]

    return output


# ============================================================
# Historical baseline comparison
# ============================================================


def calculate_legacy_deltas(
    results: dict,
    budgets: tuple[int, ...],
    thresholds: tuple[float, ...],
) -> dict:
    """Compare V2 with V0/V1 for diagnosis, not candidate selection."""

    deltas = {}

    for budget in budgets:
        budget_key = str(budget)
        deltas[budget_key] = {}

        v2 = results["V2"]["budgets"][budget_key]

        for mode in ("single", "union"):
            deltas[budget_key][mode] = {}

            for threshold in thresholds:
                threshold_key = str(threshold)

                v2_metrics = v2["retrieval"][mode][threshold_key]

                v2_context_hit = v2["retrieval"]["context_hit_rate"][threshold_key][
                    mode
                ]

                deltas[budget_key][mode][threshold_key] = {}

                for baseline in ("V0", "V1"):
                    baseline_result = results[baseline]["budgets"][budget_key]

                    baseline_metrics = baseline_result["retrieval"][mode][threshold_key]

                    baseline_context_hit = baseline_result["retrieval"][
                        "context_hit_rate"
                    ][threshold_key][mode]

                    deltas[budget_key][mode][threshold_key][baseline] = {
                        "hit@10_delta": (
                            v2_metrics["hit@10"] - baseline_metrics["hit@10"]
                        ),
                        "mrr@10_delta": (
                            v2_metrics["mrr@10"] - baseline_metrics["mrr@10"]
                        ),
                        "context_hit_rate_delta": (
                            v2_context_hit - baseline_context_hit
                        ),
                    }

    return deltas


# ============================================================
# Main
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate V0/V1/V2 retrieval contexts under fixed token budgets. "
            "V0 and V1 are diagnostic references only; V2 is the only "
            "sentence-integrity-eligible candidate."
        )
    )

    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="train",
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

    print("\n========== Configuration ==========")
    print("Dataset:", split)
    print("Versions:", VERSIONS)
    print("Version roles:", VERSION_ROLES)
    print("Context budgets:", budgets)
    print("Evidence thresholds:", thresholds)
    print("Context separator:", repr(CHUNK_SEPARATOR))
    print("Ranking scope: all parent chunks within the same paper")

    # --------------------------------------------------------
    # Step 1: Load QA and evidence locations.
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
        "\nWARNING: The embedding tokenizer is used as a proxy "
        "because the final generation model has not been selected."
    )

    # --------------------------------------------------------
    # Step 3: Load indexes.
    # --------------------------------------------------------

    print("\n========== Step 3: Loading Indexes ==========")

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

    # --------------------------------------------------------
    # Step 4: Build a common evaluation set.
    # --------------------------------------------------------

    eligible = [
        item
        for item in questions
        if all(item["paper_id"] in indexes[version] for version in VERSIONS)
    ]

    excluded_missing_index = len(questions) - len(eligible)

    print("\n========== Step 4: Evaluation Set ==========")
    print("Eligible questions:", len(eligible))
    print("Excluded missing index:", excluded_missing_index)

    if not eligible:
        raise ValueError("No common evaluable questions.")

    # --------------------------------------------------------
    # Step 5: Encode each query once.
    # --------------------------------------------------------

    print("\n========== Step 5: Encoding Queries ==========")

    query_vectors = embedder.encode_documents(
        [item["question"] for item in eligible],
        batch_size=32,
        show_progress_bar=True,
    )

    print("Query embeddings:", query_vectors.shape)

    # --------------------------------------------------------
    # Step 6: Evaluate all versions and budgets.
    # --------------------------------------------------------

    print("\n========== Step 6: Token Budget Evaluation ==========")

    token_cache: dict[tuple[str, str], int] = {}

    results = {}

    for version in VERSIONS:
        results[version] = evaluate_version(
            version=version,
            questions=eligible,
            query_vectors=query_vectors,
            index=indexes[version],
            tokenizer=tokenizer,
            budgets=budgets,
            thresholds=thresholds,
            token_cache=token_cache,
            separator_tokens=separator_tokens,
        )

    # --------------------------------------------------------
    # Step 7: Compare historical references.
    # --------------------------------------------------------

    print("\n========== Step 7: Diagnostic Comparison ==========")

    legacy_deltas = calculate_legacy_deltas(
        results=results,
        budgets=budgets,
        thresholds=thresholds,
    )

    report_threshold = 0.5 if 0.5 in thresholds else thresholds[0]
    threshold_key = str(report_threshold)

    for budget in budgets:
        print(f"\n--- Context Budget: {budget} Tokens ---")

        for version in VERSIONS:
            metrics = results[version]["budgets"][str(budget)]

            union_metrics = metrics["retrieval"]["union"][threshold_key]

            union_context_hit = metrics["retrieval"]["context_hit_rate"][threshold_key][
                "union"
            ]

            print(
                f"{version} ({results[version]['role']}): "
                f"Hit@10={union_metrics['hit@10']:.4f}, "
                f"MRR@10={union_metrics['mrr@10']:.4f}, "
                f"Context Hit Rate={union_context_hit:.4f}, "
                f"Mean Tokens={metrics['mean_context_tokens']:.1f}, "
                f"Mean Chunks={metrics['mean_selected_chunks']:.2f}"
            )

    # --------------------------------------------------------
    # Step 8: Save results.
    # --------------------------------------------------------

    output_path = args.output or Path(
        f"storage/metadata/context_budget_eval_{split}.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "split": split,
        "embedding_model": embedder.model_name,
        "context_tokenizer": embedder.model_name,
        "context_tokenizer_is_proxy": True,
        "token_counting": {
            "add_special_tokens": False,
            "truncation": False,
            "scope": (
                "Retrieved parent chunk text and chunk separators only. "
                "Question, system prompt, instructions and metadata "
                "are excluded."
            ),
        },
        "evaluation_goal": (
            "Evaluate V2 as the only sentence-integrity-eligible "
            "candidate. V0 and V1 are historical diagnostic references."
        ),
        "ranking_scope": (
            "All parent chunks within the corresponding paper are ranked. "
            "There is no fixed Top-K cutoff during context packing."
        ),
        "packing_policy": (
            "Greedy ranking-order packing of complete parent chunks. "
            "Non-fitting chunks are skipped, later smaller chunks may "
            "still be selected. No chunk is truncated or split. "
            "The assembled context is re-tokenized before evaluation."
        ),
        "context_separator": CHUNK_SEPARATOR,
        "budgets": list(budgets),
        "thresholds": list(thresholds),
        "question_statistics": {
            **question_stats,
            "excluded_missing_index": excluded_missing_index,
            "final_evaluable_questions": len(eligible),
        },
        "v2_index_manifest": v2_manifest,
        "version_roles": VERSION_ROLES,
        "results": results,
        "legacy_reference_deltas": legacy_deltas,
        "token_cache_entries": len(token_cache),
    }

    output_path.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n========== Evaluation Completed ==========")
    print("Results saved to:", output_path)
    print("Token cache entries:", len(token_cache))


if __name__ == "__main__":
    main()
