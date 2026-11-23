import importlib.util
import sys
from pathlib import Path

import pytest

from paper_rag.evaluation.retrieval_evaluator import EvidenceLocation
from paper_rag.schemas import ChunkSpan, SectionChunk, SentenceTokenChunk

# ============================================================
# Load the standalone experiment script by file path.
# Do not import scripts as a Python package.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "evaluate_retrieval_token_budget.py"

SPEC = importlib.util.spec_from_file_location(
    "_evaluate_retrieval_token_budget",
    SCRIPT_PATH,
)

if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load experiment script: {SCRIPT_PATH}")

SCRIPT_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT_MODULE
SPEC.loader.exec_module(SCRIPT_MODULE)

RankedChunk = SCRIPT_MODULE.RankedChunk
count_tokens = SCRIPT_MODULE.count_tokens
evaluate_packed_context = SCRIPT_MODULE.evaluate_packed_context
pack_ranked_chunks = SCRIPT_MODULE.pack_ranked_chunks
summarize_budget_results = SCRIPT_MODULE.summarize_budget_results


class FakeTokenizer:
    """Deterministic whitespace tokenizer for unit tests."""

    def __call__(
        self,
        text,
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    ):
        return {"input_ids": text.split()}


class BoundaryTokenizer:
    """Simulate non-additive token counts at text boundaries."""

    def __call__(
        self,
        text,
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    ):
        if text == "\n\n":
            tokens = ["SEP"]
        elif text == "A":
            tokens = ["A"]
        elif text == "B":
            tokens = ["B"]
        elif text == "A\n\nB":
            tokens = ["A", "SEP", "B", "EXTRA"]
        else:
            tokens = text.split()

        return {"input_ids": tokens}


def make_chunk(
    chunk_id: str,
    text: str,
    start_char: int = 0,
) -> SectionChunk:
    """Create a chunk with valid source offsets."""

    return SectionChunk(
        chunk_id=chunk_id,
        paper_id="paper-1",
        split="train",
        section_index=0,
        section_name="Introduction",
        text=text,
        spans=[
            ChunkSpan(
                paragraph_index=0,
                start_char=start_char,
                end_char=start_char + len(text),
                chunk_start=0,
                chunk_end=len(text),
            )
        ],
    )


def make_sentence_chunk(
    chunk_id: str,
    text: str,
    oversize_sentence: bool = False,
) -> SentenceTokenChunk:
    """Create a complete-sentence parent chunk."""

    return SentenceTokenChunk(
        chunk_id=chunk_id,
        paper_id="paper-1",
        split="train",
        section_index=0,
        section_name="Introduction",
        text=text,
        spans=[
            ChunkSpan(
                paragraph_index=0,
                start_char=0,
                end_char=len(text),
                chunk_start=0,
                chunk_end=len(text),
            )
        ],
        token_count=1,
        sentence_count=1,
        oversize_sentence=oversize_sentence,
    )


def make_ranked(chunk, rank: int) -> RankedChunk:
    return RankedChunk(
        rank=rank,
        score=1.0 / rank,
        chunk=chunk,
    )


def make_packed(
    chunks,
    tokenizer=None,
    budget=100,
    version="V2",
):
    """Convenience helper for context packing tests."""

    if tokenizer is None:
        tokenizer = FakeTokenizer()

    return pack_ranked_chunks(
        ranked_chunks=[
            make_ranked(chunk, rank) for rank, chunk in enumerate(chunks, start=1)
        ],
        tokenizer=tokenizer,
        budget=budget,
        version=version,
        token_cache={},
        separator_tokens=count_tokens(tokenizer, "\n\n"),
    )


# ============================================================
# Token counting
# ============================================================


def test_count_tokens_without_special_tokens():
    tokenizer = FakeTokenizer()

    assert count_tokens(tokenizer, "one two three") == 3


def test_count_tokens_empty_text():
    tokenizer = FakeTokenizer()

    assert count_tokens(tokenizer, "") == 0


# ============================================================
# Context packing
# ============================================================


def test_pack_complete_chunks_within_budget():
    tokenizer = FakeTokenizer()

    chunks = [
        make_ranked(make_chunk("c1", "one two"), 1),
        make_ranked(make_chunk("c2", "three four five"), 2),
        make_ranked(make_chunk("c3", "six"), 3),
    ]

    packed = pack_ranked_chunks(
        ranked_chunks=chunks,
        tokenizer=tokenizer,
        budget=3,
        version="V0",
        token_cache={},
        separator_tokens=0,
    )

    assert [item.chunk.chunk_id for item in packed.selected] == [
        "c1",
        "c3",
    ]

    assert packed.text == "one two\n\nsix"
    assert packed.token_count == 3
    assert packed.token_count <= packed.budget


def test_oversized_chunk_is_skipped():
    tokenizer = FakeTokenizer()

    chunks = [
        make_ranked(make_chunk("large", "one two three"), 1),
        make_ranked(make_chunk("small", "four"), 2),
    ]

    packed = pack_ranked_chunks(
        ranked_chunks=chunks,
        tokenizer=tokenizer,
        budget=2,
        version="V2",
        token_cache={},
        separator_tokens=0,
    )

    assert [item.chunk.chunk_id for item in packed.selected] == ["small"]
    assert packed.text == "four"
    assert packed.token_count == 1
    assert packed.oversized_candidates == 1
    assert packed.skipped_candidates == 1


def test_later_smaller_chunks_are_still_considered():
    tokenizer = FakeTokenizer()

    chunks = [
        make_ranked(make_chunk("c1", "one two"), 1),
        make_ranked(make_chunk("c2", "three four five"), 2),
        make_ranked(make_chunk("c3", "six"), 3),
    ]

    packed = pack_ranked_chunks(
        ranked_chunks=chunks,
        tokenizer=tokenizer,
        budget=3,
        version="V2",
        token_cache={},
        separator_tokens=0,
    )

    assert [item.chunk.chunk_id for item in packed.selected] == [
        "c1",
        "c3",
    ]


def test_exact_combined_token_count_is_verified():
    tokenizer = BoundaryTokenizer()

    chunks = [
        make_ranked(make_chunk("a", "A"), 1),
        make_ranked(make_chunk("b", "B"), 2),
    ]

    packed = pack_ranked_chunks(
        ranked_chunks=chunks,
        tokenizer=tokenizer,
        budget=3,
        version="V2",
        token_cache={},
        separator_tokens=1,
    )

    assert [item.chunk.chunk_id for item in packed.selected] == ["a"]
    assert packed.text == "A"
    assert packed.token_count == 1
    assert packed.token_count <= packed.budget


def test_empty_candidate_list():
    packed = make_packed(chunks=[], budget=10)

    assert packed.selected == ()
    assert packed.text == ""
    assert packed.token_count == 0
    assert packed.candidate_count == 0
    assert packed.skipped_candidates == 0


def test_invalid_token_budget():
    with pytest.raises(ValueError, match="positive"):
        make_packed(chunks=[], budget=0)


def test_token_cache_is_reused():
    tokenizer = FakeTokenizer()
    cache = {}

    chunk = make_chunk("c1", "one two")
    ranked = [make_ranked(chunk, 1)]

    first = pack_ranked_chunks(
        ranked,
        tokenizer,
        budget=10,
        version="V2",
        token_cache=cache,
        separator_tokens=0,
    )

    assert cache[("V2", "c1")] == 2

    second = pack_ranked_chunks(
        ranked,
        tokenizer,
        budget=10,
        version="V2",
        token_cache=cache,
        separator_tokens=0,
    )

    assert first.token_count == second.token_count
    assert len(cache) == 1


# ============================================================
# Evidence coverage
# ============================================================


def test_single_chunk_full_evidence_coverage():
    chunk = make_sentence_chunk("sentence-1", "abcdefghij")
    packed = make_packed([chunk])

    gold = EvidenceLocation(
        section_index=0,
        paragraph_index=0,
        start_char=2,
        end_char=8,
    )

    result = evaluate_packed_context(
        packed=packed,
        locations=[gold],
        thresholds=(0.25, 0.5, 0.75, 1.0),
    )

    assert result["best_single_evidence_coverage"] == 1.0
    assert result["best_union_evidence_coverage"] == 1.0
    assert result["single_ranks"]["0.5"] == 1
    assert result["union_ranks"]["0.5"] == 1


def test_union_combines_multiple_complete_chunks():
    chunk_a = make_chunk("a", "abcde", start_char=0)
    chunk_b = make_chunk("b", "fghij", start_char=5)

    packed = make_packed([chunk_a, chunk_b])

    gold = EvidenceLocation(
        section_index=0,
        paragraph_index=0,
        start_char=0,
        end_char=10,
    )

    result = evaluate_packed_context(
        packed=packed,
        locations=[gold],
        thresholds=(0.5, 1.0),
    )

    assert result["best_single_evidence_coverage"] == 0.5
    assert result["best_union_evidence_coverage"] == 1.0
    assert result["single_ranks"]["1.0"] is None
    assert result["union_ranks"]["1.0"] == 2


def test_empty_context_has_zero_evidence_coverage():
    packed = make_packed(chunks=[])

    gold = EvidenceLocation(
        section_index=0,
        paragraph_index=0,
        start_char=0,
        end_char=10,
    )

    result = evaluate_packed_context(
        packed=packed,
        locations=[gold],
        thresholds=(0.5,),
    )

    assert result["context_tokens"] == 0
    assert result["selected_chunk_count"] == 0
    assert result["best_single_evidence_coverage"] == 0.0
    assert result["best_union_evidence_coverage"] == 0.0
    assert result["single_ranks"]["0.5"] is None
    assert result["union_ranks"]["0.5"] is None


def test_oversize_sentence_statistics():
    chunk = make_sentence_chunk(
        "long-sentence",
        "one",
        oversize_sentence=True,
    )

    packed = make_packed([chunk], budget=10)

    assert packed.selected_oversize_sentences == 1


def test_invalid_empty_evidence_locations():
    packed = make_packed(chunks=[])

    with pytest.raises(ValueError, match="cannot be empty"):
        evaluate_packed_context(
            packed=packed,
            locations=[],
            thresholds=(0.5,),
        )


# ============================================================
# Summary metrics
# ============================================================


def test_summarize_budget_results():
    chunk = make_sentence_chunk("sentence-1", "abcdefghij")
    packed = make_packed([chunk], budget=10)

    gold = EvidenceLocation(
        section_index=0,
        paragraph_index=0,
        start_char=0,
        end_char=10,
    )

    result = evaluate_packed_context(
        packed=packed,
        locations=[gold],
        thresholds=(0.5,),
    )

    summary = summarize_budget_results(
        question_results=[result, result],
        thresholds=(0.5,),
    )

    assert summary["questions"] == 2
    assert summary["mean_context_tokens"] == 1
    assert summary["mean_selected_chunks"] == 1

    assert summary["retrieval"]["single"]["0.5"]["hit@1"] == 1.0
    assert summary["retrieval"]["union"]["0.5"]["hit@10"] == 1.0

    assert summary["retrieval"]["context_hit_rate"]["0.5"]["union"] == 1.0


def test_context_budget_utilization():
    chunk = make_sentence_chunk("sentence-1", "one two")
    packed = make_packed([chunk], budget=10)

    gold = EvidenceLocation(
        section_index=0,
        paragraph_index=0,
        start_char=0,
        end_char=7,
    )

    result = evaluate_packed_context(
        packed=packed,
        locations=[gold],
        thresholds=(0.5,),
    )

    assert result["context_tokens"] == 2
    assert result["budget_utilization"] == pytest.approx(0.2)
