import importlib.util
import sys
from pathlib import Path

import pytest

from paper_rag.evaluation.retrieval_evaluator import (
    EvidenceLocation,
    chunk_evidence_coverage,
)
from paper_rag.schemas import (
    ChunkSpan,
    SentenceTokenChunk,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "evaluate_retrieval_token_budget.py"

SPEC = importlib.util.spec_from_file_location(
    "_day6_dense_baseline",
    SCRIPT_PATH,
)

if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load experiment script: {SCRIPT_PATH}")

SCRIPT_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SCRIPT_MODULE
SPEC.loader.exec_module(SCRIPT_MODULE)

RankedChunk = SCRIPT_MODULE.RankedChunk
evaluate_ranked_chunks = SCRIPT_MODULE.evaluate_ranked_chunks
summarize_ranking_results = SCRIPT_MODULE.summarize_ranking_results


def make_sentence_chunk(
    chunk_id: str,
    start_char: int,
    end_char: int,
) -> SentenceTokenChunk:
    """Create a V2 parent chunk mapped to an original paragraph."""

    text = "x" * (end_char - start_char)

    return SentenceTokenChunk(
        chunk_id=chunk_id,
        paper_id="paper-1",
        split="validation",
        section_index=0,
        section_name="Introduction",
        text=text,
        spans=[
            ChunkSpan(
                paragraph_index=0,
                start_char=start_char,
                end_char=end_char,
                chunk_start=0,
                chunk_end=len(text),
            )
        ],
        token_count=10,
        sentence_count=1,
        oversize_sentence=False,
    )


def test_v2_sentence_chunk_evidence_coverage():
    chunk = make_sentence_chunk(
        "chunk-1",
        0,
        50,
    )

    location = EvidenceLocation(
        section_index=0,
        paragraph_index=0,
        start_char=20,
        end_char=80,
    )

    assert chunk_evidence_coverage(
        chunk,
        location,
    ) == pytest.approx(0.5)


def test_raw_ranking_single_and_union():
    """UNION should be able to combine evidence across parent chunks."""

    location = EvidenceLocation(
        section_index=0,
        paragraph_index=0,
        start_char=0,
        end_char=100,
    )

    chunks = [
        make_sentence_chunk("chunk-1", 0, 40),
        make_sentence_chunk("chunk-2", 40, 80),
        make_sentence_chunk("chunk-3", 0, 100),
    ]

    ranked_chunks = [
        RankedChunk(
            rank=rank,
            score=1.0 / rank,
            chunk=chunk,
        )
        for rank, chunk in enumerate(chunks, start=1)
    ]

    result = evaluate_ranked_chunks(
        ranked_chunks=ranked_chunks,
        locations=[location],
        thresholds=(0.25, 0.5, 0.75),
    )

    assert result["single_ranks"] == {
        "0.25": 1,
        "0.5": 3,
        "0.75": 3,
    }

    assert result["union_ranks"] == {
        "0.25": 1,
        "0.5": 2,
        "0.75": 2,
    }


def test_raw_ranking_without_evidence():
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluate_ranked_chunks(
            ranked_chunks=[],
            locations=[],
            thresholds=(0.5,),
        )


def test_summarize_raw_ranking_results():
    question_results = [
        {
            "single_ranks": {"0.5": 3},
            "union_ranks": {"0.5": 2},
        },
        {
            "single_ranks": {"0.5": None},
            "union_ranks": {"0.5": None},
        },
    ]

    summary = summarize_ranking_results(
        question_results=question_results,
        thresholds=(0.5,),
    )

    assert summary["single"]["0.5"]["hit@1"] == 0.0
    assert summary["single"]["0.5"]["hit@5"] == 0.5

    assert summary["single"]["0.5"]["mrr@10"] == pytest.approx(1 / 6)

    assert summary["union"]["0.5"]["hit@5"] == 0.5

    assert summary["union"]["0.5"]["mrr@10"] == pytest.approx(1 / 4)
