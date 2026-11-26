"""Tests for Day 7 RRF search with optimized BM25 rankings."""

import importlib.util
import sys
from pathlib import Path

import pytest

from paper_rag.evaluation.retrieval_evaluator import EvidenceLocation
from paper_rag.retrieval.bm25_retriever import BM25Retriever
from paper_rag.schemas import ChunkSpan, SentenceTokenChunk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "search_hybrid_rrf_params.py"

SPEC = importlib.util.spec_from_file_location(
    "_day7_optimized_bm25_rrf_search",
    SCRIPT_PATH,
)

if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load RRF search script: {SCRIPT_PATH}")

MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_chunk(
    chunk_id: str,
    text: str,
    paragraph_index: int,
) -> SentenceTokenChunk:
    """Create a V2 Parent Chunk for testing."""

    return SentenceTokenChunk(
        chunk_id=chunk_id,
        paper_id="paper-1",
        split="validation",
        section_index=0,
        section_name="Introduction",
        text=text,
        spans=[
            ChunkSpan(
                paragraph_index=paragraph_index,
                start_char=0,
                end_char=len(text),
                chunk_start=0,
                chunk_end=len(text),
            )
        ],
        token_count=len(text.split()),
        sentence_count=1,
        oversize_sentence=False,
    )


def make_dense_fixture():
    """Build frozen Dense and Day 6 Hybrid reference records."""

    chunk_a = make_chunk(
        "chunk-a",
        "neural network evidence",
        paragraph_index=0,
    )

    chunk_b = make_chunk(
        "chunk-b",
        "graph convolution model",
        paragraph_index=1,
    )

    location = EvidenceLocation(
        section_index=0,
        paragraph_index=0,
        start_char=0,
        end_char=7,
    )

    question = {
        "paper_id": "paper-1",
        "question_id": "q1",
        "question": "What is neural network?",
        "locations": [location],
    }

    dense_results = [
        MODULE.DENSE.RankedChunk(
            rank=1,
            score=0.9,
            chunk=chunk_b,
        ),
        MODULE.DENSE.RankedChunk(
            rank=2,
            score=0.8,
            chunk=chunk_a,
        ),
    ]

    thresholds = (0.5,)

    raw_ranking = MODULE.DENSE.evaluate_ranked_chunks(
        ranked_chunks=dense_results,
        locations=question["locations"],
        thresholds=thresholds,
    )

    dense_record = {
        "paper_id": "paper-1",
        "question_id": "q1",
        "question": question["question"],
        "top_10": [
            {
                "rank": item.rank,
                "chunk_id": item.chunk.chunk_id,
                "score": item.score,
            }
            for item in dense_results
        ],
        "raw_ranking": raw_ranking,
    }

    # Deliberately provide old BM25 ranks.
    # The new search must ignore them and recompute BM25.
    hybrid_record = {
        "paper_id": "paper-1",
        "question_id": "q1",
        "question": question["question"],
        "dense_candidate_count": 2,
        "hybrid_candidate_count": 2,
        "ranked_candidates": [
            {
                "rank": 1,
                "chunk_id": "chunk-a",
                "dense_rank": 2,
                "dense_score": 0.8,
                "bm25_rank": 1,
                "bm25_score": 9.0,
            },
            {
                "rank": 2,
                "chunk_id": "chunk-b",
                "dense_rank": 1,
                "dense_score": 0.9,
                "bm25_rank": 2,
                "bm25_score": 2.0,
            },
        ],
    }

    indexes = {
        "paper-1": {
            "chunks_by_id": {
                "chunk-a": chunk_a,
                "chunk-b": chunk_b,
            }
        }
    }

    return (
        question,
        dense_record,
        hybrid_record,
        indexes,
        thresholds,
        chunk_a,
        chunk_b,
    )


def test_parameter_grid_contains_48_configurations():
    """The original RRF parameter grid must remain unchanged."""

    configs = MODULE.build_parameter_grid()

    assert len(configs) == 48

    assert {item.rrf_k for item in configs} == {
        10,
        20,
        40,
        60,
        100,
        200,
    }

    assert {item.dense_weight for item in configs} == {1.0}

    assert {item.bm25_weight for item in configs} == {
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        2.0,
        3.0,
    }


def test_config_id():
    """Configuration IDs must be stable."""

    config = MODULE.FusionConfig(
        rrf_k=100,
        dense_weight=1.0,
        bm25_weight=0.75,
    )

    assert MODULE.config_id(config) == "k100_wb0p75"


def test_calculate_objective():
    """The RRF objective must average all evidence thresholds."""

    summary = {
        "union": {
            "0.25": {
                "mrr@10": 0.6,
                "hit@1": 0.4,
            },
            "0.5": {
                "mrr@10": 0.5,
                "hit@1": 0.3,
            },
            "0.75": {
                "mrr@10": 0.4,
                "hit@1": 0.2,
            },
        }
    }

    result = MODULE.calculate_objective(
        summary=summary,
        thresholds=(0.25, 0.5, 0.75),
    )

    assert result["mean_union_mrr_at_10"] == pytest.approx(0.5)
    assert result["mean_union_hit_at_1"] == pytest.approx(0.3)
    assert result["primary_threshold"] == 0.5
    assert result["primary_union_mrr_at_10"] == pytest.approx(0.5)


def test_prepare_question_records_restores_dense_ranking():
    """Dense ranking must be restored without using old BM25 ranks."""

    (
        question,
        dense_record,
        hybrid_record,
        indexes,
        thresholds,
        chunk_a,
        chunk_b,
    ) = make_dense_fixture()

    prepared = MODULE.prepare_question_records(
        questions=[question],
        dense_records=[dense_record],
        hybrid_records=[hybrid_record],
        indexes=indexes,
        thresholds=thresholds,
    )

    assert len(prepared) == 1

    dense_results = prepared[0]["dense_results"]

    assert [item.chunk.chunk_id for item in dense_results] == [
        "chunk-b",
        "chunk-a",
    ]

    assert [item.rank for item in dense_results] == [1, 2]

    assert dense_results[0].chunk == chunk_b
    assert dense_results[1].chunk == chunk_a

    # BM25 is attached only after fresh retrieval.
    assert "bm25_results" not in prepared[0]


def test_attach_optimized_bm25_rankings():
    """New BM25 rankings must match the optimized BM25 evaluation."""

    (
        question,
        dense_record,
        hybrid_record,
        indexes,
        thresholds,
        chunk_a,
        chunk_b,
    ) = make_dense_fixture()

    prepared = MODULE.prepare_question_records(
        questions=[question],
        dense_records=[dense_record],
        hybrid_records=[hybrid_record],
        indexes=indexes,
        thresholds=thresholds,
    )

    retriever = BM25Retriever(
        k1=2.25,
        b=0.10,
    )

    retriever.build_index([chunk_a, chunk_b])

    bm25_results = MODULE.BM25_EVALUATION.rank_bm25_question(
        retriever=retriever,
        question=question["question"],
        paper_id=question["paper_id"],
    )

    assert [item.chunk.chunk_id for item in bm25_results] == ["chunk-a"]

    raw_ranking = MODULE.DENSE.evaluate_ranked_chunks(
        ranked_chunks=bm25_results,
        locations=question["locations"],
        thresholds=thresholds,
    )

    optimized_record = {
        "paper_id": question["paper_id"],
        "question_id": question["question_id"],
        "question": question["question"],
        "matched_chunk_count": len(bm25_results),
        "top_10": [
            {
                "rank": item.rank,
                "chunk_id": item.chunk.chunk_id,
                "score": item.score,
            }
            for item in bm25_results
        ],
        "raw_ranking": raw_ranking,
    }

    MODULE.attach_optimized_bm25_rankings(
        prepared=prepared,
        retriever=retriever,
        optimized_records=[optimized_record],
        thresholds=thresholds,
    )

    assert [item.chunk.chunk_id for item in prepared[0]["bm25_results"]] == ["chunk-a"]

    assert prepared[0]["bm25_results"][0].rank == 1


def test_attach_rejects_mismatched_optimized_bm25_result():
    """BM25 rankings must not silently use a different question."""

    (
        question,
        dense_record,
        hybrid_record,
        indexes,
        thresholds,
        chunk_a,
        chunk_b,
    ) = make_dense_fixture()

    prepared = MODULE.prepare_question_records(
        questions=[question],
        dense_records=[dense_record],
        hybrid_records=[hybrid_record],
        indexes=indexes,
        thresholds=thresholds,
    )

    retriever = BM25Retriever(k1=2.25, b=0.10)
    retriever.build_index([chunk_a, chunk_b])

    wrong_record = {
        "paper_id": "paper-1",
        "question_id": "different-question",
        "question": question["question"],
        "matched_chunk_count": 0,
        "top_10": [],
        "raw_ranking": {},
    }

    with pytest.raises(
        ValueError,
        match="question alignment mismatch",
    ):
        MODULE.attach_optimized_bm25_rankings(
            prepared=prepared,
            retriever=retriever,
            optimized_records=[wrong_record],
            thresholds=thresholds,
        )


def test_output_protection(tmp_path):
    """Existing experiment results must never be overwritten."""

    output_path = tmp_path / "result.json"
    output_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError):
        MODULE.save_json_exclusive(
            output_path,
            {"new": True},
        )
