import importlib.util
import sys
from pathlib import Path

import pytest

from paper_rag.schemas import ChunkSpan, SentenceTokenChunk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "search_hybrid_rrf_params.py"

SPEC = importlib.util.spec_from_file_location(
    "_day7_rrf_grid_search",
    SCRIPT_PATH,
)

if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load RRF grid search script: {SCRIPT_PATH}")

MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_chunk(chunk_id: str) -> SentenceTokenChunk:
    source_start = {"a": 0, "b": 8, "c": 16}[chunk_id]
    text = "evidence"

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
                start_char=source_start,
                end_char=source_start + len(text),
                chunk_start=0,
                chunk_end=len(text),
            )
        ],
        token_count=3,
        sentence_count=1,
    )


def make_candidate(
    chunk_id: str,
    rank: int,
    dense_rank: int,
    bm25_rank: int,
    dense_score: float,
    bm25_score: float,
) -> dict:
    dense_contribution = 1 / (60 + dense_rank)
    bm25_contribution = 1 / (60 + bm25_rank)

    return {
        "rank": rank,
        "chunk_id": chunk_id,
        "rrf_score": dense_contribution + bm25_contribution,
        "dense_rank": dense_rank,
        "bm25_rank": bm25_rank,
        "dense_score": dense_score,
        "bm25_score": bm25_score,
        "dense_contribution": dense_contribution,
        "bm25_contribution": bm25_contribution,
    }


def make_top_10(ranked_chunks: list) -> list[dict]:
    return [
        {
            "rank": item.rank,
            "chunk_id": item.chunk.chunk_id,
            "score": item.score,
        }
        for item in ranked_chunks[: MODULE.TOP_K]
    ]


def make_raw_ranking(question: dict, ranked_chunks: list) -> dict:
    return MODULE.DENSE.evaluate_ranked_chunks(
        ranked_chunks=ranked_chunks,
        locations=question["locations"],
        thresholds=(0.5,),
    )


def make_question() -> dict:
    return {
        "paper_id": "paper-1",
        "question_id": "q1",
        "question": "neural network",
        "locations": [
            MODULE.DENSE.EvidenceLocation(
                section_index=0,
                paragraph_index=0,
                start_char=0,
                end_char=8,
            )
        ],
    }


def test_parameter_grid_contains_48_configurations():
    configs = MODULE.build_parameter_grid()

    assert len(configs) == 48
    assert len({MODULE.config_id(item) for item in configs}) == 48

    assert (
        MODULE.FusionConfig(
            rrf_k=60,
            dense_weight=1.0,
            bm25_weight=1.0,
        )
        in configs
    )


def test_config_id_is_stable():
    config = MODULE.FusionConfig(
        rrf_k=60,
        dense_weight=1.0,
        bm25_weight=0.75,
    )

    assert MODULE.config_id(config) == "k60_wb0p75"


def test_bm25_weight_changes_fusion_ranking():
    chunk_a = make_chunk("a")
    chunk_b = make_chunk("b")

    dense = [
        MODULE.DENSE.RankedChunk(1, 0.9, chunk_a),
        MODULE.DENSE.RankedChunk(2, 0.8, chunk_b),
    ]

    bm25 = [
        MODULE.DENSE.RankedChunk(1, 2.0, chunk_b),
        MODULE.DENSE.RankedChunk(2, 1.0, chunk_a),
    ]

    low_weight = MODULE.FusionConfig(
        rrf_k=10,
        dense_weight=1.0,
        bm25_weight=0.5,
    )

    high_weight = MODULE.FusionConfig(
        rrf_k=10,
        dense_weight=1.0,
        bm25_weight=2.0,
    )

    low_results = MODULE.fuse_rankings(dense, bm25, low_weight)
    high_results = MODULE.fuse_rankings(dense, bm25, high_weight)

    assert [item.chunk.chunk_id for item in low_results] == ["a", "b"]
    assert [item.chunk.chunk_id for item in high_results] == ["b", "a"]


def test_cached_dense_and_optimized_bm25_rankings_are_reconstructed(
    monkeypatch,
):
    chunks = {chunk_id: make_chunk(chunk_id) for chunk_id in ("a", "b", "c")}
    question = make_question()

    # Day 6 Hybrid records retain the complete candidate set and original
    # Dense ranks/scores. Day 7 deliberately rebuilds BM25 with optimized
    # parameters instead of reusing the old BM25 ranks.
    candidates = [
        make_candidate("b", 1, 2, 1, 0.8, 2.0),
        make_candidate("a", 2, 1, 3, 0.9, 1.0),
        make_candidate("c", 3, 3, 2, 0.7, 1.5),
    ]

    dense_results = [
        MODULE.DENSE.RankedChunk(1, 0.9, chunks["a"]),
        MODULE.DENSE.RankedChunk(2, 0.8, chunks["b"]),
        MODULE.DENSE.RankedChunk(3, 0.7, chunks["c"]),
    ]

    dense_record = {
        "paper_id": "paper-1",
        "question_id": "q1",
        "question": "neural network",
        "top_10": make_top_10(dense_results),
        "raw_ranking": make_raw_ranking(question, dense_results),
    }

    hybrid_record = {
        "paper_id": "paper-1",
        "question_id": "q1",
        "question": "neural network",
        "dense_candidate_count": 3,
        "hybrid_candidate_count": 3,
        "ranked_candidates": candidates,
    }

    prepared = MODULE.prepare_question_records(
        questions=[question],
        dense_records=[dense_record],
        hybrid_records=[hybrid_record],
        indexes={"paper-1": {"chunks_by_id": chunks}},
        thresholds=(0.5,),
    )

    assert len(prepared) == 1
    assert [result.chunk.chunk_id for result in prepared[0]["dense_results"]] == [
        "a",
        "b",
        "c",
    ]

    # The optimized BM25 ranking intentionally differs from the old cached
    # BM25 order stored in the Day 6 Hybrid candidate records.
    optimized_bm25_results = [
        MODULE.DENSE.RankedChunk(1, 2.0, chunks["c"]),
        MODULE.DENSE.RankedChunk(2, 1.5, chunks["a"]),
        MODULE.DENSE.RankedChunk(3, 1.0, chunks["b"]),
    ]

    optimized_bm25_record = {
        "paper_id": "paper-1",
        "question_id": "q1",
        "question": "neural network",
        "matched_chunk_count": len(optimized_bm25_results),
        "top_10": make_top_10(optimized_bm25_results),
        "raw_ranking": make_raw_ranking(question, optimized_bm25_results),
    }

    def fake_rank_bm25_question(*, retriever, question, paper_id):
        assert retriever is fake_retriever
        assert question == "neural network"
        assert paper_id == "paper-1"
        return optimized_bm25_results

    fake_retriever = object()

    monkeypatch.setattr(
        MODULE.BM25_EVALUATION,
        "rank_bm25_question",
        fake_rank_bm25_question,
    )

    MODULE.attach_optimized_bm25_rankings(
        prepared=prepared,
        retriever=fake_retriever,
        optimized_records=[optimized_bm25_record],
        thresholds=(0.5,),
    )

    assert [result.chunk.chunk_id for result in prepared[0]["bm25_results"]] == [
        "c",
        "a",
        "b",
    ]

    assert prepared[0]["dense_results"][0].chunk.chunk_id == "a"
    assert prepared[0]["bm25_results"][0].chunk.chunk_id == "c"


def test_calculate_objective_averages_all_thresholds():
    summary = {
        "union": {
            "0.25": {
                "hit@1": 0.4,
                "mrr@10": 0.6,
            },
            "0.5": {
                "hit@1": 0.3,
                "mrr@10": 0.5,
            },
            "0.75": {
                "hit@1": 0.2,
                "mrr@10": 0.4,
            },
        }
    }

    result = MODULE.calculate_objective(
        summary,
        thresholds=(0.25, 0.5, 0.75),
    )

    assert result["mean_union_mrr_at_10"] == pytest.approx(0.5)
    assert result["mean_union_hit_at_1"] == pytest.approx(0.3)
    assert result["primary_threshold"] == 0.5
    assert result["primary_union_mrr_at_10"] == pytest.approx(0.5)


def test_nested_metric_comparison_accepts_small_float_error():
    actual = {
        "ranking": {
            "hit@1": 0.5,
            "mrr@10": 0.60000000001,
        }
    }

    expected = {
        "ranking": {
            "hit@1": 0.5,
            "mrr@10": 0.6,
        }
    }

    MODULE.assert_nested_close(actual, expected)


def test_nested_metric_comparison_rejects_mismatch():
    actual = {"hit@1": 0.6}
    expected = {"hit@1": 0.5}

    with pytest.raises(ValueError, match="Metric mismatch"):
        MODULE.assert_nested_close(actual, expected)
