"""Tests for optimized retrieval context-budget evaluation."""

import importlib.util
import sys
from pathlib import Path

import pytest

from paper_rag.evaluation.retrieval_evaluator import EvidenceLocation
from paper_rag.schemas import ChunkSpan, SentenceTokenChunk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "evaluate_hybrid_context_budget.py"
SPEC = importlib.util.spec_from_file_location("_hybrid_context_budget", SCRIPT_PATH)

if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load context-budget script: {SCRIPT_PATH}")

MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


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


def make_chunk(chunk_id: str, text: str):
    """Create a V2 parent chunk with a valid source span."""

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


def make_question():
    """Create one question with a body-evidence location."""

    return {
        "paper_id": "paper-1",
        "question_id": "q1",
        "question": "What is neural evidence?",
        "locations": [
            EvidenceLocation(
                section_index=0,
                paragraph_index=0,
                start_char=0,
                end_char=6,
            )
        ],
    }


def test_selected_hybrid_configurations():
    assert [item.rrf_k for item in MODULE.HYBRID_CONFIGS] == [60, 100, 10, 20]
    assert [item.dense_weight for item in MODULE.HYBRID_CONFIGS] == [1.0] * 4
    assert [item.bm25_weight for item in MODULE.HYBRID_CONFIGS] == [
        1.0,
        1.0,
        0.75,
        1.0,
    ]


def test_same_hybrid_config():
    config = MODULE.HYBRID_CONFIGS[0]
    assert MODULE.same_hybrid_config(
        {
            "rrf_k": 60,
            "dense_weight": 1.0,
            "bm25_weight": 1.0,
        },
        config,
    )
    assert not MODULE.same_hybrid_config(
        {
            "rrf_k": 60,
            "dense_weight": 1.0,
            "bm25_weight": 0.75,
        },
        config,
    )


def test_build_method_rankings_preserves_dense_candidate_set():
    chunk_a = make_chunk("chunk-a", "neural evidence")
    chunk_b = make_chunk("chunk-b", "graph model")
    dense_results = [
        MODULE.DENSE.RankedChunk(rank=1, score=0.9, chunk=chunk_a),
        MODULE.DENSE.RankedChunk(rank=2, score=0.8, chunk=chunk_b),
    ]
    bm25_results = [MODULE.DENSE.RankedChunk(rank=1, score=1.2, chunk=chunk_b)]
    prepared = [
        {
            "dense_results": dense_results,
            "bm25_results": bm25_results,
        }
    ]

    method_rankings = MODULE.build_method_rankings(prepared)

    assert len(method_rankings) == 6
    assert method_rankings["dense_v2"] == [dense_results]
    assert method_rankings["bm25_v2_optimized"] == [bm25_results]
    for config in MODULE.HYBRID_CONFIGS:
        fused = method_rankings[config.method_id][0]
        assert {item.chunk.chunk_id for item in fused} == {"chunk-a", "chunk-b"}
        assert [item.rank for item in fused] == [1, 2]


def test_evaluate_method_records_context_metrics():
    chunk = make_chunk("chunk-a", "neural evidence")
    question = make_question()
    ranked = [MODULE.DENSE.RankedChunk(rank=1, score=0.9, chunk=chunk)]

    result = MODULE.evaluate_method(
        method_id="dense_v2",
        questions=[question],
        ranked_lists=[ranked],
        tokenizer=FakeTokenizer(),
        budgets=(10,),
        thresholds=(0.5,),
        token_cache={},
        separator_tokens=0,
    )

    budget_result = result["budgets"]["10"]
    assert budget_result["mean_context_tokens"] == 2
    assert budget_result["mean_selected_chunks"] == 1
    assert budget_result["retrieval"]["context_hit_rate"]["0.5"]["union"] == 1.0
    context = result["per_question"][0]["context_by_budget"]["10"]
    assert context["selected_chunk_ids"] == ["chunk-a"]


def test_evaluate_method_handles_no_lexical_candidates():
    result = MODULE.evaluate_method(
        method_id="bm25_v2_optimized",
        questions=[make_question()],
        ranked_lists=[[]],
        tokenizer=FakeTokenizer(),
        budgets=(10,),
        thresholds=(0.5,),
        token_cache={},
        separator_tokens=0,
    )

    budget_result = result["budgets"]["10"]
    assert budget_result["mean_context_tokens"] == 0
    assert budget_result["mean_selected_chunks"] == 0
    assert budget_result["retrieval"]["context_hit_rate"]["0.5"]["union"] == 0.0
    assert result["candidate_statistics"]["questions_without_candidates"] == 1


def test_output_protection(tmp_path):
    output_path = tmp_path / "result.json"
    output_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        MODULE.ensure_output_available(output_path)
