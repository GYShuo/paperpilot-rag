import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from paper_rag.evaluation.retrieval_evaluator import EvidenceLocation
from paper_rag.retrieval.bm25_retriever import BM25Retriever
from paper_rag.schemas import ChunkSpan, SentenceTokenChunk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "evaluate_hybrid_retrieval.py"

SPEC = importlib.util.spec_from_file_location(
    "_day6_hybrid_evaluation",
    SCRIPT_PATH,
)

if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load Hybrid evaluation script: {SCRIPT_PATH}")

MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_chunk(chunk_id: str, text: str) -> SentenceTokenChunk:
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
    )


class FakeTokenizer:
    def __call__(
        self,
        text,
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    ):
        return {"input_ids": text.split()}


def make_question(question: str = "neural network") -> dict:
    return {
        "paper_id": "paper-1",
        "question_id": "q1",
        "question": question,
        "locations": [
            EvidenceLocation(
                section_index=0,
                paragraph_index=0,
                start_char=0,
                end_char=6,
            )
        ],
    }


def make_baseline() -> dict:
    return {
        "experiment": "day6_bm25_v2_baseline",
        "split": "validation",
        "retrieval_method": "bm25_v2",
        "retrieval_scope": "within_paper",
        "bm25": {
            "k1": 1.5,
            "b": 0.75,
            "idf_scope": "global_validation_parent_chunks",
        },
        "context_budget_evaluation": {
            "budgets": [1024, 2048, 4096],
            "thresholds": [0.25, 0.5, 0.75],
            "tokenizer": "sentence-transformers/all-MiniLM-L6-v2",
            "tokenizer_is_proxy": True,
            "special_tokens": False,
            "truncation": False,
            "context_separator": "\n\n",
            "packing_policy": "same policy",
        },
        "question_statistics": {
            "final_evaluable_questions": 1,
        },
        "per_question": [
            {
                "paper_id": "paper-1",
                "question_id": "q1",
                "question": "neural network",
            }
        ],
    }


def test_load_bm25_baseline(tmp_path):
    path = tmp_path / "bm25.json"
    path.write_text(json.dumps(make_baseline()), encoding="utf-8")

    result = MODULE.load_bm25_baseline(path)

    assert result["bm25"]["k1"] == 1.5
    assert result["bm25"]["b"] == 0.75


def test_load_bm25_baseline_rejects_wrong_method(tmp_path):
    baseline = make_baseline()
    baseline["retrieval_method"] = "dense_v2"

    path = tmp_path / "bm25.json"
    path.write_text(json.dumps(baseline), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid BM25 baseline metadata"):
        MODULE.load_bm25_baseline(path)


def test_validate_baseline_compatibility():
    bm25 = make_baseline()

    dense = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "context_budget_evaluation": bm25["context_budget_evaluation"],
        "per_question": bm25["per_question"],
    }

    MODULE.validate_baseline_compatibility(dense, bm25)


def test_validate_baseline_compatibility_rejects_budget_mismatch():
    bm25 = make_baseline()

    dense = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "context_budget_evaluation": {
            **bm25["context_budget_evaluation"],
            "budgets": [512, 1024],
        },
        "per_question": bm25["per_question"],
    }

    with pytest.raises(ValueError, match="budgets differ"):
        MODULE.validate_baseline_compatibility(dense, bm25)


def test_evaluate_hybrid_reuses_shared_evaluation(monkeypatch):
    chunk_a = make_chunk("a", "neural network model")
    chunk_b = make_chunk("b", "graph convolution model")

    dense_results = [
        MODULE.DENSE.RankedChunk(
            rank=1,
            score=0.9,
            chunk=chunk_a,
        ),
        MODULE.DENSE.RankedChunk(
            rank=2,
            score=0.5,
            chunk=chunk_b,
        ),
    ]

    monkeypatch.setattr(
        MODULE.DENSE,
        "rank_windowed_chunks",
        lambda query_vector, index: dense_results,
    )

    bm25 = BM25Retriever()
    bm25.build_index([chunk_a, chunk_b])

    results = MODULE.evaluate_hybrid(
        questions=[make_question()],
        query_vectors=np.array([[0.1, 0.2]], dtype=np.float32),
        indexes={"paper-1": {}},
        bm25_retriever=bm25,
        hybrid_retriever=MODULE.HybridRetriever(rrf_k=60),
        tokenizer=FakeTokenizer(),
        budgets=(4,),
        thresholds=(0.5,),
        separator_tokens=0,
    )

    record = results["per_question"][0]

    assert record["dense_candidate_count"] == 2
    assert record["bm25_candidate_count"] == 1
    assert record["hybrid_candidate_count"] == 2

    assert record["top_10"][0]["chunk_id"] == "a"

    assert record["context_by_budget"]["4"]["selected_chunk_ids"] == ["a"]

    assert results["questions_without_lexical_match"] == 0


def test_hybrid_keeps_dense_candidates_without_bm25_match(monkeypatch):
    chunk = make_chunk("a", "neural network model")

    dense_results = [
        MODULE.DENSE.RankedChunk(
            rank=1,
            score=0.9,
            chunk=chunk,
        )
    ]

    monkeypatch.setattr(
        MODULE.DENSE,
        "rank_windowed_chunks",
        lambda query_vector, index: dense_results,
    )

    bm25 = BM25Retriever()
    bm25.build_index([chunk])

    results = MODULE.evaluate_hybrid(
        questions=[make_question("quantum physics")],
        query_vectors=np.array([[0.1, 0.2]], dtype=np.float32),
        indexes={"paper-1": {}},
        bm25_retriever=bm25,
        hybrid_retriever=MODULE.HybridRetriever(rrf_k=60),
        tokenizer=FakeTokenizer(),
        budgets=(4,),
        thresholds=(0.5,),
        separator_tokens=0,
    )

    record = results["per_question"][0]

    assert record["bm25_candidate_count"] == 0
    assert record["hybrid_candidate_count"] == 1
    assert record["top_10"][0]["bm25_rank"] is None
    assert results["questions_without_lexical_match"] == 1


def test_evaluate_hybrid_rejects_query_count_mismatch():
    with pytest.raises(ValueError, match="do not match"):
        MODULE.evaluate_hybrid(
            questions=[make_question()],
            query_vectors=np.empty((0, 2), dtype=np.float32),
            indexes={},
            bm25_retriever=BM25Retriever(),
            hybrid_retriever=MODULE.HybridRetriever(),
            tokenizer=FakeTokenizer(),
            budgets=(4,),
            thresholds=(0.5,),
            separator_tokens=0,
        )


def test_output_protection(tmp_path):
    output_path = tmp_path / "result.json"

    MODULE.ensure_output_available(output_path)

    output_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        MODULE.ensure_output_available(output_path)


def test_validate_index_compatibility():
    manifest = {"num_chunks": 10}
    build_config = {"chunk_size": 500}

    dense = {
        "v2_index_manifest": manifest,
        "v2_index_build_config": build_config,
    }

    bm25 = {
        "v2_index_manifest": manifest,
        "v2_index_build_config": build_config,
    }

    MODULE.validate_index_compatibility(
        dense,
        bm25,
        manifest,
        build_config,
    )


def test_validate_bm25_index_statistics():
    chunk = make_chunk("a", "neural network model")

    retriever = BM25Retriever()
    retriever.build_index([chunk])

    baseline = {
        "bm25_index_statistics": {
            "num_documents": 1,
            "vocabulary_size": 3,
            "average_document_length": 3.0,
        }
    }

    MODULE.validate_bm25_index_statistics(baseline, retriever)
