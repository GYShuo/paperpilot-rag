import importlib.util
import sys
from pathlib import Path

import pytest

from paper_rag.evaluation.retrieval_evaluator import EvidenceLocation
from paper_rag.retrieval.bm25_retriever import BM25Retriever
from paper_rag.schemas import ChunkSpan, SentenceTokenChunk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "evaluate_bm25_retrieval.py"

SPEC = importlib.util.spec_from_file_location(
    "_day6_bm25_evaluation",
    SCRIPT_PATH,
)

if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load BM25 evaluation script: {SCRIPT_PATH}")

MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_chunk(
    chunk_id: str,
    text: str,
    paper_id: str = "paper-1",
    split: str = "validation",
) -> SentenceTokenChunk:
    """Create a V2 parent chunk with an original paragraph span."""

    return SentenceTokenChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        split=split,
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


class FakeTokenizer:
    """A deterministic whitespace tokenizer for unit tests."""

    def __call__(
        self,
        text,
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    ):
        return {
            "input_ids": text.split(),
        }


def test_collect_v2_parent_chunks():
    chunk_a = make_chunk("a", "neural network")
    chunk_b = make_chunk("b", "graph network", paper_id="paper-2")

    indexes = {
        "paper-1": {"chunks": [chunk_a]},
        "paper-2": {"chunks": [chunk_b]},
    }

    result = MODULE.collect_v2_parent_chunks(indexes)

    assert result == [chunk_a, chunk_b]


def test_collect_rejects_wrong_paper_id():
    chunk = make_chunk("a", "neural network")

    indexes = {
        "paper-2": {"chunks": [chunk]},
    }

    with pytest.raises(ValueError, match="Paper ID mismatch"):
        MODULE.collect_v2_parent_chunks(indexes)


def test_collect_rejects_wrong_split():
    chunk = make_chunk(
        "a",
        "neural network",
        split="train",
    )

    indexes = {
        "paper-1": {"chunks": [chunk]},
    }

    with pytest.raises(ValueError, match="Unexpected split"):
        MODULE.collect_v2_parent_chunks(indexes)


def test_question_alignment_passes():
    questions = [
        {
            "paper_id": "paper-1",
            "question_id": "q1",
            "question": "What is neural network?",
        },
        {
            "paper_id": "paper-2",
            "question_id": "q2",
            "question": "What is graph learning?",
        },
    ]

    dense_records = [
        {
            "paper_id": "paper-1",
            "question_id": "q1",
            "question": "What is neural network?",
        },
        {
            "paper_id": "paper-2",
            "question_id": "q2",
            "question": "What is graph learning?",
        },
    ]

    MODULE.validate_question_alignment(
        questions,
        dense_records,
    )


def test_question_alignment_rejects_different_order():
    questions = [
        {
            "paper_id": "paper-1",
            "question_id": "q1",
            "question": "Question 1",
        },
        {
            "paper_id": "paper-2",
            "question_id": "q2",
            "question": "Question 2",
        },
    ]

    dense_records = list(reversed(questions))

    with pytest.raises(ValueError, match="ordering differ"):
        MODULE.validate_question_alignment(
            questions,
            dense_records,
        )


def test_question_alignment_rejects_question_text_mismatch():
    questions = [
        {
            "paper_id": "paper-1",
            "question_id": "q1",
            "question": "Original question",
        }
    ]

    dense_records = [
        {
            "paper_id": "paper-1",
            "question_id": "q1",
            "question": "Modified question",
        }
    ]

    with pytest.raises(ValueError, match="Question text mismatch"):
        MODULE.validate_question_alignment(
            questions,
            dense_records,
        )


def test_question_alignment_rejects_duplicate_ids():
    question = {
        "paper_id": "paper-1",
        "question_id": "q1",
        "question": "Question",
    }

    with pytest.raises(ValueError, match="duplicate"):
        MODULE.validate_question_alignment(
            [question, question.copy()],
            [question],
        )


def test_rank_bm25_question():
    chunks = [
        make_chunk("a", "neural network model"),
        make_chunk("b", "graph convolution network"),
    ]

    retriever = BM25Retriever()
    retriever.build_index(chunks)

    results = MODULE.rank_bm25_question(
        retriever,
        "neural network",
        "paper-1",
    )

    assert results
    assert results[0].rank == 1
    assert results[0].chunk.chunk_id == "a"
    assert results[0].score > 0


def test_rank_bm25_question_without_matches():
    retriever = BM25Retriever()
    retriever.build_index(
        [
            make_chunk("a", "neural network model"),
        ]
    )

    results = MODULE.rank_bm25_question(
        retriever,
        "quantum physics",
        "paper-1",
    )

    assert results == []


def test_evaluate_bm25_context_budgets():
    chunk = make_chunk(
        "a",
        "neural network model",
    )

    retriever = BM25Retriever()
    retriever.build_index([chunk])

    questions = [
        {
            "paper_id": "paper-1",
            "question_id": "q1",
            "question": "neural network",
            "locations": [
                EvidenceLocation(
                    section_index=0,
                    paragraph_index=0,
                    start_char=0,
                    end_char=6,
                )
            ],
        }
    ]

    results = MODULE.evaluate_bm25(
        questions=questions,
        retriever=retriever,
        tokenizer=FakeTokenizer(),
        budgets=(2, 4),
        thresholds=(0.5,),
        separator_tokens=0,
    )

    assert len(results["per_question"]) == 1

    question_result = results["per_question"][0]

    assert question_result["matched_chunk_count"] == 1

    assert question_result["raw_ranking"]["single_ranks"]["0.5"] == 1

    assert question_result["context_by_budget"]["2"]["selected_chunk_count"] == 0

    assert question_result["context_by_budget"]["4"]["selected_chunk_count"] == 1

    assert (
        results["budgets"]["4"]["retrieval"]["context_hit_rate"]["0.5"]["union"] == 1.0
    )


def test_output_protection(tmp_path):
    output_path = tmp_path / "result.json"

    MODULE.ensure_output_available(output_path)

    output_path.write_text(
        "{}",
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        MODULE.ensure_output_available(output_path)


def test_invalid_dense_baseline_metadata(tmp_path):
    baseline_path = tmp_path / "baseline.json"

    baseline_path.write_text(
        '{"experiment": "wrong_experiment"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid Dense baseline metadata"):
        MODULE.load_dense_baseline(baseline_path)
