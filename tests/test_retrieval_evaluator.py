import numpy as np
import pytest

from paper_rag.evaluation.retrieval_evaluator import (
    EvidenceLocation,
    chunk_evidence_coverage,
    evidence_hit,
    first_hit_rank,
    first_union_hit_rank,
    rank_chunks,
    summarize_ranks,
    union_evidence_coverage,
)
from paper_rag.schemas import ChunkSpan, SectionChunk


def make_chunk(start, end, chunk_id="c1"):
    return SectionChunk(
        chunk_id=chunk_id,
        paper_id="p1",
        split="train",
        section_index=0,
        section_name="Method",
        text="x" * (end - start),
        spans=[
            ChunkSpan(
                paragraph_index=0,
                start_char=start,
                end_char=end,
                chunk_start=0,
                chunk_end=end - start,
            )
        ],
    )


def test_full_evidence_coverage():
    chunk = make_chunk(0, 100)
    gold = EvidenceLocation(0, 0, 20, 80)

    assert chunk_evidence_coverage(chunk, gold) == 1.0


def test_partial_evidence_coverage():
    chunk = make_chunk(0, 50)
    gold = EvidenceLocation(0, 0, 20, 80)

    assert chunk_evidence_coverage(chunk, gold) == pytest.approx(0.5)


def test_unrelated_paragraph():
    chunk = make_chunk(0, 100)
    gold = EvidenceLocation(0, 1, 20, 80)

    assert chunk_evidence_coverage(chunk, gold) == 0.0


def test_evidence_hit_threshold():
    chunk = make_chunk(0, 50)
    gold = EvidenceLocation(0, 0, 20, 80)

    assert evidence_hit(chunk, [gold], threshold=0.5)
    assert not evidence_hit(chunk, [gold], threshold=0.75)


def test_first_hit_rank():
    gold = EvidenceLocation(0, 0, 20, 80)

    chunks = [
        make_chunk(100, 150, "c1"),
        make_chunk(0, 50, "c2"),
        make_chunk(20, 80, "c3"),
    ]

    assert first_hit_rank(chunks, [gold]) == 2


def test_no_hit():
    gold = EvidenceLocation(0, 0, 20, 80)
    chunks = [make_chunk(100, 150)]

    assert first_hit_rank(chunks, [gold]) is None


def test_rank_chunks():
    chunks = [
        make_chunk(0, 10, "c1"),
        make_chunk(10, 20, "c2"),
    ]

    embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )

    query = np.array([0.0, 1.0], dtype=np.float32)

    ranked = rank_chunks(query, chunks, embeddings, top_k=2)

    assert [chunk.chunk_id for chunk in ranked] == ["c2", "c1"]


def test_summarize_ranks():
    metrics = summarize_ranks([1, 3, None])

    assert metrics["hit@1"] == pytest.approx(1 / 3)
    assert metrics["hit@5"] == pytest.approx(2 / 3)
    assert metrics["hit@10"] == pytest.approx(2 / 3)
    assert metrics["mrr@10"] == pytest.approx((1 + 1 / 3) / 3)


def test_union_coverage_across_chunks():
    gold = EvidenceLocation(0, 0, 0, 100)
    chunks = [
        make_chunk(0, 40, "a"),
        make_chunk(40, 100, "b"),
    ]

    assert union_evidence_coverage(chunks, gold) == 1.0
    assert first_union_hit_rank(chunks, [gold], threshold=0.75) == 2


def test_union_does_not_double_count_overlap():
    gold = EvidenceLocation(0, 0, 0, 100)
    chunks = [
        make_chunk(0, 60, "a"),
        make_chunk(30, 80, "b"),
    ]

    assert union_evidence_coverage(chunks, gold) == pytest.approx(0.8)


def test_union_cannot_combine_different_paragraphs():
    gold = EvidenceLocation(0, 0, 0, 100)

    chunk_a = make_chunk(0, 40, "a")
    chunk_b = make_chunk(40, 100, "b")
    chunk_b.spans[0].paragraph_index = 1

    assert union_evidence_coverage(
        [chunk_a, chunk_b],
        gold,
    ) == pytest.approx(0.4)
