from dataclasses import dataclass

import pytest

from paper_rag.retrieval.hybrid_retriever import HybridRetriever
from paper_rag.schemas import SentenceTokenChunk


@dataclass(frozen=True)
class RankedItem:
    """Lightweight ranked chunk for testing."""

    rank: int
    score: float
    chunk: SentenceTokenChunk


def make_chunk(
    chunk_id: str,
    paper_id: str = "paper-1",
    split: str = "validation",
    text: str | None = None,
) -> SentenceTokenChunk:
    return SentenceTokenChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        split=split,
        section_index=0,
        section_name="Introduction",
        text=text or f"Text for {chunk_id}",
        token_count=3,
        sentence_count=1,
    )


def make_ranked(
    chunks: list[SentenceTokenChunk],
    scores: list[float] | None = None,
) -> list[RankedItem]:
    if scores is None:
        scores = [1.0 / rank for rank in range(1, len(chunks) + 1)]

    return [
        RankedItem(
            rank=rank,
            score=score,
            chunk=chunk,
        )
        for rank, (chunk, score) in enumerate(
            zip(chunks, scores, strict=True),
            start=1,
        )
    ]


def test_rrf_fusion_ranking():
    chunk_a = make_chunk("a")
    chunk_b = make_chunk("b")
    chunk_c = make_chunk("c")

    dense = make_ranked([chunk_a, chunk_b, chunk_c])
    bm25 = make_ranked([chunk_b, chunk_c])

    retriever = HybridRetriever(rrf_k=60)

    results = retriever.fuse(dense, bm25)

    assert [item.chunk.chunk_id for item in results] == [
        "b",
        "c",
        "a",
    ]

    assert [item.rank for item in results] == [1, 2, 3]

    assert results[0].score == pytest.approx(1 / 62 + 1 / 61)

    assert results[1].score == pytest.approx(1 / 63 + 1 / 62)

    assert results[2].score == pytest.approx(1 / 61)


def test_rrf_preserves_all_dense_candidates():
    chunks = [
        make_chunk("a"),
        make_chunk("b"),
        make_chunk("c"),
        make_chunk("d"),
    ]

    dense = make_ranked(chunks)
    bm25 = make_ranked([chunks[1]])

    results = HybridRetriever().fuse(dense, bm25)

    assert len(results) == len(dense)

    assert {item.chunk.chunk_id for item in results} == {
        item.chunk.chunk_id for item in dense
    }


def test_empty_bm25_results_preserve_dense_order():
    chunks = [
        make_chunk("a"),
        make_chunk("b"),
        make_chunk("c"),
    ]

    dense = make_ranked(chunks)

    results = HybridRetriever().fuse(dense, [])

    assert [item.chunk.chunk_id for item in results] == [
        "a",
        "b",
        "c",
    ]

    assert all(item.bm25_rank is None for item in results)
    assert all(item.bm25_score is None for item in results)

    assert all(item.bm25_contribution == 0.0 for item in results)


def test_bm25_match_receives_second_contribution():
    chunks = [
        make_chunk("a"),
        make_chunk("b"),
    ]

    dense = make_ranked(chunks)
    bm25 = make_ranked([chunks[1]])

    results = HybridRetriever().fuse(dense, bm25)

    result_a = next(item for item in results if item.chunk.chunk_id == "a")
    result_b = next(item for item in results if item.chunk.chunk_id == "b")

    assert result_a.bm25_contribution == 0.0
    assert result_a.bm25_rank is None

    assert result_b.bm25_contribution == pytest.approx(1 / 61)
    assert result_b.bm25_rank == 1


def test_rrf_uses_weights():
    chunks = [
        make_chunk("a"),
        make_chunk("b"),
    ]

    dense = make_ranked(chunks)
    bm25 = make_ranked([chunks[1]])

    retriever = HybridRetriever(
        rrf_k=60,
        dense_weight=2.0,
        bm25_weight=3.0,
    )

    results = retriever.fuse(dense, bm25)

    result_b = results[0]

    assert result_b.dense_contribution == pytest.approx(2 / 62)
    assert result_b.bm25_contribution == pytest.approx(3 / 61)


def test_rrf_uses_dense_rank_as_tie_breaker():
    chunks = [
        make_chunk("a"),
        make_chunk("b"),
    ]

    dense = make_ranked(chunks)
    bm25 = make_ranked([chunks[1]])

    results = HybridRetriever().fuse(dense, bm25)

    assert results[0].chunk.chunk_id == "b"


def test_rrf_result_contains_source_scores():
    chunks = [
        make_chunk("a"),
        make_chunk("b"),
    ]

    dense = make_ranked(chunks, scores=[0.9, 0.7])
    bm25 = make_ranked([chunks[1]], scores=[2.5])

    results = HybridRetriever().fuse(dense, bm25)

    result_b = next(item for item in results if item.chunk.chunk_id == "b")

    assert result_b.dense_score == 0.7
    assert result_b.bm25_score == 2.5
    assert result_b.dense_rank == 2
    assert result_b.bm25_rank == 1


def test_empty_dense_results():
    retriever = HybridRetriever()

    with pytest.raises(ValueError, match="Dense results cannot be empty"):
        retriever.fuse([], [])


def test_non_consecutive_dense_ranks():
    chunks = [
        make_chunk("a"),
        make_chunk("b"),
    ]

    dense = [
        RankedItem(rank=1, score=1.0, chunk=chunks[0]),
        RankedItem(rank=3, score=0.5, chunk=chunks[1]),
    ]

    with pytest.raises(ValueError, match="consecutive"):
        HybridRetriever().fuse(dense, [])


def test_duplicate_dense_chunk_ids():
    chunk = make_chunk("a")

    dense = [
        RankedItem(rank=1, score=1.0, chunk=chunk),
        RankedItem(rank=2, score=0.5, chunk=chunk.model_copy()),
    ]

    with pytest.raises(ValueError, match="duplicate chunk ID"):
        HybridRetriever().fuse(dense, [])


def test_bm25_chunk_must_exist_in_dense_results():
    dense = make_ranked([make_chunk("a")])
    bm25 = make_ranked([make_chunk("unknown")])

    with pytest.raises(ValueError, match="missing from Dense"):
        HybridRetriever().fuse(dense, bm25)


def test_mismatched_chunk_metadata():
    dense_chunk = make_chunk("a", text="original text")
    bm25_chunk = make_chunk("a", text="different text")

    dense = make_ranked([dense_chunk])
    bm25 = make_ranked([bm25_chunk])

    with pytest.raises(ValueError, match="metadata mismatch"):
        HybridRetriever().fuse(dense, bm25)


def test_cross_paper_results_are_rejected():
    dense = make_ranked(
        [
            make_chunk("a", paper_id="paper-1"),
        ]
    )

    bm25 = make_ranked(
        [
            make_chunk("a", paper_id="paper-2"),
        ]
    )

    with pytest.raises(ValueError, match="same paper"):
        HybridRetriever().fuse(dense, bm25)


@pytest.mark.parametrize("rrf_k", [0, -1])
def test_invalid_rrf_k(rrf_k):
    with pytest.raises(ValueError, match="rrf_k"):
        HybridRetriever(rrf_k=rrf_k)


def test_invalid_rrf_k_type():
    with pytest.raises(TypeError, match="integer"):
        HybridRetriever(rrf_k=60.5)


@pytest.mark.parametrize(
    "weight",
    [0, -1, float("inf"), float("nan")],
)
def test_invalid_weights(weight):
    with pytest.raises(ValueError, match="positive and finite"):
        HybridRetriever(dense_weight=weight)


def test_invalid_weight_type():
    with pytest.raises(TypeError, match="numeric"):
        HybridRetriever(dense_weight="1.0")


def test_non_finite_dense_score():
    dense = [
        RankedItem(
            rank=1,
            score=float("nan"),
            chunk=make_chunk("a"),
        )
    ]

    with pytest.raises(ValueError, match="non-finite"):
        HybridRetriever().fuse(dense, [])


def test_non_positive_bm25_score():
    chunk = make_chunk("a")

    dense = make_ranked([chunk])

    bm25 = [
        RankedItem(
            rank=1,
            score=0.0,
            chunk=chunk,
        )
    ]

    with pytest.raises(ValueError, match="BM25 scores must be positive"):
        HybridRetriever().fuse(dense, bm25)
