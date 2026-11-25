import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from paper_rag.schemas import SentenceTokenChunk


class RankedChunkLike(Protocol):
    """Common interface for ranked retrieval results."""

    rank: int
    score: float
    chunk: SentenceTokenChunk


@dataclass(frozen=True)
class HybridRetrievalResult:
    """A parent chunk with its fused RRF score and source rankings."""

    rank: int
    score: float
    chunk: SentenceTokenChunk

    dense_rank: int
    bm25_rank: int | None

    dense_score: float
    bm25_score: float | None

    dense_contribution: float
    bm25_contribution: float


class HybridRetriever:
    """Fuse Dense and BM25 rankings using Reciprocal Rank Fusion."""

    def __init__(
        self,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ):
        if not isinstance(rrf_k, int) or isinstance(rrf_k, bool):
            raise TypeError("rrf_k must be an integer.")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive.")

        self._validate_weight(dense_weight, "dense_weight")
        self._validate_weight(bm25_weight, "bm25_weight")

        self.rrf_k = rrf_k
        self.dense_weight = float(dense_weight)
        self.bm25_weight = float(bm25_weight)

    @staticmethod
    def _validate_weight(weight: float, name: str) -> None:
        """Validate a positive finite fusion weight."""

        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise TypeError(f"{name} must be a numeric value.")

        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"{name} must be positive and finite.")

    @staticmethod
    def _validate_ranking(
        results: Sequence[RankedChunkLike],
        source_name: str,
    ) -> tuple[dict[str, RankedChunkLike], str | None, str | None]:
        """Validate a ranked list and index it by parent chunk ID."""

        indexed = {}
        paper_id = None
        split = None

        for expected_rank, result in enumerate(results, start=1):
            if not isinstance(result.chunk, SentenceTokenChunk):
                raise TypeError(
                    f"{source_name} results must contain SentenceTokenChunk."
                )

            if not isinstance(result.rank, int) or isinstance(result.rank, bool):
                raise TypeError(f"{source_name} ranks must be integers.")

            if result.rank != expected_rank:
                raise ValueError(
                    f"{source_name} ranks must be consecutive and ordered "
                    "starting from 1."
                )

            if isinstance(result.score, bool) or not isinstance(
                result.score, (int, float)
            ):
                raise TypeError(f"{source_name} scores must be numeric.")

            if not math.isfinite(result.score):
                raise ValueError(f"{source_name} contains a non-finite score.")

            if source_name == "BM25" and result.score <= 0:
                raise ValueError("BM25 scores must be positive.")

            chunk = result.chunk

            if not chunk.chunk_id.strip():
                raise ValueError(f"{source_name} contains an empty chunk ID.")

            if chunk.chunk_id in indexed:
                raise ValueError(
                    f"{source_name} contains duplicate chunk ID: {chunk.chunk_id}"
                )

            if paper_id is None:
                paper_id = chunk.paper_id
                split = chunk.split
            elif chunk.paper_id != paper_id or chunk.split != split:
                raise ValueError(
                    f"{source_name} results must belong to the same paper "
                    "and dataset split."
                )

            indexed[chunk.chunk_id] = result

        return indexed, paper_id, split

    def fuse(
        self,
        dense_results: Sequence[RankedChunkLike],
        bm25_results: Sequence[RankedChunkLike],
    ) -> list[HybridRetrievalResult]:
        """Fuse complete Dense and partial BM25 rankings.

        Dense is expected to rank every parent chunk in the paper.
        BM25 may rank only a subset of those chunks.

        The final candidate set contains every Dense candidate.
        Missing BM25 rankings contribute zero to the RRF score.

        Results are sorted by descending RRF score. Ties are
        resolved by the original Dense rank and then chunk ID.
        """

        if not dense_results:
            raise ValueError("Dense results cannot be empty.")

        dense_by_id, dense_paper_id, dense_split = self._validate_ranking(
            dense_results, "Dense"
        )

        bm25_by_id, bm25_paper_id, bm25_split = self._validate_ranking(
            bm25_results, "BM25"
        )

        if bm25_paper_id is not None and (
            bm25_paper_id != dense_paper_id or bm25_split != dense_split
        ):
            raise ValueError(
                "Dense and BM25 results must belong to the same "
                "paper and dataset split."
            )

        for chunk_id, bm25_item in bm25_by_id.items():
            dense_item = dense_by_id.get(chunk_id)

            if dense_item is None:
                raise ValueError(
                    f"BM25 chunk is missing from Dense results: {chunk_id}"
                )

            if dense_item.chunk != bm25_item.chunk:
                raise ValueError(
                    f"Chunk metadata mismatch between retrieval methods: {chunk_id}"
                )

        fused_candidates = []

        for chunk_id, dense_item in dense_by_id.items():
            bm25_item = bm25_by_id.get(chunk_id)

            dense_contribution = self.dense_weight / (self.rrf_k + dense_item.rank)

            bm25_contribution = (
                self.bm25_weight / (self.rrf_k + bm25_item.rank)
                if bm25_item is not None
                else 0.0
            )

            fused_score = dense_contribution + bm25_contribution

            fused_candidates.append(
                (
                    fused_score,
                    dense_item,
                    bm25_item,
                    dense_contribution,
                    bm25_contribution,
                )
            )

        fused_candidates.sort(
            key=lambda item: (
                -item[0],
                item[1].rank,
                item[1].chunk.chunk_id,
            )
        )

        return [
            HybridRetrievalResult(
                rank=rank,
                score=float(score),
                chunk=dense_item.chunk,
                dense_rank=dense_item.rank,
                bm25_rank=(bm25_item.rank if bm25_item is not None else None),
                dense_score=float(dense_item.score),
                bm25_score=(float(bm25_item.score) if bm25_item is not None else None),
                dense_contribution=float(dense_contribution),
                bm25_contribution=float(bm25_contribution),
            )
            for rank, (
                score,
                dense_item,
                bm25_item,
                dense_contribution,
                bm25_contribution,
            ) in enumerate(fused_candidates, start=1)
        ]
