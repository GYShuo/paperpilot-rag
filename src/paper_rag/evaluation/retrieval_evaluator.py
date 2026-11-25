from dataclasses import dataclass

import numpy as np

from paper_rag.schemas import (
    PaperDocument,
    SectionChunk,
    SentenceTokenChunk,
)

RetrievableChunk = SectionChunk | SentenceTokenChunk


@dataclass(frozen=True)
class EvidenceLocation:
    """A labeled evidence occurrence in the source paper."""

    section_index: int
    paragraph_index: int
    start_char: int
    end_char: int


def locate_body_evidence(
    paper: PaperDocument,
    evidence: str,
) -> list[EvidenceLocation]:
    """Find all exact occurrences of evidence in body paragraphs."""

    locations = []

    if not evidence:
        return locations

    for section in paper.sections:
        for paragraph in section.paragraphs:
            offset = 0

            while True:
                position = paragraph.text.find(evidence, offset)

                if position == -1:
                    break

                locations.append(
                    EvidenceLocation(
                        section_index=section.section_index,
                        paragraph_index=paragraph.paragraph_index,
                        start_char=position,
                        end_char=position + len(evidence),
                    )
                )

                offset = position + 1

    return locations


def chunk_evidence_coverage(
    chunk: RetrievableChunk,
    location: EvidenceLocation,
) -> float:
    """Fraction of one evidence occurrence covered by one chunk."""

    if chunk.section_index != location.section_index:
        return 0.0

    intervals = []

    for span in chunk.spans:
        if span.paragraph_index != location.paragraph_index:
            continue

        left = max(span.start_char, location.start_char)
        right = min(span.end_char, location.end_char)

        if left < right:
            intervals.append((left, right))

    if not intervals:
        return 0.0

    intervals.sort()

    merged = []

    for left, right in intervals:
        if merged and left <= merged[-1][1]:
            merged[-1] = (
                merged[-1][0],
                max(merged[-1][1], right),
            )
        else:
            merged.append((left, right))

    covered = sum(right - left for left, right in merged)

    evidence_length = location.end_char - location.start_char

    return covered / evidence_length


def evidence_hit(
    chunk: RetrievableChunk,
    evidence_locations: list[EvidenceLocation],
    threshold: float = 0.5,
) -> bool:
    """Whether one chunk sufficiently covers any gold evidence."""

    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")

    return any(
        chunk_evidence_coverage(chunk, location) >= threshold
        for location in evidence_locations
    )


def rank_chunks(
    query_embedding: np.ndarray,
    chunks: list[RetrievableChunk],
    embeddings: np.ndarray,
    top_k: int = 10,
) -> list[RetrievableChunk]:
    """Rank chunks using normalized embedding dot products."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    query = np.asarray(query_embedding, dtype=np.float32)
    vectors = np.asarray(embeddings, dtype=np.float32)

    if query.ndim != 1 or vectors.ndim != 2:
        raise ValueError("Invalid embedding dimensions")

    if vectors.shape != (len(chunks), len(query)):
        raise ValueError("Chunk and embedding dimensions mismatch")

    if not chunks:
        return []

    scores = vectors @ query

    indices = np.argsort(-scores, kind="stable")[:top_k]

    return [chunks[int(index)] for index in indices]


def first_hit_rank(
    ranked_chunks: list[RetrievableChunk],
    evidence_locations: list[EvidenceLocation],
    threshold: float = 0.5,
) -> int | None:
    """Return the first relevant rank, or None when not found."""

    for rank, chunk in enumerate(ranked_chunks, start=1):
        if evidence_hit(
            chunk,
            evidence_locations,
            threshold,
        ):
            return rank

    return None


def summarize_ranks(
    ranks: list[int | None],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Calculate question-level Hit@K and truncated MRR."""

    if not ranks:
        raise ValueError("No evaluable questions")

    metrics = {}

    for k in ks:
        if k <= 0:
            raise ValueError("K must be positive")

        metrics[f"hit@{k}"] = sum(
            rank is not None and rank <= k for rank in ranks
        ) / len(ranks)

    max_k = max(ks)

    metrics[f"mrr@{max_k}"] = sum(
        1.0 / rank if rank is not None and rank <= max_k else 0.0 for rank in ranks
    ) / len(ranks)

    return metrics


def union_evidence_coverage(
    chunks: list[RetrievableChunk],
    location: EvidenceLocation,
) -> float:
    """Fraction of an evidence occurrence covered by chunk union."""

    if location.end_char <= location.start_char:
        raise ValueError("Evidence interval must be nonempty.")

    intervals = []

    for chunk in chunks:
        if chunk.section_index != location.section_index:
            continue

        for span in chunk.spans:
            if span.paragraph_index != location.paragraph_index:
                continue

            left = max(span.start_char, location.start_char)
            right = min(span.end_char, location.end_char)

            if left < right:
                intervals.append((left, right))

    if not intervals:
        return 0.0

    intervals.sort()
    merged = []

    for left, right in intervals:
        if merged and left <= merged[-1][1]:
            previous_left, previous_right = merged[-1]

            merged[-1] = (
                previous_left,
                max(previous_right, right),
            )
        else:
            merged.append((left, right))

    covered = sum(right - left for left, right in merged)

    return covered / (location.end_char - location.start_char)


def first_union_hit_rank(
    ranked_chunks: list[RetrievableChunk],
    evidence_locations: list[EvidenceLocation],
    threshold: float = 0.5,
) -> int | None:
    """First rank where retrieved chunks jointly cover gold evidence."""

    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")

    for k in range(1, len(ranked_chunks) + 1):
        retrieved = ranked_chunks[:k]

        if any(
            union_evidence_coverage(retrieved, location) >= threshold
            for location in evidence_locations
        ):
            return k

    return None
