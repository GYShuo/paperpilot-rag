import re

from paper_rag.schemas import PaperChunk, PaperDocument


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 0,
) -> list[tuple[int, int, str]]:
    """Split text into overlapping character windows."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")

    if not text.strip():
        return []

    chunks = []
    step = chunk_size - chunk_overlap  # 滑动步长

    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]

        if chunk_text.strip():
            chunks.append((start, end, chunk_text))

        if end == len(text):
            break

    return chunks


def chunk_paper(
    paper: PaperDocument,
    chunk_size: int = 500,
    chunk_overlap: int = 0,
) -> list[PaperChunk]:
    """Chunk paper paragraphs while preserving provenance."""

    chunks = []

    for section in paper.sections:
        for paragraph in section.paragraphs:
            windows = split_text(
                text=paragraph.text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            for start, end, text in windows:
                chunk_id = (
                    f"{paper.paper_id}:"
                    f"{section.section_index}:"
                    f"{paragraph.paragraph_index}:"
                    f"{start}:{end}"
                )

                chunk = PaperChunk(
                    chunk_id=chunk_id,
                    paper_id=paper.paper_id,
                    split=paper.split,
                    section_index=section.section_index,
                    section_name=section.section_name,
                    paragraph_index=paragraph.paragraph_index,
                    start_char=start,
                    end_char=end,
                    text=text,
                )

                chunks.append(chunk)

    return chunks


# Common abbreviations that should not normally end a sentence.
ABBREVIATIONS = {
    "e.g.",
    "i.e.",
    "dr.",
    "mr.",
    "mrs.",
    "prof.",
    "vs.",
    "etc.",
}


def _find_boundaries(text: str) -> tuple[list[int], list[int]]:
    """Find sentence and whitespace boundaries in original text."""

    sentence_boundaries = []
    word_boundaries = []

    for match in re.finditer(r"\s+", text):
        position = match.end()

        if position >= len(text):
            continue

        word_boundaries.append(position)

        prefix = text[: match.start()].rstrip()

        if not prefix:
            continue

        last_word = prefix.split()[-1].lower()

        if last_word in ABBREVIATIONS:
            continue

        if prefix[-1] in ".!?。！？":
            sentence_boundaries.append(position)

    return sentence_boundaries, word_boundaries


def _choose_boundary(
    start: int,
    limit: int,
    sentence_boundaries: list[int],
    word_boundaries: list[int],
    chunk_size: int,
) -> int:
    """Prefer a sentence boundary, then a word boundary."""

    # Avoid tiny chunks caused by very early sentence boundaries.
    minimum_preferred = start + max(1, chunk_size // 2)

    sentence_candidates = [
        pos for pos in sentence_boundaries if minimum_preferred <= pos <= limit
    ]

    if sentence_candidates:
        return sentence_candidates[-1]

    word_candidates = [pos for pos in word_boundaries if start < pos <= limit]

    if word_candidates:
        return word_candidates[-1]

    # A single token may exceed the maximum length.
    return limit


def _rebalance_short_tail(
    text: str,
    spans: list[tuple[int, int]],
    sentence_boundaries: list[int],
    word_boundaries: list[int],
    chunk_size: int,
    min_tail_size: int,
) -> list[tuple[int, int]]:
    """Rebalance the last two chunks if the final chunk is too short."""

    if len(spans) < 2 or min_tail_size == 0:
        return spans

    previous_start, _ = spans[-2]
    tail_start, final_end = spans[-1]

    if final_end - tail_start >= min_tail_size:
        return spans

    # Legal boundaries must keep both chunks within chunk_size.
    lower = max(
        previous_start + 1,
        final_end - chunk_size,
    )
    upper = min(
        previous_start + chunk_size,
        final_end - 1,
    )

    if lower > upper:
        return spans

    # Prefer both final chunks to meet the tail threshold.
    balanced_lower = max(lower, previous_start + min_tail_size)
    balanced_upper = min(upper, final_end - min_tail_size)

    if balanced_lower <= balanced_upper:
        lower, upper = balanced_lower, balanced_upper

    midpoint = (previous_start + final_end) / 2

    def nearest(candidates: list[int]) -> int | None:
        valid = [pos for pos in candidates if lower <= pos <= upper]
        if not valid:
            return None
        return min(valid, key=lambda pos: abs(pos - midpoint))

    boundary = nearest(sentence_boundaries)

    if boundary is None:
        boundary = nearest(word_boundaries)

    if boundary is None:
        boundary = round(midpoint)
        boundary = max(lower, min(boundary, upper))

    spans[-2] = (previous_start, boundary)
    spans[-1] = (boundary, final_end)

    return spans


def split_text_semantic(
    text: str,
    chunk_size: int = 500,
    min_tail_size: int = 100,
) -> list[tuple[int, int, str]]:
    """Split text using sentence and word boundaries.

    Return (start, end, text_piece) tuples using original offsets.
    This version intentionally does not implement overlap.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= min_tail_size <= chunk_size:
        raise ValueError("min_tail_size must be between 0 and chunk_size")

    if not text.strip():
        return []

    if len(text) <= chunk_size:
        return [(0, len(text), text)]

    sentence_boundaries, word_boundaries = _find_boundaries(text)

    spans = []
    start = 0

    while start < len(text):
        limit = min(start + chunk_size, len(text))

        if limit == len(text):
            end = limit
        else:
            end = _choose_boundary(
                start,
                limit,
                sentence_boundaries,
                word_boundaries,
                chunk_size,
            )

        spans.append((start, end))
        start = end

    spans = _rebalance_short_tail(
        text,
        spans,
        sentence_boundaries,
        word_boundaries,
        chunk_size,
        min_tail_size,
    )

    return [(start, end, text[start:end]) for start, end in spans]
