from dataclasses import dataclass
from typing import Protocol

import sentencesplit

from paper_rag.schemas import (
    ChunkSpan,
    PaperDocument,
    SentenceTokenChunk,
)


class TokenCounter(Protocol):
    """Interface for model-specific token counting."""

    def count(self, text: str) -> int:
        """Return the number of tokens in text."""
        ...


@dataclass(frozen=True)
class SentenceUnit:
    """One complete sentence with its source location."""

    paragraph_index: int
    start_char: int
    end_char: int
    text: str


def split_paragraph_into_sentences(
    text: str,
    paragraph_index: int,
    segmenter: sentencesplit.Segmenter,
) -> list[SentenceUnit]:
    """Split a paragraph while preserving exact source offsets."""

    if not text:
        return []

    raw_spans = segmenter.segment_spans(text)

    if not raw_spans:
        return [
            SentenceUnit(
                paragraph_index=paragraph_index,
                start_char=0,
                end_char=len(text),
                text=text,
            )
        ]

    sentences: list[SentenceUnit] = []
    cursor = 0

    for span in raw_spans:
        raw_start = int(span.start)
        end = int(span.end)

        if not 0 <= raw_start < end <= len(text):
            raise ValueError("Invalid sentence offsets.")

        if text[raw_start:end] != span.sent:
            raise ValueError("Sentence text does not match source offsets.")

        if raw_start < cursor:
            raise ValueError("Overlapping sentence offsets.")

        start = raw_start

        # Preserve whitespace between sentence spans.
        if start > cursor:
            gap = text[cursor:start]

            if gap.strip():
                raise ValueError("Uncovered non-whitespace text.")

            if sentences:
                previous = sentences[-1]
                sentences[-1] = SentenceUnit(
                    paragraph_index=paragraph_index,
                    start_char=previous.start_char,
                    end_char=start,
                    text=text[previous.start_char : start],
                )
            else:
                start = 0

        sentences.append(
            SentenceUnit(
                paragraph_index=paragraph_index,
                start_char=start,
                end_char=end,
                text=text[start:end],
            )
        )

        cursor = end

    # Preserve remaining source text.
    if cursor < len(text):
        tail = text[cursor:]

        if sentences and not tail.strip():
            previous = sentences[-1]
            sentences[-1] = SentenceUnit(
                paragraph_index=paragraph_index,
                start_char=previous.start_char,
                end_char=len(text),
                text=text[previous.start_char :],
            )
        else:
            sentences.append(
                SentenceUnit(
                    paragraph_index=paragraph_index,
                    start_char=cursor,
                    end_char=len(text),
                    text=tail,
                )
            )

    # Verify complete source reconstruction.
    reconstructed = "".join(sentence.text for sentence in sentences)

    if reconstructed != text:
        raise ValueError("Sentence segmentation lost source text.")

    return sentences


def _create_chunk(
    paper: PaperDocument,
    section_index: int,
    section_name: str,
    chunk_index: int,
    text: str,
    spans: list[ChunkSpan],
    token_count: int,
    sentence_count: int,
    oversize_sentence: bool,
) -> SentenceTokenChunk:
    """Create a chunk with its source metadata."""

    return SentenceTokenChunk(
        chunk_id=(f"{paper.paper_id}:{section_index}:sentence-token:{chunk_index}"),
        paper_id=paper.paper_id,
        split=paper.split,
        section_index=section_index,
        section_name=section_name,
        text=text,
        spans=spans.copy(),
        token_count=token_count,
        sentence_count=sentence_count,
        oversize_sentence=oversize_sentence,
    )


def chunk_paper_by_sentence(
    paper: PaperDocument,
    token_counter: TokenCounter,
    max_tokens: int,
    language: str = "en",
) -> list[SentenceTokenChunk]:
    """Pack complete sentences into token-budgeted chunks.

    A sentence exceeding max_tokens is preserved as one
    standalone oversized chunk. It is never silently truncated.
    """

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    segmenter = sentencesplit.Segmenter(
        language=language,
        clean=False,
        split_mode="conservative",
    )

    chunks: list[SentenceTokenChunk] = []

    for section in paper.sections:
        section_index = section.section_index
        section_name = section.section_name

        current_text = ""
        current_spans: list[ChunkSpan] = []
        current_sentence_count = 0
        current_token_count = 0
        current_oversize = False
        chunk_index = 0

        for paragraph in section.paragraphs:
            if not paragraph.text.strip():
                continue

            sentences = split_paragraph_into_sentences(
                text=paragraph.text,
                paragraph_index=paragraph.paragraph_index,
                segmenter=segmenter,
            )

            for sentence in sentences:
                same_paragraph = (
                    bool(current_spans)
                    and current_spans[-1].paragraph_index == sentence.paragraph_index
                )

                separator = "" if not current_text or same_paragraph else "\n\n"

                candidate = current_text + separator + sentence.text

                candidate_tokens = token_counter.count(candidate)

                if candidate_tokens < 0:
                    raise ValueError("Token count cannot be negative.")

                # Current chunk exceeds the token budget.
                if current_text and candidate_tokens > max_tokens:
                    chunks.append(
                        _create_chunk(
                            paper=paper,
                            section_index=section_index,
                            section_name=section_name,
                            chunk_index=chunk_index,
                            text=current_text,
                            spans=current_spans,
                            token_count=current_token_count,
                            sentence_count=current_sentence_count,
                            oversize_sentence=current_oversize,
                        )
                    )

                    chunk_index += 1

                    # Reset chunk state.
                    current_text = ""
                    current_spans = []
                    current_sentence_count = 0
                    current_token_count = 0
                    current_oversize = False

                    separator = ""
                    candidate = sentence.text
                    candidate_tokens = token_counter.count(candidate)

                    if candidate_tokens < 0:
                        raise ValueError("Token count cannot be negative.")

                # A single sentence exceeds the budget.
                oversize = candidate_tokens > max_tokens

                chunk_start = len(current_text) + len(separator)
                chunk_end = chunk_start + len(sentence.text)

                current_text += separator + sentence.text

                current_spans.append(
                    ChunkSpan(
                        paragraph_index=sentence.paragraph_index,
                        start_char=sentence.start_char,
                        end_char=sentence.end_char,
                        chunk_start=chunk_start,
                        chunk_end=chunk_end,
                    )
                )

                current_sentence_count += 1
                current_token_count = candidate_tokens

                if oversize:
                    current_oversize = True

                    # Oversized sentences must remain standalone.
                    chunks.append(
                        _create_chunk(
                            paper=paper,
                            section_index=section_index,
                            section_name=section_name,
                            chunk_index=chunk_index,
                            text=current_text,
                            spans=current_spans,
                            token_count=current_token_count,
                            sentence_count=current_sentence_count,
                            oversize_sentence=True,
                        )
                    )

                    chunk_index += 1

                    current_text = ""
                    current_spans = []
                    current_sentence_count = 0
                    current_token_count = 0
                    current_oversize = False

        # Flush remaining text at the end of each section.
        if current_text:
            chunks.append(
                _create_chunk(
                    paper=paper,
                    section_index=section_index,
                    section_name=section_name,
                    chunk_index=chunk_index,
                    text=current_text,
                    spans=current_spans,
                    token_count=current_token_count,
                    sentence_count=current_sentence_count,
                    oversize_sentence=current_oversize,
                )
            )

    return chunks
