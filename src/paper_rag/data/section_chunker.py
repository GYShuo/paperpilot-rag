from typing import Literal

from paper_rag.data.chunker import split_text, split_text_semantic
from paper_rag.schemas import (
    ChunkSpan,
    PaperDocument,
    SectionChunk,
)


def has_meaningful_content(text: str) -> bool:
    """Check whether text contains at least one letter or digit."""
    return any(char.isalnum() for char in text)


def chunk_paper_by_section(
    paper: PaperDocument,
    chunk_size: int = 500,
    strategy: Literal["fixed", "semantic"] = "fixed",
    min_tail_size: int = 100,
) -> list[SectionChunk]:
    """Chunk a paper by section with greedy paragraph packing."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if strategy not in ("fixed", "semantic"):
        raise ValueError(f"Unsupported chunking strategy: {strategy}")

    if strategy == "semantic" and not 0 <= min_tail_size <= chunk_size:
        raise ValueError("min_tail_size must be between 0 and chunk_size")

    chunks: list[SectionChunk] = []

    for section in paper.sections:
        current_text = ""
        current_spans: list[ChunkSpan] = []
        chunk_index = 0

        def flush_chunk(
            section_index: int = section.section_index,
            section_name: str = section.section_name,
        ) -> None:
            """Save the current chunk and reset the buffer."""
            nonlocal current_text, current_spans, chunk_index

            if not current_text:
                return

            chunks.append(
                SectionChunk(
                    chunk_id=f"{paper.paper_id}:{section_index}:{chunk_index}",
                    paper_id=paper.paper_id,
                    split=paper.split,
                    section_index=section_index,
                    section_name=section_name,
                    text=current_text,
                    spans=current_spans.copy(),
                )
            )

            chunk_index += 1
            current_text = ""
            current_spans = []

        for paragraph in section.paragraphs:
            text = paragraph.text

            if not has_meaningful_content(text):
                continue

            # Long paragraphs are split into separate windows.
            if len(text) > chunk_size:
                flush_chunk()

                if strategy == "semantic":  # V1：语义感知切分
                    windows = split_text_semantic(
                        text,
                        chunk_size=chunk_size,
                        min_tail_size=min_tail_size,
                    )
                else:  # strategy == "fixed":  V0：固定字符窗口
                    windows = split_text(
                        text,
                        chunk_size=chunk_size,
                        chunk_overlap=0,
                    )

                for start, end, piece in windows:
                    if not has_meaningful_content(piece):
                        continue

                    current_text = piece
                    current_spans = [
                        ChunkSpan(
                            paragraph_index=paragraph.paragraph_index,
                            start_char=start,
                            end_char=end,
                            chunk_start=0,
                            chunk_end=len(piece),
                        )
                    ]
                    flush_chunk()

                continue

            # Joining two paragraphs requires one newline.
            separator = "\n" if current_text else ""
            proposed_length = len(current_text) + len(separator) + len(text)

            if proposed_length > chunk_size:
                flush_chunk()
                separator = ""

            chunk_start = len(current_text) + len(separator)
            chunk_end = chunk_start + len(text)

            current_text += separator + text

            current_spans.append(
                ChunkSpan(
                    paragraph_index=paragraph.paragraph_index,
                    start_char=0,
                    end_char=len(text),
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                )
            )

        # Save remaining text at the end of each section.
        flush_chunk()

    return chunks
