from paper_rag.data.chunker import split_text
from paper_rag.schemas import (
    ChunkSpan,
    PaperDocument,
    SectionChunk,
)


def chunk_paper_by_section(
    paper: PaperDocument,
    chunk_size: int = 500,
) -> list[SectionChunk]:
    """Chunk a paper by section with greedy paragraph packing."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

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

            if not text.strip():
                continue

            # Long paragraphs are split into separate windows.
            if len(text) > chunk_size:
                flush_chunk()

                for start, end, piece in split_text(
                    text,
                    chunk_size=chunk_size,
                    chunk_overlap=0,
                ):
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
