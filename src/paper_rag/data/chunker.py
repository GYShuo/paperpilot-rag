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
