import pytest

from paper_rag.data.section_chunker import chunk_paper_by_section
from paper_rag.schemas import PaperDocument, PaperParagraph, PaperSection


def make_paper(sections_data: list[list[str]]) -> PaperDocument:
    """Build a small paper for unit tests."""
    sections = []

    for section_index, texts in enumerate(sections_data):
        paragraphs = [
            PaperParagraph(paragraph_index=i, text=text) for i, text in enumerate(texts)
        ]

        sections.append(
            PaperSection(
                section_index=section_index,
                section_name=f"Section {section_index}",
                paragraphs=paragraphs,
            )
        )

    return PaperDocument(
        paper_id="test-paper",
        title="Test",
        abstract="",
        split="train",
        sections=sections,
    )


def test_short_paragraph_packing():
    paper = make_paper([["AAAA", "BBBB", "CCCC"]])

    chunks = chunk_paper_by_section(paper, chunk_size=10)

    assert len(chunks) == 2
    assert chunks[0].text == "AAAA\nBBBB"
    assert chunks[1].text == "CCCC"


def test_span_positions():
    paper = make_paper([["AAAA", "BBBB"]])

    chunks = chunk_paper_by_section(paper, chunk_size=10)
    chunk = chunks[0]

    assert len(chunk.spans) == 2

    assert chunk.spans[0].paragraph_index == 0
    assert chunk.spans[0].chunk_start == 0
    assert chunk.spans[0].chunk_end == 4

    assert chunk.spans[1].paragraph_index == 1
    assert chunk.spans[1].chunk_start == 5
    assert chunk.spans[1].chunk_end == 9


def test_long_paragraph():
    paper = make_paper([["A" * 12]])

    chunks = chunk_paper_by_section(paper, chunk_size=5)

    assert [chunk.text for chunk in chunks] == [
        "AAAAA",
        "AAAAA",
        "AA",
    ]

    assert [chunk.spans[0].start_char for chunk in chunks] == [
        0,
        5,
        10,
    ]


def test_section_boundaries():
    paper = make_paper([["AAAA"], ["BBBB"]])

    chunks = chunk_paper_by_section(paper, chunk_size=10)

    assert len(chunks) == 2
    assert chunks[0].section_index == 0
    assert chunks[1].section_index == 1


def test_empty_paragraphs():
    paper = make_paper([["", "   ", "AAAA"]])

    chunks = chunk_paper_by_section(paper, chunk_size=10)

    assert len(chunks) == 1
    assert chunks[0].text == "AAAA"


def test_invalid_size():
    paper = make_paper([["AAAA"]])

    with pytest.raises(ValueError):
        chunk_paper_by_section(paper, chunk_size=0)


def test_span_reconstruction():
    paper = make_paper([["AAAA", "BBBB", "C" * 12]])

    chunks = chunk_paper_by_section(paper, chunk_size=10)

    original = {p.paragraph_index: p.text for p in paper.sections[0].paragraphs}

    reconstructed = {i: "" for i in original}

    for chunk in chunks:
        for span in chunk.spans:
            source_piece = original[span.paragraph_index][
                span.start_char : span.end_char
            ]
            chunk_piece = chunk.text[span.chunk_start : span.chunk_end]

            assert source_piece == chunk_piece
            reconstructed[span.paragraph_index] += source_piece

    assert reconstructed == original
