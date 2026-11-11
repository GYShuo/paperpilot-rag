import pytest

from paper_rag.data.chunker import chunk_paper, split_text
from paper_rag.schemas import (
    PaperDocument,
    PaperParagraph,
    PaperSection,
)


def test_fixed_size_chunking():
    chunks = split_text(
        "ABCDEFGHIJKL",
        chunk_size=5,
        chunk_overlap=0,
    )

    assert [x[2] for x in chunks] == [
        "ABCDE",
        "FGHIJ",
        "KL",
    ]


def test_sliding_window_chunking():
    chunks = split_text(
        "ABCDEFGHIJKL",
        chunk_size=5,
        chunk_overlap=2,
    )

    assert [x[2] for x in chunks] == [
        "ABCDE",
        "DEFGH",
        "GHIJK",
        "JKL",
    ]

    assert chunks[1][0] == 3
    assert chunks[-1][1] == 12


def test_empty_text():
    assert split_text("", 5, 0) == []
    assert split_text("   ", 5, 0) == []


def test_invalid_parameters():
    with pytest.raises(ValueError):
        split_text("hello", chunk_size=0)

    with pytest.raises(ValueError):
        split_text("hello", chunk_size=5, chunk_overlap=5)

    with pytest.raises(ValueError):
        split_text("hello", chunk_size=5, chunk_overlap=-1)


def test_chunk_metadata():
    paper = PaperDocument(
        paper_id="paper_001",
        title="Test Paper",
        abstract="Test abstract",
        split="train",
        sections=[
            PaperSection(
                section_index=0,
                section_name="Method",
                paragraphs=[
                    PaperParagraph(
                        paragraph_index=2,
                        text="ABCDEFGHIJKL",
                    )
                ],
            )
        ],
    )

    chunks = chunk_paper(
        paper,
        chunk_size=5,
        chunk_overlap=2,
    )

    assert len(chunks) == 4

    first = chunks[0]

    assert first.paper_id == "paper_001"
    assert first.section_name == "Method"
    assert first.paragraph_index == 2
    assert first.start_char == 0
    assert first.end_char == 5
    assert first.text == "ABCDE"

    for chunk in chunks:
        assert len(chunk.text) <= 5
        assert chunk.chunk_id
