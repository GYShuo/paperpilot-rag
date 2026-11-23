import pytest

from paper_rag.data.sentence_token_chunker import (
    chunk_paper_by_sentence,
)
from paper_rag.schemas import (
    PaperDocument,
    PaperParagraph,
    PaperSection,
)


class WordTokenCounter:
    """Deterministic fake tokenizer for unit tests."""

    def count(self, text: str) -> int:
        return len(text.split())


class CharacterTokenCounter:
    """Deterministic counter for CJK boundary tests."""

    def count(self, text: str) -> int:
        return len(text)


def make_paper(sections_data):
    sections = []

    for section_index, paragraphs in enumerate(sections_data):
        sections.append(
            PaperSection(
                section_index=section_index,
                section_name=f"Section {section_index}",
                paragraphs=[
                    PaperParagraph(
                        paragraph_index=i,
                        text=text,
                    )
                    for i, text in enumerate(paragraphs)
                ],
            )
        )

    return PaperDocument(
        paper_id="test-paper",
        title="Test",
        abstract="",
        split="train",
        sections=sections,
    )


def test_complete_sentences_are_preserved():
    text = "Alpha beta. Gamma delta. Epsilon zeta."
    paper = make_paper([[text]])

    chunks = chunk_paper_by_sentence(
        paper,
        token_counter=WordTokenCounter(),
        max_tokens=2,
    )

    assert [chunk.text for chunk in chunks] == [
        "Alpha beta. ",
        "Gamma delta. ",
        "Epsilon zeta.",
    ]

    assert all(chunk.sentence_count == 1 for chunk in chunks)


def test_normal_chunks_respect_token_budget():
    paper = make_paper([["Alpha beta. Gamma delta. Epsilon zeta."]])

    chunks = chunk_paper_by_sentence(
        paper,
        token_counter=WordTokenCounter(),
        max_tokens=4,
    )

    assert all(
        chunk.token_count <= 4 for chunk in chunks if not chunk.oversize_sentence
    )


def test_oversized_sentence_is_preserved():
    long_sentence = "This sentence contains six words."
    short_sentence = "Short one."

    paper = make_paper([[long_sentence + " " + short_sentence]])

    chunks = chunk_paper_by_sentence(
        paper,
        token_counter=WordTokenCounter(),
        max_tokens=3,
    )

    assert len(chunks) == 2

    assert chunks[0].text == long_sentence + " "
    assert chunks[0].oversize_sentence is True

    assert chunks[1].text == short_sentence
    assert chunks[1].oversize_sentence is False


def test_source_offsets_are_correct():
    text = "Alpha beta. Gamma delta. Epsilon zeta."
    paper = make_paper([[text]])

    chunks = chunk_paper_by_sentence(
        paper,
        token_counter=WordTokenCounter(),
        max_tokens=4,
    )

    for chunk in chunks:
        for span in chunk.spans:
            source = text[span.start_char : span.end_char]
            extracted = chunk.text[span.chunk_start : span.chunk_end]

            assert source == extracted


def test_paragraphs_can_be_packed_together():
    paper = make_paper([["First paragraph.", "Second paragraph."]])

    chunks = chunk_paper_by_sentence(
        paper,
        token_counter=WordTokenCounter(),
        max_tokens=10,
    )

    assert len(chunks) == 1
    assert chunks[0].text == ("First paragraph.\n\nSecond paragraph.")

    assert len(chunks[0].spans) == 2


def test_sections_are_never_merged():
    paper = make_paper([["First section. "], ["Second section."]])

    chunks = chunk_paper_by_sentence(
        paper,
        token_counter=WordTokenCounter(),
        max_tokens=100,
    )

    assert len(chunks) == 2
    assert chunks[0].section_index == 0
    assert chunks[1].section_index == 1


def test_chinese_sentence_segmentation():
    paper = make_paper([["本文提出了一种新方法。实验结果表明，准确率达到92.5%。"]])

    chunks = chunk_paper_by_sentence(
        paper,
        token_counter=CharacterTokenCounter(),
        max_tokens=20,
        language="zh",
    )

    assert len(chunks) == 2
    assert all(chunk.sentence_count == 1 for chunk in chunks)


def test_japanese_sentence_segmentation():
    paper = make_paper([["本研究では新しい手法を提案する。実験結果を示す。"]])

    chunks = chunk_paper_by_sentence(
        paper,
        token_counter=CharacterTokenCounter(),
        max_tokens=20,
        language="ja",
    )

    assert len(chunks) == 2
    assert all(chunk.sentence_count == 1 for chunk in chunks)


def test_empty_paper():
    paper = make_paper([["", "   "]])

    chunks = chunk_paper_by_sentence(
        paper,
        token_counter=WordTokenCounter(),
        max_tokens=100,
    )

    assert chunks == []


def test_invalid_token_budget():
    paper = make_paper([["A sentence."]])

    with pytest.raises(ValueError):
        chunk_paper_by_sentence(
            paper,
            token_counter=WordTokenCounter(),
            max_tokens=0,
        )
