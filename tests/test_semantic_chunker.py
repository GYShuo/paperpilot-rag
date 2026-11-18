from itertools import pairwise

import pytest

from paper_rag.data.chunker import split_text_semantic


def test_short_text():
    text = "The model uses Adam."

    result = split_text_semantic(text, chunk_size=100)

    assert result == [(0, len(text), text)]


def test_sentence_boundary():
    text = "Alpha ends. Beta ends. Gamma ends."

    result = split_text_semantic(
        text,
        chunk_size=20,
        min_tail_size=0,
    )

    assert [piece for _, _, piece in result] == [
        "Alpha ends. ",
        "Beta ends. ",
        "Gamma ends.",
    ]


def test_word_boundary():
    text = "apple banana cherry"

    result = split_text_semantic(
        text,
        chunk_size=13,
        min_tail_size=0,
    )

    assert [piece for _, _, piece in result] == [
        "apple banana ",
        "cherry",
    ]


def test_long_word_fallback():
    text = "ABCDEFGHIJKL"

    result = split_text_semantic(
        text,
        chunk_size=5,
        min_tail_size=0,
    )

    assert [piece for _, _, piece in result] == [
        "ABCDE",
        "FGHIJ",
        "KL",
    ]


def test_short_tail_rebalancing():
    text = "word " * 20 + "end"

    result = split_text_semantic(
        text,
        chunk_size=50,
        min_tail_size=20,
    )

    lengths = [len(piece) for _, _, piece in result]

    assert max(lengths) <= 50
    assert lengths[-1] >= 20


def test_abbreviation_not_sentence_boundary():
    text = "We use e.g. neural networks. This is effective."

    result = split_text_semantic(
        text,
        chunk_size=30,
        min_tail_size=0,
    )

    # A split after "e.g." must not be treated as a sentence split.
    assert result[0][2] != "We use e.g. "


def test_full_text_reconstruction():
    text = (
        "The model uses Adam optimizer. "
        "We evaluate it on three datasets. "
        "Experimental results are promising."
    )

    result = split_text_semantic(
        text,
        chunk_size=45,
        min_tail_size=15,
    )

    reconstructed = "".join(piece for _, _, piece in result)

    assert reconstructed == text

    for start, end, piece in result:
        assert piece == text[start:end]
        assert len(piece) <= 45


def test_no_gaps_or_overlaps():
    text = "A short sentence. " * 30

    result = split_text_semantic(
        text,
        chunk_size=50,
        min_tail_size=15,
    )

    assert result[0][0] == 0
    assert result[-1][1] == len(text)

    for current, following in pairwise(result):
        assert current[1] == following[0]


def test_invalid_parameters():
    with pytest.raises(ValueError):
        split_text_semantic("hello", chunk_size=0)

    with pytest.raises(ValueError):
        split_text_semantic(
            "hello",
            chunk_size=10,
            min_tail_size=11,
        )

    with pytest.raises(ValueError):
        split_text_semantic(
            "hello",
            chunk_size=10,
            min_tail_size=-1,
        )


def test_empty_text():
    assert split_text_semantic("", 10, min_tail_size=0) == []
    assert split_text_semantic("   ", 10, min_tail_size=0) == []


def test_empty_text_with_default_parameters():
    assert split_text_semantic("") == []
    assert split_text_semantic("   ") == []
