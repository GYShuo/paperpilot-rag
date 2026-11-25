import re

import pytest

from paper_rag.data.embedding_window import build_embedding_windows
from paper_rag.schemas import SentenceTokenChunk


class FakeFastTokenizer:
    """Deterministic tokenizer for embedding window tests."""

    is_fast = True

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(
        self,
        text,
        add_special_tokens=True,
        truncation=False,
        return_offsets_mapping=False,
    ):
        matches = list(re.finditer(r"\S+", text))

        input_ids = [index + 1 for index in range(len(matches))]

        if add_special_tokens:
            input_ids = [101, *input_ids, 102]

        result = {"input_ids": input_ids}

        if return_offsets_mapping:
            result["offset_mapping"] = [
                (match.start(), match.end()) for match in matches
            ]

        return result


class SlowTokenizer:
    """Tokenizer without reliable character offsets."""

    is_fast = False


def make_chunk(text):
    return SentenceTokenChunk(
        chunk_id="paper-1:0:sentence-token:0",
        paper_id="paper-1",
        split="train",
        section_index=0,
        section_name="Introduction",
        text=text,
        token_count=len(text.split()),
        sentence_count=1,
        oversize_sentence=len(text.split()) > 4,
    )


def test_short_chunk_produces_one_window():
    chunk = make_chunk("one two three")

    windows = build_embedding_windows(
        chunks=[chunk],
        tokenizer=FakeFastTokenizer(),
        max_sequence_length=6,
        overlap_tokens=1,
        show_progress_bar=False,
    )

    assert len(windows) == 1

    window = windows[0]

    assert window.text == chunk.text
    assert window.parent_chunk_id == chunk.chunk_id
    assert window.token_start == 0
    assert window.token_end == 3
    assert window.token_count == 5


def test_long_chunk_is_split_into_overlapping_windows():
    text = "one two three four five six seven eight"
    chunk = make_chunk(text)

    windows = build_embedding_windows(
        chunks=[chunk],
        tokenizer=FakeFastTokenizer(),
        max_sequence_length=6,
        overlap_tokens=1,
        show_progress_bar=False,
    )

    assert len(windows) == 3

    assert [(window.token_start, window.token_end) for window in windows] == [
        (0, 4),
        (3, 7),
        (6, 8),
    ]

    assert [window.text for window in windows] == [
        "one two three four",
        "four five six seven",
        "seven eight",
    ]


def test_all_windows_respect_model_input_limit():
    chunk = make_chunk("one two three four five six seven eight nine ten")

    max_length = 6

    windows = build_embedding_windows(
        chunks=[chunk],
        tokenizer=FakeFastTokenizer(),
        max_sequence_length=max_length,
        overlap_tokens=1,
        show_progress_bar=False,
    )

    assert windows

    assert all(window.token_count <= max_length for window in windows)


def test_all_parent_tokens_are_covered():
    chunk = make_chunk("one two three four five six seven eight")

    windows = build_embedding_windows(
        chunks=[chunk],
        tokenizer=FakeFastTokenizer(),
        max_sequence_length=6,
        overlap_tokens=1,
        show_progress_bar=False,
    )

    covered_tokens = set()

    for window in windows:
        covered_tokens.update(range(window.token_start, window.token_end))

    assert covered_tokens == set(range(8))


def test_window_character_offsets_are_correct():
    chunk = make_chunk("one two three four five six seven eight")

    windows = build_embedding_windows(
        chunks=[chunk],
        tokenizer=FakeFastTokenizer(),
        max_sequence_length=6,
        overlap_tokens=1,
        show_progress_bar=False,
    )

    for window in windows:
        assert chunk.text[window.start_char : window.end_char] == window.text


def test_window_ids_are_unique():
    chunk = make_chunk("one two three four five six seven eight")

    windows = build_embedding_windows(
        chunks=[chunk],
        tokenizer=FakeFastTokenizer(),
        max_sequence_length=6,
        overlap_tokens=1,
        show_progress_bar=False,
    )

    ids = [window.window_id for window in windows]

    assert len(ids) == len(set(ids))


def test_window_indices_are_sequential():
    chunk = make_chunk("one two three four five six seven eight")

    windows = build_embedding_windows(
        chunks=[chunk],
        tokenizer=FakeFastTokenizer(),
        max_sequence_length=6,
        overlap_tokens=1,
        show_progress_bar=False,
    )

    assert [window.window_index for window in windows] == [0, 1, 2]


def test_slow_tokenizer_is_rejected():
    chunk = make_chunk("one two three")

    with pytest.raises(ValueError, match="fast tokenizer"):
        build_embedding_windows(
            chunks=[chunk],
            tokenizer=SlowTokenizer(),
            max_sequence_length=6,
            show_progress_bar=False,
        )


def test_invalid_overlap_is_rejected():
    chunk = make_chunk("one two three")

    with pytest.raises(ValueError, match="overlap_tokens"):
        build_embedding_windows(
            chunks=[chunk],
            tokenizer=FakeFastTokenizer(),
            max_sequence_length=6,
            overlap_tokens=4,
            show_progress_bar=False,
        )


def test_invalid_sequence_length_is_rejected():
    chunk = make_chunk("one two three")

    with pytest.raises(ValueError):
        build_embedding_windows(
            chunks=[chunk],
            tokenizer=FakeFastTokenizer(),
            max_sequence_length=2,
            show_progress_bar=False,
        )


def test_empty_chunk_list():
    windows = build_embedding_windows(
        chunks=[],
        tokenizer=FakeFastTokenizer(),
        max_sequence_length=6,
        show_progress_bar=False,
    )

    assert windows == []
