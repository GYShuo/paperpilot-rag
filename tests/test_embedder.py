import numpy as np
import pytest

from paper_rag.retrieval.embedder import TextEmbedder
from paper_rag.schemas import EmbeddingWindow


class FakeTokenizer:
    """A deterministic tokenizer for unit tests."""

    def __call__(
        self,
        text,
        add_special_tokens=True,
        truncation=False,
        verbose=True,
    ):
        tokens = text.split()

        if add_special_tokens:
            tokens = ["[CLS]", *tokens, "[SEP]"]

        return {"input_ids": tokens}


class FakeModel:
    """A lightweight fake embedding model."""

    max_seq_length = 256
    tokenizer = FakeTokenizer()

    def __init__(self):
        self.last_texts = None
        self.last_batch_size = None
        self.last_show_progress_bar = None

    def get_embedding_dimension(self):
        return 384

    def encode(
        self,
        sentences,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ):
        if isinstance(sentences, str):
            texts = [sentences]
            single_input = True
        else:
            texts = list(sentences)
            single_input = False

        self.last_texts = texts
        self.last_batch_size = batch_size
        self.last_show_progress_bar = show_progress_bar

        vectors = []

        for index, _ in enumerate(texts):
            vector = np.full(
                384,
                fill_value=float(index + 1),
                dtype=np.float32,
            )

            vector /= np.linalg.norm(vector)
            vectors.append(vector)

        if single_input:
            return vectors[0]

        if not vectors:
            return np.empty((0, 384), dtype=np.float32)

        return np.stack(vectors)


def make_embedder():
    embedder = TextEmbedder.__new__(TextEmbedder)
    embedder.model = FakeModel()
    embedder.dimension = 384
    embedder.model_name = "fake-model"
    return embedder


def make_window(
    window_id,
    text,
    parent_chunk_id="paper-1:0:sentence-token:0",
    window_index=0,
):
    return EmbeddingWindow(
        window_id=window_id,
        parent_chunk_id=parent_chunk_id,
        paper_id="paper-1",
        split="train",
        section_index=0,
        section_name="Introduction",
        window_index=window_index,
        text=text,
        start_char=0,
        end_char=len(text),
        token_start=0,
        token_end=len(text.split()),
        token_count=len(text.split()) + 2,
    )


def test_count_includes_special_tokens():
    embedder = make_embedder()

    assert embedder.count("hello world") == 4


def test_count_empty_text():
    embedder = make_embedder()

    assert embedder.count("") == 2


def test_max_seq_length():
    embedder = make_embedder()

    assert embedder.max_seq_length == 256


def test_count_does_not_truncate():
    embedder = make_embedder()

    text = "word " * 500

    assert embedder.count(text) == 502


def test_count_requires_string():
    embedder = make_embedder()

    with pytest.raises(TypeError):
        embedder.count(123)


def test_encode_windows_shape_and_dtype():
    embedder = make_embedder()

    windows = [
        make_window("window-0", "hello world"),
        make_window("window-1", "another window"),
    ]

    embeddings = embedder.encode_windows(
        windows,
        show_progress_bar=False,
    )

    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32


def test_encode_windows_preserves_input_order():
    embedder = make_embedder()

    windows = [
        make_window("window-0", "first text"),
        make_window("window-1", "second text"),
        make_window("window-2", "third text"),
    ]

    embedder.encode_windows(
        windows,
        show_progress_bar=False,
    )

    assert embedder.model.last_texts == [
        "first text",
        "second text",
        "third text",
    ]


def test_encode_windows_passes_batch_configuration():
    embedder = make_embedder()

    windows = [
        make_window("window-0", "hello world"),
    ]

    embedder.encode_windows(
        windows,
        batch_size=8,
        show_progress_bar=True,
    )

    assert embedder.model.last_batch_size == 8
    assert embedder.model.last_show_progress_bar is True


def test_encode_windows_normalized():
    embedder = make_embedder()

    windows = [
        make_window("window-0", "hello world"),
        make_window("window-1", "another window"),
    ]

    embeddings = embedder.encode_windows(
        windows,
        show_progress_bar=False,
    )

    norms = np.linalg.norm(embeddings, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-6)


def test_encode_windows_rejects_overlength():
    embedder = make_embedder()

    text = "word " * 255

    window = make_window(
        "window-too-long",
        text,
    )

    with pytest.raises(ValueError, match="exceeds model maximum length"):
        embedder.encode_windows(
            [window],
            show_progress_bar=False,
        )


def test_encode_windows_rejects_token_count_mismatch():
    embedder = make_embedder()

    window = make_window(
        "window-0",
        "hello world",
    )

    window.token_count = 10

    with pytest.raises(ValueError, match="Token count mismatch"):
        embedder.encode_windows(
            [window],
            show_progress_bar=False,
        )


def test_encode_windows_rejects_empty_text():
    embedder = make_embedder()

    window = make_window(
        "window-0",
        "   ",
    )

    with pytest.raises(ValueError, match="is empty"):
        embedder.encode_windows(
            [window],
            show_progress_bar=False,
        )


def test_encode_windows_empty_input():
    embedder = make_embedder()

    embeddings = embedder.encode_windows(
        [],
        show_progress_bar=False,
    )

    assert embeddings.shape == (0, 384)
    assert embeddings.dtype == np.float32


def test_encode_windows_invalid_batch_size():
    embedder = make_embedder()

    window = make_window(
        "window-0",
        "hello world",
    )

    with pytest.raises(ValueError, match="batch_size"):
        embedder.encode_windows(
            [window],
            batch_size=0,
            show_progress_bar=False,
        )


def test_encode_documents_validates_output_shape():
    embedder = make_embedder()

    embeddings = embedder.encode_documents(
        ["first text", "second text"],
    )

    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32


def test_encode_documents_empty_input():
    embedder = make_embedder()

    embeddings = embedder.encode_documents([])

    assert embeddings.shape == (0, 384)
    assert embeddings.dtype == np.float32
