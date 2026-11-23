from paper_rag.retrieval.embedder import TextEmbedder


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


def make_embedder():
    embedder = TextEmbedder.__new__(TextEmbedder)
    embedder.model = FakeModel()
    embedder.dimension = 384
    embedder.model_name = "fake-model"
    return embedder


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

    try:
        embedder.count(123)
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError.")
