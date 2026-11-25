import re
from types import SimpleNamespace

import numpy as np
import pytest

from paper_rag.pipeline.windowed_index_builder import (
    build_windowed_index,
)
from paper_rag.retrieval.windowed_index_store import WindowedIndexStore
from paper_rag.schemas import (
    PaperDocument,
    PaperParagraph,
    PaperSection,
)


class FakeFastTokenizer:
    """A deterministic fast tokenizer for integration tests."""

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

        input_ids = list(range(1, len(matches) + 1))

        if add_special_tokens:
            input_ids = [101, *input_ids, 102]

        result = {"input_ids": input_ids}

        if return_offsets_mapping:
            result["offset_mapping"] = [
                (match.start(), match.end()) for match in matches
            ]

        return result


class FakeEmbedder:
    """Fake embedding model with deterministic vectors."""

    def __init__(self):
        self.model = SimpleNamespace(
            tokenizer=FakeFastTokenizer(),
        )

        self.model_name = "fake-embedding-model"
        self.dimension = 2
        self.max_seq_length = 6

    def count(self, text):
        return len(re.findall(r"\S+", text)) + 2

    def encode_windows(
        self,
        windows,
        batch_size=32,
        show_progress_bar=True,
    ):
        return np.tile(
            np.array([[1.0, 0.0]], dtype=np.float32),
            (len(windows), 1),
        )


class FakeTokenCounter:
    def count(self, text):
        return len(re.findall(r"\S+", text)) + 2


def make_paper(
    paper_id="paper-1",
    paragraphs=None,
):
    if paragraphs is None:
        paragraphs = ["Alpha beta. Gamma delta. Epsilon zeta."]

    section = PaperSection(
        section_index=0,
        section_name="Introduction",
        paragraphs=[
            PaperParagraph(
                paragraph_index=index,
                text=text,
            )
            for index, text in enumerate(paragraphs)
        ],
    )

    return PaperDocument(
        paper_id=paper_id,
        title="Test Paper",
        abstract="Test abstract",
        split="train",
        sections=[section],
    )


def make_builder(tmp_path):
    embedder = FakeEmbedder()
    store = WindowedIndexStore(tmp_path / "windowed_index")

    return embedder, store


def test_build_complete_windowed_index(tmp_path):
    embedder, store = make_builder(tmp_path)

    paper = make_paper()

    report = build_windowed_index(
        papers=[paper],
        embedder=embedder,
        store=store,
        chunk_token_counter=FakeTokenCounter(),
        max_chunk_tokens=20,
        overlap_tokens=1,
        batch_size=2,
        show_progress_bar=False,
    )

    assert report.paper_count == 1
    assert report.chunk_count == 1
    assert report.window_count == 2

    assert report.embedding_shape == (2, 2)

    assert store.exists()
    assert report.build_config_path.is_file()

    chunks, windows, embeddings, manifest = store.load(
        expected_model_name=embedder.model_name,
        show_progress_bar=False,
    )

    assert len(chunks) == 1
    assert len(windows) == 2

    assert embeddings.shape == (2, 2)
    assert manifest["num_chunks"] == 1
    assert manifest["num_windows"] == 2

    assert chunks[0].text == paper.sections[0].paragraphs[0].text


def test_oversized_sentence_is_preserved(tmp_path):
    embedder, store = make_builder(tmp_path)

    text = "This sentence contains exactly six words."

    paper = make_paper(
        paragraphs=[text],
    )

    report = build_windowed_index(
        papers=[paper],
        embedder=embedder,
        store=store,
        chunk_token_counter=FakeTokenCounter(),
        max_chunk_tokens=5,
        overlap_tokens=1,
        show_progress_bar=False,
    )

    assert report.chunk_count == 1
    assert report.window_count >= 1
    assert len(report.oversize_chunk_ids) == 1

    chunks, _, _, _ = store.load(
        show_progress_bar=False,
    )

    assert chunks[0].text == text
    assert chunks[0].oversize_sentence is True


def test_empty_papers_are_rejected(tmp_path):
    embedder, store = make_builder(tmp_path)

    with pytest.raises(ValueError, match="empty papers"):
        build_windowed_index(
            papers=[],
            embedder=embedder,
            store=store,
            chunk_token_counter=FakeTokenCounter(),
            show_progress_bar=False,
        )


def test_paper_without_chunks_is_reported(tmp_path):
    embedder, store = make_builder(tmp_path)

    paper = make_paper(
        paragraphs=["", "   "],
    )

    with pytest.raises(ValueError, match="No valid chunks"):
        build_windowed_index(
            papers=[paper],
            embedder=embedder,
            store=store,
            chunk_token_counter=FakeTokenCounter(),
            show_progress_bar=False,
        )


def test_duplicate_paper_ids_are_rejected(tmp_path):
    embedder, store = make_builder(tmp_path)

    paper1 = make_paper("duplicate")
    paper2 = make_paper("duplicate")

    with pytest.raises(ValueError, match="Duplicate paper IDs"):
        build_windowed_index(
            papers=[paper1, paper2],
            embedder=embedder,
            store=store,
            chunk_token_counter=FakeTokenCounter(),
            show_progress_bar=False,
        )


def test_existing_index_directory_is_not_overwritten(tmp_path):
    embedder, store = make_builder(tmp_path)

    store.index_dir.mkdir(parents=True)

    with pytest.raises(FileExistsError):
        build_windowed_index(
            papers=[make_paper()],
            embedder=embedder,
            store=store,
            chunk_token_counter=FakeTokenCounter(),
            show_progress_bar=False,
        )


def test_build_config_records_configuration(tmp_path):
    embedder, store = make_builder(tmp_path)

    report = build_windowed_index(
        papers=[make_paper()],
        embedder=embedder,
        store=store,
        chunk_token_counter=FakeTokenCounter(),
        max_chunk_tokens=20,
        overlap_tokens=1,
        language="en",
        show_progress_bar=False,
    )

    import json

    with report.build_config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    assert config["chunking_strategy"] == "sentence"
    assert config["max_chunk_tokens"] == 20
    assert config["overlap_tokens"] == 1
    assert config["language"] == "en"
    assert config["embedding_model"] == embedder.model_name
    assert config["chunk_count"] == report.chunk_count
    assert config["window_count"] == report.window_count
