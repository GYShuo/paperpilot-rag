import numpy as np
import pytest

from paper_rag.retrieval.index_store import IndexStore
from paper_rag.schemas import SectionChunk


def make_chunks():
    return [
        SectionChunk(
            chunk_id="test:0",
            paper_id="test-paper",
            split="train",
            section_index=0,
            section_name="Introduction",
            text="Introduction content",
        ),
        SectionChunk(
            chunk_id="test:1",
            paper_id="test-paper",
            split="train",
            section_index=1,
            section_name="Method",
            text="Method content",
        ),
    ]


def make_embeddings():
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )


def test_save_and_load(tmp_path):
    store = IndexStore(tmp_path / "dense")

    chunks = make_chunks()
    embeddings = make_embeddings()

    store.save(
        chunks,
        embeddings,
        model_name="fake-model",
    )

    loaded_chunks, loaded_embeddings, manifest = store.load(
        expected_model_name="fake-model"
    )

    assert len(loaded_chunks) == 2

    assert loaded_chunks[0].chunk_id == "test:0"
    assert loaded_chunks[1].chunk_id == "test:1"

    np.testing.assert_array_equal(
        loaded_embeddings,
        embeddings,
    )

    assert manifest["model_name"] == "fake-model"
    assert manifest["embedding_dimension"] == 3


def test_index_files_exist(tmp_path):
    store = IndexStore(tmp_path / "dense")

    store.save(
        make_chunks(),
        make_embeddings(),
        model_name="fake-model",
    )

    assert store.exists()


def test_load_missing_index(tmp_path):
    store = IndexStore(tmp_path / "missing")

    with pytest.raises(FileNotFoundError):
        store.load()


def test_save_mismatched_count(tmp_path):
    store = IndexStore(tmp_path / "dense")

    with pytest.raises(ValueError):
        store.save(
            make_chunks(),
            make_embeddings()[:1],
            model_name="fake-model",
        )


def test_load_wrong_model(tmp_path):
    store = IndexStore(tmp_path / "dense")

    store.save(
        make_chunks(),
        make_embeddings(),
        model_name="fake-model",
    )

    with pytest.raises(ValueError):
        store.load(expected_model_name="another-model")


def test_save_empty_chunks(tmp_path):
    store = IndexStore(tmp_path / "dense")

    with pytest.raises(ValueError):
        store.save(
            [],
            np.empty((0, 3), dtype=np.float32),
            model_name="fake-model",
        )
