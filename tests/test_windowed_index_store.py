import json

import numpy as np
import pytest

from paper_rag.retrieval.windowed_index_store import WindowedIndexStore
from paper_rag.schemas import EmbeddingWindow, SentenceTokenChunk


def make_chunk():
    text = "one two three four five six"

    return SentenceTokenChunk(
        chunk_id="paper-1:0:sentence-token:0",
        paper_id="paper-1",
        split="train",
        section_index=0,
        section_name="Introduction",
        text=text,
        token_count=6,
        sentence_count=3,
    )


def make_windows():
    chunk = make_chunk()

    return [
        EmbeddingWindow(
            window_id=f"{chunk.chunk_id}:embedding:0",
            parent_chunk_id=chunk.chunk_id,
            paper_id=chunk.paper_id,
            split=chunk.split,
            section_index=chunk.section_index,
            section_name=chunk.section_name,
            window_index=0,
            text=chunk.text[0:18],
            start_char=0,
            end_char=18,
            token_start=0,
            token_end=4,
            token_count=6,
        ),
        EmbeddingWindow(
            window_id=f"{chunk.chunk_id}:embedding:1",
            parent_chunk_id=chunk.chunk_id,
            paper_id=chunk.paper_id,
            split=chunk.split,
            section_index=chunk.section_index,
            section_name=chunk.section_name,
            window_index=1,
            text=chunk.text[8:],
            start_char=8,
            end_char=len(chunk.text),
            token_start=2,
            token_end=6,
            token_count=6,
        ),
    ]


def make_embeddings():
    return np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )


def make_store(tmp_path):
    return WindowedIndexStore(tmp_path / "windowed")


def save_index(store):
    store.save(
        chunks=[make_chunk()],
        windows=make_windows(),
        embeddings=make_embeddings(),
        model_name="fake-model",
        max_sequence_length=8,
        overlap_tokens=2,
        show_progress_bar=False,
    )


def test_save_and_load(tmp_path):
    store = make_store(tmp_path)

    chunks = [make_chunk()]
    windows = make_windows()
    embeddings = make_embeddings()

    store.save(
        chunks=chunks,
        windows=windows,
        embeddings=embeddings,
        model_name="fake-model",
        max_sequence_length=8,
        overlap_tokens=2,
        show_progress_bar=False,
    )

    loaded_chunks, loaded_windows, loaded_embeddings, manifest = store.load(
        expected_model_name="fake-model",
        show_progress_bar=False,
    )

    assert loaded_chunks == chunks
    assert loaded_windows == windows

    np.testing.assert_array_equal(
        loaded_embeddings,
        embeddings,
    )

    assert manifest["index_type"] == "windowed"
    assert manifest["num_chunks"] == 1
    assert manifest["num_windows"] == 2
    assert manifest["embedding_dimension"] == 2
    assert manifest["model_name"] == "fake-model"
    assert manifest["max_sequence_length"] == 8
    assert manifest["overlap_tokens"] == 2


def test_index_files_exist(tmp_path):
    store = make_store(tmp_path)

    save_index(store)

    assert store.exists()
    assert store.chunks_path.is_file()
    assert store.windows_path.is_file()
    assert store.embeddings_path.is_file()
    assert store.manifest_path.is_file()


def test_load_missing_index(tmp_path):
    store = make_store(tmp_path)

    with pytest.raises(FileNotFoundError):
        store.load(show_progress_bar=False)


def test_save_empty_chunks(tmp_path):
    store = make_store(tmp_path)

    with pytest.raises(ValueError, match="empty chunk"):
        store.save(
            chunks=[],
            windows=make_windows(),
            embeddings=make_embeddings(),
            model_name="fake-model",
            show_progress_bar=False,
        )


def test_save_empty_windows(tmp_path):
    store = make_store(tmp_path)

    with pytest.raises(ValueError, match="empty window"):
        store.save(
            chunks=[make_chunk()],
            windows=[],
            embeddings=np.empty((0, 2), dtype=np.float32),
            model_name="fake-model",
            show_progress_bar=False,
        )


def test_save_mismatched_embedding_count(tmp_path):
    store = make_store(tmp_path)

    with pytest.raises(ValueError, match="do not match"):
        store.save(
            chunks=[make_chunk()],
            windows=make_windows(),
            embeddings=make_embeddings()[:1],
            model_name="fake-model",
            show_progress_bar=False,
        )


def test_save_invalid_embedding_dimension(tmp_path):
    store = make_store(tmp_path)

    with pytest.raises(ValueError, match="2D"):
        store.save(
            chunks=[make_chunk()],
            windows=make_windows(),
            embeddings=np.array([1.0, 0.0]),
            model_name="fake-model",
            show_progress_bar=False,
        )


def test_save_non_finite_embeddings(tmp_path):
    store = make_store(tmp_path)

    embeddings = make_embeddings()
    embeddings[0, 0] = np.nan

    with pytest.raises(ValueError, match="invalid values"):
        store.save(
            chunks=[make_chunk()],
            windows=make_windows(),
            embeddings=embeddings,
            model_name="fake-model",
            show_progress_bar=False,
        )


def test_duplicate_chunk_ids(tmp_path):
    store = make_store(tmp_path)

    chunk = make_chunk()

    with pytest.raises(ValueError, match="Duplicate chunk ID"):
        store.save(
            chunks=[chunk, chunk.model_copy()],
            windows=make_windows(),
            embeddings=make_embeddings(),
            model_name="fake-model",
            show_progress_bar=False,
        )


def test_duplicate_window_ids(tmp_path):
    store = make_store(tmp_path)

    windows = make_windows()
    windows[1] = windows[1].model_copy(update={"window_id": windows[0].window_id})

    with pytest.raises(ValueError, match="Duplicate window ID"):
        store.save(
            chunks=[make_chunk()],
            windows=windows,
            embeddings=make_embeddings(),
            model_name="fake-model",
            show_progress_bar=False,
        )


def test_unknown_parent_chunk(tmp_path):
    store = make_store(tmp_path)

    windows = make_windows()
    windows[0] = windows[0].model_copy(update={"parent_chunk_id": "unknown-parent"})

    with pytest.raises(ValueError, match="unknown parent"):
        store.save(
            chunks=[make_chunk()],
            windows=windows,
            embeddings=make_embeddings(),
            model_name="fake-model",
            show_progress_bar=False,
        )


def test_window_metadata_mismatch(tmp_path):
    store = make_store(tmp_path)

    windows = make_windows()
    windows[0] = windows[0].model_copy(update={"paper_id": "another-paper"})

    with pytest.raises(ValueError, match="Metadata mismatch"):
        store.save(
            chunks=[make_chunk()],
            windows=windows,
            embeddings=make_embeddings(),
            model_name="fake-model",
            show_progress_bar=False,
        )


def test_window_text_mismatch(tmp_path):
    store = make_store(tmp_path)

    windows = make_windows()
    windows[0] = windows[0].model_copy(update={"text": "incorrect text"})

    with pytest.raises(ValueError, match="does not match parent"):
        store.save(
            chunks=[make_chunk()],
            windows=windows,
            embeddings=make_embeddings(),
            model_name="fake-model",
            show_progress_bar=False,
        )


def test_token_coverage_gap(tmp_path):
    store = make_store(tmp_path)

    windows = make_windows()
    windows[1] = windows[1].model_copy(update={"token_start": 5})

    with pytest.raises(ValueError, match="coverage gap"):
        store.save(
            chunks=[make_chunk()],
            windows=windows,
            embeddings=make_embeddings(),
            model_name="fake-model",
            show_progress_bar=False,
        )


def test_non_sequential_window_indices(tmp_path):
    store = make_store(tmp_path)

    windows = make_windows()
    windows[1] = windows[1].model_copy(update={"window_index": 3})

    with pytest.raises(ValueError, match="not sequential"):
        store.save(
            chunks=[make_chunk()],
            windows=windows,
            embeddings=make_embeddings(),
            model_name="fake-model",
            show_progress_bar=False,
        )


def test_window_exceeds_max_sequence_length(tmp_path):
    store = make_store(tmp_path)

    with pytest.raises(ValueError, match="exceeds maximum"):
        store.save(
            chunks=[make_chunk()],
            windows=make_windows(),
            embeddings=make_embeddings(),
            model_name="fake-model",
            max_sequence_length=5,
            show_progress_bar=False,
        )


def test_invalid_model_name(tmp_path):
    store = make_store(tmp_path)

    with pytest.raises(ValueError, match="Model name"):
        store.save(
            chunks=[make_chunk()],
            windows=make_windows(),
            embeddings=make_embeddings(),
            model_name=" ",
            show_progress_bar=False,
        )


def test_invalid_window_configuration(tmp_path):
    store = make_store(tmp_path)

    with pytest.raises(ValueError, match="max_sequence_length"):
        store.save(
            chunks=[make_chunk()],
            windows=make_windows(),
            embeddings=make_embeddings(),
            model_name="fake-model",
            max_sequence_length=0,
            show_progress_bar=False,
        )


def test_load_wrong_model(tmp_path):
    store = make_store(tmp_path)

    save_index(store)

    with pytest.raises(ValueError, match="does not match"):
        store.load(
            expected_model_name="another-model",
            show_progress_bar=False,
        )


def test_load_manifest_count_mismatch(tmp_path):
    store = make_store(tmp_path)

    save_index(store)

    with store.manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    manifest["num_windows"] = 100

    with store.manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file)

    with pytest.raises(ValueError, match="Window count"):
        store.load(show_progress_bar=False)


def test_load_detects_wrong_index_type(tmp_path):
    store = make_store(tmp_path)

    save_index(store)

    with store.manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    manifest["index_type"] = "ordinary"

    with store.manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file)

    with pytest.raises(ValueError, match="Index type"):
        store.load(show_progress_bar=False)
