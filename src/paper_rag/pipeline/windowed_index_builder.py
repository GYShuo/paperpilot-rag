import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from paper_rag.data.embedding_window import build_embedding_windows
from paper_rag.data.sentence_token_chunker import (
    TokenCounter,
    chunk_paper_by_sentence,
)
from paper_rag.retrieval.embedder import TextEmbedder
from paper_rag.retrieval.windowed_dense_retriever import (
    WindowedDenseRetriever,
)
from paper_rag.retrieval.windowed_index_store import WindowedIndexStore
from paper_rag.schemas import PaperDocument, SentenceTokenChunk


@dataclass(frozen=True)
class WindowedIndexBuildReport:
    """Summary of a completed windowed index build."""

    paper_count: int
    chunk_count: int
    window_count: int
    oversize_chunk_ids: tuple[str, ...]
    empty_paper_ids: tuple[str, ...]
    embedding_shape: tuple[int, int]
    index_dir: Path
    build_config_path: Path
    manifest: dict[str, Any]


def _write_build_config(
    index_dir: Path,
    config: dict[str, Any],
) -> Path:
    """Persist the reproducible index build configuration."""

    config_path = index_dir / "build_config.json"
    temporary_path = index_dir / "build_config.json.tmp"

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(temporary_path, config_path)

    return config_path


def build_windowed_index(
    papers: Sequence[PaperDocument],
    embedder: TextEmbedder,
    store: WindowedIndexStore,
    chunk_token_counter: TokenCounter,
    max_chunk_tokens: int = 512,
    overlap_tokens: int = 32,
    batch_size: int = 32,
    language: str = "en",
    show_progress_bar: bool = True,
    source_label: str | None = None,
) -> WindowedIndexBuildReport:
    """Build, persist, reload, and verify a sentence-aware index.

    Pipeline:
        PaperDocument
        -> SentenceTokenChunk
        -> EmbeddingWindow
        -> Embeddings
        -> WindowedIndexStore
        -> WindowedDenseRetriever

    The chunk token counter is independent of the embedding tokenizer.
    This allows a future LLM tokenizer to be injected.

    Oversized sentences remain intact in their parent chunks.
    Their embedding windows may contain smaller token ranges.
    """

    if not papers:
        raise ValueError("Cannot build an index from empty papers.")

    if max_chunk_tokens <= 0:
        raise ValueError("max_chunk_tokens must be positive.")

    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if not isinstance(language, str) or not language.strip():
        raise ValueError("Language cannot be empty.")

    if not callable(getattr(chunk_token_counter, "count", None)):
        raise TypeError("Token counter must implement count().")

    if store.index_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing index: {store.index_dir}"
        )

    paper_ids = [paper.paper_id for paper in papers]

    if any(not paper_id.strip() for paper_id in paper_ids):
        raise ValueError("Paper IDs cannot be empty.")

    if len(paper_ids) != len(set(paper_ids)):
        raise ValueError("Duplicate paper IDs detected.")

    model = embedder.model
    tokenizer = getattr(model, "tokenizer", None)

    if tokenizer is None:
        raise TypeError("Embedding model must expose its tokenizer.")

    max_sequence_length = embedder.max_seq_length
    embedding_dimension = embedder.dimension
    model_name = embedder.model_name

    if max_sequence_length <= 0:
        raise ValueError("Embedding model maximum sequence length is invalid.")

    if embedding_dimension <= 0:
        raise ValueError("Embedding dimension must be positive.")

    if not model_name.strip():
        raise ValueError("Embedding model name cannot be empty.")

    # --------------------------------------------------
    # Step 1: Sentence-aware chunking
    # --------------------------------------------------

    print("\n========== Step 1: Sentence Chunking ==========")

    chunks: list[SentenceTokenChunk] = []
    empty_paper_ids: list[str] = []

    iterator = tqdm(
        papers,
        desc="Chunking papers",
        total=len(papers),
        disable=not show_progress_bar,
    )

    for paper in iterator:
        paper_chunks = chunk_paper_by_sentence(
            paper=paper,
            token_counter=chunk_token_counter,
            max_tokens=max_chunk_tokens,
            language=language,
        )

        if not paper_chunks:
            empty_paper_ids.append(paper.paper_id)

        chunks.extend(paper_chunks)

    if not chunks:
        raise ValueError(f"No valid chunks generated. Empty papers: {empty_paper_ids}")

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    oversize_chunk_ids = tuple(
        chunk.chunk_id for chunk in chunks if chunk.oversize_sentence
    )

    print("Papers:", len(papers))
    print("Empty papers:", len(empty_paper_ids))
    print("Parent chunks:", len(chunks))
    print("Oversized sentence chunks:", len(oversize_chunk_ids))

    # --------------------------------------------------
    # Step 2: Build embedding windows
    # --------------------------------------------------

    print("\n========== Step 2: Building Embedding Windows ==========")

    windows = build_embedding_windows(
        chunks=chunks,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        overlap_tokens=overlap_tokens,
        show_progress_bar=show_progress_bar,
    )

    if not windows:
        raise ValueError("No embedding windows generated.")

    window_ids = [window.window_id for window in windows]

    if len(window_ids) != len(set(window_ids)):
        raise ValueError("Duplicate embedding window IDs detected.")

    print("Parent chunks:", len(chunks))
    print("Embedding windows:", len(windows))
    print("Maximum sequence length:", max_sequence_length)
    print("Window overlap:", overlap_tokens)

    # --------------------------------------------------
    # Step 3: Generate embeddings
    # --------------------------------------------------

    print("\n========== Step 3: Generating Embeddings ==========")

    embeddings = embedder.encode_windows(
        windows=windows,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    expected_shape = (
        len(windows),
        embedding_dimension,
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Embedding shape mismatch: "
            f"expected {expected_shape}, got {embeddings.shape}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid values.")

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Embeddings are not L2 normalized.")

    print("Embedding shape:", embeddings.shape)
    print("Embedding dtype:", embeddings.dtype)
    print(
        "Embedding memory:",
        round(embeddings.nbytes / (1024**2), 2),
        "MiB",
    )

    # --------------------------------------------------
    # Step 4: Persist the index
    # --------------------------------------------------

    print("\n========== Step 4: Saving Index ==========")

    store.save(
        chunks=chunks,
        windows=windows,
        embeddings=embeddings,
        model_name=model_name,
        max_sequence_length=max_sequence_length,
        overlap_tokens=overlap_tokens,
        show_progress_bar=show_progress_bar,
    )

    # --------------------------------------------------
    # Step 5: Reload and verify the index
    # --------------------------------------------------

    print("\n========== Step 5: Verifying Persisted Index ==========")

    (
        restored_chunks,
        restored_windows,
        restored_embeddings,
        manifest,
    ) = store.load(
        expected_model_name=model_name,
        show_progress_bar=show_progress_bar,
    )

    if restored_chunks != chunks:
        raise ValueError("Restored parent chunks do not match.")

    if restored_windows != windows:
        raise ValueError("Restored embedding windows do not match.")

    if restored_embeddings.shape != expected_shape:
        raise ValueError("Restored embedding shape does not match.")

    # Verify that the persisted index can be consumed by the retriever.
    retriever = WindowedDenseRetriever(embedder)

    retriever.load_index(
        chunks=restored_chunks,
        windows=restored_windows,
        embeddings=restored_embeddings,
    )

    if len(retriever.chunks) != len(chunks):
        raise ValueError("Retriever parent chunk count mismatch.")

    if len(retriever.windows) != len(windows):
        raise ValueError("Retriever window count mismatch.")

    # --------------------------------------------------
    # Step 6: Save reproducibility configuration
    # --------------------------------------------------

    print("\n========== Step 6: Saving Build Configuration ==========")

    counter_name = (
        getattr(chunk_token_counter, "model_name", None)
        or getattr(chunk_token_counter, "name_or_path", None)
        or type(chunk_token_counter).__name__
    )

    build_config = {
        "schema_version": 1,
        "chunking_strategy": "sentence",
        "max_chunk_tokens": max_chunk_tokens,
        "language": language,
        "chunk_token_counter": str(counter_name),
        "embedding_model": model_name,
        "embedding_dimension": embedding_dimension,
        "max_sequence_length": max_sequence_length,
        "overlap_tokens": overlap_tokens,
        "batch_size": batch_size,
        "paper_count": len(papers),
        "chunk_count": len(chunks),
        "window_count": len(windows),
        "oversize_chunk_count": len(oversize_chunk_ids),
        "source": source_label,
    }

    config_path = _write_build_config(
        store.index_dir,
        build_config,
    )

    print("Build configuration:", config_path)

    # --------------------------------------------------
    # Final report
    # --------------------------------------------------

    print("\n========== Windowed Index Build Completed ==========")

    print("Papers:", len(papers))
    print("Parent chunks:", len(restored_chunks))
    print("Embedding windows:", len(restored_windows))
    print("Embedding shape:", restored_embeddings.shape)
    print("Oversized sentence chunks:", len(oversize_chunk_ids))
    print("Index directory:", store.index_dir)

    return WindowedIndexBuildReport(
        paper_count=len(papers),
        chunk_count=len(restored_chunks),
        window_count=len(restored_windows),
        oversize_chunk_ids=oversize_chunk_ids,
        empty_paper_ids=tuple(empty_paper_ids),
        embedding_shape=tuple(restored_embeddings.shape),
        index_dir=store.index_dir,
        build_config_path=config_path,
        manifest=manifest,
    )
