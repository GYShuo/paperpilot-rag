import json
import os
from itertools import pairwise
from pathlib import Path
from typing import TypeVar

import numpy as np
from pydantic import BaseModel
from tqdm import tqdm

from paper_rag.schemas import EmbeddingWindow, SentenceTokenChunk

ModelT = TypeVar("ModelT", bound=BaseModel)


class WindowedIndexStore:
    """Persist and restore window-based dense retrieval indexes."""

    SCHEMA_VERSION = 1
    INDEX_TYPE = "windowed"

    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir)

        self.embeddings_path = self.index_dir / "embeddings.npy"
        self.chunks_path = self.index_dir / "chunks.jsonl"
        self.windows_path = self.index_dir / "windows.jsonl"
        self.manifest_path = self.index_dir / "manifest.json"

    def exists(self) -> bool:
        """Check whether all index files exist."""

        return all(
            path.is_file()
            for path in (
                self.embeddings_path,
                self.chunks_path,
                self.windows_path,
                self.manifest_path,
            )
        )

    @staticmethod
    def _validate_config(
        max_sequence_length: int | None,
        overlap_tokens: int | None,
    ) -> None:
        """Validate optional window configuration metadata."""

        if max_sequence_length is not None and max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive.")

        if overlap_tokens is not None and overlap_tokens < 0:
            raise ValueError("overlap_tokens cannot be negative.")

    @staticmethod
    def _validate_chunks_and_windows(
        chunks: list[SentenceTokenChunk],
        windows: list[EmbeddingWindow],
        max_sequence_length: int | None = None,
    ) -> None:
        """Validate parent chunks and their embedding window mappings."""

        if not chunks:
            raise ValueError("Cannot save an empty chunk collection.")

        if not windows:
            raise ValueError("Cannot save an empty window collection.")

        chunk_by_id = {}

        for chunk in chunks:
            if chunk.chunk_id in chunk_by_id:
                raise ValueError(f"Duplicate chunk ID: {chunk.chunk_id}")

            if not chunk.text.strip():
                raise ValueError(f"Parent chunk {chunk.chunk_id} contains empty text.")

            chunk_by_id[chunk.chunk_id] = chunk

        window_ids = set()
        windows_by_parent: dict[str, list[EmbeddingWindow]] = {}

        for window in windows:
            if window.window_id in window_ids:
                raise ValueError(f"Duplicate window ID: {window.window_id}")

            window_ids.add(window.window_id)

            parent = chunk_by_id.get(window.parent_chunk_id)

            if parent is None:
                raise ValueError(
                    f"Window {window.window_id} references "
                    f"unknown parent chunk {window.parent_chunk_id}."
                )

            if (
                window.paper_id != parent.paper_id
                or window.split != parent.split
                or window.section_index != parent.section_index
                or window.section_name != parent.section_name
            ):
                raise ValueError(f"Metadata mismatch for window {window.window_id}.")

            if not 0 <= window.start_char < window.end_char <= len(parent.text):
                raise ValueError(
                    f"Invalid character offsets for window {window.window_id}."
                )

            expected_text = parent.text[window.start_char : window.end_char]

            if expected_text != window.text:
                raise ValueError(
                    f"Window text does not match parent chunk: {window.window_id}."
                )

            if not window.text.strip():
                raise ValueError(f"Window {window.window_id} contains empty text.")

            if (
                window.token_start < 0
                or window.token_end <= window.token_start
                or window.token_count <= 0
            ):
                raise ValueError(
                    f"Invalid token metadata for window {window.window_id}."
                )

            if (
                max_sequence_length is not None
                and window.token_count > max_sequence_length
            ):
                raise ValueError(
                    f"Window {window.window_id} exceeds maximum "
                    f"sequence length: {window.token_count} > "
                    f"{max_sequence_length}."
                )

            windows_by_parent.setdefault(
                window.parent_chunk_id,
                [],
            ).append(window)

        if set(windows_by_parent) != set(chunk_by_id):
            raise ValueError(
                "Every parent chunk must have at least one embedding window."
            )

        for chunk_id, parent_windows in windows_by_parent.items():
            parent = chunk_by_id[chunk_id]

            parent_windows.sort(key=lambda item: item.window_index)

            indices = [window.window_index for window in parent_windows]

            if indices != list(range(len(parent_windows))):
                raise ValueError(
                    f"Window indices are not sequential for chunk {chunk_id}."
                )

            first = parent_windows[0]

            if first.token_start != 0:
                raise ValueError(
                    f"Token coverage does not start at zero for {chunk_id}."
                )

            if first.start_char != 0:
                raise ValueError(
                    f"Character coverage does not start at zero for {chunk_id}."
                )

            for previous, current in pairwise(parent_windows):
                # Overlap is allowed, but token gaps are not.
                if current.token_start > previous.token_end:
                    raise ValueError(
                        f"Token coverage gap detected in chunk {chunk_id}."
                    )

                if current.token_end <= previous.token_end:
                    raise ValueError(
                        f"Windows do not make forward progress in {chunk_id}."
                    )

                if (
                    current.start_char < previous.start_char
                    or current.end_char <= previous.end_char
                ):
                    raise ValueError(f"Invalid character ordering in chunk {chunk_id}.")

            last = parent_windows[-1]

            if last.end_char != len(parent.text):
                raise ValueError(
                    f"Character coverage does not reach the end "
                    f"of parent chunk {chunk_id}."
                )

    @staticmethod
    def _validate_embeddings(
        embeddings: np.ndarray,
        expected_count: int,
    ) -> np.ndarray:
        """Validate embedding matrix shape and values."""

        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if vectors.shape[0] != expected_count:
            raise ValueError("Window count and embedding count do not match.")

        if vectors.shape[1] == 0:
            raise ValueError("Embedding dimension cannot be zero.")

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain invalid values.")

        return vectors

    @staticmethod
    def _read_jsonl(
        path: Path,
        model_type: type[ModelT],
        description: str,
        show_progress_bar: bool,
    ) -> list[ModelT]:
        """Read and validate JSONL records."""

        records: list[ModelT] = []

        with path.open("r", encoding="utf-8") as file:
            iterator = tqdm(
                file,
                desc=description,
                disable=not show_progress_bar,
            )

            for line_number, line in enumerate(iterator, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {path.name}, line {line_number}."
                    ) from exc

                records.append(model_type.model_validate(record))

        return records

    def save(
        self,
        chunks: list[SentenceTokenChunk],
        windows: list[EmbeddingWindow],
        embeddings: np.ndarray,
        model_name: str,
        max_sequence_length: int | None = None,
        overlap_tokens: int | None = None,
        show_progress_bar: bool = True,
    ) -> None:
        """Save parent chunks, windows, embeddings, and metadata."""

        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Model name cannot be empty.")

        self._validate_config(
            max_sequence_length,
            overlap_tokens,
        )

        self._validate_chunks_and_windows(
            chunks,
            windows,
            max_sequence_length=max_sequence_length,
        )

        vectors = self._validate_embeddings(
            embeddings,
            expected_count=len(windows),
        )

        self.index_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        embeddings_tmp = self.embeddings_path.with_name("embeddings.npy.tmp")
        chunks_tmp = self.chunks_path.with_name("chunks.jsonl.tmp")
        windows_tmp = self.windows_path.with_name("windows.jsonl.tmp")
        manifest_tmp = self.manifest_path.with_name("manifest.json.tmp")

        try:
            with embeddings_tmp.open("wb") as file:
                np.save(
                    file,
                    vectors,
                    allow_pickle=False,
                )

            with chunks_tmp.open("w", encoding="utf-8") as file:
                iterator = tqdm(
                    chunks,
                    desc="Saving parent chunks",
                    disable=not show_progress_bar,
                )

                for chunk in iterator:
                    record = chunk.model_dump(mode="json")

                    file.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            with windows_tmp.open("w", encoding="utf-8") as file:
                iterator = tqdm(
                    windows,
                    desc="Saving embedding windows",
                    disable=not show_progress_bar,
                )

                for window in iterator:
                    record = window.model_dump(mode="json")

                    file.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            manifest = {
                "schema_version": self.SCHEMA_VERSION,
                "index_type": self.INDEX_TYPE,
                "model_name": model_name,
                "num_chunks": len(chunks),
                "num_windows": len(windows),
                "embedding_dimension": int(vectors.shape[1]),
                "dtype": str(vectors.dtype),
                "normalization": "l2",
                "max_sequence_length": max_sequence_length,
                "overlap_tokens": overlap_tokens,
            }

            with manifest_tmp.open("w", encoding="utf-8") as file:
                json.dump(
                    manifest,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            self.manifest_path.unlink(missing_ok=True)

            os.replace(embeddings_tmp, self.embeddings_path)
            os.replace(chunks_tmp, self.chunks_path)
            os.replace(windows_tmp, self.windows_path)
            os.replace(manifest_tmp, self.manifest_path)

        finally:
            for path in (
                embeddings_tmp,
                chunks_tmp,
                windows_tmp,
                manifest_tmp,
            ):
                path.unlink(missing_ok=True)

    def load(
        self,
        expected_model_name: str | None = None,
        show_progress_bar: bool = True,
    ) -> tuple[
        list[SentenceTokenChunk],
        list[EmbeddingWindow],
        np.ndarray,
        dict,
    ]:
        """Load and validate a persisted windowed index."""

        if not self.exists():
            raise FileNotFoundError(f"Index files not found in {self.index_dir}")

        with self.manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)

        if not isinstance(manifest, dict):
            raise TypeError("Index manifest must be a dictionary.")

        if manifest.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("Unsupported index schema version.")

        if manifest.get("index_type") != self.INDEX_TYPE:
            raise ValueError("Index type does not match WindowedIndexStore.")

        if (
            expected_model_name is not None
            and manifest.get("model_name") != expected_model_name
        ):
            raise ValueError("Embedding model does not match saved index.")

        max_sequence_length = manifest.get("max_sequence_length")
        overlap_tokens = manifest.get("overlap_tokens")

        self._validate_config(
            max_sequence_length,
            overlap_tokens,
        )

        chunks = self._read_jsonl(
            self.chunks_path,
            SentenceTokenChunk,
            description="Loading parent chunks",
            show_progress_bar=show_progress_bar,
        )

        windows = self._read_jsonl(
            self.windows_path,
            EmbeddingWindow,
            description="Loading embedding windows",
            show_progress_bar=show_progress_bar,
        )

        embeddings = np.load(
            self.embeddings_path,
            allow_pickle=False,
        )

        if len(chunks) != manifest.get("num_chunks"):
            raise ValueError("Chunk count does not match manifest.")

        if len(windows) != manifest.get("num_windows"):
            raise ValueError("Window count does not match manifest.")

        if embeddings.ndim != 2:
            raise ValueError("Saved embeddings must be 2D.")

        if embeddings.dtype != np.float32:
            raise ValueError("Saved embeddings must be float32.")

        if embeddings.shape[1] != manifest.get("embedding_dimension"):
            raise ValueError("Embedding dimension does not match manifest.")

        if str(embeddings.dtype) != manifest.get("dtype"):
            raise ValueError("Embedding dtype does not match manifest.")

        if manifest.get("normalization") != "l2":
            raise ValueError("Unsupported embedding normalization.")

        self._validate_chunks_and_windows(
            chunks,
            windows,
            max_sequence_length=max_sequence_length,
        )

        embeddings = self._validate_embeddings(
            embeddings,
            expected_count=len(windows),
        )

        return chunks, windows, embeddings, manifest
