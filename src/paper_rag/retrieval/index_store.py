import json
import os
from pathlib import Path

import numpy as np

from paper_rag.schemas import SectionChunk


class IndexStore:
    """Persist and restore dense retrieval indexes."""

    SCHEMA_VERSION = 1

    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir)

        self.embeddings_path = self.index_dir / "embeddings.npy"
        self.chunks_path = self.index_dir / "chunks.jsonl"
        self.manifest_path = self.index_dir / "manifest.json"

    def exists(self) -> bool:
        """Check whether all index files exist."""

        return all(
            path.is_file()
            for path in (
                self.embeddings_path,
                self.chunks_path,
                self.manifest_path,
            )
        )

    def save(
        self,
        chunks: list[SectionChunk],
        embeddings: np.ndarray,
        model_name: str,
    ) -> None:
        """Save chunks, embeddings, and index metadata."""

        if not chunks:
            raise ValueError("Cannot save an empty index.")

        if not model_name.strip():
            raise ValueError("Model name cannot be empty.")

        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if vectors.shape[0] != len(chunks):
            raise ValueError("Chunk count and embedding count do not match.")

        if vectors.shape[1] == 0:
            raise ValueError("Embedding dimension cannot be zero.")

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain invalid values.")

        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Temporary files prevent incomplete writes from being
        # mistaken for a complete index.
        embeddings_tmp = self.embeddings_path.with_name("embeddings.npy.tmp")
        chunks_tmp = self.chunks_path.with_name("chunks.jsonl.tmp")
        manifest_tmp = self.manifest_path.with_name("manifest.json.tmp")

        # 1. Save embeddings.
        with embeddings_tmp.open("wb") as file:
            np.save(file, vectors, allow_pickle=False)

        # 2. Save chunks.
        with chunks_tmp.open("w", encoding="utf-8") as file:
            for chunk in chunks:
                record = chunk.model_dump(mode="json")
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 3. Save manifest.
        manifest = {
            "schema_version": self.SCHEMA_VERSION,
            "model_name": model_name,
            "num_chunks": len(chunks),
            "embedding_dimension": int(vectors.shape[1]),
            "dtype": str(vectors.dtype),
            "normalization": "l2",
        }

        with manifest_tmp.open("w", encoding="utf-8") as file:
            json.dump(
                manifest,
                file,
                ensure_ascii=False,
                indent=2,
            )

        # Write manifest last, so incomplete indexes are not accepted.
        self.manifest_path.unlink(missing_ok=True)

        os.replace(embeddings_tmp, self.embeddings_path)
        os.replace(chunks_tmp, self.chunks_path)
        os.replace(manifest_tmp, self.manifest_path)

    def load(
        self,
        expected_model_name: str | None = None,
    ) -> tuple[list[SectionChunk], np.ndarray, dict]:
        """Load and validate a persisted index."""

        if not self.exists():
            raise FileNotFoundError(f"Index files not found in {self.index_dir}")

        # 1. Load manifest.
        with self.manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)

        if manifest.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("Unsupported index schema version.")

        if (
            expected_model_name is not None
            and manifest.get("model_name") != expected_model_name
        ):
            raise ValueError("Embedding model does not match saved index.")

        # 2. Load embeddings.
        embeddings = np.load(
            self.embeddings_path,
            allow_pickle=False,
        )

        # 3. Load chunks.
        chunks = []

        with self.chunks_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    record = json.loads(line)
                    chunks.append(SectionChunk.model_validate(record))

        # 4. Validate consistency.
        if embeddings.ndim != 2:
            raise ValueError("Saved embeddings must be 2D.")

        if embeddings.shape[0] != len(chunks):
            raise ValueError("Saved chunk and embedding counts do not match.")

        if len(chunks) != manifest["num_chunks"]:
            raise ValueError("Chunk count does not match manifest.")

        if embeddings.shape[1] != manifest["embedding_dimension"]:
            raise ValueError("Embedding dimension does not match manifest.")

        if not np.isfinite(embeddings).all():
            raise ValueError("Saved embeddings contain invalid values.")

        return chunks, embeddings, manifest
