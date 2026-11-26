from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import sentencesplit
from tqdm import tqdm

from paper_rag.data.sentence_token_chunker import (
    TokenCounter,
    split_paragraph_into_sentences,
)
from paper_rag.retrieval.embedder import TextEmbedder
from paper_rag.retrieval.windowed_index_store import WindowedIndexStore
from paper_rag.schemas import (
    EmbeddingWindow,
    PaperDocument,
    SentenceTokenChunk,
)


@dataclass
class AuditCollector:
    """Collect audit errors while allowing the remaining checks to continue."""

    counts: Counter[str] = field(default_factory=Counter)
    samples: list[dict[str, Any]] = field(default_factory=list)
    max_samples: int = 100

    def add(self, code: str, message: str, **context: Any) -> None:
        self.counts[code] += 1
        if len(self.samples) < self.max_samples:
            self.samples.append({"code": code, "message": message, "context": context})

    @property
    def error_count(self) -> int:
        return sum(self.counts.values())


def compare_sentence_spans(
    expected: list[tuple[int, int, str]],
    actual: list[tuple[int, int, str]],
) -> tuple[list[tuple[int, int, str]], list[tuple[int, int, str]]]:
    """Return missing and unexpected spans, preserving duplicate occurrences."""

    expected_counts = Counter(expected)
    actual_counts = Counter(actual)
    missing = list((expected_counts - actual_counts).elements())
    unexpected = list((actual_counts - expected_counts).elements())
    return missing, unexpected


def validate_window_coverage(
    windows: list[EmbeddingWindow],
    total_tokens: int,
    total_chars: int,
    overlap_tokens: int,
) -> list[str]:
    """Validate complete, gap-free token and character coverage for one parent."""

    if total_tokens <= 0 or total_chars <= 0:
        return ["Parent chunk has no token or character content."]
    if overlap_tokens < 0:
        return ["Configured overlap cannot be negative."]
    if not windows:
        return ["No embedding windows found."]

    errors: list[str] = []
    ordered = sorted(windows, key=lambda item: item.window_index)
    indices = [window.window_index for window in ordered]

    if indices != list(range(len(ordered))):
        errors.append("Window indices are not sequential.")

    for window in ordered:
        if not 0 <= window.token_start < window.token_end <= total_tokens:
            errors.append("Window token offsets are outside the parent token range.")
        if not 0 <= window.start_char < window.end_char <= total_chars:
            errors.append("Window character offsets are outside the parent text.")

    first = ordered[0]
    if first.token_start != 0:
        errors.append("Token coverage does not start at zero.")
    if first.start_char != 0:
        errors.append("Character coverage does not start at zero.")

    for previous, current in pairwise(ordered):
        if current.token_start > previous.token_end:
            errors.append("Token coverage contains a gap.")
        if current.token_end <= previous.token_end:
            errors.append("Windows do not make forward progress in token space.")

        expected_start = previous.token_end - overlap_tokens
        if current.token_start != expected_start:
            errors.append("Window overlap differs from the configured overlap.")

        if current.start_char > previous.end_char:
            errors.append("Character coverage contains a gap.")
        if current.end_char <= previous.end_char:
            errors.append("Windows do not make forward progress in character space.")

    last = ordered[-1]
    if last.token_end != total_tokens:
        errors.append("Token coverage does not reach the parent end.")
    if last.end_char != total_chars:
        errors.append("Character coverage does not reach the parent end.")

    return errors


def _distribution(values: list[int]) -> dict[str, int | float | None]:
    """Return a compact descriptive distribution for integer measurements."""

    if not values:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }

    array = np.asarray(values, dtype=np.int64)
    return {
        "count": int(array.size),
        "min": int(array.min()),
        "mean": round(float(array.mean()), 2),
        "median": float(np.median(array)),
        "p90": round(float(np.percentile(array, 90)), 2),
        "p95": round(float(np.percentile(array, 95)), 2),
        "p99": round(float(np.percentile(array, 99)), 2),
        "max": int(array.max()),
    }


def audit_source_chunks(
    papers: list[PaperDocument],
    chunks: list[SentenceTokenChunk],
    token_counter: TokenCounter,
    language: str,
    max_chunk_tokens: int,
    collector: AuditCollector,
    show_progress_bar: bool = True,
) -> dict[str, Any]:
    """Audit sentence spans, source reconstruction, and parent chunk budgets."""

    if max_chunk_tokens <= 0:
        raise ValueError("max_chunk_tokens must be positive.")

    papers_by_id: dict[str, PaperDocument] = {}
    sections_by_key: dict[tuple[str, int], Any] = {}
    paragraphs_by_key: dict[tuple[str, int, int], Any] = {}
    paragraph_order: dict[tuple[str, int, int], tuple[int, int]] = {}
    paragraph_items: list[tuple[tuple[str, int, int], Any]] = []
    empty_paper_ids: list[str] = []

    for paper in papers:
        if paper.paper_id in papers_by_id:
            collector.add(
                "duplicate_paper_id",
                "Source data contains a duplicate paper ID.",
                paper_id=paper.paper_id,
            )
            continue

        papers_by_id[paper.paper_id] = paper
        has_body_text = False

        for section_position, section in enumerate(paper.sections):
            section_key = (paper.paper_id, section.section_index)
            if section_key in sections_by_key:
                collector.add(
                    "duplicate_section_key",
                    "A paper contains duplicate section indices.",
                    paper_id=paper.paper_id,
                    section_index=section.section_index,
                )
                continue
            sections_by_key[section_key] = section

            for paragraph_position, paragraph in enumerate(section.paragraphs):
                paragraph_key = (
                    paper.paper_id,
                    section.section_index,
                    paragraph.paragraph_index,
                )
                if paragraph_key in paragraphs_by_key:
                    collector.add(
                        "duplicate_paragraph_key",
                        "A section contains duplicate paragraph indices.",
                        paper_id=paper.paper_id,
                        section_index=section.section_index,
                        paragraph_index=paragraph.paragraph_index,
                    )
                    continue

                paragraphs_by_key[paragraph_key] = paragraph
                paragraph_order[paragraph_key] = (
                    section_position,
                    paragraph_position,
                )
                paragraph_items.append((paragraph_key, paragraph))
                has_body_text = has_body_text or bool(paragraph.text.strip())

        if not has_body_text:
            empty_paper_ids.append(paper.paper_id)

    segmenter = sentencesplit.Segmenter(
        language=language,
        clean=False,
        split_mode="conservative",
    )

    expected_by_paragraph: dict[tuple[str, int, int], list[tuple[int, int, str]]] = {}
    sentence_token_counts: list[int] = []
    oversized_sentence_samples: list[dict[str, Any]] = []

    for paragraph_key, paragraph in tqdm(
        paragraph_items,
        desc="Auditing source sentences",
        disable=not show_progress_bar,
    ):
        if not paragraph.text.strip():
            expected_by_paragraph[paragraph_key] = []
            continue

        sentence_units = split_paragraph_into_sentences(
            text=paragraph.text,
            paragraph_index=paragraph.paragraph_index,
            segmenter=segmenter,
        )
        reconstructed = "".join(unit.text for unit in sentence_units)
        if reconstructed != paragraph.text:
            collector.add(
                "segmenter_source_reconstruction_failure",
                "Sentence segmentation did not reconstruct the source paragraph.",
                paper_id=paragraph_key[0],
                section_index=paragraph_key[1],
                paragraph_index=paragraph_key[2],
            )

        expected_by_paragraph[paragraph_key] = [
            (unit.start_char, unit.end_char, unit.text) for unit in sentence_units
        ]

        for unit in sentence_units:
            token_count = token_counter.count(unit.text)
            sentence_token_counts.append(token_count)
            if token_count > max_chunk_tokens:
                oversized_sentence_samples.append(
                    {
                        "paper_id": paragraph_key[0],
                        "section_index": paragraph_key[1],
                        "paragraph_index": paragraph_key[2],
                        "token_count": token_count,
                        "character_count": len(unit.text),
                        "text_preview": unit.text[:240],
                    }
                )

    observed_by_paragraph: dict[tuple[str, int, int], list[tuple[int, int, str]]] = (
        defaultdict(list)
    )
    chunk_ids: set[str] = set()
    chunk_token_counts: list[int] = []
    oversized_chunk_count = 0

    for chunk in tqdm(
        chunks,
        desc="Auditing parent chunks",
        disable=not show_progress_bar,
    ):
        if chunk.chunk_id in chunk_ids:
            collector.add(
                "duplicate_chunk_id",
                "Index contains a duplicate parent chunk ID.",
                chunk_id=chunk.chunk_id,
            )
            continue
        chunk_ids.add(chunk.chunk_id)

        paper = papers_by_id.get(chunk.paper_id)
        if paper is None:
            collector.add(
                "unknown_chunk_paper",
                "Chunk references a paper absent from the source dataset.",
                chunk_id=chunk.chunk_id,
                paper_id=chunk.paper_id,
            )
            continue
        if chunk.split != paper.split:
            collector.add(
                "chunk_split_mismatch",
                "Chunk split differs from its source paper split.",
                chunk_id=chunk.chunk_id,
            )

        section = sections_by_key.get((chunk.paper_id, chunk.section_index))
        if section is None:
            collector.add(
                "unknown_chunk_section",
                "Chunk references an unknown source section.",
                chunk_id=chunk.chunk_id,
            )
        elif chunk.section_name != section.section_name:
            collector.add(
                "section_name_mismatch",
                "Chunk section name differs from the source section.",
                chunk_id=chunk.chunk_id,
            )

        if not chunk.spans:
            collector.add(
                "chunk_without_spans",
                "Parent chunk contains no source spans.",
                chunk_id=chunk.chunk_id,
            )
            continue

        if chunk.sentence_count != len(chunk.spans):
            collector.add(
                "sentence_count_mismatch",
                "Stored sentence_count differs from the number of spans.",
                chunk_id=chunk.chunk_id,
                stored=chunk.sentence_count,
                actual=len(chunk.spans),
            )

        previous_span = None
        previous_paragraph_key = None

        for span_index, span in enumerate(chunk.spans):
            paragraph_key = (
                chunk.paper_id,
                chunk.section_index,
                span.paragraph_index,
            )
            paragraph = paragraphs_by_key.get(paragraph_key)
            if paragraph is None:
                collector.add(
                    "unknown_source_paragraph",
                    "Chunk span references an unknown source paragraph.",
                    chunk_id=chunk.chunk_id,
                    paragraph_index=span.paragraph_index,
                )
                previous_span = span
                previous_paragraph_key = paragraph_key
                continue

            source_offsets_valid = (
                0 <= span.start_char < span.end_char <= len(paragraph.text)
            )
            chunk_offsets_valid = (
                0 <= span.chunk_start < span.chunk_end <= len(chunk.text)
            )

            if not source_offsets_valid:
                collector.add(
                    "invalid_source_offsets",
                    "Span offsets are outside the source paragraph.",
                    chunk_id=chunk.chunk_id,
                    paragraph_index=span.paragraph_index,
                    offsets=(span.start_char, span.end_char),
                )
            else:
                source_text = paragraph.text[span.start_char : span.end_char]
                observed_by_paragraph[paragraph_key].append(
                    (span.start_char, span.end_char, source_text)
                )

            if not chunk_offsets_valid:
                collector.add(
                    "invalid_chunk_offsets",
                    "Span offsets are outside the parent chunk text.",
                    chunk_id=chunk.chunk_id,
                    offsets=(span.chunk_start, span.chunk_end),
                )
            elif source_offsets_valid:
                chunk_text = chunk.text[span.chunk_start : span.chunk_end]
                if chunk_text != source_text:
                    collector.add(
                        "source_text_mismatch",
                        "Chunk span text differs from the original source text.",
                        chunk_id=chunk.chunk_id,
                        paragraph_index=span.paragraph_index,
                        source_preview=source_text[:120],
                        chunk_preview=chunk_text[:120],
                    )

            if span_index == 0 and span.chunk_start != 0:
                collector.add(
                    "chunk_leading_gap",
                    "First span does not begin at parent chunk offset zero.",
                    chunk_id=chunk.chunk_id,
                )
            elif previous_span is not None:
                same_paragraph = previous_paragraph_key == paragraph_key
                separator = "" if same_paragraph else "\n\n"
                expected_chunk_start = previous_span.chunk_end + len(separator)

                if span.chunk_start != expected_chunk_start:
                    collector.add(
                        "chunk_span_layout_mismatch",
                        "Chunk spans are not contiguous with the expected separator.",
                        chunk_id=chunk.chunk_id,
                        expected_start=expected_chunk_start,
                        actual_start=span.chunk_start,
                    )
                elif (
                    chunk.text[previous_span.chunk_end : span.chunk_start] != separator
                ):
                    collector.add(
                        "invalid_chunk_separator",
                        "Text between spans differs from the expected separator.",
                        chunk_id=chunk.chunk_id,
                        expected_separator=separator,
                    )

                previous_order = paragraph_order.get(previous_paragraph_key)
                current_order = paragraph_order.get(paragraph_key)
                if (
                    previous_order is not None
                    and current_order is not None
                    and current_order < previous_order
                ):
                    collector.add(
                        "source_order_violation",
                        "Chunk spans are not in original paragraph order.",
                        chunk_id=chunk.chunk_id,
                    )
                elif (
                    previous_order is not None
                    and current_order is not None
                    and current_order == previous_order
                    and span.start_char < previous_span.end_char
                ):
                    collector.add(
                        "source_span_overlap",
                        "Sentence spans overlap or are out of order.",
                        chunk_id=chunk.chunk_id,
                    )

            previous_span = span
            previous_paragraph_key = paragraph_key

        if chunk.spans[-1].chunk_end != len(chunk.text):
            collector.add(
                "chunk_trailing_gap",
                "Last span does not reach the end of parent chunk text.",
                chunk_id=chunk.chunk_id,
            )

        actual_chunk_tokens = token_counter.count(chunk.text)
        chunk_token_counts.append(actual_chunk_tokens)
        if actual_chunk_tokens != chunk.token_count:
            collector.add(
                "chunk_token_count_mismatch",
                "Stored chunk token count differs from tokenizer output.",
                chunk_id=chunk.chunk_id,
                stored=chunk.token_count,
                actual=actual_chunk_tokens,
            )

        is_oversized = actual_chunk_tokens > max_chunk_tokens
        if chunk.oversize_sentence:
            oversized_chunk_count += 1
        if chunk.oversize_sentence != is_oversized:
            collector.add(
                "oversize_flag_mismatch",
                "Oversize flag differs from the actual chunk token count.",
                chunk_id=chunk.chunk_id,
                stored=chunk.oversize_sentence,
                actual=is_oversized,
            )
        if is_oversized and len(chunk.spans) != 1:
            collector.add(
                "oversized_chunk_not_standalone",
                "An oversized parent chunk must contain exactly one sentence span.",
                chunk_id=chunk.chunk_id,
                span_count=len(chunk.spans),
                token_count=actual_chunk_tokens,
            )

    for paragraph_key, expected in expected_by_paragraph.items():
        actual = observed_by_paragraph.get(paragraph_key, [])
        missing, unexpected = compare_sentence_spans(expected, actual)
        if missing:
            collector.add(
                "missing_sentence_spans",
                "One or more source sentences are absent from the parent chunks.",
                paper_id=paragraph_key[0],
                section_index=paragraph_key[1],
                paragraph_index=paragraph_key[2],
                count=len(missing),
                examples=missing[:3],
            )
        if unexpected:
            collector.add(
                "unexpected_sentence_spans",
                "Parent chunks contain duplicate or unexpected source spans.",
                paper_id=paragraph_key[0],
                section_index=paragraph_key[1],
                paragraph_index=paragraph_key[2],
                count=len(unexpected),
                examples=unexpected[:3],
            )

    oversized_sentence_samples.sort(
        key=lambda item: item["token_count"],
        reverse=True,
    )
    if len(oversized_sentence_samples) != oversized_chunk_count:
        collector.add(
            "oversized_sentence_count_mismatch",
            "Oversized source sentence count differs from oversized chunk count.",
            source_sentences=len(oversized_sentence_samples),
            oversized_chunks=oversized_chunk_count,
        )

    return {
        "paper_count": len(papers),
        "section_count": len(sections_by_key),
        "paragraph_count": len(paragraph_items),
        "nonempty_paragraph_count": sum(
            bool(paragraph.text.strip()) for _, paragraph in paragraph_items
        ),
        "sentence_count": len(sentence_token_counts),
        "parent_chunk_count": len(chunks),
        "empty_paper_ids": sorted(empty_paper_ids),
        "sentence_token_distribution": _distribution(sentence_token_counts),
        "parent_chunk_token_distribution": _distribution(chunk_token_counts),
        "oversized_sentence_count": len(oversized_sentence_samples),
        "oversized_chunk_count": oversized_chunk_count,
        "oversized_sentence_samples": oversized_sentence_samples[:10],
    }


def _to_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        value = list(value)
    return value


def _tokenize_parent(
    tokenizer: Any,
    text: str,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenize one parent without special tokens or truncation."""

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
    )
    token_ids = _to_list(encoded["input_ids"])
    raw_offsets = _to_list(encoded["offset_mapping"])

    if token_ids and isinstance(token_ids[0], list):
        raise TypeError("Expected one unbatched tokenizer input.")
    if raw_offsets and isinstance(raw_offsets[0], list) and len(raw_offsets[0]) != 2:
        raise TypeError("Expected one unbatched offset mapping.")

    offsets = [(int(start), int(end)) for start, end in raw_offsets]
    if len(token_ids) != len(offsets):
        raise ValueError("Token IDs and offset mapping lengths differ.")

    for start, end in offsets:
        if not 0 <= start < end <= len(text):
            raise ValueError(
                "Invalid tokenizer offset "
                f"({start}, {end}) for text length {len(text)}."
            )

    return [int(token_id) for token_id in token_ids], offsets


def audit_embedding_windows(
    chunks: list[SentenceTokenChunk],
    windows: list[EmbeddingWindow],
    embeddings: np.ndarray,
    embedder: TextEmbedder,
    manifest: dict[str, Any],
    collector: AuditCollector,
    show_progress_bar: bool = True,
) -> dict[str, Any]:
    """Audit window-to-parent mappings, token coverage, and saved vectors."""

    tokenizer = embedder.model.tokenizer
    if not getattr(tokenizer, "is_fast", False):
        collector.add(
            "fast_tokenizer_required",
            "A fast tokenizer is required to audit character offsets.",
        )
        return {
            "window_count": len(windows),
            "embedding_shape": list(embeddings.shape),
        }

    max_sequence_length = int(manifest["max_sequence_length"])
    overlap_tokens = int(manifest["overlap_tokens"])
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    windows_by_parent: dict[str, list[EmbeddingWindow]] = defaultdict(list)
    window_ids: set[str] = set()

    for window in windows:
        if window.window_id in window_ids:
            collector.add(
                "duplicate_window_id",
                "Index contains a duplicate embedding window ID.",
                window_id=window.window_id,
            )
            continue
        window_ids.add(window.window_id)

        parent = chunks_by_id.get(window.parent_chunk_id)
        if parent is None:
            collector.add(
                "unknown_window_parent",
                "Embedding window references an unknown parent chunk.",
                window_id=window.window_id,
                parent_chunk_id=window.parent_chunk_id,
            )
            continue

        if (
            window.paper_id != parent.paper_id
            or window.split != parent.split
            or window.section_index != parent.section_index
            or window.section_name != parent.section_name
        ):
            collector.add(
                "window_metadata_mismatch",
                "Embedding window metadata differs from its parent chunk.",
                window_id=window.window_id,
            )
        windows_by_parent[window.parent_chunk_id].append(window)

    window_counts = [len(windows_by_parent.get(chunk.chunk_id, [])) for chunk in chunks]
    parent_token_counts: list[int] = []
    window_token_counts: list[int] = []

    for chunk in tqdm(
        chunks,
        desc="Auditing embedding windows",
        disable=not show_progress_bar,
    ):
        try:
            token_ids, offsets = _tokenize_parent(tokenizer, chunk.text)
        except (KeyError, TypeError, ValueError) as exc:
            collector.add(
                "parent_tokenization_failure",
                str(exc),
                chunk_id=chunk.chunk_id,
            )
            continue

        total_tokens = len(token_ids)
        parent_token_counts.append(total_tokens)
        if total_tokens == 0:
            collector.add(
                "empty_parent_tokenization",
                "Parent chunk produced no non-special tokens.",
                chunk_id=chunk.chunk_id,
            )
            continue

        parent_windows = sorted(
            windows_by_parent.get(chunk.chunk_id, []),
            key=lambda item: item.window_index,
        )
        for message in validate_window_coverage(
            windows=parent_windows,
            total_tokens=total_tokens,
            total_chars=len(chunk.text),
            overlap_tokens=overlap_tokens,
        ):
            collector.add(
                "window_coverage_failure",
                message,
                chunk_id=chunk.chunk_id,
            )

        for window in parent_windows:
            if not 0 <= window.token_start < window.token_end <= total_tokens:
                continue

            expected_start_char = (
                0 if window.token_start == 0 else offsets[window.token_start][0]
            )
            expected_end_char = (
                len(chunk.text)
                if window.token_end == total_tokens
                else offsets[window.token_end - 1][1]
            )
            if (
                window.start_char != expected_start_char
                or window.end_char != expected_end_char
            ):
                collector.add(
                    "window_character_offset_mismatch",
                    "Stored window character offsets differ from tokenizer offsets.",
                    window_id=window.window_id,
                    stored=(window.start_char, window.end_char),
                    expected=(expected_start_char, expected_end_char),
                )

            if not 0 <= window.start_char < window.end_char <= len(chunk.text):
                continue
            expected_text = chunk.text[window.start_char : window.end_char]
            if window.text != expected_text:
                collector.add(
                    "window_text_mismatch",
                    "Window text differs from the corresponding parent substring.",
                    window_id=window.window_id,
                )

            actual_token_count = embedder.count(window.text)
            window_token_counts.append(actual_token_count)
            if actual_token_count != window.token_count:
                collector.add(
                    "window_token_count_mismatch",
                    "Stored window token count differs from tokenizer output.",
                    window_id=window.window_id,
                    stored=window.token_count,
                    actual=actual_token_count,
                )
            if actual_token_count > max_sequence_length:
                collector.add(
                    "window_exceeds_model_limit",
                    "Window input exceeds the embedding model maximum sequence length.",
                    window_id=window.window_id,
                    actual=actual_token_count,
                    maximum=max_sequence_length,
                )

    expected_embedding_shape = (len(windows), embedder.dimension)
    if embeddings.shape != expected_embedding_shape:
        collector.add(
            "embedding_shape_mismatch",
            "Embedding matrix shape differs from window count and model dimension.",
            actual=list(embeddings.shape),
            expected=list(expected_embedding_shape),
        )
    if embeddings.dtype != np.float32:
        collector.add(
            "embedding_dtype_mismatch",
            "Embedding matrix dtype is not float32.",
            actual=str(embeddings.dtype),
        )

    embedding_norms: dict[str, Any] = {
        "count": 0,
        "min": None,
        "mean": None,
        "max": None,
        "all_l2_normalized": False,
    }
    has_embedding_rows = embeddings.ndim == 2 and embeddings.shape[0] > 0
    embeddings_are_finite = has_embedding_rows and bool(np.isfinite(embeddings).all())
    if has_embedding_rows and not embeddings_are_finite:
        collector.add(
            "non_finite_embeddings",
            "Embedding matrix contains NaN or infinite values.",
        )
    elif embeddings_are_finite:
        norms = np.linalg.norm(embeddings, axis=1)
        normalized = bool(np.allclose(norms, 1.0, atol=1e-4))
        embedding_norms = {
            "count": int(norms.size),
            "min": round(float(norms.min()), 6),
            "mean": round(float(norms.mean()), 6),
            "max": round(float(norms.max()), 6),
            "all_l2_normalized": normalized,
        }
        if not normalized:
            collector.add(
                "embedding_normalization_failure",
                "One or more embedding vectors are not L2 normalized.",
            )

    return {
        "window_count": len(windows),
        "windows_per_parent_distribution": _distribution(window_counts),
        "parent_token_distribution": _distribution(parent_token_counts),
        "window_model_input_token_distribution": _distribution(window_token_counts),
        "maximum_window_tokens": max(window_token_counts, default=None),
        "model_max_sequence_length": max_sequence_length,
        "configured_overlap_tokens": overlap_tokens,
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "embedding_norms": embedding_norms,
    }


def audit_split(
    split: str,
    source_path: Path,
    index_dir: Path,
    embedder: TextEmbedder,
    papers: list[PaperDocument],
    show_progress_bar: bool = True,
) -> dict[str, Any]:
    """Run the read-only V2 source, chunk, window, and embedding audit."""

    collector = AuditCollector()
    store = WindowedIndexStore(index_dir)
    chunks, windows, embeddings, manifest = store.load(
        expected_model_name=embedder.model_name,
        show_progress_bar=show_progress_bar,
    )

    build_config_path = index_dir / "build_config.json"
    if not build_config_path.is_file():
        raise FileNotFoundError(build_config_path)
    with build_config_path.open("r", encoding="utf-8") as file:
        build_config = json.load(file)

    required_config_keys = (
        "chunking_strategy",
        "max_chunk_tokens",
        "language",
        "chunk_token_counter",
        "embedding_model",
        "embedding_dimension",
        "max_sequence_length",
        "overlap_tokens",
        "paper_count",
        "chunk_count",
        "window_count",
        "oversize_chunk_count",
    )
    missing_config_keys = [
        key for key in required_config_keys if key not in build_config
    ]
    if missing_config_keys:
        collector.add(
            "build_config_missing_fields",
            "Build configuration is missing required fields.",
            fields=missing_config_keys,
        )

    if build_config.get("chunking_strategy") != "sentence":
        collector.add(
            "unexpected_chunking_strategy",
            "Index was not built with the sentence chunking strategy.",
            actual=build_config.get("chunking_strategy"),
        )
    if build_config.get("embedding_model") != embedder.model_name:
        collector.add(
            "build_embedding_model_mismatch",
            "Build config model differs from the loaded embedding model.",
            stored=build_config.get("embedding_model"),
            actual=embedder.model_name,
        )
    if build_config.get("chunk_token_counter") != embedder.model_name:
        collector.add(
            "unsupported_chunk_token_counter",
            "Audit expects the embedding model tokenizer as the chunk token counter.",
            stored=build_config.get("chunk_token_counter"),
            expected=embedder.model_name,
        )
    if build_config.get("embedding_dimension") != embedder.dimension:
        collector.add(
            "build_embedding_dimension_mismatch",
            "Build config embedding dimension differs from the loaded model.",
            stored=build_config.get("embedding_dimension"),
            actual=embedder.dimension,
        )
    if manifest.get("embedding_dimension") != embedder.dimension:
        collector.add(
            "manifest_embedding_dimension_mismatch",
            "Manifest embedding dimension differs from the loaded model.",
            stored=manifest.get("embedding_dimension"),
            actual=embedder.dimension,
        )
    if manifest.get("max_sequence_length") != embedder.max_seq_length:
        collector.add(
            "manifest_sequence_length_mismatch",
            "Manifest maximum sequence length differs from the loaded model.",
            stored=manifest.get("max_sequence_length"),
            actual=embedder.max_seq_length,
        )
    if build_config.get("max_sequence_length") != manifest.get("max_sequence_length"):
        collector.add(
            "sequence_length_config_mismatch",
            "Build config and manifest maximum sequence lengths differ.",
        )
    if build_config.get("overlap_tokens") != manifest.get("overlap_tokens"):
        collector.add(
            "overlap_config_mismatch",
            "Build config and manifest overlap values differ.",
        )
    if build_config.get("paper_count") != len(papers):
        collector.add(
            "paper_count_mismatch",
            "Build config paper count differs from the source dataset.",
            stored=build_config.get("paper_count"),
            actual=len(papers),
        )
    if build_config.get("chunk_count") != len(chunks):
        collector.add(
            "chunk_count_mismatch",
            "Build config chunk count differs from the persisted chunks.",
            stored=build_config.get("chunk_count"),
            actual=len(chunks),
        )
    if build_config.get("window_count") != len(windows):
        collector.add(
            "window_count_mismatch",
            "Build config window count differs from the persisted windows.",
            stored=build_config.get("window_count"),
            actual=len(windows),
        )

    source_result = audit_source_chunks(
        papers=papers,
        chunks=chunks,
        token_counter=embedder,
        language=str(build_config["language"]),
        max_chunk_tokens=int(build_config["max_chunk_tokens"]),
        collector=collector,
        show_progress_bar=show_progress_bar,
    )
    window_result = audit_embedding_windows(
        chunks=chunks,
        windows=windows,
        embeddings=embeddings,
        embedder=embedder,
        manifest=manifest,
        collector=collector,
        show_progress_bar=show_progress_bar,
    )

    actual_oversized_chunks = sum(chunk.oversize_sentence for chunk in chunks)
    if build_config.get("oversize_chunk_count") != actual_oversized_chunks:
        collector.add(
            "oversize_chunk_count_mismatch",
            "Build config oversized chunk count differs from stored chunks.",
            stored=build_config.get("oversize_chunk_count"),
            actual=actual_oversized_chunks,
        )

    return {
        "split": split,
        "passed": collector.error_count == 0,
        "source_path": str(source_path),
        "index_dir": str(index_dir),
        "model_name": embedder.model_name,
        "manifest": manifest,
        "build_config": build_config,
        **source_result,
        **window_result,
        "error_count": collector.error_count,
        "error_counts": dict(collector.counts),
        "error_samples": collector.samples,
    }
