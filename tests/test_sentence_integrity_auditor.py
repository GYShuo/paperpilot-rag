import re

import numpy as np

from paper_rag.data.sentence_token_chunker import chunk_paper_by_sentence
from paper_rag.evaluation.sentence_integrity_auditor import (
    AuditCollector,
    audit_embedding_windows,
    audit_source_chunks,
    compare_sentence_spans,
    validate_window_coverage,
)
from paper_rag.schemas import (
    EmbeddingWindow,
    PaperDocument,
    PaperParagraph,
    PaperSection,
    SentenceTokenChunk,
)


class WordTokenCounter:
    def count(self, text: str) -> int:
        return len(text.split()) + 2


class FakeFastTokenizer:
    is_fast = True

    def __call__(
        self,
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=False,
    ):
        matches = list(re.finditer(r"\S+", text))
        result = {"input_ids": [index + 1 for index in range(len(matches))]}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (match.start(), match.end()) for match in matches
            ]
        return result


class FakeModel:
    tokenizer = FakeFastTokenizer()


class FakeEmbedder:
    model = FakeModel()
    model_name = "fake-model"
    dimension = 2

    def count(self, text: str) -> int:
        return len(re.findall(r"\S+", text)) + 2


def make_paper(text: str) -> PaperDocument:
    return PaperDocument(
        paper_id="paper-1",
        title="Test paper",
        abstract="",
        split="train",
        sections=[
            PaperSection(
                section_index=0,
                section_name="Introduction",
                paragraphs=[PaperParagraph(paragraph_index=0, text=text)],
            )
        ],
    )


def make_window(
    chunk: SentenceTokenChunk,
    window_index: int,
    start_char: int,
    end_char: int,
    token_start: int,
    token_end: int,
) -> EmbeddingWindow:
    text = chunk.text[start_char:end_char]
    return EmbeddingWindow(
        window_id=f"{chunk.chunk_id}:embedding:{window_index}",
        parent_chunk_id=chunk.chunk_id,
        paper_id=chunk.paper_id,
        split=chunk.split,
        section_index=chunk.section_index,
        section_name=chunk.section_name,
        window_index=window_index,
        text=text,
        start_char=start_char,
        end_char=end_char,
        token_start=token_start,
        token_end=token_end,
        token_count=len(re.findall(r"\S+", text)) + 2,
    )


def test_compare_sentence_spans_detects_missing_and_duplicate_spans():
    expected = [(0, 5, "Hello"), (6, 11, "world")]
    actual = [(0, 5, "Hello"), (0, 5, "Hello")]

    missing, unexpected = compare_sentence_spans(expected, actual)

    assert missing == [(6, 11, "world")]
    assert unexpected == [(0, 5, "Hello")]


def test_validate_window_coverage_passes_for_complete_overlapping_windows():
    chunk = SentenceTokenChunk(
        chunk_id="paper-1:0:sentence-token:0",
        paper_id="paper-1",
        split="train",
        section_index=0,
        section_name="Introduction",
        text="one two three four five six",
        token_count=8,
        sentence_count=1,
    )
    windows = [
        make_window(chunk, 0, 0, 18, 0, 4),
        make_window(chunk, 1, 8, 27, 2, 6),
    ]

    errors = validate_window_coverage(
        windows=windows,
        total_tokens=6,
        total_chars=len(chunk.text),
        overlap_tokens=2,
    )

    assert errors == []


def test_validate_window_coverage_detects_token_gap():
    chunk = SentenceTokenChunk(
        chunk_id="parent",
        paper_id="paper-1",
        split="train",
        section_index=0,
        section_name="Introduction",
        text="one two three four five six",
        token_count=8,
        sentence_count=1,
    )
    windows = [
        make_window(chunk, 0, 0, 18, 0, 4),
        make_window(chunk, 1, 19, 27, 5, 6),
    ]

    errors = validate_window_coverage(
        windows=windows,
        total_tokens=6,
        total_chars=len(chunk.text),
        overlap_tokens=2,
    )

    assert "Token coverage contains a gap." in errors


def test_validate_window_coverage_detects_wrong_overlap():
    chunk = SentenceTokenChunk(
        chunk_id="parent",
        paper_id="paper-1",
        split="train",
        section_index=0,
        section_name="Introduction",
        text="one two three four five six",
        token_count=8,
        sentence_count=1,
    )
    windows = [
        make_window(chunk, 0, 0, 18, 0, 4),
        make_window(chunk, 1, 13, 27, 3, 6),
    ]

    errors = validate_window_coverage(
        windows=windows,
        total_tokens=6,
        total_chars=len(chunk.text),
        overlap_tokens=2,
    )

    assert "Window overlap differs from the configured overlap." in errors


def test_validate_window_coverage_detects_missing_tail():
    chunk = SentenceTokenChunk(
        chunk_id="parent",
        paper_id="paper-1",
        split="train",
        section_index=0,
        section_name="Introduction",
        text="one two three four five six",
        token_count=8,
        sentence_count=1,
    )
    windows = [make_window(chunk, 0, 0, 23, 0, 5)]

    errors = validate_window_coverage(
        windows=windows,
        total_tokens=6,
        total_chars=len(chunk.text),
        overlap_tokens=2,
    )

    assert "Token coverage does not reach the parent end." in errors
    assert "Character coverage does not reach the parent end." in errors


def test_audit_source_chunks_passes_for_complete_sentence_chunks():
    paper = make_paper("Alpha beta. Gamma delta.")
    counter = WordTokenCounter()
    chunks = chunk_paper_by_sentence(
        paper=paper,
        token_counter=counter,
        max_tokens=10,
        language="en",
    )
    collector = AuditCollector()

    result = audit_source_chunks(
        papers=[paper],
        chunks=chunks,
        token_counter=counter,
        language="en",
        max_chunk_tokens=10,
        collector=collector,
        show_progress_bar=False,
    )

    assert collector.error_count == 0
    assert result["sentence_count"] == 2
    assert result["parent_chunk_count"] == 1


def test_audit_source_chunks_accepts_standalone_oversized_sentence():
    paper = make_paper("This sentence contains many words and exceeds the budget.")
    counter = WordTokenCounter()
    chunks = chunk_paper_by_sentence(
        paper=paper,
        token_counter=counter,
        max_tokens=4,
        language="en",
    )
    collector = AuditCollector()

    result = audit_source_chunks(
        papers=[paper],
        chunks=chunks,
        token_counter=counter,
        language="en",
        max_chunk_tokens=4,
        collector=collector,
        show_progress_bar=False,
    )

    assert collector.error_count == 0
    assert result["oversized_sentence_count"] == 1
    assert result["oversized_chunk_count"] == 1
    assert chunks[0].oversize_sentence is True
    assert chunks[0].sentence_count == 1


def test_audit_source_chunks_detects_missing_sentence():
    paper = make_paper("Alpha beta. Gamma delta.")
    counter = WordTokenCounter()
    chunks = chunk_paper_by_sentence(
        paper=paper,
        token_counter=counter,
        max_tokens=5,
        language="en",
    )
    collector = AuditCollector()

    audit_source_chunks(
        papers=[paper],
        chunks=chunks[:1],
        token_counter=counter,
        language="en",
        max_chunk_tokens=5,
        collector=collector,
        show_progress_bar=False,
    )

    assert collector.counts["missing_sentence_spans"] == 1


def test_audit_embedding_windows_checks_offsets_and_embeddings():
    chunk = SentenceTokenChunk(
        chunk_id="paper-1:0:sentence-token:0",
        paper_id="paper-1",
        split="train",
        section_index=0,
        section_name="Introduction",
        text="one two three four five six",
        token_count=8,
        sentence_count=1,
    )
    windows = [
        make_window(chunk, 0, 0, 18, 0, 4),
        make_window(chunk, 1, 8, 27, 2, 6),
    ]
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    collector = AuditCollector()

    result = audit_embedding_windows(
        chunks=[chunk],
        windows=windows,
        embeddings=embeddings,
        embedder=FakeEmbedder(),
        manifest={"max_sequence_length": 8, "overlap_tokens": 2},
        collector=collector,
        show_progress_bar=False,
    )

    assert collector.error_count == 0
    assert result["window_count"] == 2
    assert result["embedding_norms"]["all_l2_normalized"] is True
