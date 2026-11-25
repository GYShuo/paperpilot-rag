import pytest

from paper_rag.retrieval.bm25_retriever import BM25Retriever
from paper_rag.schemas import SentenceTokenChunk


def make_chunk(
    chunk_id: str,
    text: str,
    paper_id: str = "paper-1",
    split: str = "validation",
) -> SentenceTokenChunk:
    return SentenceTokenChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        split=split,
        section_index=0,
        section_name="Introduction",
        text=text,
        token_count=max(len(text.split()), 1),
        sentence_count=1,
    )


def make_retriever(chunks):
    retriever = BM25Retriever()
    retriever.build_index(chunks)
    return retriever


def test_tokenize_case_insensitive():
    assert BM25Retriever.tokenize("Transformer TRANSFORMER Attention") == [
        "transformer",
        "transformer",
        "attention",
    ]


def test_tokenize_unicode_and_numbers():
    assert BM25Retriever.tokenize("BERT-base 3D 模型 1.5") == [
        "bert",
        "base",
        "3d",
        "模型",
        "1",
        "5",
    ]


def test_build_index_statistics():
    chunks = [
        make_chunk("a", "neural network model"),
        make_chunk("b", "neural network neural network model"),
        make_chunk("c", "graph convolution network"),
    ]

    retriever = make_retriever(chunks)

    assert retriever.num_documents == 3
    assert retriever.split == "validation"
    assert retriever.vocabulary_size == 5
    assert retriever.average_document_length == pytest.approx(11 / 3)


def test_search_ranking():
    chunks = [
        make_chunk("a", "neural network model"),
        make_chunk("b", "neural network neural network model"),
        make_chunk("c", "graph convolution network"),
    ]

    retriever = make_retriever(chunks)

    results = retriever.search("neural network", "paper-1")

    assert [item.chunk.chunk_id for item in results] == ["b", "a", "c"]
    assert [item.rank for item in results] == [1, 2, 3]
    assert all(item.score > 0 for item in results)


def test_search_filters_by_paper_id():
    chunks = [
        make_chunk("a", "transformer attention mechanism", "paper-1"),
        make_chunk("b", "transformer attention mechanism", "paper-2"),
    ]

    retriever = make_retriever(chunks)
    results = retriever.search("transformer attention", "paper-1")

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "a"


def test_search_returns_empty_when_no_terms_match():
    retriever = make_retriever(
        [
            make_chunk("a", "neural network model"),
            make_chunk("b", "graph convolution"),
        ]
    )

    assert retriever.search("quantum physics", "paper-1") == []


def test_search_returns_empty_for_punctuation_query():
    retriever = make_retriever(
        [
            make_chunk("a", "neural network model"),
        ]
    )

    assert retriever.search("!!! ???", "paper-1") == []


def test_search_top_k():
    retriever = make_retriever(
        [
            make_chunk("a", "neural network model"),
            make_chunk("b", "neural network architecture"),
            make_chunk("c", "neural network learning"),
        ]
    )

    results = retriever.search("neural network", "paper-1", top_k=2)

    assert len(results) == 2
    assert [item.rank for item in results] == [1, 2]


def test_search_stable_tie_breaking():
    retriever = make_retriever(
        [
            make_chunk("a", "neural network"),
            make_chunk("b", "neural network"),
        ]
    )

    results = retriever.search("neural network", "paper-1")

    assert [item.chunk.chunk_id for item in results] == ["a", "b"]


def test_search_before_build_index():
    retriever = BM25Retriever()

    with pytest.raises(RuntimeError, match="build"):
        retriever.search("neural network", "paper-1")


def test_build_empty_index():
    retriever = BM25Retriever()

    with pytest.raises(ValueError, match="empty"):
        retriever.build_index([])


def test_build_duplicate_chunk_ids():
    chunk = make_chunk("a", "neural network")
    retriever = BM25Retriever()

    with pytest.raises(ValueError, match="Duplicate"):
        retriever.build_index([chunk, chunk.model_copy()])


def test_build_rejects_mixed_splits():
    chunks = [
        make_chunk("a", "neural network", split="train"),
        make_chunk("b", "graph network", split="validation"),
    ]

    retriever = BM25Retriever()

    with pytest.raises(ValueError, match="same dataset split"):
        retriever.build_index(chunks)


def test_build_rejects_tokenless_corpus():
    chunks = [
        make_chunk("a", "!!! ???"),
        make_chunk("b", "... ---"),
    ]

    retriever = BM25Retriever()

    with pytest.raises(ValueError, match="searchable tokens"):
        retriever.build_index(chunks)


@pytest.mark.parametrize(
    "k1,b",
    [
        (0, 0.75),
        (-1, 0.75),
        (1.5, -0.1),
        (1.5, 1.1),
    ],
)
def test_invalid_bm25_parameters(k1, b):
    with pytest.raises(ValueError):
        BM25Retriever(k1=k1, b=b)


def test_invalid_top_k():
    retriever = make_retriever(
        [
            make_chunk("a", "neural network"),
        ]
    )

    with pytest.raises(ValueError, match="top_k"):
        retriever.search("neural", "paper-1", top_k=0)


def test_search_rejects_empty_query():
    retriever = make_retriever(
        [
            make_chunk("a", "neural network"),
        ]
    )

    with pytest.raises(ValueError, match="empty"):
        retriever.search("   ", "paper-1")


def test_search_rejects_empty_paper_id():
    retriever = make_retriever(
        [
            make_chunk("a", "neural network"),
        ]
    )

    with pytest.raises(ValueError, match="paper_id"):
        retriever.search("neural", " ")
