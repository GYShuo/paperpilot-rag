import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from paper_rag.schemas import SentenceTokenChunk


@dataclass(frozen=True)
class BM25RetrievalResult:
    """A retrieved parent chunk with its BM25 score."""

    rank: int
    score: float
    chunk: SentenceTokenChunk


class BM25Retriever:
    """BM25 sparse retriever over V2 parent chunks."""

    TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be a positive finite number.")

        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")

        self.k1 = float(k1)
        self.b = float(b)

        self.chunks: list[SentenceTokenChunk] = []
        self._paper_ids: list[str] = []
        self._doc_lengths: list[int] = []

        self._postings: dict[str, dict[int, int]] = {}
        self._idf: dict[str, float] = {}

        self._avgdl = 0.0
        self._num_documents = 0
        self._split: str | None = None

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        """Convert text into lowercase Unicode word tokens."""

        if not isinstance(text, str):
            raise TypeError("Text must be a string.")

        return [token.casefold() for token in cls.TOKEN_PATTERN.findall(text)]

    @property
    def num_documents(self) -> int:
        """Return the number of indexed parent chunks."""

        return self._num_documents

    @property
    def vocabulary_size(self) -> int:
        """Return the number of unique indexed terms."""

        return len(self._postings)

    @property
    def average_document_length(self) -> float:
        """Return the average number of tokens per parent chunk."""

        return self._avgdl

    @property
    def split(self) -> str | None:
        """Return the dataset split represented by this index."""

        return self._split

    def build_index(
        self,
        chunks: list[SentenceTokenChunk],
    ) -> None:
        """Build an in-memory inverted index from V2 parent chunks.

        IDF is calculated over the complete supplied corpus.
        Paper filtering is applied later during search.

        The index is updated only after all input validation
        and index construction have succeeded.
        """

        if not chunks:
            raise ValueError("Cannot build BM25 index from empty chunks.")

        chunk_ids = set()
        splits = set()

        postings: dict[str, dict[int, int]] = defaultdict(dict)
        doc_lengths = []
        paper_ids = []
        total_tokens = 0

        for doc_index, chunk in enumerate(chunks):
            if not isinstance(chunk, SentenceTokenChunk):
                raise TypeError("BM25Retriever requires SentenceTokenChunk objects.")

            if not chunk.chunk_id.strip():
                raise ValueError("Chunk ID cannot be empty.")

            if chunk.chunk_id in chunk_ids:
                raise ValueError(f"Duplicate chunk ID: {chunk.chunk_id}")

            if not chunk.paper_id.strip():
                raise ValueError(f"Paper ID cannot be empty: {chunk.chunk_id}")

            if not chunk.text.strip():
                raise ValueError(f"Empty parent chunk: {chunk.chunk_id}")

            chunk_ids.add(chunk.chunk_id)
            splits.add(chunk.split)

            if not chunk.split.strip():
                raise ValueError(f"Split cannot be empty: {chunk.chunk_id}")

            tokens = self.tokenize(chunk.text)
            term_frequencies = Counter(tokens)

            doc_length = len(tokens)

            doc_lengths.append(doc_length)
            paper_ids.append(chunk.paper_id)
            total_tokens += doc_length

            for term, frequency in term_frequencies.items():
                postings[term][doc_index] = frequency

        if len(splits) != 1:
            raise ValueError("All chunks must belong to the same dataset split.")

        if total_tokens == 0:
            raise ValueError(
                "Cannot build BM25 index: corpus contains no searchable tokens."
            )

        num_documents = len(chunks)
        average_length = total_tokens / num_documents

        # Positive IDF formula.
        idf = {
            term: math.log(
                1.0
                + (num_documents - len(document_frequencies) + 0.5)
                / (len(document_frequencies) + 0.5)
            )
            for term, document_frequencies in postings.items()
        }

        # Commit the newly built index only after successful validation.
        self.chunks = list(chunks)
        self._paper_ids = paper_ids
        self._doc_lengths = doc_lengths
        self._postings = dict(postings)
        self._idf = idf

        self._avgdl = average_length
        self._num_documents = num_documents
        self._split = next(iter(splits))

    def search(
        self,
        query: str,
        paper_id: str,
        top_k: int | None = None,
    ) -> list[BM25RetrievalResult]:
        """Search matching parent chunks within one paper.

        Only documents with positive BM25 scores are returned.
        Documents without lexical matches are not treated as
        retrieved candidates.
        """

        if self._num_documents == 0:
            raise RuntimeError("Please build the BM25 index before searching.")

        if not isinstance(query, str):
            raise TypeError("Query must be a string.")

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not isinstance(paper_id, str):
            raise TypeError("paper_id must be a string.")

        if not paper_id.strip():
            raise ValueError("paper_id cannot be empty.")

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_terms = self.tokenize(query)

        if not query_terms:
            return []

        query_frequencies = Counter(query_terms)
        scores: dict[int, float] = {}

        for term, query_frequency in query_frequencies.items():
            term_postings = self._postings.get(term)

            if not term_postings:
                continue

            term_idf = self._idf[term]

            for doc_index, term_frequency in term_postings.items():
                if self._paper_ids[doc_index] != paper_id:
                    continue

                document_length = self._doc_lengths[doc_index]

                length_normalization = self.k1 * (
                    1.0 - self.b + self.b * document_length / self._avgdl
                )

                denominator = term_frequency + length_normalization

                term_score = term_idf * term_frequency * (self.k1 + 1.0) / denominator

                scores[doc_index] = (
                    scores.get(doc_index, 0.0) + query_frequency * term_score
                )

        # Stable tie-breaking by original corpus order.
        ranked_indices = sorted(
            scores,
            key=lambda index: (-scores[index], index),
        )

        if top_k is not None:
            ranked_indices = ranked_indices[:top_k]

        return [
            BM25RetrievalResult(
                rank=rank,
                score=float(scores[doc_index]),
                chunk=self.chunks[doc_index],
            )
            for rank, doc_index in enumerate(
                ranked_indices,
                start=1,
            )
        ]
