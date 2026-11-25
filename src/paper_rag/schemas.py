from pydantic import BaseModel, Field


class PaperParagraph(BaseModel):
    """A paragraph in a research paper."""

    paragraph_index: int
    text: str


class PaperSection(BaseModel):
    """A section containing multiple paragraphs."""

    section_index: int
    section_name: str
    paragraphs: list[PaperParagraph] = Field(default_factory=list)


class FigureTable(BaseModel):
    """Metadata of a figure or table."""

    item_index: int
    caption: str | None = None
    file: str | None = None


class PaperDocument(BaseModel):
    """Normalized research paper."""

    paper_id: str
    title: str
    abstract: str
    split: str
    sections: list[PaperSection] = Field(default_factory=list)
    figures_and_tables: list[FigureTable] = Field(default_factory=list)


class AnswerAnnotation(BaseModel):
    """One annotator's answer to a question."""

    annotation_id: str | None = None
    worker_id: str | None = None

    unanswerable: bool | None = None
    yes_no: bool | None = None

    free_form_answer: str = ""
    extractive_spans: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    highlighted_evidence: list[str] = Field(default_factory=list)


class QAExample(BaseModel):
    """A question and its answer annotations."""

    question_id: str
    paper_id: str
    question: str
    split: str

    answers: list[AnswerAnnotation] = Field(default_factory=list)


class PaperChunk(BaseModel):
    """A retrievable text chunk with source metadata."""

    chunk_id: str
    paper_id: str
    split: str

    section_index: int
    section_name: str
    paragraph_index: int

    start_char: int
    end_char: int

    text: str


class ChunkSpan(BaseModel):
    """Location of source text within a paragraph and chunk."""

    paragraph_index: int

    # Offsets in the original paragraph.
    start_char: int
    end_char: int

    # Offsets in the final chunk.
    chunk_start: int
    chunk_end: int


class SectionChunk(BaseModel):
    """A chunk that may contain multiple paragraphs."""

    chunk_id: str
    paper_id: str
    split: str

    section_index: int
    section_name: str

    text: str
    spans: list[ChunkSpan] = Field(default_factory=list)


class SentenceTokenChunk(BaseModel):
    """A token-budgeted chunk containing complete sentences."""

    chunk_id: str
    paper_id: str
    split: str

    section_index: int
    section_name: str

    text: str
    spans: list[ChunkSpan] = Field(default_factory=list)

    token_count: int
    sentence_count: int

    # True when this chunk contains one sentence exceeding
    # the normal token budget.
    oversize_sentence: bool = False


class EmbeddingWindow(BaseModel):
    """A token-limited window used for dense embedding.

    Each window maps back to one complete SentenceTokenChunk.
    Character offsets refer to the parent chunk's text.
    Token offsets refer to the parent's untruncated token sequence.
    """

    window_id: str
    parent_chunk_id: str

    paper_id: str
    split: str

    section_index: int
    section_name: str

    window_index: int

    text: str

    # Character offsets in the parent SentenceTokenChunk.
    start_char: int
    end_char: int

    # Token offsets in the parent token sequence.
    # end token index is exclusive.
    token_start: int
    token_end: int

    # Actual model input length, including special tokens.
    token_count: int
