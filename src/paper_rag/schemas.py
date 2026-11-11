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


class PaperChunk(BaseModel):  # 原始paper段落不会合并的chunk
    """A retrievable text chunk with source metadata."""

    chunk_id: str
    paper_id: str
    split: str

    section_index: int
    section_name: str
    paragraph_index: int  # 原始段落索引

    start_char: int  # 清洗后段落的起始字符索引
    end_char: int

    text: str  # 实际检索文本


class ChunkSpan(BaseModel):
    """Location of source text within a paragraph and chunk."""

    paragraph_index: int

    # 在原始段落中的偏移量
    start_char: int
    end_char: int

    # 在最终生成的chunk中的偏移量
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
