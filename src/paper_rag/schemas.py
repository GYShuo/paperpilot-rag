
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
    figures_and_tables: list[FigureTable] = Field(
        default_factory=list
    )


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
