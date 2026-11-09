
from typing import Any

from paper_rag.schemas import (
    AnswerAnnotation,
    FigureTable,
    PaperDocument,
    PaperParagraph,
    PaperSection,
    QAExample,
)


def clean_text(value: Any) -> str:
    """Normalize whitespace in a text field."""
    if value is None:
        return ""

    return " ".join(str(value).split())


def as_list(value: Any) -> list:
    """Return a list, or an empty list for invalid values."""
    return value if isinstance(value, list) else []


def normalize_paper(
    record: dict,
    split: str,
) -> PaperDocument:
    """Convert one raw QASPER record into PaperDocument."""

    paper_id = clean_text(record.get("id"))

    if not paper_id:
        raise ValueError("Paper ID cannot be empty.")

    full_text = record.get("full_text") or {}

    section_names = as_list(full_text.get("section_name"))
    paragraph_groups = as_list(full_text.get("paragraphs"))

    sections = []

    section_count = max(
        len(section_names),
        len(paragraph_groups),
    )

    for section_index in range(section_count):

        # Get section name.
        raw_name = (
            section_names[section_index]
            if section_index < len(section_names)
            else None
        )

        section_name = (
            clean_text(raw_name)
            or f"Section {section_index + 1}"
        )

        # Get paragraphs belonging to this section.
        raw_paragraphs = (
            paragraph_groups[section_index]
            if section_index < len(paragraph_groups)
            else []
        )

        if not isinstance(raw_paragraphs, list):
            raw_paragraphs = []

        paragraphs = []

        for paragraph_index, raw_paragraph in enumerate(
            raw_paragraphs
        ):
            text = clean_text(raw_paragraph)

            # Skip empty paragraphs.
            if not text:
                continue

            paragraphs.append(
                PaperParagraph(
                    paragraph_index=paragraph_index,
                    text=text,
                )
            )

        sections.append(
            PaperSection(
                section_index=section_index,
                section_name=section_name,
                paragraphs=paragraphs,
            )
        )

    # Extract figure and table metadata.
    raw_figures = record.get("figures_and_tables") or {}

    captions = as_list(raw_figures.get("caption"))
    files = as_list(raw_figures.get("file"))

    figures_and_tables = []

    item_count = max(len(captions), len(files))

    for item_index in range(item_count):
        caption = (
            clean_text(captions[item_index])
            if item_index < len(captions)
            else ""
        )

        file = (
            clean_text(files[item_index])
            if item_index < len(files)
            else ""
        )

        figures_and_tables.append(
            FigureTable(
                item_index=item_index,
                caption=caption or None,
                file=file or None,
            )
        )

    return PaperDocument(
        paper_id=paper_id,
        title=clean_text(record.get("title")),
        abstract=clean_text(record.get("abstract")),
        split=split,
        sections=sections,
        figures_and_tables=figures_and_tables,
    )


def normalize_qa_examples(
    record: dict,
    split: str,
) -> list[QAExample]:
    """Extract QA examples while preserving all annotations."""

    paper_id = clean_text(record.get("id"))

    if not paper_id:
        raise ValueError("Paper ID cannot be empty.")

    qas = record.get("qas") or {}

    questions = as_list(qas.get("question"))
    question_ids = as_list(qas.get("question_id"))
    answer_groups = as_list(qas.get("answers"))

    examples = []

    for question_index, raw_question in enumerate(questions):

        question = clean_text(raw_question)

        if not question:
            continue

        question_id = (
            clean_text(question_ids[question_index])
            if question_index < len(question_ids)
            else ""
        )

        if not question_id:
            question_id = f"{paper_id}:{question_index}"

        answer_group = (
            answer_groups[question_index]
            if question_index < len(answer_groups)
            else {}
        )

        if not isinstance(answer_group, dict):
            answer_group = {}

        raw_annotations = as_list(
            answer_group.get("answer")
        )

        annotation_ids = as_list(
            answer_group.get("annotation_id")
        )

        worker_ids = as_list(
            answer_group.get("worker_id")
        )

        annotations = []

        for annotation_index, raw_annotation in enumerate(
            raw_annotations
        ):
            if not isinstance(raw_annotation, dict):
                continue

            annotation_id = (
                clean_text(annotation_ids[annotation_index])
                if annotation_index < len(annotation_ids)
                else ""
            )

            worker_id = (
                clean_text(worker_ids[annotation_index])
                if annotation_index < len(worker_ids)
                else ""
            )

            unanswerable = raw_annotation.get("unanswerable")
            yes_no = raw_annotation.get("yes_no")

            if not isinstance(unanswerable, bool):
                unanswerable = None

            if not isinstance(yes_no, bool):
                yes_no = None

            annotations.append(
                AnswerAnnotation(
                    annotation_id=annotation_id or None,
                    worker_id=worker_id or None,
                    unanswerable=unanswerable,
                    yes_no=yes_no,
                    free_form_answer=clean_text(
                        raw_annotation.get("free_form_answer")
                    ),
                    extractive_spans=[
                        clean_text(x)
                        for x in as_list(
                            raw_annotation.get("extractive_spans")
                        )
                        if clean_text(x)
                    ],
                    evidence=[
                        clean_text(x)
                        for x in as_list(
                            raw_annotation.get("evidence")
                        )
                        if clean_text(x)
                    ],
                    highlighted_evidence=[
                        clean_text(x)
                        for x in as_list(
                            raw_annotation.get("highlighted_evidence")
                        )
                        if clean_text(x)
                    ],
                )
            )

        examples.append(
            QAExample(
                question_id=question_id,
                paper_id=paper_id,
                question=question,
                split=split,
                answers=annotations,
            )
        )

    return examples
