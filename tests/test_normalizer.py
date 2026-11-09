
from paper_rag.data.normalizer import (
    normalize_paper,
    normalize_qa_examples,
)


def test_normalize_paper():
    """Test paper normalization and empty paragraph removal."""

    record = {
        "id": "paper_001",
        "title": "  Example Paper  ",
        "abstract": "An example abstract.",
        "full_text": {
            "section_name": [
                "Introduction",
                "Method",
            ],
            "paragraphs": [
                [
                    " First paragraph. ",
                    "   ",
                    "Third paragraph.",
                ],
                [],
            ],
        },
        "figures_and_tables": {
            "caption": ["Figure 1"],
            "file": ["figure1.png"],
        },
    }

    paper = normalize_paper(record, "train")

    assert paper.paper_id == "paper_001"
    assert paper.title == "Example Paper"

    assert len(paper.sections) == 2

    paragraphs = paper.sections[0].paragraphs

    assert len(paragraphs) == 2
    assert paragraphs[0].paragraph_index == 0
    assert paragraphs[1].paragraph_index == 2

    assert paper.sections[1].section_name == "Method"
    assert len(paper.sections[1].paragraphs) == 0

    assert paper.figures_and_tables[0].caption == "Figure 1"


def test_normalize_qa_examples():
    """Test multiple annotations are preserved."""

    record = {
        "id": "paper_001",
        "qas": {
            "question": [
                "What is the proposed method?"
            ],
            "question_id": [
                "question_001"
            ],
            "answers": [
                {
                    "answer": [
                        {
                            "unanswerable": False,
                            "yes_no": None,
                            "free_form_answer": "Method A",
                            "extractive_spans": [],
                            "evidence": ["Evidence A"],
                            "highlighted_evidence": ["Evidence A"],
                        },
                        {
                            "unanswerable": False,
                            "yes_no": None,
                            "free_form_answer": "",
                            "extractive_spans": ["Method A"],
                            "evidence": ["Evidence A"],
                            "highlighted_evidence": [],
                        },
                    ],
                    "annotation_id": [
                        "annotation_001",
                        "annotation_002",
                    ],
                    "worker_id": [
                        "worker_001",
                        "worker_002",
                    ],
                }
            ],
        },
    }

    examples = normalize_qa_examples(record, "train")

    assert len(examples) == 1

    qa = examples[0]

    assert qa.question_id == "question_001"
    assert qa.paper_id == "paper_001"

    assert len(qa.answers) == 2

    assert qa.answers[0].free_form_answer == "Method A"
    assert qa.answers[1].extractive_spans == ["Method A"]

    assert qa.answers[0].evidence == ["Evidence A"]


def test_empty_question_is_skipped():
    """Empty questions should not enter the QA dataset."""

    record = {
        "id": "paper_002",
        "qas": {
            "question": ["", "Valid question"],
            "question_id": ["q1", "q2"],
            "answers": [None, {"answer": []}],
        },
    }

    examples = normalize_qa_examples(record, "train")

    assert len(examples) == 1
    assert examples[0].question == "Valid question"
    assert examples[0].question_id == "q2"


def test_empty_paper_id_raises_error():
    """A paper without an ID should be rejected."""

    record = {
        "id": "",
        "title": "Example",
        "full_text": {},
    }

    try:
        normalize_paper(record, "train")
        assert False, "Expected ValueError"
    except ValueError:
        pass
