from paper_rag.data.qasper_loader import load_qasper


def main():
    dataset = load_qasper()

    sample = dataset["train"][0]

    # 1. Inspect paper structure
    print("\n========== Paper Structure ==========")

    full_text = sample["full_text"]
    sections = full_text["section_name"]
    paragraph_groups = full_text["paragraphs"]

    print("Sections:", len(sections))
    print("Paragraph groups:", len(paragraph_groups))
    print("Section alignment:", len(sections) == len(paragraph_groups))

    for i, (section, paragraphs) in enumerate(zip(sections, paragraph_groups)):
        print(f"\nSection {i}: {section}")
        print("Paragraph count:", len(paragraphs))

        if paragraphs:
            print("First paragraph:", paragraphs[0][:150])

        if i == 2:
            break

    # 2. Inspect Q&A structure
    print("\n========== QA Structure ==========")

    qas = sample["qas"]

    questions = qas["question"]
    question_ids = qas["question_id"]
    answers = qas["answers"]

    print("Questions:", len(questions))
    print("Question IDs:", len(question_ids))
    print("Answer groups:", len(answers))

    print("QA alignment:", len(questions) == len(question_ids) == len(answers))

    # 3. Inspect one question
    print("\n========== First Question ==========")

    print("Question ID:", question_ids[0])
    print("Question:", questions[0])

    answer_group = answers[0]["answer"]

    print("Number of annotations:", len(answer_group))

    for i, annotation in enumerate(answer_group):
        print(f"\nAnnotation {i + 1}")

        print("Unanswerable:", annotation["unanswerable"])
        print("Yes/No:", annotation["yes_no"])
        print("Free-form answer:", annotation["free_form_answer"])
        print("Extractive spans:", annotation["extractive_spans"])

        evidence = annotation["evidence"]

        print("Evidence count:", len(evidence))

        if evidence:
            print("First evidence:", evidence[0][:200])

    # 4. Inspect figures and tables
    print("\n========== Figures and Tables ==========")

    figures = sample["figures_and_tables"]

    print("Type:", type(figures))

    if isinstance(figures, dict):
        print("Keys:", figures.keys())
        for key, value in figures.items():
            print(key, type(value))

    elif isinstance(figures, list):
        print("Number of items:", len(figures))

    # 5. Inspect split statistics
    print("\n========== Split Statistics ==========")

    for split_name, split_data in dataset.items():
        total_questions = sum(
            len(item["qas"]["question"])
            for item in split_data
            if item["qas"] is not None
        )

        print(f"{split_name}: {len(split_data)} papers, {total_questions} questions")


if __name__ == "__main__":
    main()
