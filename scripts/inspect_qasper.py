
from paper_rag.data.qasper_loader import load_qasper


def main():
    dataset = load_qasper()

    print("\n========== Dataset Overview ==========")

    # Display all splits.
    for split_name, split_data in dataset.items():
        print(f"\nSplit: {split_name}")
        print(f"Number of papers: {len(split_data)}")
        print(f"Columns: {split_data.column_names}")

    # Inspect one paper.
    sample = dataset["train"][0]

    print("\n========== Paper Information ==========")
    print("Paper ID:", sample["id"])
    print("Title:", sample["title"])
    print("Abstract:", sample["abstract"][:300])

    # Inspect full text.
    full_text = sample["full_text"]

    print("\n========== Full Text ==========")
    print("Type:", type(full_text))
    print("Keys:", full_text.keys())

    print("Number of sections:", len(full_text["section_name"]))
    print("Number of paragraph groups:", len(full_text["paragraphs"]))

    # Inspect Q&A.
    qas = sample["qas"]

    print("\n========== Questions ==========")
    print("QAS type:", type(qas))
    print("QAS keys:", qas.keys())

    print("Number of questions:", len(qas["question"]))

    print("\nFirst question:")
    print(qas["question"][0])

    print("\nFirst answer:")
    print(qas["answers"][0])


if __name__ == "__main__":
    main()
