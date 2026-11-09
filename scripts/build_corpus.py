
from pathlib import Path

from tqdm import tqdm

from paper_rag.data.normalizer import (
    normalize_paper,
    normalize_qa_examples,
)
from paper_rag.data.qasper_loader import load_qasper

OUTPUT_DIR = Path("data/processed")


def main():
    dataset = load_qasper()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in dataset.items():

        paper_path = OUTPUT_DIR / f"{split_name}_papers.jsonl"
        qa_path = OUTPUT_DIR / f"{split_name}_qa.jsonl"

        paper_count = 0
        qa_count = 0

        with (
            open(paper_path, "w", encoding="utf-8") as paper_file,
            open(qa_path, "w", encoding="utf-8") as qa_file,
        ):

            for record in tqdm(
                split_data,
                desc=f"Processing {split_name}",
            ):
                # Normalize paper.
                paper = normalize_paper(
                    record,
                    split=split_name,
                )

                paper_file.write(
                    paper.model_dump_json() + "\n"
                )

                paper_count += 1

                # Normalize QA examples.
                examples = normalize_qa_examples(
                    record,
                    split=split_name,
                )

                for example in examples:
                    qa_file.write(
                        example.model_dump_json() + "\n"
                    )

                    qa_count += 1

        print(f"\nSplit: {split_name}")
        print(f"Papers saved: {paper_count}")
        print(f"QA examples saved: {qa_count}")
        print(f"Paper file: {paper_path}")
        print(f"QA file: {qa_path}")


if __name__ == "__main__":
    main()
