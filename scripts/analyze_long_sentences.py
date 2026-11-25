import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from paper_rag.data.sentence_token_chunker import (
    split_paragraph_into_sentences,
)
from paper_rag.retrieval.embedder import TextEmbedder
from paper_rag.schemas import PaperDocument

PAPER_PATH = Path("data/processed/train_papers.jsonl")

CHUNK_BUDGET = 240
TOP_N = 20


def main():
    print("========== Step 1: Loading Model ==========")

    embedder = TextEmbedder()

    max_seq_length = embedder.max_seq_length

    print("Model:", embedder.model_name)
    print("Maximum sequence length:", max_seq_length)
    print("Analysis budget:", CHUNK_BUDGET)

    print("\n========== Step 2: Analyzing Sentences ==========")

    segmenter = __import__("sentencesplit").Segmenter(
        language="en",
        clean=False,
        split_mode="conservative",
    )

    token_counts = []
    long_sentences = []

    paper_count = 0
    papers_with_long_sentences = set()
    papers_without_sections = 0

    with PAPER_PATH.open("r", encoding="utf-8") as file:
        for line in tqdm(file, desc="Processing papers"):
            if not line.strip():
                continue

            paper = PaperDocument.model_validate(json.loads(line))

            paper_count += 1

            if not paper.sections:
                papers_without_sections += 1

            for section in paper.sections:
                for paragraph in section.paragraphs:
                    sentences = split_paragraph_into_sentences(
                        text=paragraph.text,
                        paragraph_index=paragraph.paragraph_index,
                        segmenter=segmenter,
                    )

                    for sentence in sentences:
                        count = embedder.count(sentence.text)

                        token_counts.append(count)

                        if count > CHUNK_BUDGET:
                            papers_with_long_sentences.add(paper.paper_id)

                            long_sentences.append(
                                {
                                    "paper_id": paper.paper_id,
                                    "title": paper.title,
                                    "section": section.section_name,
                                    "paragraph_index": (paragraph.paragraph_index),
                                    "token_count": count,
                                    "char_count": len(sentence.text),
                                    "text": sentence.text,
                                }
                            )

    if not token_counts:
        print("No sentences found.")
        return

    counts = np.asarray(token_counts)

    over_budget = counts > CHUNK_BUDGET
    over_model_limit = counts > max_seq_length

    print("\n========== Step 3: Corpus Statistics ==========")

    print("Papers processed:", paper_count)
    print("Papers without sections:", papers_without_sections)
    print("Total sentences:", len(counts))

    print("\n========== Step 4: Token Distribution ==========")

    print("Average:", round(float(np.mean(counts)), 2))
    print("Median:", float(np.median(counts)))

    for percentile in [75, 90, 95, 99]:
        print(
            f"P{percentile}:",
            round(float(np.percentile(counts, percentile)), 2),
        )

    print("Maximum:", int(np.max(counts)))

    print("\n========== Step 5: Long Sentence Statistics ==========")

    print(
        f"Sentences > {CHUNK_BUDGET} tokens:",
        int(np.sum(over_budget)),
    )

    print(
        "Ratio:",
        f"{np.mean(over_budget) * 100:.2f}%",
    )

    print(
        "Affected papers:",
        len(papers_with_long_sentences),
    )

    print(
        f"Sentences > model limit ({max_seq_length}):",
        int(np.sum(over_model_limit)),
    )

    print(
        "Model-limit violation ratio:",
        f"{np.mean(over_model_limit) * 100:.2f}%",
    )

    print("\n========== Step 6: Longest Sentences ==========")

    long_sentences.sort(
        key=lambda item: item["token_count"],
        reverse=True,
    )

    for index, item in enumerate(
        long_sentences[:TOP_N],
        start=1,
    ):
        print(f"\n--- Rank {index} ---")
        print("Paper:", item["paper_id"])
        print("Title:", item["title"])
        print("Section:", item["section"])
        print("Paragraph:", item["paragraph_index"])
        print("Tokens:", item["token_count"])
        print("Characters:", item["char_count"])

        text = item["text"]

        if len(text) > 500:
            text = text[:500] + "..."

        print("Text:", repr(text))

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
