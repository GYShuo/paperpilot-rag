import json
from collections import Counter
from pathlib import Path

RESULT_PATH = Path("storage/metadata/retrieval_eval_train_v0_v1.json")

TOP_K = 10
SAMPLES_PER_CATEGORY = 5


def classify_result(r0, r1):
    """Classify paired retrieval outcomes."""

    if r0 is None and r1 is None:
        return "both_missed"

    if r0 is not None and r1 is None:
        return "v0_only_hit"

    if r0 is None and r1 is not None:
        return "v1_only_hit"

    if r0 < r1:
        return "v0_better_rank"

    if r1 < r0:
        return "v1_better_rank"

    return "same_rank"


def main():
    with RESULT_PATH.open(encoding="utf-8") as f:
        data = json.load(f)

    records = data["per_question"]
    counts = Counter()
    samples = {}

    for row in records:
        r0 = row["V0_first_hit_rank"]
        r1 = row["V1_first_hit_rank"]

        category = classify_result(r0, r1)
        counts[category] += 1
        samples.setdefault(category, [])

        if len(samples[category]) < SAMPLES_PER_CATEGORY:
            samples[category].append(row)

    print("========== Paired Error Analysis ==========")
    print("Total questions:", len(records))

    for category, count in counts.most_common():
        ratio = count / len(records) * 100
        print(f"{category}: {count} ({ratio:.2f}%)")

    print("\n========== Hit@10 Comparison ==========")

    v0_hit = sum(row["V0_first_hit_rank"] is not None for row in records)
    v1_hit = sum(row["V1_first_hit_rank"] is not None for row in records)

    print("V0 hits:", v0_hit)
    print("V1 hits:", v1_hit)
    print("Difference (V1 - V0):", v1_hit - v0_hit)

    assert counts["v0_only_hit"] - counts["v1_only_hit"] == (v0_hit - v1_hit)

    assert sum(counts.values()) == len(records)

    print("\n========== Example Questions ==========")

    for category in (
        "v0_only_hit",
        "v1_only_hit",
        "v0_better_rank",
        "v1_better_rank",
        "both_missed",
    ):
        print(f"\n--- {category} ---")

        for row in samples.get(category, []):
            print("Paper:", row["paper_id"])
            print("Question:", row["question"])
            print("V0 rank:", row["V0_first_hit_rank"])
            print("V1 rank:", row["V1_first_hit_rank"])
            print("Evidence locations:", row["evidence_locations"])
            print()

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
