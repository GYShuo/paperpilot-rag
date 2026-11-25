import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from paper_rag.evaluation.sentence_integrity_auditor import audit_split
from paper_rag.retrieval.embedder import TextEmbedder
from paper_rag.schemas import PaperDocument

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_CONFIG = {
    "train": {
        "source": PROJECT_ROOT / "data/processed/train_papers.jsonl",
        "index": PROJECT_ROOT / "data/indexes/qasper_train_windowed_v2",
    },
    "validation": {
        "source": PROJECT_ROOT / "data/processed/validation_papers.jsonl",
        "index": PROJECT_ROOT / "data/indexes/qasper_validation_windowed_v2",
    },
}


def load_papers_jsonl(path: Path) -> list[PaperDocument]:
    """Load normalized papers and report progress with tqdm."""

    if not path.is_file():
        raise FileNotFoundError(path)

    papers: list[PaperDocument] = []
    with path.open("r", encoding="utf-8") as file:
        total_lines = sum(1 for _ in file)
        file.seek(0)
        for line_number, line in enumerate(
            tqdm(file, total=total_lines, desc=f"Loading {path.stem}"),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                papers.append(PaperDocument.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid paper record at {path}:{line_number}."
                ) from exc

    if not papers:
        raise ValueError(f"No papers found in {path}.")

    return papers


def load_model_name(index_dir: Path) -> str:
    """Read the model name from a persisted windowed index manifest."""

    manifest_path = index_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest: dict[str, Any] = json.load(file)

    if not isinstance(manifest, dict):
        raise TypeError(f"Manifest in {manifest_path} must be a JSON object.")

    model_name = manifest.get("model_name")
    if not isinstance(model_name, str):
        raise TypeError(f"model_name in {manifest_path} must be a string.")
    if not model_name.strip():
        raise ValueError(f"model_name in {manifest_path} cannot be empty.")

    return model_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit V2 sentence integrity and embedding window coverage."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLIT_CONFIG),
        default=["train", "validation"],
        help="Dataset splits to audit (default: train validation).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Embedding model device (default: cpu).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT
        / "storage/metadata/sentence_integrity_audit.json",
        help="Path for the JSON audit report.",
    )
    args = parser.parse_args()

    print("\n========== V2 Sentence Integrity Audit ==========")
    print("Splits:", args.splits)
    print("Device:", args.device)

    first_index_dir = SPLIT_CONFIG[args.splits[0]]["index"]
    model_name = load_model_name(first_index_dir)

    print("\n========== Loading Embedding Model ==========")
    embedder = TextEmbedder(model_name=model_name, device=args.device)
    print("Model:", embedder.model_name)
    print("Dimension:", embedder.dimension)
    print("Maximum sequence length:", embedder.max_seq_length)

    split_results: dict[str, dict[str, Any]] = {}
    for split in args.splits:
        config = SPLIT_CONFIG[split]
        print(f"\n========== Auditing {split.upper()} ==========")
        try:
            papers = load_papers_jsonl(config["source"])
            result = audit_split(
                split=split,
                source_path=config["source"],
                index_dir=config["index"],
                embedder=embedder,
                papers=papers,
                show_progress_bar=True,
            )
        except (OSError, ValueError, TypeError, KeyError, EOFError) as exc:
            result = {
                "split": split,
                "passed": False,
                "fatal_error": f"{type(exc).__name__}: {exc}",
                "source_path": str(config["source"]),
                "index_dir": str(config["index"]),
            }

        split_results[split] = result
        print("Result:", "PASS" if result["passed"] else "FAIL")
        if "error_count" in result:
            print("Errors:", result["error_count"])
        if "fatal_error" in result:
            print("Fatal error:", result["fatal_error"])

    overall_passed = all(
        result["passed"] for result in split_results.values()
    )
    report = {
        "audit_name": "V2 Sentence Integrity Audit",
        "model_name": embedder.model_name,
        "splits": split_results,
        "overall_passed": overall_passed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print("\n========== Final Audit Result ==========")
    for split, result in split_results.items():
        print(f"{split}: {'PASS' if result['passed'] else 'FAIL'}")
    print("Overall:", "PASS" if overall_passed else "FAIL")
    print("Report:", args.output)

    if not overall_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
