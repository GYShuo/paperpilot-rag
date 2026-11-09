
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk

# Dataset configuration
DEFAULT_DATA_DIR = Path("data/raw/qasper")

DATA_FILES = {
    "train": (
        "https://huggingface.co/datasets/allenai/qasper/"
        "resolve/91bc63e221475d59a6e17e4299fbfc5a919d7c7a/"
        "qasper/train-00000-of-00001.parquet"
    ),
    "validation": (
        "https://huggingface.co/datasets/allenai/qasper/"
        "resolve/3065362e337ded696bbb0171b073c73e513c9410/"
        "qasper/qasper-validation.parquet"
    ),
    "test": (
        "https://huggingface.co/datasets/allenai/qasper/"
        "resolve/ff28ef00406baa196597e25a0b2cb7b0a0e1f96d/"
        "qasper/qasper-test.parquet"
    ),
}


def load_qasper(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> DatasetDict:
    """
    Load QASPER dataset from local cache or remote Parquet files.

    Args:
        data_dir: Directory used to cache the downloaded dataset.

    Returns:
        DatasetDict containing train, validation and test splits.
    """
    data_dir = Path(data_dir)
    dataset_marker = data_dir / "dataset_dict.json"

    # 1. Load local dataset if a valid saved DatasetDict exists.
    if dataset_marker.is_file():
        print(f"Loading local dataset from: {data_dir}")
        dataset = load_from_disk(str(data_dir))

    else:
        # 2. Load remote Parquet files.
        print("Downloading QASPER Parquet files...")

        dataset = load_dataset(
            "parquet",
            data_files=DATA_FILES,
        )

        # 3. Save the dataset locally.
        data_dir.parent.mkdir(parents=True, exist_ok=True)

        dataset.save_to_disk(str(data_dir))

        print(f"Dataset saved to: {data_dir}")

    # 4. Validate the returned dataset.
    if not isinstance(dataset, DatasetDict):
        raise TypeError(
            f"Expected DatasetDict, got {type(dataset).__name__}"
        )

    return dataset
