# Development Log

## Day 2 - QASPER Dataset Loading

### Problem

The original implementation used:

load_dataset("allenai/qasper")

It failed because the QASPER repository contained
a custom dataset loading script (qasper.py),
which is unsupported by the installed datasets version.

### Root Cause

The installed datasets version is 5.1.0.
Modern versions no longer support loading
datasets through remote Python scripts.

### Solution

Load QASPER directly from its Parquet files:

load_dataset("parquet", data_files=DATA_FILES)

### Engineering Improvement

- Pin dataset file revisions using commit hashes.
- Cache the downloaded DatasetDict locally.
- Check dataset_dict.json before loading local data.

### Status

Pending verification.