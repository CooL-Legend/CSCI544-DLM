# sudoku_dataset.py
from pathlib import Path
import re

from datasets import (Dataset as HFDataset, Features, Value,
                      concatenate_datasets, config as datasets_config, load_dataset)
from datasets.naming import camelcase_to_snakecase
from torch.utils.data import Dataset

BGPO_DATA_REVISION = "2d3e819a65b42ff5bd9cce958fb4412d57c1609d"
BGPO_DATA_URL = (
    f"https://raw.githubusercontent.com/THU-KEG/BGPO/{BGPO_DATA_REVISION}/"
    "data/preprocessed/test/sudoku_1.parquet"
)
BGPO_PROVENANCE = {
    "source": "bgpo_4x4", "url": BGPO_DATA_URL,
    "revision": BGPO_DATA_REVISION, "split": "test", "board_size": 4,
}

SUDOKU_REPO_ID = "certen/Sudoku-1M-Puzzles"


def _load_cached_sudoku(n, cache_dir=None):
    """Read prepared Arrow shards without asking the Hub to resolve a config."""
    cache_root = Path(cache_dir or datasets_config.HF_DATASETS_CACHE).expanduser()
    namespace, name = SUDOKU_REPO_ID.split("/", 1)
    dataset_root = cache_root / f"{namespace}___{camelcase_to_snakecase(name)}"
    prepared = []
    for info_path in dataset_root.glob("*/*/*/dataset_info.json"):
        train_files = sorted(info_path.parent.glob("*-train*.arrow"))
        if train_files:
            prepared.append((info_path.stat().st_mtime_ns, train_files))
    if not prepared:
        raise FileNotFoundError(
            f"Prepared offline cache for {SUDOKU_REPO_ID} was not found under {cache_root}. "
            "Populate the dataset cache before starting the experiment job."
        )
    train_files = max(prepared, key=lambda item: item[0])[1]
    shards = [HFDataset.from_file(str(path)) for path in train_files]
    data = shards[0] if len(shards) == 1 else concatenate_datasets(shards)
    if not {"puzzle", "solution"}.issubset(data.column_names):
        raise ValueError("Cached Sudoku dataset must contain puzzle and solution columns.")
    if len(data) < n:
        raise ValueError(f"Cached Sudoku dataset contains {len(data)} rows; requested {n}.")
    return data.select(range(n))


class BGPOSudokuDataset(Dataset):
    """The authors' 500 held-out 4x4 puzzles, in their original order."""

    provenance = BGPO_PROVENANCE

    def __init__(self, n=3, cache_dir=None):
        if type(n) is not int or not 1 <= n <= 500:
            raise ValueError("n must be between 1 and 500 for bgpo_4x4")
        ds = load_dataset("parquet", data_files={"test": BGPO_DATA_URL},
                          split=f"test[:{n}]", cache_dir=cache_dir)
        if len(ds) != n:
            raise ValueError(f"Expected {n} BGPO test puzzles, received {len(ds)}.")
        self.data = []
        for row in ds:
            puzzle = row["extra_info"]["puzzle"]
            solution = row["reward_model"]["ground_truth"]
            messages = row["prompt"]
            if not isinstance(puzzle, str) or not re.fullmatch(r"[0-4]{16}", puzzle):
                raise ValueError("BGPO puzzles must be 16-character strings of digits 0–4.")
            if not isinstance(solution, str) or not re.fullmatch(r"[1-4]{16}", solution):
                raise ValueError("BGPO references must be 16-character strings of digits 1–4.")
            if (len(messages) != 1 or messages[0]["role"] != "user"
                    or not isinstance(messages[0]["content"], str)):
                raise ValueError("Expected exactly one original BGPO user prompt.")
            self.data.append({
                "puzzle": puzzle, "solution": solution, "num_blanks": puzzle.count("0"),
                "source_index": row["extra_info"]["index"],
                "prompt": messages[0]["content"], "board_size": 4,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return dict(self.data[idx])


def load_sudoku_dataset(data_config):
    source = data_config.get("source", "sudoku_1m")
    if source not in {"sudoku_1m", "bgpo_4x4"}:
        raise ValueError(f"Unknown Sudoku data.source: {source}")
    dataset_type = BGPOSudokuDataset if source == "bgpo_4x4" else SudokuDataset
    return dataset_type(n=data_config["num_samples"], cache_dir=data_config.get("cache_dir"))


def item_prompt(item):
    """Pass the released prompt unchanged; references are never model inputs."""
    return item["prompt"] if "prompt" in item else format_prompt(item["puzzle"])

class SudokuDataset(Dataset):
    def __init__(self, n=100, cache_dir=None):
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError("n must be a positive integer")
        if datasets_config.HF_DATASETS_OFFLINE:
            ds = _load_cached_sudoku(n, cache_dir)
        else:
            ds = load_dataset(
                SUDOKU_REPO_ID,
                split=f"train[:{n}]",
                cache_dir=cache_dir,
                # Preserve leading zeros and avoid integer overflow while preparing the cache.
                features=Features({
                    "puzzle": Value("string"),
                    "solution": Value("string"),
                }),
            )
        self.data = ds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]

        return {
            "puzzle": x["puzzle"],
            "solution": x["solution"],
            "num_blanks": x["puzzle"].count("0"),
        }

def format_prompt(puzzle):
    rows = [puzzle[i:i+9] for i in range(0, 81, 9)]

    board = "\n".join(
        " ".join("." if x == "0" else x for x in row)
        for row in rows
    )

    return f"""Solve this Sudoku puzzle.

{board}

Fill every empty cell while preserving all given numbers.

Return ONLY the completed Sudoku as exactly 81 digits.
Do not include spaces, explanations, labels, or any other text.
"""
