# sudoku_dataset.py
from datasets import Features, Value, load_dataset
from torch.utils.data import Dataset

class SudokuDataset(Dataset):
    def __init__(self, n=100, cache_dir=None):
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError("n must be a positive integer")
        ds = load_dataset(
            "certen/Sudoku-1M-Puzzles",
            split=f"train[:{n}]",
            cache_dir=cache_dir,
            # Boards are 81-character strings; preserve leading zeros and avoid integer overflow.
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
