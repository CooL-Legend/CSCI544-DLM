"""Parse and validate generated Sudoku answers without repairing them."""

import re


def score_prediction(output, target):
    """Score only a complete parsed answer; partial output earns no cell credit."""
    compact = re.sub(r"\s+", "", output)

    if re.fullmatch(r"[1-9]{81}", compact):
        prediction = compact
    else:
        match = re.search(r"(?<!\d)([1-9]{81})(?!\d)", output)
        prediction = match.group(1) if match else ""

    return {
        "prediction": prediction,
        "parse_success": len(prediction) == 81,
        "generated_digit_count": len(re.findall(r"[0-9]", output)),
        "exact": prediction == target,
        "correct_cells": sum(
            p == t for p, t in zip(prediction, target)
        ),
    }


def preserves_clues(puzzle, prediction):
    if not re.fullmatch(r"[0-9]{81}", puzzle) or not re.fullmatch(r"[1-9]{81}", prediction):
        return False

    return all(
        given == "0" or given == pred
        for given, pred in zip(puzzle, prediction)
    )


def is_valid_sudoku(prediction):
    if not re.fullmatch(r"[1-9]{81}", prediction):
        return False

    grid = [
        [int(x) for x in prediction[i:i+9]]
        for i in range(0, 81, 9)
    ]

    required = set(range(1, 10))

    # Rows
    if any(set(row) != required for row in grid):
        return False

    # Columns
    if any(
        {grid[r][c] for r in range(9)} != required
        for c in range(9)
    ):
        return False

    # 3x3 boxes
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            box = {
                grid[r][c]
                for r in range(br, br + 3)
                for c in range(bc, bc + 3)
            }
            if box != required:
                return False

    return True
