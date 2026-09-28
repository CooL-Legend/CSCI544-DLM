"""Parse and validate generated Sudoku answers without repairing them."""

import re


def box_size(board_size):
    if board_size not in (4, 9):
        raise ValueError("Sudoku board_size must be 4 or 9.")
    return 2 if board_size == 4 else 3


def answer_body(output, board_size=9):
    """Return the sole tagged 4x4 answer, or unchanged legacy 9x9 text."""
    box_size(board_size)
    if board_size == 9:
        return output
    if output.count("<answer>") != 1 or output.count("</answer>") != 1:
        return None
    match = re.search(r"<answer>(.*?)</answer>", output, flags=re.DOTALL)
    return match.group(1) if match else None


def score_prediction(output, target, board_size=9):
    """Score only a complete parsed answer; partial output earns no cell credit."""
    body = answer_body(output, board_size)
    compact = re.sub(r"\s+", "", body or "")
    cells = board_size ** 2
    digits = rf"[1-{board_size}]{{{cells}}}"

    if re.fullmatch(digits, compact):
        prediction = compact
    elif board_size == 9:
        match = re.search(r"(?<!\d)([1-9]{81})(?!\d)", output)
        prediction = match.group(1) if match else ""
    else:
        prediction = ""

    return {
        "prediction": prediction,
        "parse_success": len(prediction) == cells,
        "generated_digit_count": len(re.findall(r"[0-9]", output)),
        "exact": prediction == target,
        "correct_cells": sum(
            p == t for p, t in zip(prediction, target)
        ),
    }


def preserves_clues(puzzle, prediction, board_size=9):
    box_size(board_size)
    cells = board_size ** 2
    if (not re.fullmatch(rf"[0-{board_size}]{{{cells}}}", puzzle)
            or not re.fullmatch(rf"[1-{board_size}]{{{cells}}}", prediction)):
        return False

    return all(
        given == "0" or given == pred
        for given, pred in zip(puzzle, prediction)
    )


def is_valid_sudoku(prediction, board_size=9):
    box = box_size(board_size)
    if not re.fullmatch(rf"[1-{board_size}]{{{board_size ** 2}}}", prediction):
        return False

    grid = [
        [int(x) for x in prediction[i:i+board_size]]
        for i in range(0, board_size ** 2, board_size)
    ]

    required = set(range(1, board_size + 1))

    # Rows
    if any(set(row) != required for row in grid):
        return False

    # Columns
    if any(
        {grid[r][c] for r in range(board_size)} != required
        for c in range(board_size)
    ):
        return False

    # Square subgrids (2x2 or 3x3).
    for br in range(0, board_size, box):
        for bc in range(0, board_size, box):
            values = {
                grid[r][c]
                for r in range(br, br + box)
                for c in range(bc, bc + box)
            }
            if values != required:
                return False

    return True
