"""Evaluate original LLaDA decoding budgets on frozen 4x4 Sudoku examples."""

from dlm_project.experiments.sweep import main


if __name__ == "__main__":
    main("sudoku")
