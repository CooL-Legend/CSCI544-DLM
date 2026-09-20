"""Evaluate iLLaDA using the shared Sudoku experiment and artifact pipeline."""

from run_sudoku import ROOT, main


if __name__ == "__main__":
    main(default_config=ROOT / "configs/illada_sudoku.yaml", expected_family="illada")
