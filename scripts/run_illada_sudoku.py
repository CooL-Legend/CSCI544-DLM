"""Evaluate iLLaDA using the shared Sudoku experiment and artifact pipeline."""

from dlm_project.paths import PROJECT_ROOT as ROOT
from dlm_project.tasks.sudoku.runner import main


if __name__ == "__main__":
    main(default_config=ROOT / "configs/sudoku/illada.yaml", expected_family="illada")
