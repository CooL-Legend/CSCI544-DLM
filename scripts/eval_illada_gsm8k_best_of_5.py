"""Evaluate iLLaDA GSM8K accuracy using five confidence samples and voting."""

from dlm_project.paths import PROJECT_ROOT
from dlm_project.tasks.gsm8k.benchmark import main


if __name__ == "__main__":
    main("best_of_5", default_config=PROJECT_ROOT / "configs/gsm8k/illada_decoding.yaml",
         expected_family="illada")
