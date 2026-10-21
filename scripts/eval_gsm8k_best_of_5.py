"""Evaluate GSM8K accuracy using five confidence samples and answer voting."""

from dlm_project.tasks.gsm8k.benchmark import main


if __name__ == "__main__":
    main("best_of_5")
