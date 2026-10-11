"""Evaluate GSM8K accuracy using top-two-margin position selection."""

from dlm_project.tasks.gsm8k.benchmark import main


if __name__ == "__main__":
    main("top_k_margin")
