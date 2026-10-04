"""Evaluate original LLaDA decoding budgets on frozen GSM8K examples."""

from dlm_project.experiments.sweep import main


if __name__ == "__main__":
    main("gsm8k")
