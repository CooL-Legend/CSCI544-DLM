"""GSM8K exact accuracy after reference-free Best-of-5 selection."""

from dlm_project.tasks.gsm8k.decoder_evaluation import main


if __name__ == "__main__":
    main("best_of_five")
