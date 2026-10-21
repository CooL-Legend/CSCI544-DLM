"""GSM8K exact accuracy with margin-ranked LLaDA commitments."""

from dlm_project.tasks.gsm8k.decoder_evaluation import main


if __name__ == "__main__":
    main("topk_margin")
