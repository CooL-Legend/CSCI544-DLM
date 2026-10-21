"""GSM8K exact accuracy with token-probability confidence decoding."""

from dlm_project.tasks.gsm8k.benchmark import main


if __name__ == "__main__":
    main("confidence")
