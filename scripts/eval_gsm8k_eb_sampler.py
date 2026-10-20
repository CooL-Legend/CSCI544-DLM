"""LLaDA GSM8K accuracy and adaptive decoding cost with EB-Sampler."""

from dlm_project.tasks.gsm8k.benchmark import main


if __name__ == "__main__":
    main("eb_sampler")
