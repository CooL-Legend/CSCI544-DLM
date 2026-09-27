# Third-party notices

`src/dlm_project/decoding/baseline.py` is derived from the sampling implementation in
[ML-GSAI/LLaDA](https://github.com/ML-GSAI/LLaDA/blob/main/generate.py).
Credit belongs to the upstream LLaDA authors. The surrounding Sudoku evaluation,
Colab workflow, and project configuration are maintained in this repository.

The pretrained weights and tokenizer are downloaded separately from
[GSAI-ML/LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct).
The upstream project identifies LLaDA-8B-Instruct as MIT-licensed in its
[license section](https://github.com/ML-GSAI/LLaDA#license).

The dataset is downloaded separately from
[certen/Sudoku-1M-Puzzles](https://huggingface.co/datasets/certen/Sudoku-1M-Puzzles),
whose dataset card lists an MIT license.

The GSM8K prompt walkthrough downloads question/answer examples separately from
[openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k). Its dataset card lists
the MIT license; the original source is [openai/grade-school-math](https://github.com/openai/grade-school-math).

This repository does not redistribute model weights or dataset files. No new
license is granted here for the project's original coursework code.
