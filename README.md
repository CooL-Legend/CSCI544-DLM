# CSCI544-DLM

Evaluate **LLaDA-8B-Instruct** on Sudoku puzzles for CSCI 544 Applied NLP. Develop locally, run model inference on **Google Colab Pro**, and keep model weights and experiment results in **Google Drive**.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CooL-Legend/CSCI544-DLM/blob/main/notebooks/LLaDA_Sudoku_Colab.ipynb)

This is a private repository. Sign in to GitHub and authorize Colab to read private repositories when opening the notebook. If that route is unavailable, download `notebooks/LLaDA_Sudoku_Colab.ipynb` and use **File → Upload notebook** in Colab.

## Run on Colab Pro

1. Open the notebook using the badge above and save a working copy in Drive if desired.
2. Choose **Runtime → Change runtime type → GPU → L4 or A100**, subject to availability. A T4 does not have enough memory headroom for this unquantized setup. Additional system RAM can help during model loading.
3. Give the clone cell read access to this private repository. Create a fine-grained GitHub token scoped to this repository with **Contents: read-only**. Add it to Colab's **Secrets** as `GITHUB_TOKEN` and grant this notebook access, or use the cell's hidden prompt. Do not paste a token into notebook source or a repository URL.
4. Run the cells in order. Approve the Google Drive mount when Colab asks.
5. Keep `NUM_SAMPLES = 1` for the first run. After it succeeds, change it to 10 or 100 and rerun the configuration and evaluation cells.

The notebook installs the project dependencies while keeping Colab's CUDA-enabled PyTorch. It runs evaluation in a fresh Python process, so updated packages are used without keeping old model objects in the notebook kernel.

### Persistent files

```text
MyDrive/CSCI544-DLM/
├── models/LLaDA-8B-Instruct/  # tokenizer, configuration, code, and weight shards
├── cache/                   # dataset download/preparation cache
└── results/<run timestamp>/
    ├── config.json
    ├── predictions.jsonl
    └── metrics.json
```

Allow at least **20 GB of free Drive storage** for the model and dataset, with extra room for results. Colab Pro compute and Google Drive storage are separate allowances.

The first model download writes approximately 16 GB to Drive. Later sessions check Hugging Face's metadata and reuse the same fixed snapshot. Keep the hidden `.cache/huggingface` directory inside the model folder. The snapshot revision is recorded in `scripts/cache_model.py` and in notebook run metadata.

By default, the notebook copies the model files from Drive to the runtime's `/content/models` directory before loading. This takes time each fresh session but avoids repeated weight reads from mounted Drive during subsequent model loads. Set `COPY_MODEL_TO_RUNTIME = False` to load directly from the Drive copy instead. This does **not** reduce GPU memory requirements: the weights still load onto the GPU once for each evaluation invocation. Runtime disk copies disappear when Colab deletes the VM; the Drive copy remains.

## Local setup

From the repository root, create a Python 3.11 environment:

```bash
conda env create -f environment.yaml
conda activate diff-dev
```

If `diff-dev` already exists, install the project packages inside it instead of creating another environment:

```bash
conda activate diff-dev
python -m pip install -r requirements.txt -r requirements-dev.txt
```

The project uses `transformers==4.38.2`, matching upstream LLaDA's instructions. This may downgrade a newer Transformers installation; use an isolated environment. For an NVIDIA machine, ensure its PyTorch installation includes CUDA support.

Check data loading on a Mac or other CPU machine without downloading model weights:

```bash
python Llada-Sudoku/run_sudoku.py --dataset-only --num-samples 1
```

Run inference on a suitable CUDA GPU:

```bash
python Llada-Sudoku/run_sudoku.py --config configs/sudoku.yaml --num-samples 100
```

The current model setup targets NVIDIA CUDA. Apple MPS inference has not been configured or tested. CPU checks and data preparation work independently of model inference.

### Use an existing model directory

Download the complete model once to a persistent directory:

```bash
python scripts/cache_model.py --drive-dir /path/to/persistent/LLaDA-8B-Instruct
```

Run with that local copy:

```bash
python Llada-Sudoku/run_sudoku.py \
  --model-path /path/to/persistent/LLaDA-8B-Instruct \
  --local-files-only \
  --num-samples 10
```

`--local-files-only` applies to model loading. Dataset loading may still need network access unless its files are already cached.

## Configuration and outputs

Edit `configs/sudoku.yaml` to set the number of puzzles, model directory, device, precision, generation parameters, and output directory. Command-line overrides include `--num-samples`, `--model-path`, and `--output-dir`. Run `python Llada-Sudoku/run_sudoku.py --help` for all options.

Defaults are one puzzle, BF16, batch size one, 128 generation tokens, 128 sampling steps, 32-token blocks, and temperature zero. The generation length must be divisible by block length; the sampling steps must be divisible by the number of blocks and no greater than the generation length.

Each run creates a new timestamped folder. The configuration is saved before evaluation. Each completed puzzle adds a JSON line containing its board, target, raw model output, extracted prediction, and scores. Metrics update after every puzzle and show `status: completed` only after the loop finishes. Interrupted runs keep completed records and remain marked `running`; rerunning starts a new run rather than automatically resuming.

- **Exact accuracy:** fraction of predictions matching all 81 digits of the reference solution.
- **Cell accuracy:** matching cells divided by 81 times the number of processed puzzles. Predictions shorter than 81 digits count as zero correct cells.
- The parser preserves the original experiment: extract digits 1–9, then use the first 81. It does not independently validate Sudoku constraints, and extra explanatory digits can affect scoring. Raw responses are retained for inspection.

The dataset uses explicit string features for `puzzle` and `solution`. This prevents integer overflow and preserves leading zeros. Even `--num-samples 1` can prepare all one million CSV rows on the first load before selecting the requested slice.

## Repository layout

```text
Llada-Sudoku/                  # model wrapper, sampler, dataset, evaluation CLI
notebooks/LLaDA_Sudoku_Colab.ipynb
configs/sudoku.yaml           # experiment settings
scripts/cache_model.py        # persistent download and optional runtime copy
tests/                       # CPU checks without real model downloads
environment.yaml             # Python 3.11 Conda environment
requirements*.txt             # local, Colab, and development dependencies
.github/workflows/checks.yaml # GitHub Actions checks
THIRD_PARTY_NOTICES.md        # upstream sampler attribution
```

Model weights, datasets, tokens, local environments, and generated results are excluded from Git. Keep the full model and dataset outside the repository or in ignored directories.

## Validation

```bash
python -m pytest -q
```

GitHub Actions runs CPU checks for CSV string handling, prompt formatting, interrupted-run persistence, configuration validation, model file staging, and notebook structure. These checks do not download the 8B model and do not establish GPU performance or Sudoku accuracy. Run the notebook's one-puzzle experiment on Colab to verify end-to-end inference.

## References

- [LLaDA source and inference instructions](https://github.com/ML-GSAI/LLaDA)
- [LLaDA-8B-Instruct model](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
- [Sudoku-1M-Puzzles dataset](https://huggingface.co/datasets/certen/Sudoku-1M-Puzzles)
- [Hugging Face persistent downloads](https://huggingface.co/docs/huggingface_hub/guides/download#download-files-to-a-local-folder)
- [Colab resources and storage FAQ](https://research.google.com/colaboratory/faq.html)
