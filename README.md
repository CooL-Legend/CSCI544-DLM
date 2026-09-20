# CSCI544-DLM

Evaluate **LLaDA-8B-Instruct** on Sudoku puzzles for CSCI 544 Applied NLP. Develop locally, run model inference on **Google Colab Pro**, and keep model weights and experiment results in **Google Drive**.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CooL-Legend/CSCI544-DLM/blob/main/notebooks/LLaDA_Sudoku_Colab.ipynb)

This is a private repository. Sign in to GitHub and authorize Colab to read private repositories when opening the notebook. If that route is unavailable, download `notebooks/LLaDA_Sudoku_Colab.ipynb` and use **File → Upload notebook** in Colab.

## Run on Colab Pro

1. Open the notebook using the badge above and save a working copy in Drive if desired.
2. Choose **Runtime → Change runtime type → GPU → L4 or A100**, subject to availability. A T4 does not have enough memory headroom for this unquantized setup. Additional system RAM can help during model loading.
3. Give the clone cell read access to this private repository. Create a fine-grained GitHub token scoped to this repository with **Contents: read-only**. Add it to Colab's **Secrets** as `GITHUB_TOKEN` and grant this notebook access, or use the cell's hidden prompt. Do not paste a token into notebook source or a repository URL.
4. Run notebook steps **1–5** in order to get the code, install dependencies, mount Drive, prepare the configuration, and cache the model. Approve the Google Drive mount when Colab asks.
5. **Verify one puzzle:** keep `NUM_SAMPLES = 1`, run step **6**, and inspect the metrics, raw response, and chart in step **7**. `status: completed` means the run finished; it does not mean the puzzle was solved.
6. **Compare generation lengths:** run step **8**. It evaluates the same first **three puzzles** at **128 and 256 tokens**, with the prompt, **128 sampling steps**, **32-token blocks**, model, temperature, and remasking settings held fixed. The model loads once for both runs. This comparison always uses three puzzles, regardless of `NUM_SAMPLES`.
7. **Inspect saved results:** step **9** displays the comparison table and all six raw responses/charts. Review completion, clue preservation, Sudoku validity, exact accuracy, and cell accuracy together. Check each run's saved configuration and predictions. More generated tokens do not guarantee a correct Sudoku solution.
8. **Scale only after review:** if appropriate, change `NUM_SAMPLES` to 10 or 100 in step **3**, rerun configuration step **4**, then evaluation and inspection steps **6–7**. To change the generation length for this larger run, edit `config["generation"]["gen_length"]` in step **4** before it writes `RUN_CONFIG`. The three-puzzle comparison is a diagnostic, not a reliable benchmark.

Open the updated notebook from GitHub (or upload the updated file) to get these new cells; an older copy saved in Drive does not automatically receive notebook edits. All committed cell outputs and execution counts are cleared. The verification inspection selects the run just created, and the comparison table reads its explicitly named folder, so neither silently falls back to an older run.

The notebook installs the project dependencies while keeping Colab's CUDA-enabled PyTorch. It runs evaluation in a fresh Python process, so updated packages are used without keeping old model objects in the notebook kernel.

### Persistent files

```text
MyDrive/CSCI544-DLM/
├── models/LLaDA-8B-Instruct/  # tokenizer, configuration, code, and weight shards
├── cache/                   # dataset download/preparation cache
├── results/<run timestamp>/
│   ├── config.json
│   ├── predictions.jsonl
│   └── metrics.json
└── comparisons/<comparison timestamp>/
    ├── puzzles.json          # same three inputs for both runs
    ├── comparison.json       # status, metrics, and exact result paths
    ├── comparison.csv        # comparison table
    ├── length-128/<run timestamp>/  # config, predictions, metrics
    └── length-256/<run timestamp>/  # config, predictions, metrics
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

Each run creates a new timestamped folder. The configuration is saved before evaluation. Each completed puzzle adds a JSON line containing its board, target, raw model output, extracted prediction, digit count, scores, and `parse_success`, `clues_preserved`, and `valid_sudoku` flags. Metrics update after every puzzle and show `status: completed` only after the loop finishes. Interrupted runs keep completed records and remain marked `running`; rerunning starts a new run rather than automatically resuming. Comparison summaries update after each completed length, preserving the first run if the second is interrupted.

- **Exact accuracy:** fraction of predictions matching all 81 digits of the reference solution.
- **Cell accuracy:** matching cells divided by 81 times the number of processed puzzles. Unparseable responses count as zero correct cells, retaining the existing scoring definition.
- **Completion rate** (`completion_rate`): fraction of processed responses with a complete parsed 81-digit answer. Completion does not imply correctness.
- **Clue-preservation rate** (`clue_preservation_rate`): fraction of processed responses whose parsed answer preserves every given clue.
- **Sudoku-validity rate** (`sudoku_validity_rate`): fraction of processed responses whose parsed answer has digits 1–9 exactly once in every row, column, and 3×3 box. This checks Sudoku structure independently of the clues; a valid grid for a different puzzle can still fail clue preservation.

All rates use **processed puzzles**, including incomplete answers, as the denominator. The parser accepts exactly 81 digits in the range 1–9 after removing whitespace, or a standalone contiguous 81-digit answer embedded in text. It does not truncate excess digits or pad missing ones. Failed parsing produces an empty `prediction`, zero matching cells, and false validation flags. Raw responses are always retained.

The chart preserves incomplete numeric lines in their original row positions. For example, eight rows of eight digits appear as eight rows of eight, with the ninth column and ninth row blank and the label **“Incomplete: 64/81 digits”**. Complete parsed grids are visibly flagged when they conflict with clues or break Sudoku rules. Responses whose layout cannot fit in nine rows of nine columns appear as raw text rather than being reshaped. Red digits indicate clue conflicts at the displayed positions.

This remains **LLaDA-only evaluation**: Sudoku rules report validity; no solver repairs, fills, or replaces model answers.

To run the same fixed three-puzzle comparison outside the notebook, use a configuration pointing to your model and dataset cache:

```bash
python Llada-Sudoku/compare_generation.py --config /path/to/sudoku-run.yaml
```

The default comparison destination is `results/comparisons/<timestamp>` in the repository. Use `--output-dir /path/to/new/comparison` to choose a new folder. The notebook sets this to a timestamped folder on Drive.

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

GitHub Actions runs CPU checks for CSV string handling, prompt formatting, interrupted-run persistence, configuration validation, model file staging, and notebook structure. Regression checks cover the supplied 64-digit response and its plotted coordinates, valid and invalid complete grids, clue conflicts, aggregate metrics, and identical inputs/settings across the two comparison runs. These checks do not download the 8B model and do not establish GPU performance or Sudoku accuracy. Run notebook steps **6–9** on Colab Pro to verify actual one-puzzle and comparison behavior.

## References

- [LLaDA source and inference instructions](https://github.com/ML-GSAI/LLaDA)
- [LLaDA-8B-Instruct model](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
- [Sudoku-1M-Puzzles dataset](https://huggingface.co/datasets/certen/Sudoku-1M-Puzzles)
- [Hugging Face persistent downloads](https://huggingface.co/docs/huggingface_hub/guides/download#download-files-to-a-local-folder)
- [Colab resources and storage FAQ](https://research.google.com/colaboratory/faq.html)
