# CSCI544-DLM

Evaluate **LLaDA-8B-Instruct** and **iLLaDA-8B-Instruct** on Sudoku for CSCI 544 Applied NLP. Develop locally, run inference on **Google Colab Pro**, and keep model weights and experiment artifacts in **Google Drive**.

| Model | Notebook on `adi` | Dependency profile |
| --- | --- | --- |
| LLaDA | [Open LLaDA in Colab](https://colab.research.google.com/github/CooL-Legend/CSCI544-DLM/blob/adi/notebooks/LLaDA_Sudoku_Colab.ipynb) | Transformers 4.38.2 |
| iLLaDA | [Open iLLaDA in Colab](https://colab.research.google.com/github/CooL-Legend/CSCI544-DLM/blob/adi/notebooks/iLLaDA_Sudoku_Colab.ipynb) | Transformers 4.57.1, Tokenizers 0.22.2, Hugging Face Hub 0.36.2 |

This repository is private. Authorize Colab to read it through GitHub, or download the desired notebook and use **File → Upload notebook**. An older notebook copy saved in Drive does not automatically receive repository changes. Both notebooks check out `adi` and display its commit before running.

## Run on Colab Pro

1. Open the notebook for the model you want. Select **Runtime → Change runtime type → GPU → L4 or A100**, subject to availability. The notebooks check BF16 support and GPU memory before loading unquantized weights.
2. Give the clone cell read access to the private repository: use a fine-grained GitHub token with **Contents: read** access, stored in Colab **Secrets** as `GITHUB_TOKEN`, or use the hidden prompt. Do not paste tokens into notebook source or Git URLs.
3. Run steps **1–5** to get `adi`, install the selected model's dependencies, mount Drive, load the base configuration, and download/stage the model. A clean checkout can switch to `adi`; local modifications or divergent commits stop the update without discarding work.
4. In **Config Updater**, set `NUM_SAMPLES = 1` for the first verification. It writes the actual configuration and checks the dataset. The first dataset preparation can process the million-row CSV before selecting your slice.
5. Run **step 6**, then inspect all saved predictions and plots in **step 7**. `status: completed` means inference and artifact saving finished, not that the puzzles were solved.
6. Set `NUM_SAMPLES = 3`, rerun **Config Updater**, and rerun **steps 6–7** for the configured experiment. Defaults preserve the `adi` notebook's generation length **256**, **256 steps**, **128-token blocks**, temperature **0.0**, and **low_confidence** remasking.
7. Check cache reuse by rerunning **step 5** and then evaluation. Unchanged weight files are reused; each evaluation creates a new experiment timestamp. Inspect Sudoku validity and clue preservation before increasing to 10 or 100 puzzles.

Change generation settings in **Config Updater** before rerunning evaluation. The displayed configuration name is derived from those settings. The removed length-comparison cells are not reintroduced.

Each evaluation uses a fresh Python process and loads the model once. If you switch model notebooks in the same runtime, rerun the selected notebook's dependency installation cell first. Inference sees the freshly installed packages; the notebook kernel does not load either 8B model. Keep Colab's CUDA-enabled PyTorch.

## Drive layout and artifacts

```text
MyDrive/CSCI544-DLM/
├── models/
│   ├── LLaDA-8B-Instruct/
│   └── iLLaDA-8B-Instruct/
├── cache/
│   ├── hub/
│   └── datasets/
├── LLaDA/Sudoku/
│   └── gen256-steps256-blk128-temp0-lowconf/
│       └── 20:09:26:14:35:07/
│           ├── config.json
│           ├── metrics.json
│           ├── predictions.jsonl
│           └── plots/
│               ├── puzzle_000.png
│               ├── puzzle_001.png
│               └── puzzle_002.png
└── iLLaDA/Sudoku/
    └── gen256-steps256-blk128-temp0-lowconf/
        └── 20:09:26:14:35:07/  # same artifact structure
```

The naming pattern is `gen<length>-steps<steps>-blk<block>-temp<temperature>-<remasking>`, with `low_confidence` shortened to `lowconf`. Timestamps use **DD:MM:YY:HH:mm:SS**, including minutes, in **America/Los_Angeles** by default. Set `EXPERIMENT_TIMEZONE` in the mount cell before loading the configuration to change the timezone. Same-second collisions receive `-02`, `-03`, etc.; existing runs are never overwritten. Inspection follows the exact directory returned by that invocation, not alphabetical timestamp order.

Every processed prediction is appended to `predictions.jsonl`; its record includes the puzzle, reference, unchanged raw response, parsed answer, validation flags, scores, and relative `plot_file`. PNGs are saved automatically during evaluation, including invalid and incomplete outputs. The notebook displays those same saved images, with your puzzle-index titles. Saving plots does not depend on running the display cell.

`config.json` records the resolved settings, actual sample count, model identity/revision, Git commit, experiment name, timezone, and local/UTC start times. `metrics.json` updates after each prediction and saved plot; `plots_saved` reports progress. The run becomes `completed` only after every expected plot is saved. Interrupted runs retain completed artifacts and remain `running`; rerunning creates another folder rather than resuming.

This layout applies to **new runs only**. Older Drive result folders and existing LLaDA model files are not moved or deleted.

## Persistent model caching

Allow at least **20 GB of free Drive storage per model snapshot**, with extra space for the shared dataset cache and experiment artifacts. Keeping both models requires storage for both snapshots. Colab Pro compute and Drive storage are separate allowances.

Each notebook downloads its model to a separate persistent directory. Downloads are pinned to:

- LLaDA: `08b83a6feb34df1a6011b80c3c00c7563e963b07`
- iLLaDA: `5769f049224b2cff61106ebce0429ede5deeb434`

The downloader includes tokenizer files, chat templates (including `.jinja`), configuration, Python model code, and the safetensors shards listed in the checkpoint's index. It checks the selected model family and snapshot identity. Keep the hidden `.cache/huggingface` directory for download reuse. Interrupted downloads can be rerun.

By default, model files are staged to `/content/models/<model-name>` for faster subsequent loads in the same session. Unchanged runtime files are skipped; incomplete copies are repaired. `COPY_MODEL_TO_RUNTIME = False` loads directly from Drive. Runtime files disappear when Colab discards its VM; Drive files remain. This copying choice does not reduce GPU memory requirements.

## Configuration and inference

LLaDA uses `configs/sudoku.yaml`; iLLaDA uses `configs/illada_sudoku.yaml`. Both expose the same generation settings. The shared loader accepts `model.family: llada` or `illada`; older configurations without `family` default to LLaDA.

The iLLaDA tokenizer needs the newer dependency profile and its own chat template. The wrapper uses mask token **5** for iLLaDA and **126336** for LLaDA. Both use the same **fixed-length masked sampler**, single-puzzle inference, and unchanged Sudoku prompt. iLLaDA's optional variable-length decoding is not enabled. Sudoku rules only validate model responses; no solver fills, repairs, or replaces them.

- `gen_length`: output token positions, excluding the prompt.
- `steps`: total denoising iterations.
- `block_length`: token positions per block.
- `temperature`: token-sampling randomness; zero uses no sampling noise.
- `remasking`: `low_confidence` or `random` position selection.

Generation length must divide into whole blocks. Steps must be positive, no greater than generation length, and divisible by the number of blocks. Increasing output length or steps does not guarantee correct Sudoku answers.

`experiment_root` selects the directory above `<model>/Sudoku/<config-name>`. Local runs default to the repository's ignored `results/` directory; Colab sets this to `MyDrive/CSCI544-DLM`. Legacy configurations containing only `output_dir` still treat that directory as the experiment parent. The explicit `--output-dir` CLI override also bypasses model/dataset grouping.

## Scores and plots

All rates use **processed puzzles**, including incomplete responses, as the denominator:

- **Exact accuracy:** fraction matching the reference solution in all 81 cells.
- **Cell accuracy:** matching cells divided by `81 × processed puzzles`. Unparseable outputs receive zero matching cells.
- **Completion rate:** fraction with a parsed 81-digit answer.
- **Clue-preservation rate:** fraction whose parsed answer preserves every given clue.
- **Sudoku-validity rate:** fraction whose parsed answer contains digits 1–9 exactly once in each row, column, and 3×3 box. This is independent of clue preservation.

The parser accepts exactly 81 digits in 1–9 after removing whitespace, or a standalone contiguous 81-digit answer embedded in text. It does not truncate excess digits or pad missing ones. Raw outputs are retained. The dataset loads `puzzle` and `solution` as strings, preventing integer overflow and preserving leading zeros.

Incomplete numeric lines remain at their original row positions in the plot. Eight rows of eight digits are labeled **Incomplete: 64/81 digits**, with the ninth row and column blank. Layouts that cannot be represented without reshaping appear as raw text. Complete invalid grids are flagged; red digits conflict with given clues.

## Local commands

Use separate Python environments for the two dependency profiles. For the existing LLaDA environment:

```bash
conda env create -f environment.yaml
conda activate diff-dev
# Or, inside an existing environment:
python -m pip install -r requirements.txt -r requirements-dev.txt
```

For iLLaDA, in a separate Python 3.11 environment with suitable PyTorch installed:

```bash
python -m pip install -r requirements-illada-colab.txt -r requirements-dev.txt
```

Dataset-only checks need no model weights or GPU:

```bash
python Llada-Sudoku/run_sudoku.py --dataset-only --num-samples 1
python Llada-Sudoku/run_illada_sudoku.py --dataset-only --num-samples 1
```

Download iLLaDA and evaluate it on a suitable CUDA GPU:

```bash
python scripts/cache_model.py --model illada --drive-dir /path/to/models/iLLaDA-8B-Instruct
python Llada-Sudoku/run_illada_sudoku.py \
  --model-path /path/to/models/iLLaDA-8B-Instruct \
  --local-files-only --num-samples 3
```

Use `--model llada` and `run_sudoku.py` for LLaDA. The downloader defaults to LLaDA when `--model` is omitted. Add `--runtime-dir` to stage downloaded files locally. `--local-files-only` applies to model loading; uncached datasets can still require network access. Apple MPS inference is not configured.

The earlier `compare_generation.py` CLI remains available for the fixed 128/256-token comparison; it is not part of the current notebooks and uses its explicitly selected comparison directory.

## Validation

Run in each dependency environment:

```bash
python -m pytest -q
```

GitHub Actions tests both profiles on CPU. Checks cover string data loading, scoring, incomplete/invalid plots, model and mask selection, snapshot files, warm staging, experiment names and timezones, same-second collisions, partial artifact persistence, safe branch checkout, and both notebooks' structure and cleared outputs.

CPU tests do not establish 8B model quality or GPU behavior. Verify each notebook's one-puzzle run, three-puzzle run, warm-cache rerun, and saved Drive artifacts on Colab Pro.

Model weights, datasets, credentials, environments, and generated results remain excluded from Git. See `THIRD_PARTY_NOTICES.md` for upstream sampler attribution.

## References

- [LLaDA source and sampler](https://github.com/ML-GSAI/LLaDA)
- [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
- [iLLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/iLLaDA-8B-Instruct)
- [Sudoku dataset](https://huggingface.co/datasets/certen/Sudoku-1M-Puzzles)
