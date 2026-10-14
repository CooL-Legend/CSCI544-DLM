# Project structure

The project uses an editable Python package, `dlm_project`, shared by local commands and Colab notebooks. Run setup commands from the repository root:

```bash
# LLaDA: requirements.txt also installs PyTorch for local environments.
python -m pip install -r requirements.txt -r requirements/dev.txt
python -m pip install --no-deps -e .
```

On Colab, the notebooks install `requirements/llada-colab.txt`, `requirements/illada-colab.txt`, or `requirements/bgpo-colab.txt`, then install the package with `--no-deps -e`. This keeps Colab's CUDA-enabled PyTorch. The package is intended for an editable checkout: its default config and results paths are relative to that checkout, independent of the launch directory.

## Where to read and edit code

| Area | Location | Responsibility |
| --- | --- | --- |
| Model interface | `src/dlm_project/models/llada.py` | Load the HF model/tokenizer, encode a prompt, dispatch decoding, report predictions and statistics. |
| Model identity/cache | `src/dlm_project/models/profiles.py`, `cache.py` | Model IDs, revisions, mask IDs, download validation and staging. |
| MCTS | `src/dlm_project/decoding/mcts_decoding.py` | All search components, all four cycle stages, immediate-child entropy evaluation, adjusted-score sampling and trace events in one file. |
| Baseline | `src/dlm_project/decoding/baseline.py` | Upstream fixed-length sampler. |
| Experiment utilities | `src/dlm_project/experiments/` | Shared configuration validation, run names/directories and atomic JSON saving. |
| Decoder sweeps | `src/dlm_project/experiments/{manifests,sweep,sweep_report,sweep_plots}.py` | Frozen examples, paired baseline/confidence/MCTS runs, incremental records, offline tables and figures. |
| Sudoku | `src/dlm_project/tasks/sudoku/` | Dataset/prompt, scoring, plotting, runner, inspection and length comparison. |
| Paired 4×4 comparison | `src/dlm_project/tasks/sudoku/model_comparison.py` | Sequential original LLaDA/BGPO subprocesses, alignment checks, combined scores and four-panel plots. |
| GSM8K | `src/dlm_project/tasks/gsm8k/` | Single-question loading/prompt construction, numeric answer scoring and runner. |
| Commands | `scripts/` | Thin entry points that call package functions. |
| Settings | `configs/sudoku/`, `configs/gsm8k/` | Model, generation, search and task settings. |
| Notebooks | `notebooks/sudoku/`, `notebooks/gsm8k/` | Colab setup, subprocess generation and inspection of saved artifacts. |
| Tests | `tests/decoding/`, `tests/models/`, `tests/tasks/`, `tests/workflows/` | Tests grouped by the corresponding code area. |

The dependency direction is: **notebook → script → task runner → shared model/decoder/experiment code**. Dataset references enter task scoring only. GSM8K does not import the Sudoku runner. Notebook kernels read saved artifacts; subprocesses load the model.

To follow one MCTS cycle, begin at `MCTSSearch.step()` and read `select_node()`, `expand_node()`, `evaluate_action()`, and `backpropagate()` in the same file. See [the MCTS walkthrough](mcts_walkthrough.md) for a carried example.

## Commands

```bash
# Inspect one expansion, then a complete initialization search.
python scripts/inspect_mcts.py --phase one_step \
  --config configs/sudoku/llada_mcts.yaml --trace-path results/one_step.jsonl
python scripts/inspect_mcts.py --phase search \
  --config configs/sudoku/llada_mcts.yaml --trace-path results/search.jsonl

# Complete a Sudoku prompt using the selected initialization state.
python scripts/run_sudoku.py --config configs/sudoku/llada_mcts.yaml

# View one GSM8K user prompt without loading model weights.
python scripts/run_gsm8k.py --config configs/gsm8k/llada.yaml --prompt-only

# Preview the authors' 4×4 Sudoku test prompts, then compare both models on CUDA.
python scripts/run_bgpo_sudoku.py --dataset-only --num-samples 1
python scripts/run_bgpo_sudoku.py --num-samples 3

# Same original LLaDA checkpoint; five decoder conditions on frozen examples.
python scripts/run_gsm8k_sweep.py --profile smoke
python scripts/run_sudoku_sweep.py --profile smoke

# Run the tests in the installed dependency environment.
python -m pytest -q
```

Use each script's `--help` for additional arguments. iLLaDA uses `scripts/run_illada_sudoku.py` and its separate dependency profile. The cache and fixed-length comparison commands are `scripts/cache_model.py` and `scripts/compare_generation.py`.

See [the milestone sweep guide](milestone1_sweeps.md) for the two sweep configurations,
budget stopping, Colab launches and saved-metric reporting. Sweep entry points call
shared experiment code, which reuses task prompts/scorers and the existing decoder.

Python imports now use the package name:

```python
from dlm_project.models.llada import LLaDA
from dlm_project.decoding.mcts_decoding import MCTSConfig, MCTSSearch
from dlm_project.experiments.config import load_config
from dlm_project.tasks.gsm8k.dataset import build_prompt
```

## Path migration

| Previous location | Current location |
| --- | --- |
| `Llada-Sudoku/Llada.py` | `src/dlm_project/models/llada.py` |
| `Llada-Sudoku/mcts_decoding.py` | `src/dlm_project/decoding/mcts_decoding.py` |
| `Llada-Sudoku/generate.py` | `src/dlm_project/decoding/baseline.py` |
| `Llada-Sudoku/model_profiles.py` | `src/dlm_project/models/profiles.py` |
| `Llada-Sudoku/experiment_paths.py` | `src/dlm_project/experiments/paths.py` |
| `Llada-Sudoku/sudoku_*.py` | `src/dlm_project/tasks/sudoku/{dataset,evaluation,visualization}.py` |
| `Llada-Sudoku/{run_sudoku,run_illada_sudoku,inspect_mcts,compare_generation}.py` | Matching entry points in `scripts/`; implementation in the package. |
| `Llada-GSM8K/run_gsm8k.py` | `scripts/run_gsm8k.py`; implementation split into `tasks/gsm8k/{dataset,evaluation,runner}.py`. |
| `configs/sudoku.yaml`, `sudoku_mcts.yaml`, `illada_sudoku.yaml` | `configs/sudoku/llada.yaml`, `llada_mcts.yaml`, `illada.yaml` |
| `configs/gsm8k.yaml` | `configs/gsm8k/llada.yaml` |
| `notebooks/*Sudoku_Colab.ipynb` | `notebooks/sudoku/` |
| `notebooks/LLaDA_GSM8K_Colab.ipynb` | `notebooks/gsm8k/` |
| `requirements-colab.txt`, `requirements-illada-colab.txt`, `requirements-dev.txt` | `requirements/llada-colab.txt`, `requirements/illada-colab.txt`, `requirements/dev.txt` |

Update personal scripts and notebook bookmarks to the current paths. The previous source paths are no longer entry points. The root `requirements.txt` and `environment.yaml` remain convenience setup files. Imports and Colab commands in this repository already use the new locations.

Downloaded weights, dataset caches, local environments and previous experiment artifacts retain their locations. New local runs still default to `results/`; Drive runs retain their existing model/task/config/timestamp hierarchy. Source files in `src/dlm_project/models/` are tracked; downloaded weights in the root `models/` directory are ignored.
