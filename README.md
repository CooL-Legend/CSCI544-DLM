# CSCI544-DLM

Evaluate **LLaDA-8B-Instruct**, **iLLaDA-8B-Instruct**, and **LLaDA-8B-BGPO-sudoku** on Sudoku for CSCI 544 Applied NLP. Develop locally, run inference on **Google Colab Pro**, and keep model weights and experiment artifacts in **Google Drive**.

The **LLaDA and iLLaDA GSM8K notebooks** evaluate four shared decoding methods and display accuracy, inference cost, and randomly selected saved answers. LLaDA also includes the documented MEDAL reconstruction with immediate-child information gain, cached candidate ranking, and adjusted-score sampling.

| Model | Notebook on `adi` | Dependency profile |
| --- | --- | --- |
| LLaDA | [Open LLaDA in Colab](https://colab.research.google.com/github/CooL-Legend/CSCI544-DLM/blob/adi/notebooks/sudoku/LLaDA_Sudoku_Colab.ipynb) | Transformers 4.38.2 |
| iLLaDA | [Open iLLaDA in Colab](https://colab.research.google.com/github/CooL-Legend/CSCI544-DLM/blob/adi/notebooks/sudoku/iLLaDA_Sudoku_Colab.ipynb) | Transformers 4.57.1, Tokenizers 0.22.2, Hugging Face Hub 0.36.2 |
| Original LLaDA vs BGPO — 4×4 Sudoku | [Open comparison in Colab](https://colab.research.google.com/github/CooL-Legend/CSCI544-DLM/blob/adi/notebooks/sudoku/LLaDA_BGPO_Sudoku_Colab.ipynb) | Transformers 4.53.2 |
| LLaDA — GSM8K decoders | [Open LLaDA GSM8K in Colab](https://colab.research.google.com/github/CooL-Legend/CSCI544-DLM/blob/adi/notebooks/gsm8k/LLaDA_GSM8K_Colab.ipynb) | Transformers 4.38.2 |
| iLLaDA — GSM8K decoders | [Open iLLaDA GSM8K in Colab](https://colab.research.google.com/github/CooL-Legend/CSCI544-DLM/blob/adi/notebooks/gsm8k/iLLaDA_GSM8K_Colab.ipynb) | Transformers 4.57.1, Tokenizers 0.22.2, Hugging Face Hub 0.36.2 |

This repository is private. Authorize Colab to read it through GitHub, or download the desired notebook and use **File → Upload notebook**. An older notebook copy saved in Drive does not automatically receive repository changes. Both GSM8K notebooks check out `adi` and display its commit before running.

## Repository structure

```text
src/dlm_project/
├── models/                 # LLaDA wrapper, model profiles, snapshot caching
├── decoding/               # Baseline sampler and the complete MCTS module
├── experiments/            # Shared configuration, naming, artifact saving
└── tasks/
    ├── sudoku/             # Dataset, scoring, plots, runner, MCTS inspection
    └── gsm8k/              # Prompt construction, answer scoring, runner
scripts/                    # Thin command-line entry points
configs/{sudoku,gsm8k}/      # Task settings
notebooks/{sudoku,gsm8k}/    # Colab workflows
requirements/               # LLaDA, iLLaDA, BGPO comparison, and test dependencies
tests/{decoding,models,tasks,workflows}/
docs/                       # Structure guide and MCTS walkthrough
```

Install the package with `python -m pip install --no-deps -e .` after installing a dependency profile. The Colab notebooks do this automatically. See [the structure guide](docs/project_structure.md) for file locations, imports, and commands. Model weights and run artifacts stay outside the source package.

## Single-prompt MEDAL demo

Use [MEDAL_MCTS_Demo.ipynb](notebooks/gsm8k/MEDAL_MCTS_Demo.ipynb) to see the current method without running an evaluation. It loads one model from the runtime, Drive, or the pinned Hub checkpoint; shows the inline three-subtask guidance; runs MCTS initialization; and displays the top three partial-state candidates with their unmasked tokens, selection values, and root-to-candidate entropy reductions. Short cells report actual NFE by MCTS phase and synchronized wall-clock time, then complete the selected state.

The completion cell reuses the same initialization and accepts `method="medal"`, an imported callable, or `"my_sampler:decode"`. Copy that cell to try a sampler implemented in your own Python module. Each sampler receives a fresh partial-state copy; the helper counts its actual forwards and checks that it preserves the prompt and initialized tokens. Setup and validation stay outside the main action cells. Decomposition is inline prompting, so its generated reasoning appears in the final response and adds no separate planning NFE.

The same flow is available as a script after installing the LLaDA dependencies and package:

```bash
python scripts/demo_medal.py --model-path /path/to/LLaDA-8B-Instruct --local-files-only \
  --prompt-file /path/to/question.txt
```

Omit `--model-path` to load the pinned Hub model, or use `--question "..."` instead of a file. `--demonstrations-path` can point to the existing Drive `manifests/gsm8k_decomposition_train.json`; otherwise the demo caches the two training examples separately. No test dataset, accuracy calculation, or evaluation artifacts are produced. See `--help` for initialization settings and `--decoding-method my_sampler:decode`.

## Milestone 1 decoder sweeps

Four separate search-free GSM8K decoders and their evaluation commands are in the
[decoding methods README](src/dlm_project/decoding/README.md): random MDM,
Top-K Margin, token-probability confidence decoding, and majority-vote Best-of-5.
Both LLaDA and iLLaDA use their pinned checkpoints and the same frozen GSM8K examples; the guide defines
their accuracy, NFE, runtime and method-specific metrics.

The LLaDA GSM8K notebook now has an **Accuracy, compute, and MCTS simulation comparison** section. It loads one model in a subprocess for all four baselines and MCTS caps of 4, 8, 16, 32, and 64 completed simulations. It uses shared MEDAL prompting and generation settings, saves uncertainty signals without additional forwards, and displays a paired metric table plus accuracy-versus-simulations/NFEs/time plots. The [metric definitions and commands](src/dlm_project/decoding/README.md#accuracy-compute-and-search-metrics) explain decoding-step counts, immediate-child evaluation cost, and fallback behavior. Start with two examples; simulation caps are distinct from NFE caps.

The [milestone sweep guide](docs/milestone1_sweeps.md) provides two scripts using the **same pinned original LLaDA** model: GSM8K and BGPO's **4×4 dataset**. Each compares baseline, confidence-only, and fixed-branching MCTS with search caps of **128, 256 and 512 forwards**, using **immediate parent-to-child information gain** as explicit budget-limited MEDAL ablations. It saves frozen manifests, per-example traces/scores/compute, paired summaries, Wilson intervals, and PNG/PDF reports.

```bash
python scripts/run_gsm8k_sweep.py --profile smoke
python scripts/run_sudoku_sweep.py --profile smoke
```

The default smoke profile runs three development examples. `--profile milestone` selects 50 held-out GSM8K questions or 30 disjoint Sudoku puzzles. `--report-only RUN_DIR` rebuilds reports without weights or data downloads. See the guide for YAML controls, cached-model Colab subprocess launches, GPU verification, compute estimates and interrupted-run handling. These scripts do not change the separate original-LLaDA-versus-BGPO model comparison below.

## Original LLaDA vs BGPO on 4×4 Sudoku

[`notebooks/sudoku/LLaDA_BGPO_Sudoku_Colab.ipynb`](notebooks/sudoku/LLaDA_BGPO_Sudoku_Colab.ipynb) compares the original **LLaDA-8B-Instruct** checkpoint with **LLaDA-8B-BGPO-sudoku** on the authors' released **500-puzzle 4×4 test set**. BGPO's upstream Sudoku setup is 4×4; the existing notebooks continue evaluating 9×9 puzzles.

Both models receive the dataset's original user prompt and use identical baseline sampling: **256 output tokens, 256 steps, 32-token blocks, temperature 0, low-confidence remasking, seed 1**. The default is three puzzles. Use one for the initial GPU smoke run, then increase `NUM_SAMPLES` up to 500. Selection takes the first N test rows without shuffling, and each puzzle uses seed `1 + index`. References enter scoring only.

The notebook follows checkout → dependencies → Drive → configuration → caching → evaluation → inspection. It caches each checkpoint separately, reusing the existing original LLaDA snapshot, and loads models **sequentially in fresh subprocesses**. Allow at least 40 GB of Drive storage for both snapshots. Select a BF16-capable L4/A100 runtime as in the existing notebooks. The comparison uses `requirements/bgpo-colab.txt` (Transformers 4.53.2); rerun that installation cell when switching notebooks. BGPO uses its PyTorch attention path with `flash_attention=False` and `flash_attn_varlen=False`, so external FlashAttention is not required.

For a local dataset preview or CUDA evaluation, install the comparison dependencies in a separate environment with appropriate PyTorch already installed:

```bash
python -m pip install -r requirements/bgpo-colab.txt
python -m pip install --no-deps -e .
python scripts/run_bgpo_sudoku.py --dataset-only --num-samples 1
python scripts/run_bgpo_sudoku.py --num-samples 3
```

The runner accepts `--config`, `--num-samples`, `--dataset-only`, and `--run-manifest`. Its default configuration is `configs/sudoku/llada_bgpo_4x4.yaml`: shared data/generation settings, BGPO under `model`, and original LLaDA under `comparison.baseline_model`. For cached local weights, set each model's `model_path` and `local_files_only` separately; the notebook does this automatically. `python scripts/cache_model.py --model llada_bgpo --drive-dir /path/to/models/LLaDA-8B-BGPO-sudoku` uses revision `101ce156b05e4f9fef9688569b4ddc3241f81356`. The test file is pinned to upstream commit `2d3e819a65b42ff5bd9cce958fb4412d57c1609d`.

Per-model results are saved under `LLaDA/Sudoku4x4/<settings>/<timestamp>/` and `LLaDA-BGPO/Sudoku4x4/<settings>/<timestamp>/`. Combined results are under `Comparisons/Sudoku4x4/<settings>/<timestamp>/`:

- `comparison.json`: measured scores, BGPO-minus-original rate differences, run paths, and provenance.
- `comparison.csv`: one summary row per model, including mean generation latency and peak allocated GPU memory.
- `paired_predictions.jsonl`: matched puzzles, references, original prompts, both raw answers and scores, and per-puzzle runtime statistics.
- `plots/puzzle_000.png`, etc.: puzzle, reference, original LLaDA, and BGPO side by side.

The comparison validates puzzle IDs, prompts, decoding settings/seeds, and checkpoint identities before combining results. Failed or interrupted evaluations retain completed artifacts; only a completed pair with all plots receives `status: completed`. Each rerun creates a fresh timestamp.

4×4 scoring requires **one complete `<answer>` block containing exactly 16 digits from 1–4**, allowing whitespace inside the block. Reasoning digits do not enter the answer. Missing tags, truncated answers, extra digits, and malformed responses fail parsing without repair. Exact, cell, completion, clue-preservation, and Sudoku-validity rates include failed outputs in their denominators; cell accuracy uses 16 cells per puzzle. Validity checks rows, columns, and 2×2 boxes independently of clue preservation. Timings exclude model loading; peak memory includes weights. This workflow reports actual paired inference results using these scoring rules, rather than claiming reproduction of the paper's reported scores.

## Existing 9×9 workflows on Colab Pro

1. Open the notebook for the model you want. Select **Runtime → Change runtime type → GPU → L4 or A100**, subject to availability. The notebooks check BF16 support and GPU memory before loading unquantized weights.
2. Give the clone cell read access to the private repository: use a fine-grained GitHub token with **Contents: read** access, stored in Colab **Secrets** as `GITHUB_TOKEN`, or use the hidden prompt. Do not paste tokens into notebook source or Git URLs.
3. Run steps **1–5** to get `adi`, install the selected model's dependencies, mount Drive, load the base configuration, and download/stage the model. A clean checkout can switch to `adi`; local modifications or divergent commits stop the update without discarding work.
4. In **Config Updater**, use `NUM_SAMPLES = 1` for the first verification. LLaDA now starts with the MCTS walkthrough: **256 output positions, steps, and block length**, temperature **1**, `Lc=20`, `C=3`, and `K1=3`, `K2=5`, with no default cap. The iLLaDA notebook retains its baseline defaults (three puzzles and 128-token blocks). The first dataset preparation can process the million-row CSV before selecting your slice.
5. In the LLaDA notebook, run **Inspect one expansion**, then **Inspect initialization**. Change `CYCLE_TO_VIEW` to inspect saved selection, expansion, evaluation and backpropagation tables without another model load. These inspection commands do not complete the output.
6. Run **step 6** for full generation, then inspect predictions, traces, GPU artifact checks and plots in **step 7**. `status: completed` means inference and artifact saving finished, not that the puzzles were solved. After verification, increase `NUM_SAMPLES` and rerun Config Updater and steps 6–7.
7. Check cache reuse by rerunning **step 5** and then evaluation. Unchanged weight files are reused; each evaluation creates a new experiment timestamp. Inspect Sudoku validity and clue preservation before increasing to 10 or 100 puzzles.

Change generation settings in **Config Updater** before rerunning evaluation. The displayed configuration name is derived from those settings. The removed length-comparison cells are not reintroduced.

Each evaluation uses a fresh Python process and loads the model once. If you switch model notebooks in the same runtime, rerun the selected notebook's dependency installation cell first. Inference sees the freshly installed packages; the notebook kernel does not load either 8B model. Keep Colab's CUDA-enabled PyTorch.

The LLaDA Sudoku notebook includes **Check the search on CUDA before loading LLaDA** after dependency installation. It runs three tiny controlled-model tests on the GPU, then the later inspection/generation cells verify the actual HF model reports a CUDA device. Forward passes, softmax, entropy and confidence factors use the GPU; compact tree states and trace bookkeeping remain on CPU to conserve GPU memory. Saved statistics distinguish `model_device` from `search_state_device`.

## GSM8K decoder evaluations

The [LLaDA notebook](notebooks/gsm8k/LLaDA_GSM8K_Colab.ipynb) and [iLLaDA notebook](notebooks/gsm8k/iLLaDA_GSM8K_Colab.ipynb) follow the same checkout → dependencies → Drive → configuration → model cache → four decoding methods → answer inspection flow. Run setup steps 1–5, configure the benchmark, and run the methods you want. The iLLaDA notebook installs `requirements/illada-colab.txt` and caches its own pinned snapshot with mask token **5**.

The iLLaDA configuration controls default to temperature **1.0**, 256 output positions and steps, 128-token blocks, and seed 1. Set `PROFILE` to `smoke` (20 questions), `initial` (100), or `full` (1,319); `NUM_SAMPLES = None` uses that profile's count. Both models reuse the ordered `openai/gsm8k`, `main`, `test` manifest on Drive. References enter only post-generation scoring, using the strict final `#### <number>` line parser.

Each iLLaDA method streams its subprocess output, writes a unique run receipt, and saves artifacts under `iLLaDA/GSM8K/<method>/<profile>/<settings>/<timestamp>/`. Preview cells display metrics and five randomly selected responses from that exact run; rerunning a preview performs no inference. All five Best-of-5 candidates count toward inference cost. The original LLaDA notebook and its saved responses are retained.

The LLaDA notebook appends **MCTS initialization + confidence decoding** as a fifth method. Run its configuration, execution, and preview cells after the shared setup and benchmark configuration. It inherits the active prompt, model/cache paths, generation controls, profile, seed, and current `NUM_SAMPLES = 2` override; set that override to `None` for profile counts. Its run streams errors and selects previews through a unique receipt, displaying accuracy, random saved answers, initialization depth, fallback status, and search/completion costs.

MCTS now uses the [MEDAL paper reconstruction](docs/mcts_walkthrough.md) across GSM8K, Sudoku, inspection, and sweeps. Defaults are Lc=20, C=3, K1=3, K2=5, temperature 1.0, 256 output tokens/steps/global block length, seed 1, and **no forward cap**. Each simulation evaluates one immediate child and backs up its normalized entropy reduction. Cached candidate rewards select the partial state, and literal adjusted-score softmax fills its remaining masks. Inline three-subtask prompting uses disjoint training/synthetic demonstrations. Gamma 10 and exploration constant 0.01 remain repository defaults, not verified author settings. Author-code parity is unverified.

Search costs one root forward plus one forward per evaluated child; candidate ranking adds no forwards. Final completion costs 236 forwards at depth 20. A direct-path example costs **335 forwards**, with more possible under broader exploration. Strict candidate mode fails instead of silently falling back. Explicit finite-budget sweeps remain labeled ablations, retaining only completed simulation rewards. New artifact labels include `medal-v2`; historical runs and the four standalone decoding sections are preserved.

On a configured CUDA machine with the LLaDA dependency profile installed:

```bash
python scripts/eval_gsm8k_mcts_confidence.py --profile smoke --num-samples 2
```

The new CLI defaults to [`configs/gsm8k/mcts_confidence.yaml`](configs/gsm8k/mcts_confidence.yaml), requires `model.family: llada`, and supports the same benchmark overrides and run receipts as the other methods. Edit `decoding.mcts` in YAML or `MCTS_SETTINGS` in the notebook to change the search settings. Results go to `LLaDA/GSM8K/mcts_confidence/<profile>/<generation-and-search-settings>/<timestamp>/`, sharing the frozen test manifest and strict scorer. The four original notebook sections and all their saved outputs remain unchanged.

On a configured CUDA machine with the iLLaDA dependency profile installed:

```bash
python scripts/eval_illada_gsm8k_random_mdm.py --profile smoke
python scripts/eval_illada_gsm8k_top_k_margin.py --profile smoke
python scripts/eval_illada_gsm8k_confidence.py --profile smoke
python scripts/eval_illada_gsm8k_best_of_5.py --profile smoke
```

Each iLLaDA entrypoint defaults to `configs/gsm8k/illada_decoding.yaml` and rejects configurations for other model families. Add `--run-manifest /path/to/receipt.json` to save its exact completed run directory. See [the decoding README](src/dlm_project/decoding/README.md) for all CLI options and metrics. New notebooks and scripts must be available on `adi` before the Colab checkout can use them.

The single-question LLaDA walkthrough remains available through its original CLI. To preview without GPU weights:

```bash
python scripts/run_gsm8k.py --config configs/gsm8k/llada.yaml --prompt-only
```

Run the same command without `--prompt-only` on a configured CUDA machine for generation and prompt-token inspection.

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

The baseline naming pattern is `gen<length>-steps<steps>-blk<block>-temp<temperature>-<remasking>`, with `low_confidence` shortened to `lowconf`. Confidence and MCTS names append their method and settings, including the MCTS depth, branching limits, budget, exploration constant and seed. Timestamps use **DD:MM:YY:HH:mm:SS**, including minutes, in **America/Los_Angeles** by default. Set `EXPERIMENT_TIMEZONE` in the mount cell before loading the configuration to change the timezone. Same-second collisions receive `-02`, `-03`, etc.; existing runs are never overwritten. Inspection follows the exact directory returned by that invocation, not alphabetical timestamp order.

Every processed prediction is appended to `predictions.jsonl`; its record includes the puzzle, reference, unchanged raw response, parsed answer, validation flags, scores, and relative `plot_file`. It also records `decoding_method`, `decoding_stats`, and `trace_file`. Traces are streamed to `traces/puzzle_000.jsonl`, etc.; each event is flushed, so interrupted searches retain completed events. PNGs are saved automatically during evaluation, including invalid and incomplete outputs. The notebook displays those same saved images, with your puzzle-index titles. Saving plots does not depend on running the display cell.

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

The 9×9 baseline configurations are `configs/sudoku/llada.yaml` and `configs/sudoku/illada.yaml`. The LLaDA notebook loads `configs/sudoku/llada_mcts.yaml` for the one-puzzle walkthrough. The shared loader accepts `model.family: llada`, `illada`, or `llada_bgpo`; older configurations without `family` default to LLaDA. The profile's `model_type` records its architecture, so both original LLaDA and BGPO use `llada` model code with separate checkpoint identities. An absent `decoding` section preserves baseline behavior, and `predict(...)` still returns a string.

The iLLaDA tokenizer needs the newer dependency profile and its own chat template. The wrapper uses mask token **5** for iLLaDA and **126336** for LLaDA. Both share the original **fixed-length masked baseline sampler**, single-puzzle inference, and unchanged Sudoku prompt. iLLaDA's optional variable-length decoding is not enabled. Sudoku rules only validate model responses; no solver fills, repairs, or replaces them.

- `gen_length`: output token positions, excluding the prompt.
- `steps`: total denoising iterations.
- `block_length`: token positions per block.
- `temperature`: token-sampling randomness; zero uses no sampling noise.
- `remasking`: `low_confidence` or `random` position selection.

Generation length must divide into whole blocks. Steps must be positive, no greater than generation length, and divisible by the number of blocks. Increasing output length or steps does not guarantee correct Sudoku answers.

## Inspectable MCTS

All search components live in [`src/dlm_project/decoding/mcts_decoding.py`](src/dlm_project/decoding/mcts_decoding.py). Read [`docs/mcts_walkthrough.md`](docs/mcts_walkthrough.md) for the four stages, a carried example, and direct use with an already loaded model.

- `baseline`: the existing sampler, unchanged by default.
- `confidence`: greedy confidence-adjusted decoding from an entirely masked output.
- `mcts_entropy`: MCTS with immediate-child information gain and cached candidate ranking initializes the output; literal adjusted-score sampling completes it.

This is a versioned paper-based MEDAL reconstruction, with explicit choices for immediate entropy-change rewards, cached candidate ranking, and completion sampling. Each action fixes one output token; evaluation performs one forward on that child and leaves its remaining masks in place. Rewards can be negative; references and Sudoku validators never guide search. See the [function flowchart, definitions, and cost accounting](docs/mcts_walkthrough.md).

`predict_with_stats(...)` returns text, stats, and streamed trace events. It reuses one model for every branch, candidate, and final completion. Metrics separate tree analysis, simulation, candidate evaluation, and completion forwards and latency, plus peak allocated GPU memory. Failure preserves completed records and partial traces. Inline prompting is prepared by task runners; the standalone greedy confidence baseline retains its behavior.

On a configured CUDA machine, run each inspection in a fresh process:

```bash
python scripts/inspect_mcts.py --phase one_step \
  --config configs/sudoku/llada_mcts.yaml --trace-path results/one_step.jsonl
python scripts/inspect_mcts.py --phase search \
  --config configs/sudoku/llada_mcts.yaml --trace-path results/search.jsonl
python scripts/run_sudoku.py --config configs/sudoku/llada_mcts.yaml
```

Each command loads one model. The full-generation command performs its own initialization and continues its selected state directly; it does not import the independent inspection tree. The Colab notebook runs these commands and reads their saved traces, keeping weights out of the notebook kernel.

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

Use separate Python environments for the LLaDA, iLLaDA, and BGPO comparison dependency profiles. For the existing LLaDA environment:

```bash
conda env create -f environment.yaml
conda activate diff-dev
# Or, inside an existing environment:
python -m pip install -r requirements.txt -r requirements/dev.txt
python -m pip install --no-deps -e .
```

For iLLaDA, in a separate Python 3.11 environment with suitable PyTorch installed:

```bash
python -m pip install -r requirements/illada-colab.txt -r requirements/dev.txt
python -m pip install --no-deps -e .
```

Dataset-only checks need no model weights or GPU:

```bash
python scripts/run_sudoku.py --dataset-only --num-samples 1
python scripts/run_illada_sudoku.py --dataset-only --num-samples 1
```

Download iLLaDA and evaluate it on a suitable CUDA GPU:

```bash
python scripts/cache_model.py --model illada --drive-dir /path/to/models/iLLaDA-8B-Instruct
python scripts/run_illada_sudoku.py \
  --model-path /path/to/models/iLLaDA-8B-Instruct \
  --local-files-only --num-samples 3
```

Use `--model llada` and `run_sudoku.py` for LLaDA. The downloader defaults to LLaDA when `--model` is omitted. Add `--runtime-dir` to stage downloaded files locally. `--local-files-only` applies to model loading; uncached datasets can still require network access. Apple MPS inference is not configured.

The `scripts/compare_generation.py` CLI remains available for the fixed 128/256-token comparison; it is not part of the current notebooks and uses its explicitly selected comparison directory.

## Validation

Run in each dependency environment:

```bash
python -m pytest -q
```

GitHub Actions tests all three dependency profiles on CPU. Checks cover string data loading, 4×4 and 9×9 scoring, incomplete/invalid plots, model and mask selection, snapshot files, warm staging, experiment names and timezones, same-second collisions, partial artifact persistence, safe branch checkout, and notebook structure. The new iLLaDA GSM8K notebook has cleared outputs; the supplied LLaDA GSM8K reference retains its saved responses. GSM8K checks cover both model families, the four shared decoders, MCTS-to-confidence state handoff, exact run receipts, shared manifests, streamed errors, and previews without inference. Paired-comparison tests cover prompt/ID/seed alignment, separate checkpoint identities, sequential subprocesses, failed comparisons, and the notebook's shared settings. Controlled fake-model tests additionally cover confidence filtering, negative/near-zero entropy rewards, UCB and backups, independent states, fixed commitments, deduplication, exhausted branches, hard call budgets, fallback ranking, trace accounting, and equivalence of `step()` and `run()`.

CPU tests do not establish 8B model quality or GPU behavior. Verify each notebook's one-puzzle run, three-puzzle run, warm-cache rerun, and saved Drive artifacts on Colab Pro.

`tests/decoding/test_mcts_cuda.py` is skipped when CUDA is unavailable. The Colab CUDA-check cell explicitly requires CUDA before invoking these tests; a skipped local run does not count as GPU validation.

Model weights, datasets, credentials, environments, and generated results remain excluded from Git. See `THIRD_PARTY_NOTICES.md` for upstream sampler attribution.

## References

- [LLaDA source and sampler](https://github.com/ML-GSAI/LLaDA)
- [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct)
- [iLLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/iLLaDA-8B-Instruct)
- [Sudoku dataset](https://huggingface.co/datasets/certen/Sudoku-1M-Puzzles)
