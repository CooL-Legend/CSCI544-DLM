# Offline LLaDA / MEDAL / AdaLook experiments

This entry point follows Sean's `.env` / sequential-run / artifact layout, but
uses **chong's checked-in `gsm8k_shared` manifest**. It does not download datasets,
modify the frozen E0/E1 protocol, train a model, or merge Sean's older experiment
implementation. The optional first-N selection is a diagnostic subset.

## Run

Use the existing LLaDA Python environment (see `requirements/llada-colab.txt`).
From the repository root, copy `.env.example` to `.env` if no `.env` exists.
Set `MODEL_CACHE_DIR`, `DATASET_DIR`, and `RESULTS_DIR`. The model path is the HF
Hub cache root, containing `models--GSAI-ML--LLaDA-8B-Instruct/snapshots/` and the
pinned checkpoint. A parent directory containing `huggingface/hub` also works.
The launcher sets offline cache variables before importing Transformers.

`DATASET_MANIFEST` defaults to `configs/gsm8k/experiment_samples.json` in this
checkout. An optional override must contain the same frozen shared manifest.
Sean's older `gsm8k_e0` manifest is incompatible. `DATASET_DIR` configures the HF
cache for compatibility; this runner reads the questions directly from the
checked-in manifest and does not need cached train/test splits.

```bash
python scripts/run_gsm8k_experiments.py --dry-run

# Pure LLaDA, then the adaptive MEDAL variant; model loads once.
python scripts/run_gsm8k_experiments.py --method both --num-samples 5 \
  --search-budget 256 --reward-mode adalook_adaptive

# Controlled original-IG comparison on identical questions and seed.
python scripts/run_gsm8k_experiments.py --method mcts --num-samples 5 \
  --search-budget 256 --reward-mode ig

# Score-only ablation.
python scripts/run_gsm8k_experiments.py --method mcts --num-samples 5 \
  --search-budget 256 --reward-mode adalook_one_step

# Tune only on development questions before held-out evaluation.
python scripts/run_gsm8k_experiments.py --method mcts --stage dev --num-samples 5 \
  --reward-mode adalook_adaptive --adalook-max-depth 2 \
  --adalook-variance-threshold 0.01 --adalook-confidence-threshold 0.8
```

Default stage is `test`; seed is 1, Lc is 10, search budget is 256, K1=3,
K2=5 and shortlist size is 3. `--lc`, `--num-candidates`, `--top-k-tokens`,
`--top-k-actions`, `--exploration-c`, and `--seed` are configurable. Generation
uses the shared 256-token length, prompt, pinned model and T=0 LLaDA completion.
`--method llada` runs only the pure baseline. Every separate invocation reuses
the same frozen first-N examples. `--help` lists all parameters.

## Algorithm and boundaries

Source: [Adaptive Multi-Step Lookahead Decoding for Diffusion Language Models,
arXiv:2607.15655v1](https://arxiv.org/html/2607.15655v1), Section 4 and Algorithm 1.
The paper's introduction contradicts its method about the variance direction;
we follow Section 4.2 and Algorithm 1: **continue if variance < tau**.

This implementation is an **AdaLook-inspired MEDAL hybrid**, not a reproduction
of the paper's ETE / fast-block decoder. It retains MEDAL candidate filtering,
UCB, one-token tree edges, final candidate selection, and completion. At each
selected parent, a sibling group receives one-step analyses before optional
temporary rollouts. The reward for an original action a is:

```
alpha * log(max(p(a | parent), epsilon))
    + log(max(cumulative downstream confidence mass, epsilon))
```

The first analysis counts as rollout depth 1. In `adalook_one_step` mode, mass
is the sum of native maximum probabilities >= the confidence threshold at
still-masked positions, excluding proposals equal to MASK/EOS/EOT. No extra
forward is used. Zero mass uses the epsilon floor and remains finite.

Adaptive mode temporarily commits those positions, evaluates the resulting
state, and adds newly available confidence mass. Each position contributes once
along a trajectory; mass for already committed positions is never recounted.
Terminal temporary states need no forward. The tree node itself is never changed.

The global masked suffix replaces the paper's block frontier. A local retrigger
requires at least `adalook_min_masks` positions and average native confidence
below `adalook_trigger_threshold`. With no retriggers, stable trajectories
continue; with mixed retriggers, uncertain trajectories stop receiving compute;
with all retriggers, the best temporary trajectory (score + log mean confidence)
branches through its cached MEDAL actions. This reuses its forward and respects
the same total depth limit. Exploratory commits do not invent downstream mass.
Outer candidates removed from the temporary pool keep their last valid score;
multiple local alternatives for one original action contribute their maximum.
Each original child receives exactly one finalized backup.

Rollout stops at sufficient score variance, the depth limit, one remaining
stable candidate, no legal progress, or insufficient remaining budget. All
rollout sequence evaluations consume the same search cap as root/child analysis.
An additional round runs only if **all its active candidates** fit before the
next checkpoint/cap. Otherwise their preceding complete-round scores are used.
Checkpoint boundaries can split sibling groups; finalized snapshots never use
later evidence. A checkpoint sweep is therefore a distinct configuration from
a run without intermediate checkpoints. Interruptions retain provisional trace
evidence; provisional children have not been backed up.

Defaults: alpha=0.1, max depth=2, confidence threshold=0.8, variance threshold=0.01,
trigger threshold=0.2, minimum masks=2. These are starting settings for this
hybrid, **not a claim of tuned paper-equivalent hyperparameters**. In particular
tau needs development calibration. The reward scale also changes relative to
UCB exploration; log its constant and calibrate separately when needed.

## Artifacts and recovery

Each run gets a new UTC timestamp directory:

```
RESULTS_DIR/LLaDA/GSM8K/llada_vs_mcts/<both|llada|mcts>/<stage>/budget-lc<Lc>/<timestamp>/
  config.json
  manifest.json
  selection.json
  predictions.jsonl
  traces/<condition>_<example>_attempt<N>.jsonl
  summary.json
  report/report.md
  report/summary.csv
  report/comparison.png
```

`config.json` includes all AdaLook parameters and an implementation fingerprint.
Condition IDs include reward mode, budget and a search-configuration hash.
Each completed or interrupted question appends a durable prediction immediately;
trace events stream during decoding. Summaries distinguish attempted/completed
questions and report accuracy on completed questions only. Incomplete runs are
not final method comparisons. Plots are generated after completion.

Resume using the **same command arguments and `.env`**, adding:

```bash
python scripts/run_gsm8k_experiments.py --method both --num-samples 5 \
  --search-budget 256 --reward-mode adalook_adaptive --resume /path/to/run-directory
```

Completed condition/question pairs are skipped. Failed questions restart from
the question's beginning, with a fresh trace attempt; this is not mid-tree
resumption. A final incomplete JSONL append from a crash is truncated before
retry. Configuration, implementation and manifest changes reject resume.

Traces contain `adalook_gate`, `adalook_branch`, `adalook_round` and
`adalook_group` records. Predictions include original IG, finalized reward,
extra rollout NFEs, actual depth, variance, stop reasons and branch actions.
The offline tree audit separates IG from reward and rollout NFEs from tree
evaluations. Cross-question batching uses the same adaptive generator; group
metrics distinguish physical model invocations from evaluated sequences.

## Notebook

`notebooks/gsm8k/MEDAL_MCTS_Demo.ipynb` checks out `chong` and exposes
`REWARD_OPTIONS` in its settings cell. Its default selects `adalook_adaptive`
and `NOTEBOOK_MODE="comparison"`: run all cells to download/reuse the model on
Drive, run the same `both` comparison on five frozen test questions, and display
the saved report and plot. It passes one loaded model to the shared runner and
does not read `.env`. Set `STAGE`, `NUM_SAMPLES`, `CONDITION` and `RESUME_RUN` in
the settings cell; see the [Colab guide](medal_notebook.md) for output paths.
Set `reward_mode="ig"` for a separate LLaDA-versus-IG control or
`"adalook_one_step"` for the ablation. Restart the kernel and run from setup after
changing source code. `NOTEBOOK_MODE="demo"` enables the original single-question,
optional five-question and optional batching sections with the same variant.
The shared protocol hash identifies the frozen inputs; `variant_hash` identifies
the search configuration. AB-MEDAL's development workflow remains IG-only.

Tests use controlled CPU models for variance gates, branching, state isolation,
hard caps/checkpoints, batching parity, offline reconstruction, real notebook
cells, per-question saves, failure recovery and strict resume. They do not
establish LLaDA-8B accuracy or CUDA behavior; those require a cached GPU run.
