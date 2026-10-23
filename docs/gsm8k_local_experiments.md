# Pure LLaDA and fixed-budget MCTS

`scripts/run_gsm8k_experiments.py` reads the repository's `.env` and runs just the
requested methods. `both` finishes all selected questions with pure LLaDA, then
runs the same questions with MCTS. The model loads once per invocation. Both use
the E0 prompt, the same frozen question order, T=0, 256 output tokens and a
256-token global block. Answers are scored before the first EOS/EOT using the
existing strict `#### number` scorer.

## Paths and existing caches

Create `.env` from `.env.example` on RunPod and edit the paths:

```bash
cp .env.example .env
```

```dotenv
MODEL_CACHE_DIR=/workspace/model-cache/huggingface/hub
DATASET_DIR=/workspace/datasets
RESULTS_DIR=/workspace/CSCI544-DLM/results
DATASET_MANIFEST=/workspace/datasets/manifests/gsm8k_e0.json
DATASET_REVISION=740312add88f781978c0658806c59bc2815b9866
```

`.env` is ignored by Git; `.env.example` is tracked. No `source .env` is needed.
Quote paths containing spaces. Relative paths are resolved relative to the env
file, `~` is supported, and shell commands/variable interpolation are not used.
Values in the file set the Hugging Face cache locations even if your shell has
older cache variables.

`MODEL_CACHE_DIR` is the Hub directory containing
`models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07`.
The launcher also accepts a parent directory containing `huggingface/hub`.
It loads that pinned checkpoint locally via its repo ID; weights are not copied.

The launcher never downloads a model or dataset and never installs an environment.
Use your existing LLaDA environment. If `DATASET_MANIFEST` already exists, the
runner reads it directly. It must be an **E0 manifest** (`kind: gsm8k_e0`), not the
older `gsm8k_test.json`. Omit `DATASET_REVISION` to accept the revision already in
that manifest, or set it to the same revision explicitly.

If the manifest is absent, the runner creates it locally from the cached GSM8K
`main` **train and test** splits at `DATASET_REVISION`. That field pins the dataset
revision and avoids an online revision lookup. A test-only cache is insufficient
for E0's train/test partition. Missing cached data causes an offline error before
model loading; prepare the missing data separately or point `DATASET_MANIFEST`
at an existing E0 manifest. The prepared manifest is shared across separate runs.

## Commands

Run from the repository using the Python in your prepared LLaDA environment:

```bash
# Inspect effective paths and settings; no data or weights are loaded.
python scripts/run_gsm8k_experiments.py --dry-run

# Default: pure LLaDA first, then MCTS, on five held-out test questions.
python scripts/run_gsm8k_experiments.py --method both --num-samples 5

# Run either method independently on the same frozen questions.
python scripts/run_gsm8k_experiments.py --method llada --num-samples 20
python scripts/run_gsm8k_experiments.py --method mcts --num-samples 20 --search-budget 256

# Change the MCTS parameters.
python scripts/run_gsm8k_experiments.py --method both --num-samples 20 \
  --search-budget 256 --lc 5 --num-candidates 3 --top-k-tokens 3 --top-k-actions 5
```

Defaults are `Lc=5`, `C=3`, `K1=3`, `K2=5`, UCB constant `0.01`, margin gamma `10`,
seed `1` and search cap `256`. `--config` chooses an alternate E0 YAML; CLI search
parameters override that YAML. The launcher sets `completion_method=llada`,
`completion_temperature=0` and `stop_mode=budget` for this comparison.
The cap covers search forwards only, including root analysis. Search may end
earlier if the frontier is exhausted; completion costs `256 - selected_depth`
additional forwards. `C` is the shortlist/report target, not an early-stop quota
in budget mode.

`--stage test` is the default and uses E0's frozen shuffled test order.
`--stage dev` and `--stage evaluation` select the 5- and 20-question training
pools; set `--num-samples` within the chosen pool. Separate invocations with the
same manifest, stage, sample count, config and seed reuse the same inputs.

Results are written below
`RESULTS_DIR/LLaDA/GSM8K/llada_vs_mcts/<both|llada|mcts>/<stage>/budget-lc<Lc>/<timestamp>/`:

- `config.json`, `manifest.json`, `selection.json`: exact inputs/settings.
- `predictions.jsonl`: outputs, parsed answers, correctness and model calls.
- `traces/`: individual generation/search traces.
- `summary.json`: live progress and metrics.
- `report/report.md`, CSVs and plots: final accuracy/runtime/cost report.

For `both`, rows are labeled `confidence_t0` and `medal_budget_b256` (or your
selected budget). No entropy or separate no-search MEDAL run is included.
The old `scripts/experiment_0_baseline_test.py` still runs its full condition
set by default; its new `--methods confidence medal` option selects this subset.

## Reward B: IG minus pre-commit historical KL

This is an opt-in research extension, not a replacement for the original IG
baseline and not an accuracy guarantee. At current node `s`, let `g` be its
parent and `a=(i,v)` the action producing child `s_next`. The reward is:

```text
IG = (H(s) - H(s_next)) / H(s)       # Original epsilon guard is unchanged.
history_KL = KL(p_g[position i] || p_s[position i])
reward_B = IG - kl_weight * history_KL
```

Position `i` must still be masked in both `g` and `s`. This penalizes committing
an unstable position, NOT the changes the action causes elsewhere in `s_next`.
Each node uses its own parent history, never the last visited sibling's history.
Root actions have zero penalty. Different token choices at the same position
share the historical KL penalty; their IG can still differ.

Only the search reward changes: candidate action filtering, UCB exploration
constant, backups, final candidate-ranking rule, completion, prompt, and stopping
are unchanged. The new reward feeds the existing backups and ranking.
`--kl-weight 0` follows the original IG path without allocating a history cache.
The initial weight `0.05` is a starting value to tune on development data, not a
validated optimum. The default reward remains `ig` unless explicitly selected.

```bash
# Original IG control.
python scripts/run_gsm8k_experiments.py --method mcts --num-samples 5 \
  --search-budget 256 --reward-mode ig

# B: start with the same budget to isolate reward quality.
python scripts/run_gsm8k_experiments.py --method mcts --num-samples 5 \
  --search-budget 256 --reward-mode ig_kl --kl-weight 0.05 --kl-top-k 32

# Run pure LLaDA, then B on the same questions in one process.
python scripts/run_gsm8k_experiments.py --method both --num-samples 5 \
  --search-budget 256 --reward-mode ig_kl --kl-weight 0.05

# Test whether B works at a lower search budget (not adaptive early stopping).
python scripts/run_gsm8k_experiments.py --method mcts --num-samples 5 \
  --search-budget 128 --reward-mode ig_kl --kl-weight 0.05
```

`kl_top_k=32` caches the **previous** distribution's top-32 token IDs/probabilities
and a single residual-mass bucket. The new distribution is gathered at those
same IDs; its remaining mass goes in the matching bucket. This is a
**coarse-grained KL approximation**, not KL between two independently ranked
top-k vectors, and it cannot observe rearrangements within the tail.
Both distributions are floored at `epsilon` and renormalized for numerical
safety. No special tokens are removed from these probability distributions;
the existing EOS/EOT exclusion still applies only to search actions.

Use `--kl-top-k 0` for full-vocabulary KL with the same numerical smoothing.
This can require **several GB of CPU RAM** for the frontier; it is not the
memory-efficient default. A k at least as large as the vocabulary also uses
full mode. Histories live on CPU, are released after expansion or search exit,
and are never persisted as full matrices in result files. Leaf candidates do
not retain histories. Both modes reuse existing forwards, but KL calculations
and CPU/GPU transfers add runtime; fewer NFEs do not automatically mean faster.

B condition IDs include the settings, e.g.
`medal_budget_b256_ig_kl_l0.05_k32`. Trace `evaluation` events separately record
`immediate_gain` (the original IG), `action_history_kl`, `kl_penalty`, `reward`,
the history node IDs and output position. Predictions/reports include
`selected_ig_reward`, `selected_action_history_kl`, `selected_kl_penalty` and
`peak_history_cache_bytes` (retained history tensors only, not total RAM/VRAM).
`kl_estimator` records the actual mode (`full`, `top_k_tail`, or `disabled`),
and `kl_extra_model_calls` is zero. Reward/version/parameters are saved for replay.

Fixed-budget search still normally consumes its cap. Compare IG and B at both
128 and 256 on identical frozen questions, and measure accuracy **and total
NFEs/runtime**. Choose lambda using development/training data before reporting
held-out test results. This variant neither introduces automatic early stopping
nor changes final candidate scoring into a cumulative path reward.
