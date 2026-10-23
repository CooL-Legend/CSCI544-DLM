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
