# LLaDA and iLLaDA decoding methods

Four decoder modules evaluate the pinned `GSAI-ML/LLaDA-8B-Instruct` and
`GSAI-ML/iLLaDA-8B-Instruct` checkpoints on GSM8K. They share fixed-length masked sampling, block scheduling,
chat formatting, and strict final-answer scoring. A fifth LLaDA method adds MCTS
initialization before the same token-probability confidence completion. These
methods perform inference without training.

## Algorithms

| Method / module | Position selection | Default model forwards per question |
| --- | --- | ---: |
| [Random MDM](random_mdm.py) | Sample proposed tokens, then randomly choose the positions to fill. | 256 |
| [Top-K Margin](top_k_margin.py) | Fill positions with the largest difference between their top two token probabilities. | 256 |
| [Confidence](confidence.py) | Fill positions whose sampled tokens have the largest model probabilities. | 256 |
| [Best-of-5](best_of_5.py) | Generate five confidence-decoded candidates and select the most frequent normalized final answer. | 1,280 |
| [MCTS + Confidence](mcts_confidence.py) | Select an initialization with entropy-based MCTS, then confidence-decode its remaining masks. | Search calls + 256 − selected depth |

All four default to **temperature 1.0**, **256 output positions**, **256 denoising
steps**, **128-token blocks**, and **seed 1**. Tokens are sampled using the
existing LLaDA Gumbel helper. Scores use the model probabilities before sampling
noise. Mask tokens are excluded from proposals and probability rankings.
Prompt tokens and completed output positions stay fixed, and only the current
output block can receive commitments during sampling/completion. MCTS can
initialize positions anywhere in the output before block-based completion.

Top-K Margin's K is the scheduled number of positions filled at each step,
derived from the generation length and step count. For example, 256 positions
and 128 steps fill two positions per step. K does not limit vocabulary sampling.
Equal margin or confidence scores prefer the earlier output position.

Confidence here means token probability. The older `confidence_guided.py` module
and the existing `LLaDA.predict(..., decoding={"method": "confidence"})` dispatcher
use an entropy/margin-adjusted score; those interfaces retain their semantics.
The separately named legacy `topk_margin.py`, `best_of_five.py`, and their
`decoder_evaluation.py` runner also remain available. Use the exact filenames
and commands in this guide for the four methods described above.

Best-of-5 uses the same temperature and generation settings for every candidate.
The strict GSM8K parser normalizes the final `#### <number>` line before voting,
so `1,200.00` and `1200` count as the same answer. Unparseable candidates receive
no vote. Ties choose the answer that first appeared, then the earliest candidate
with that answer. If all five candidates fail parsing, the first raw output is
retained and the question scores as incorrect. Voting uses no reference answer
and adds no model forwards. Temperature zero is supported but can make all five
confidence candidates identical despite their distinct seeds.

### MCTS initialization + confidence completion

`mcts_confidence.py` reuses the existing `MCTSSearch` one-step entropy evaluator.
Each child commits one output token and costs one forward to evaluate its
remaining-mask entropy. Search defaults are `init_tokens=20` (Lc),
`num_candidates=3` (C), `top_k_tokens=3` (K1), `top_k_actions=5` (K2),
`max_model_calls=512`, `margin_gamma=10`, `exploration_c=0.01`, `epsilon=1e-8`,
and `stop_mode=candidates`. These initialization parameters use the existing
search algorithm; full-completion rollouts are not performed. See the
[search walkthrough](../../../docs/mcts_walkthrough.md) for its reward and ranking.

The **512-forward cap is per question and search-only**: it includes root analysis
and every evaluated child across all branches. Search stops earlier when it
collects the requested candidates or exhausts the frontier. If the target depth
or candidate quota cannot be reached, the deepest selected state is completed,
with `selected_depth`, `fallback_used`, `fallback_reason`, and `stop_reason` saved.
The target of 20 initialized tokens is therefore not guaranteed by a 512-call cap.

Completion preserves the prompt and every selected initialization token. It
samples at temperature 1.0 and ranks positions by sampled-token probability,
matching `confidence.py`. The selected state's probabilities are recomputed;
the legacy search analysis is not reused as a token-probability cache. Transfer
schedules use each block's remaining masks, and zero-transfer steps are skipped.
The existing `mcts_entropy` dispatcher and its greedy entropy/margin-adjusted
completion retain their behavior.

## Python API

Each module exposes `decode(model, prompt_ids, mask_id, *, gen_length, steps,
block_length, temperature, seed)` with the defaults above. Input IDs have shape
`[1, prompt_length]`; the returned `DecodeResult.token_ids` contains the prompt
and completed output. `generation_nfes` counts actual model forwards, `details`
contains method metadata, and `trace` records position commitments.

```python
from dlm_project.models.llada import LLaDA
from dlm_project.decoding import confidence, best_of_5
from dlm_project.tasks.gsm8k.evaluation import extract_answer

wrapper = LLaDA()
ids = wrapper.encode_prompt("What is 3 + 2? End with #### <number>.")
result = confidence.decode(wrapper.model, ids, wrapper.mask_id)
text = wrapper.tokenizer.batch_decode(
    result.token_ids[:, ids.shape[1]:], skip_special_tokens=True
)[0]

voted = best_of_5.decode(
    wrapper.model, ids, wrapper.mask_id,
    tokenizer=wrapper.tokenizer, answer_extractor=extract_answer,
)
print(voted.details["selected_answer"], voted.details["votes"])
```

Best-of-5 additionally requires `tokenizer` and `answer_extractor(text)`, which
returns a normalized answer string or `None`. Its metadata includes all five
raw outputs, extracted answers, seeds, forward counts, votes, and the selected
candidate index. Candidate generation is sequential and reuses the loaded model.

Confidence also accepts `initial_state`, an int64 tensor of shape
`[1, prompt_length + gen_length]`. Its prompt must match `prompt_ids`; output
mask IDs mark positions to fill. The state is cloned and existing commitments
are preserved. A fully completed state needs no model calls.

MCTS + Confidence additionally accepts `mcts` (a settings mapping or
`MCTSConfig`) and an optional `tokenizer` for readable search traces. Unspecified
search settings use this method's defaults rather than changing legacy defaults:

```python
from dlm_project.decoding import mcts_confidence

initialized = mcts_confidence.decode(
    wrapper.model, ids, wrapper.mask_id, tokenizer=wrapper.tokenizer,
    mcts={"init_tokens": 20, "max_model_calls": 512},
)
print(initialized.details["selected_depth"])
print(initialized.details["search_model_calls"],
      initialized.details["completion_model_calls"])
```

Use `LLaDA(family="illada")` with the iLLaDA dependency profile to run the same
API on iLLaDA. The wrapper supplies its checkpoint chat template and mask token
**5**; the original LLaDA mask token is **126336**. These methods use fixed-length
generation for both families, without enabling iLLaDA's variable-length option.

## GSM8K commands

In an environment with suitable PyTorch installed:

```bash
python -m pip install -r requirements/llada-colab.txt -r requirements/dev.txt
python -m pip install --no-deps -e .

python scripts/eval_gsm8k_random_mdm.py
python scripts/eval_gsm8k_top_k_margin.py
python scripts/eval_gsm8k_confidence.py
python scripts/eval_gsm8k_best_of_5.py
python scripts/eval_gsm8k_mcts_confidence.py --profile smoke --num-samples 2
```

Defaults come from [the benchmark configuration](../../../configs/gsm8k/decoding.yaml).
MCTS + Confidence defaults to its own
[configuration](../../../configs/gsm8k/mcts_confidence.yaml), requires the pinned
LLaDA family, and shares the other methods' manifest, profiles, seeds, and scorer.
Edit `decoding.mcts` in YAML to change initialization depth, branching, candidate
target, or search budget. Its CLI defaults to 20 smoke questions; the appended
section in the [LLaDA notebook](../../../notebooks/gsm8k/LLaDA_GSM8K_Colab.ipynb)
inherits the current two-question override. It streams stdout/stderr, writes a
unique run receipt, and displays metrics and up to five random saved answers with
initialization depth, fallback details, and phase costs. Rerunning its preview
reads existing artifacts. The four earlier sections and saved outputs remain.

For iLLaDA, use its dedicated entrypoints and dependency profile in a separate
environment or rerun the dependency installation in Colab before switching models:

```bash
python -m pip install -r requirements/illada-colab.txt -r requirements/dev.txt
python -m pip install --no-deps -e .

python scripts/eval_illada_gsm8k_random_mdm.py
python scripts/eval_illada_gsm8k_top_k_margin.py
python scripts/eval_illada_gsm8k_confidence.py
python scripts/eval_illada_gsm8k_best_of_5.py
```

These scripts default to [the iLLaDA configuration](../../../configs/gsm8k/illada_decoding.yaml)
and require `model.family: illada`; the original entrypoints require `llada`.
The shared runner verifies the selected family's pinned checkpoint revision.
The [iLLaDA Colab notebook](../../../notebooks/gsm8k/iLLaDA_GSM8K_Colab.ipynb)
downloads from `adi`, caches the model, streams all four runs, and displays five
randomly selected saved answers after each method.

Every command supports the same profiles:

| Profile | Default questions | Purpose |
| --- | ---: | --- |
| `smoke` (default) | 20 | Check generation, parsing, metrics, and cost accounting. |
| `initial` | 100 | Compare decoding methods. |
| `full` | 1,319 | Report full GSM8K test accuracy and inference cost. |

```bash
python scripts/eval_gsm8k_confidence.py --profile initial
python scripts/eval_gsm8k_confidence.py --profile full
python scripts/eval_gsm8k_best_of_5.py --profile smoke --num-samples 10
python scripts/eval_gsm8k_random_mdm.py --model-path /path/to/cached/LLaDA-8B-Instruct --local-files-only
```

All evaluation entrypoints support `--config`, `--profile`, `--num-samples`, `--seed`,
`--temperature`, `--model-path`, `--local-files-only`, `--manifest`,
`--output-dir`, and `--run-manifest`. `--num-samples` overrides the profile count with 1–1,319 test
questions. Local model snapshots must retain their `.model_source.json` file so
the pinned model identity can be verified. Override configuration values for
GPU device, dtype, dataset cache, generation length, steps, and block size in YAML.
`--run-manifest /path/to/receipt.json` writes a receipt after successful evaluation,
containing `run_dir`, `status`, `model_family`, `method`, and `profile`. The iLLaDA
notebook uses a unique receipt for each invocation so previews select that exact run.
The LLaDA MCTS section also uses an exact run receipt.

The first invocation freezes the complete `openai/gsm8k`, `main`, `test` split
at an immutable dataset revision in `<experiment_root>/manifests/gsm8k_test.json`.
Later invocations reuse that manifest without accessing the dataset or Hub.
Publication uses an exclusive-create lock and a temporary-file rename, so no
hard-link support is needed on mounted Google Drive.
Smoke and initial profiles use the first 20 and 100 test rows; full uses all rows.
Use the same manifest across methods to keep questions, references, revision,
and ordering identical across both model families. Set `data.revision` to an immutable SHA when a revision
must be chosen explicitly; mismatches with an existing manifest are rejected.

For source row `i`, the question seed is `(base_seed + 5*i) % 2**63`.
Best-of-5 uses that seed plus candidate indices 0–4, modulo `2**63`.
Single-sample confidence therefore matches Best-of-5's first candidate.

## Accuracy, cost, and saved artifacts

The prompt contains only the question and instructions. Reference solutions enter
post-generation scoring. Accuracy is `correct / processed` questions using the
existing strict `#### <number>` parser and decimal normalization. Missing or
malformed final answers count as incorrect. Best-of-5 contributes one selected
prediction per question.

The default artifact layout is:

```text
results/LLaDA/GSM8K/<method>/<profile>/gen256-steps256-blk128-temp1-seed1/<timestamp>/
├── config.json
├── manifest.json
├── predictions.jsonl
├── metrics.json
└── traces/test_0000.jsonl, ...
```

For iLLaDA, the artifact root is `results/iLLaDA/GSM8K/` with the same
method/profile/settings structure and separate checkpoint provenance. Predictions
and metrics identify `model_family`; predictions also include `mask_token_id`.
MCTS runs add depth, candidate count, branching limits, forward cap, margin gamma,
exploration constant, epsilon, and stopping mode to the settings directory name.

Each invocation creates a collision-safe timestamp directory in the configured
timezone. `--output-dir` replaces the artifact parent; method, profile, settings,
and timestamp directories are still added. The shared manifest remains under
`experiment_root` unless `--manifest` is supplied.

`config.json` records resolved settings, model identity, dataset revision,
manifest hash, selected example IDs, Git commit, and run times. Predictions store
raw output, reference solution, normalized answers, correctness, prompt hash,
seed, method details, actual forwards, latency, and a trace path. Best-of-5 stores
all five candidates and voting details in each prediction record.
MCTS predictions include `selected_initialization` (IDs and committed positions),
all search statistics, separate `search_model_calls` / `completion_model_calls`,
and search/completion latencies. Trace events carry `phase: search` or
`phase: completion`; completion `generation_nfes` includes preceding search calls.

`metrics.json` updates after each completed question and includes:

- `processed`, `correct`, `accuracy`, `parse_failures`, and `parse_rate`.
- `total_model_calls` and `mean_model_calls`, including all five Best-of-5 candidates.
- Total, mean, and median generation latency, plus questions per second.
- Peak allocated CUDA memory in bytes, including resident weights; `null` on CPU.
- Model loading time and load count, model/dataset revisions, run status, and errors.

MCTS metrics additionally include total and mean search/completion forwards,
`search_latency_seconds`, `completion_latency_seconds`, `fallback_count`, and
`mean_selected_depth`. With the default 256 steps, completion uses
`256 - selected_depth` actual forwards. For example, 512 search forwards and a
20-token initialization cost **512 + 236 = 748 forwards** overall. All explored
branches count, even when their tokens are absent from the selected state.
Completion does not load another model. Phase latencies synchronize CUDA and
exclude model loading; overall generation latency also includes encoding,
response decoding, and metadata construction.

Latency includes prompt encoding, decoding, response decoding, and voting, with
CUDA synchronization. It excludes model loading, dataset preparation, scoring
against the reference, and artifact writing. Compare accuracy alongside forwards
and latency: Best-of-5 uses five times the default generation budget.

Completed prediction rows are flushed before metrics updates. Failed or
interrupted runs preserve completed predictions and report partial accuracy with
`status: failed` or `interrupted`; unprocessed questions do not enter the
partial denominator. Rerunning creates a fresh directory rather than resuming.

Run `python -m pytest -q` for controlled CPU tests and existing regressions.
Actual 8B smoke and benchmark runs require an appropriate CUDA environment;
controlled tests do not measure GSM8K model accuracy.
