# LLaDA and iLLaDA decoding methods

Four decoder modules evaluate the pinned `GSAI-ML/LLaDA-8B-Instruct` and
`GSAI-ML/iLLaDA-8B-Instruct` checkpoints on GSM8K. They share fixed-length masked sampling, block scheduling,
chat formatting, and strict final-answer scoring. A fifth LLaDA method adds MCTS
initialization with immediate-child information gain and cached candidate ranking,
then completes the selected partial state with standard LLaDA confidence decoding. These
methods perform inference without training.

## Algorithms

| Method / module | Position selection | Default model forwards per question |
| --- | --- | ---: |
| [Random MDM](random_mdm.py) | Sample proposed tokens, then randomly choose the positions to fill. | 256 |
| [Top-K Margin](top_k_margin.py) | Fill positions with the largest difference between their top two token probabilities. | 256 |
| [Confidence](confidence.py) | Fill positions whose sampled tokens have the largest model probabilities. | 256 |
| [Best-of-5](best_of_5.py) | Generate five confidence-decoded candidates and select the most frequent normalized final answer. | 1,280 |
| [MCTS + Confidence](mcts_confidence.py) | MEDAL search initializes a partial state; standard LLaDA confidence fills its remaining masks. | Search calls + 256 − selected depth |

All four default to **temperature 1.0**, **256 output positions**, **256 denoising
steps**, **128-token blocks**, and **seed 1**. Tokens are sampled using the
existing LLaDA Gumbel helper. Scores use the model probabilities before sampling
noise. Mask tokens are excluded from proposals and probability rankings.
Prompt tokens and completed output positions stay fixed, and only the current
output block receives commitments in the four standalone decoders. MEDAL uses
one global output block and can choose positions anywhere in the output.

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

### MEDAL paper reconstruction

All MCTS paths now use the [documented MEDAL reconstruction](../../../docs/mcts_walkthrough.md), with implementation version `medal_reconstruction_v2`. The historical `mcts_confidence` and `mcts_entropy` identifiers remain available. Author-code parity is unverified.

Search performs one forward per immediate child and backs up `(S_parent - S_child) / S_parent`, using cached parent entropy and newly computed child entropy. It collects three distinct depth-20 states and selects the highest cached mean tree reward. Each candidate is a leaf evaluated once, so its mean reward equals its last-edge information gain. Candidate ranking adds no forwards. Final completion passes the selected partial state to the standard LLaDA confidence decoder, which preserves its committed tokens, proposes tokens with the native Gumbel sampler, and fills positions in descending sampled-token probability.

MEDAL and Adaptive MEDAL exclude `<|endoftext|>` and `<|eot_id|>` from tree-search actions, so an initialization token cannot terminate the answer. The exclusion applies only to MCTS search; the final completion policy keeps its normal vocabulary behavior. Resolved token IDs are recorded in the trace and run statistics.

General search defaults are 256 output tokens/steps, temperature 1.0, seed 1, Lc=20, C=3, K1=3, K2=5, gamma=10, exploration constant 0.01, epsilon=1e-8, and no search cap. Search can initialize any output position. The GSM8K experiment completes with the same 128-token blocks as the standalone confidence baseline. Gamma/c are repository defaults rather than verified author parameters. Immediate-child reward and cached candidate ranking remain explicit reconstruction assumptions; standard confidence completion is an experimental replacement for the unstable adjusted-score completion.

The GSM8K MCTS experiment config uses a smaller search setting: Lc=10, C=1, K1=3, K2=2, stopping at the first candidate with a 256-search-forward hard ceiling. If no depth-10 candidate is available at that ceiling, it completes from the deepest evaluated fallback instead of aborting the batch.

Three `medal_reconstruction_v3_swd` ablations based on
[Stability-Weighted Decoding](https://arxiv.org/abs/2604.17068v1) add exact parent-to-child
full-vocabulary temporal KL without retaining vocabulary distributions in tree
nodes. Each expanded parent therefore needs one additional reference forward;
that cost is recorded as `stability_reference_model_calls` and counts against
the hard search cap. All three still hand the selected partial state to standard
LLaDA confidence completion:

- `mcts_swd_puct` multiplies the MEDAL action score by
  `exp(-stability_lambda * KL)` and uses the normalized score as a PUCT prior.
- `mcts_swd_reward` keeps UCB/action ordering unchanged and subtracts
  `stability_reward_beta * KL` from the edge information-gain reward.
- `mcts_swd_adaptive` keeps the original score/reward and maps the 75th
  percentile KL to dynamic branching, target depth, candidate quota, and a soft
  forward budget. Values below `adaptive_low_uncertainty` use the configured
  minima; values above `adaptive_high_uncertainty` use the maxima; intermediate
  values are linearly interpolated.

Task runners use inline three-subtask prompting with disjoint two-shot examples. GSM8K training rows 0/1 share the frozen test revision; evaluated references never enter prompts or rewards. Low-level decoder APIs accept already prepared prompt IDs. Setting `NUM_SAMPLES=None` restores profile counts; the LLaDA notebook retains its two-question override and all four standalone sections.

Strict candidate mode fails explicitly if the candidate quota cannot finish. Finite `stop_mode: budget` runs rank any collected target-depth candidates; otherwise they select the deepest evaluated state, or root. `stop_mode: candidates_or_budget` stops as soon as the candidate quota is reached, but accepts the same deepest-state fallback if a configured cap arrives first. Every root/child forward counts against an explicit cap; unfinished evaluations do not update visits. Final completion is additional. Historical runs remain readable; new directory labels include `medal-v2`.

## Accuracy, compute, and search metrics

The LLaDA GSM8K notebook's optional comparison section runs all four baselines
and MCTS simulation caps of 4, 8, 16, 32, and 64 in **one subprocess with one model
load**. Every condition receives the same MEDAL two-shot prompt, single global
output block, temperature, frozen questions and per-question seed. These are
separate comparison runs; standalone entrypoints retain their settings.

```bash
python scripts/run_gsm8k_metrics.py --config configs/gsm8k/decoding.yaml \
  --profile smoke --num-samples 2 --budgets 4 8 16 32 64

# Rebuild the comparison without model loading or dataset downloads.
python scripts/run_gsm8k_metrics.py --report-only /path/to/comparison
```

| Measurement | Definition |
|---|---|
| Accuracy and Wilson 95% interval | Strict GSM8K exact match; malformed/empty outputs are wrong. Execution failures have no fabricated answer. |
| Total NFEs / state evaluations | All forwards, including tree analysis, full simulations, terminal candidate scoring and completion. Batch size is recorded as 1, so these counts are equal. |
| Wall-clock time | CUDA-synchronized inference, including tokenization, output decoding, signal reductions and compact trace writing; model loading, warm-up and report generation are recorded separately. |
| MCTS simulations | Immediate-child evaluations and backups. Starts, interruptions, cycles and cached candidate rankings are distinct. Per-invocation counts are saved as a list, and totals are per example. |
| Decoding steps | Actual final-output commitment iterations. Best-of-5 sums all five trajectories and separately records the selected trajectory. MCTS reports final completion steps and selected tree depth separately; internal rollout steps are zero. |
| Average search depth | Mean depth of children whose simulations completed. Selected and maximum depths are also saved. |
| Expanded nodes | Number of distinct evaluated child states, excluding root; expanded-parent and total tree-state counts are separate. |
| Entropy and margin | Mean native full-vocabulary entropy in nats and top-two probability gap over all unresolved output positions, from existing forwards. Compact traces store scalar values, never vocabulary logits. |

`max_simulations` is an optional positive cap; `stop_mode: budget` accepts a finite
simulation cap, NFE cap, or both. The first cap reached stops search. Candidate
evaluation and final completion can add NFEs after a simulation cap. Candidate
evaluation still obeys a configured NFE cap. Default quota behavior is unchanged.
Small simulation budgets may never reach Lc=20; report selected depth and fallback
rate instead of assuming all runs reached the target depth.

The three PNG/PDF plots use **the intersection of examples completed under every
selected condition**, with that denominator displayed. They show accuracy versus
actual mean completed simulations, mean total NFEs, and mean wall-clock time.
Requested simulation budgets remain in condition labels. Historical counts may
be reconstructed from saved trace events; unavailable counts remain missing.
Reports warn when effective prompts or generation settings differ.

Each comparison saves `setup.json`, exact run paths in `runs.json`, and individual
incrementally saved predictions/traces. Its `report/` directory contains
`results.csv`, `summary.csv`, `summary.json`, `plot_data.json`, `report.md`, and plots.
Set `METRICS_NUM_SAMPLES` and `METRICS_SIMULATION_BUDGETS` in the notebook before
launching. Start with two examples and `[4]` for the smallest immediate-child search check.

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
`MCTSConfig`) and an optional `tokenizer` for readable search traces. All MCTS paths share the new defaults. An optional `on_event` streams partial traces immediately:

```python
from dlm_project.decoding import mcts_confidence

initialized = mcts_confidence.decode(
    wrapper.model, ids, wrapper.mask_id, tokenizer=wrapper.tokenizer,
    mcts={"init_tokens": 20, "max_model_calls": None},
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
python scripts/eval_gsm8k_mcts_swd_puct.py --profile smoke --num-samples 2
python scripts/eval_gsm8k_mcts_swd_reward.py --profile smoke --num-samples 2
python scripts/eval_gsm8k_mcts_swd_adaptive.py --profile smoke --num-samples 2
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

On CARC or RunPod, the shared shell dispatcher accepts the same three method
names: `mcts_swd_puct`, `mcts_swd_reward`, and `mcts_swd_adaptive`.

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

MEDAL counts one root-analysis forward, one forward per evaluated child, and fresh final completion. Candidate ranking costs zero forwards; completion costs 236 at depth 20. A direct-path example totals 335 forwards, and broader exploration can cost more. The walkthrough gives the formula and explicit assumptions. No CPU or smoke-run result is a reproduction of the paper's reported accuracy.

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
