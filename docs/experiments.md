# GSM8K experiments: shared protocol and E0

The shared protocol is `gsm8k_shared_v2`. The implemented E0 runner uses
`e0_fixed_budget_v6`. E1 and E2 must consume the same sample, prompt, generation,
and candidate-selection interfaces when their experiment runners are added.
Their adaptive search and trajectory reward are outside this implementation.

## 1. Common experimental setup

| Parameter | Fixed configuration |
|---|---|
| Model | `GSAI-ML/LLaDA-8B-Instruct` |
| Model and tokenizer revision | `08b83a6feb34df1a6011b80c3c00c7563e963b07` |
| Weight dtype | BF16 (`bfloat16`); batch size 1 |
| Dataset | `openai/gsm8k`, subset `main` |
| Dataset revision | `740312add88f781978c0658806c59bc2815b9866` |
| Development | 5 fixed training-split questions |
| Evaluation | 20 different fixed training-split questions |
| Final test | 200 fixed test-split questions |
| Sample seed / decoding seed | 1 / 1 |
| Prompt | The deterministic two-shot template in Section 2 |
| Candidate filtering | K1 = 3 eligible tokens per masked position; K2 = 5 retained actions per expansion |
| Maximum depth | Fixed at Lc = 10 across all experiments; only budget varies |
| Search budgets | 256, 512, and 1024 actual model forwards |
| Final shortlist | C = 3, or fewer when fewer distinct states exist at chosen depth |
| Completion decoder | Standard LLaDA Confidence at T = 0 |
| Output length / steps / block length | 256 / 256 / 256 |
| Search simulation | Evaluate the immediate child state; no full rollouts |

The [shared YAML](../configs/gsm8k/experiments_common.yaml) is the single source
of these inputs. The [E0 YAML](../configs/gsm8k/experiment_0.yaml) references it
and adds E0's UCB settings and operational defaults. The generic config loader
resolves this reference; direct `yaml.safe_load` of the E0 YAML does not expand
it. The runtime validator rejects per-experiment changes to shared settings.
The model wrapper pins the tokenizer and chat template to the same checkpoint.
Record hardware and package versions with runs when comparing wall-clock time;
T=0 does not promise identical floating-point behavior across hardware stacks.

### Frozen sample selection

[The checked-in manifest](../configs/gsm8k/experiment_samples.json) contains
exactly 225 ordered examples with source split, row ID, question, reference
solution, and content hash. Its canonical JSON SHA256 is:

```text
75596121693b37068cc6331883089bb20be754fd5c7adc1a50cf53f16da26988
```

This hash uses the repository's `digest` function on the parsed JSON, not a
checksum of indentation or file bytes. The expected hash and immutable dataset
revision are recorded in the shared YAML. Source data comes from the
[pinned GSM8K dataset snapshot](https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866/main).

Preparation preserves the historical E0 selection algorithm: initialize
`random.Random(1)`, shuffle train indices and then test indices, exclude train
questions found anywhere in test, and take 25 distinct train questions. The
first 5 become dev, the next 20 evaluation. Take the first 200 distinct
questions in the shuffled test ordering. Normalize whitespace and case when
checking duplicate questions. All three pools are disjoint.

Every experiment loads this artifact in its saved order. Runners never query
`main`/latest, download examples, sample again, or silently regenerate a missing
manifest. They reject content/order/hash/revision mismatches and incorrect
stage sizes before loading weights. A copied manifest must match exactly.

Verify locally without network access or weights:

```bash
python scripts/prepare_gsm8k_experiments.py
```

Reconstruct from the pinned source if necessary:

```bash
python scripts/prepare_gsm8k_experiments.py --rebuild
python scripts/prepare_gsm8k_experiments.py --rebuild --output /path/to/shared-samples.json
```

Reconstruction downloads only the two `main` Parquet files at the saved
revision and checks the expected final hash. It refuses to overwrite a changed
existing artifact. `--cache-dir` controls the download cache. No command offers
a new sampling seed or a different revision under this protocol version.

Public shared interfaces in `tasks.gsm8k.experiment_protocol` are
`load_shared_protocol()`, `load_samples(stage, manifest_path=...)`,
`build_prompt(question)`, and `example_seed(row_id)`. The per-question seed is
`(1 + 5 * row_id) % 2**63`; it does not depend on the experiment or condition.
A fingerprint of the entire common protocol is saved with all results. Future
cross-experiment comparisons must match this fingerprint and the sample IDs,
prompt/token hashes, model revision, and dataset revision.

## 2. Exact prompt

The canonical template lives once in the shared YAML. Only the final
`{GSM8K_question}` placeholder is replaced, literally; braces within the
question are not interpreted. Automatic MEDAL demonstrations are disabled.

```text
Decompose the question into three steps and solve the math problem following the examples.
State the goal of each step clearly.
Show your final numerical answer after ####.

Example 1:
Problem: A shop has 12 pens. It buys 4 packs containing 6 pens each and gives away 9 pens. How many pens remain?

1. Identify how many pens the shop initially has and how many additional pens it buys.
2. Calculate the total number of pens after the purchase.
3. Subtract the pens given away to find the remaining quantity.
#### 27

Example 2:
Problem: A bakery makes 48 muffins. It sells half of them in the morning and another 8 in the afternoon. How many muffins remain?

1. Identify the total number of muffins and the portion sold in the morning.
2. Calculate how many muffins remain after the morning sales.
3. Subtract the afternoon sales to determine the final number of muffins.
#### 16

Now solve the following question using the same decomposition format.

Problem: {GSM8K_question}
```

The prompt is one user message. The existing pinned tokenizer applies its chat
template with `add_generation_prompt=True`; the rendered chat text is tokenized
with `add_special_tokens=False`. One cached encoding per question is reused
across conditions (plus the separate warm-up encoding). Save the template
hash, rendered user prompt, prompt hash, and token-ID hash. The reference
solution is never passed to the prompt builder, search, or decoder.

## 3. Common candidate selection

Search with reward R, select the depth having at least C distinct evaluated states,
rank by expected reward Q_R, retain up to three, and complete only the best one.

1. Search until budget B is exhausted or the frontier is exhausted. Never stop
   simply because three candidates exist.
2. Among successfully evaluated states, find the candidate depth d*: the deepest
   depth having at least C distinct evaluated states. If no depth has at least C
   distinct states, choose the depth with the most distinct states (breaking
   ties by preferring the deeper depth).
3. Collect every state at d*. Deduplicate exact token-ID arrays, including mask
   positions; decoded text is not an identity key.
4. Keep the first successfully evaluated node for each repeated state. Its
   statistics define the candidate value; do not pool evidence from duplicate
   tree nodes or alter UCB's tree structure.
5. Rank by Q_R(s) = reward_sum(s) / visits(s), descending. Break ties by ascending
   node ID. Do not rank by the UCB bonus or run extra model evaluations.
6. Retain min(3, number of distinct states at d*) and complete rank 1 exactly once.

Each immediate-child reward is backed up once into that child and every
ancestor. Values are the final accumulated statistics at selection time.
For the current E0 tree, deepest evaluated nodes are leaves with one backup,
so Q_IG equals their immediate IG numerically. The selector nevertheless reads
`reward_sum / visits`; later implementations can provide richer estimates.
The common helper is `decoding.mcts_decoding.select_deepest_candidates`.

### Fallback and boundaries

If Lc is not reached, apply the identical collection, deduplication, ranking,
and shortlist procedure at d*. Fallback is precisely `selected_depth < Lc`.
One or two candidates at Lc mean an underfilled shortlist, not fallback.

If only the root was analyzed, d* = 0, candidate count = 1, and shortlist size = 1.
Its zero-visit value is conventionally 0; this is not an estimated reward.
Completion starts from all 256 masks. Non-finite rewards, failed model calls,
and other implementation errors fail the run explicitly; they are not fallback.

| Experiment | Search reward | Final ranking |
|---|---|---|
| E0: MEDAL | Immediate IG | Mean backed-up IG reward |
| E1: AB-MEDAL | Immediate IG | Mean backed-up IG reward |
| E2: AB-MEDAL + Trajectory | Trajectory-aware reward | Mean backed-up trajectory reward |

These are requirements for the future E1/E2 experiment runners; the existing
adaptive demo is not yet an implementation of this fixed-budget protocol.
In E2, changing R changes both search guidance and final candidate selection.
Accuracy differences therefore measure their combined effect. A separate
common-ranking ablation may later distinguish the two effects.

## 4. E0: search-free baseline evaluation

### Objective and conditions

Establish pure search-free LLaDA baseline performance on the frozen GSM8K questions.
All MCTS search conditions (MEDAL vs. AB-MEDAL across budgets 256, 512, 1024) are unified
and evaluated in Experiment 1 (`scripts/run_e1_e3.py`).

| Condition | ID | Search B | Completion NFEs | Total NFEs | Default Status |
|---|---|---:|---:|---:|---|
| LLaDA Confidence | `confidence_t0` | 0 | 256 | 256 | **Enabled** (Primary baseline) |
| LLaDA Entropy | `entropy_t0` | 0 | 256 | 256 | Optional (`--include-entropy`) |

Both conditions perform standard search-free LLaDA decoding across 256 steps at $T=0$ using
the shared two-shot decomposition prompt.
By default, the runner runs `confidence_t0`. When `--include-entropy` (or `--methods confidence entropy`)
is passed, both baselines run under one loaded model.


### Retained MEDAL behavior

UCB uses the mean backed-up reward plus the existing exploration bonus, with
`exploration_c=0.01`. Retain `margin_gamma=10` and `epsilon=1e-8`. These are
repository defaults; this implementation is the repository's MEDAL
reconstruction, not a claim of parity with author code.

For parent p and immediate child s:

```text
R_IG(p,s) = (H(p) - H(s)) / H(p), when H(p) > epsilon
         = 0, otherwise
```

H is the cached total entropy over uncommitted output positions. The root
analysis and every immediate child evaluation each consume an actual forward.
No full completion occurs within search and final candidate ranking consumes
zero forwards. Budget checks prevent overshooting B, including midway through
an expansion. Stop reasons are `model_call_budget` and `frontier_exhausted`.
When a budget and frontier boundary coincide, the existing budget check takes
precedence. No simulation-count cap applies.

### Completion and scoring

All completions use standard LLaDA Confidence at T=0: propose the argmax token
at each remaining masked position and commit the position with highest native
probability for its proposal. Preserve the prompt and all search commitments.
One fresh forward fills each remaining mask; search logits do not provide a
free completion step.

Search actions exclude mask and resolved EOS/EOT IDs before top-K filtering.
Completion permits EOS/EOT and fills every output position, even when a
terminator is committed out of order. Score only the output prefix before the
first termination token. Preserve full output IDs and text for inspection.
The strict GSM8K parser uses the last valid `####` answer line and normalizes
its number with `Decimal`. Unparsed predictions are incorrect. Fallback outputs
use the same parsing and accuracy rules.

### Metrics and exact definitions

| Category | Per-question fields and interpretation |
|---|---|
| Performance | `exact_match`, `parse_success`; aggregate accuracy and parse rate |
| Compute | `search_nfes`, `completion_nfes`, `total_nfes`: actual invocations |
| Runtime | `search_seconds`, `completion_seconds`, `decoding_seconds` |
| Depth | `max_depth`: maximum successfully evaluated depth; `selected_depth`; `average_search_depth`: mean evaluated-child depth, excluding root |
| Branching | `expanded_parents`: parents with at least one evaluated child; `evaluated_children`; `realized_branching_factor = evaluated_children / expanded_parents` |
| Candidates | `candidate_count`: distinct states at d*; `target_depth_candidate_count`: distinct states at requested Lc; `shortlist_size` ≤ 3 |
| Fallback | `fallback_decode_used`: selected depth below Lc; aggregate mean is fallback rate |
| Diagnostics | `stop_reason`, `budget_utilization = search_nfes / B`, per-question correctness |

Empty evaluated-child/parent averages are 0. A partially expanded parent
counts if at least one child was evaluated. Duplicate tree nodes count as
individual evaluations for branching/depth metrics; final candidate counts
are deduplicated. The confidence baseline has zero search counts/depths,
zero fallback, no shortlist underfill, and unavailable budget utilization.
The historical low-level `fallback_used` field still means quota shortfall;
E0 uses `fallback_decode_used` for its depth-based fallback metric.

Required accounting checks:

```text
total_nfes = search_nfes + completion_nfes
search_nfes <= B
completion_nfes = 256 - selected_depth
```

CUDA synchronization bounds phase timers. `decoding_seconds` is the outer
search-plus-completion decoder interval, including decoder bookkeeping. It
excludes model load, warm-up, tokenization, scoring, trace disk writes, and report
generation; setup timings record those separately. Internal phase diagnostics
are nested and must not all be added together. A single warm-up forward is
outside condition NFE totals.

Report condition means, total compute, latency median/p95, stop-reason counts,
and Wilson accuracy intervals. The main comparison uses only questions with
terminal records for all four conditions. Observed summaries retain completed
records from partial stages. Interrupted/unattempted outcomes are not invented
as incorrect predictions. A complete test run contains 800 terminal records
and 200 paired questions.

## 5. Execution, artifacts, and validation

### Run in order

Use the repository's LLaDA dependency profile and a suitable CUDA GPU:

```bash
python -m pip install -r requirements/llada-colab.txt
python -m pip install --no-deps -e .
python scripts/prepare_gsm8k_experiments.py
python scripts/experiment_0_baseline_test.py --stage dev
python scripts/experiment_0_baseline_test.py --stage evaluation
python scripts/experiment_0_baseline_test.py --stage test
```

Use dev for implementation checks and evaluation for choosing reward settings
for later methods. Freeze the compared settings before reading final test
outcomes. Do not select rewards or retune the prompt using the 200 test answers.
An E0 source change can be checked with controlled models locally; it does not
constitute a measured LLaDA benchmark.

Optional operational arguments:

```bash
python scripts/experiment_0_baseline_test.py --stage test \
  --model-path /path/to/pinned/LLaDA-8B-Instruct --local-files-only \
  --manifest /path/to/shared-samples.json \
  --output-dir /path/to/e0-runs \
  --run-manifest /path/to/receipts/e0-test.json
```

`--config`, `--run-order`, stage, cache/artifact locations and a validated local
model snapshot remain configurable. `--num-samples` accepts a prefix slice (up to the stage's
frozen count: 5 for dev, 20 for evaluation, 200 for test) for rapid debugging and validation.
`--include-entropy` (or `--methods confidence entropy`) enables the optional search-free
entropy baseline. `--dataset-revision` must match the frozen revision. Old `--init-tokens`,
`--search-budgets`, and candidate-quota settings are rejected with migration guidance.

### Saved artifacts

Runs use `<experiment_root>/LLaDA/GSM8K/experiment_0/<stage>/fixed-matrix-v6/<timestamp>/`.
An explicit output directory replaces the root before stage/matrix/timestamp.

| Artifact | Contents |
|---|---|
| `config.json` | Resolved conditions, complete common protocol, hashes, model/code provenance, termination IDs |
| `manifest.json`, `selection.json` | Full shared manifest and ordered examples for this stage |
| `setup.json` | Status, warm-up/setup costs, active partial-attempt counters |
| `predictions.jsonl` | Flushed per-question records, phase metrics, outputs, scores, selection evidence |
| `traces/*.jsonl` | Compact selection/expansion/evaluation/backup/completion events |
| `summary.json` | Incremental paired and observed summaries |
| `report/` | Markdown, JSON, paired/observed/results CSVs, gain tables and accuracy-vs-NFEs/runtime PNG/PDF plots |

Search details include every deepest candidate's node ID, state hash, exact
output IDs, depth, visits, reward sum, immediate reward, and `q_value`. The
ranked shortlist retains its statistical evidence; `selected_initialization`
stores the exact chosen partial state, commitments, and mask count. The state
hash uses the full token-state tuple, including the fixed prompt. The
`common_protocol_hash`, `prompt_template_hash`, per-question prompt/token hashes,
requested depth and budget accompany each record.

`--run-manifest` writes an exact-run receipt with directory, protocol identity,
conditions and status. Use it instead of selecting the latest directory.
Ctrl-C/SIGTERM flushes partial counters and completed records. There is no
automatic resume; a new invocation creates a new run. A hard kill can lose the
active condition. Offline rebuilding ignores an incomplete trailing JSONL line
with a warning and preserves its source file.

```bash
python scripts/experiment_0_baseline_test.py --report-only /path/to/exact/run
```

Offline rebuilding needs neither weights nor network access. It checks saved
identity, hashes, accounting, termination prefixes, scores, and v5 candidate
ranking evidence. Historical E0 reports use their saved configuration and
prompt rather than being reinterpreted as v5. It does not re-tokenize saved
text, so preserve original artifacts for audit.

### Tests and acceptance

```bash
python -m pip install -r requirements/dev.txt
python -m pytest -q tests/tasks/test_experiment_protocol.py \
  tests/decoding/test_deepest_candidates.py tests/decoding/test_experiment_0.py \
  tests/decoding/test_mcts_budget.py tests/workflows/test_experiment_0.py
```

Tests cover frozen ordering and counts, overlap/tampering rejection, literal
prompt insertion, first-occurrence deduplication, backed-up means differing
from immediate reward, deterministic ties, negative rewards, root/below-target
fallback, partial expansion caps, more than three candidates, no ranking
forwards, and completion accounting. Workflow tests use stub decoders for the
full 200 × 5 schedule and actual invocation counters, plus interruption and
historical-report fixtures. Controlled decoder tests check search mathematics
and preserved commitments. None of these are LLaDA accuracy measurements.

Before the final test, inspect a GPU dev run: 25 terminal records, identical
prompt IDs per question, the exact condition matrix, consistent NFE/depth
accounting, sensible parsing, and an offline report matching the original.
Then complete evaluation and freeze the compared settings.

### Legacy notebooks

The E0 experiment notebook files remain unchanged. The three E0 notebooks target v4,
including old condition receipts and single-depth controls; use the CLI above
for v5. The committed E0 development notebook still contains unresolved merge markers
and needs separate repair before use. The [MEDAL demo](medal_notebook.md) now
mirrors the AB notebook with UCB initialization and the current shared settings.
The [historical E0 guide](experiment_0/README.md) describes previous protocols
and must not be used as the v5 execution specification.

The [AB-MEDAL implementation guide](ab_medal_walkthrough.md) describes the aligned serial development workflow and optional batching. Both MEDAL and AB now export `search_trace_v1` node identities and evaluation order for offline depth/subtree allocation analysis; this additive trace schema does not change the shared protocol fingerprint.
