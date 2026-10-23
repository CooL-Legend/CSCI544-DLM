> **Historical E0 v1–v4 reference.** The current script implements v5. Use the
> [shared protocol and E0 v5 guide](../experiments.md) for current settings and
> commands. Existing notebooks remain legacy and were not updated.

# Experiment 0: Baseline Test — implementation README

E0 measures how effectively MEDAL uses a fixed search budget on GSM8K. It
compares two greedy decoders and the current MEDAL completion decoder without
search against three independently initialized MEDAL searches. The matched
no-search MEDAL condition is the primary control for the effect of search
initialization.

Protocol version: **`e0_fixed_budget_v4`**. Shared MEDAL implementation:
**`medal_reconstruction_v2`**. This experiment preserves the repository's
immediate-information-gain reconstruction; it does not establish parity with
the authors' implementation.

## 1. Conditions and defaults

Evaluation and testing notebooks run the following six conditions, completing
all selected questions for one method before starting the next. Their search
caps are **256, 512, 1024**; 2048 is excluded for now. The development notebook
and shared YAML retain their original 512/1024/2048 caps and per-example order.

| Saved condition ID | Position/token choice | Temperature | Search cap |
|---|---|---:|---:|
| `confidence_t0` | Highest-confidence position, argmax token | 0 | None |
| `entropy_t0` | Lowest-entropy position, argmax token | 0 | None |
| `medal_no_search_t0` | Standard LLaDA argmax proposals and confidence-based position selection | 0 | None |
| `medal_budget_b256` | MEDAL initialization, then the same completion decoder | 0 | 256 |
| `medal_budget_b512` | MEDAL initialization, then the same completion decoder | 0 | 512 |
| `medal_budget_b1024` | MEDAL initialization, then the same completion decoder | 0 | 1024 |

| Setting | Default and interpretation |
|---|---|
| Model | `GSAI-ML/LLaDA-8B-Instruct` |
| Model revision | `08b83a6feb34df1a6011b80c3c00c7563e963b07`, from the repository model profile |
| Model execution | CUDA, BF16, one example/state per forward |
| Dataset | `openai/gsm8k`, `main`; immutable revision frozen when the manifest is first created |
| Generation length / steps / block length | `256 / 256 / 256`: one global output block, one committed token per forward |
| Target depth, `Lc` | 5; use 7 only for an explicitly configured extended run |
| Shortlist size, `C` | Up to 3 distinct, fully evaluated target-depth states |
| `K1`, `K2` | 3 eligible tokens per masked position, then 5 ranked actions per state |
| Gamma / exploration constant / epsilon | `10 / 0.01 / 1e-8` |
| Search stopping | Actual forward cap or exhausted frontier; no simulation-count cap |
| Search evaluation | Immediate parent-to-child information gain; no completion rollouts during search |
| Completion method | `decoding.completion_method: llada`; `medal` opts into the previous adjusted-score sampler |
| Completion temperature | `decoding.completion_temperature: 0.0`; all E0 conditions use T=0 by default |
| Base seed / dataset sampling seed | `1 / 1` |
| Prompt decomposition / demonstrations | Shared three-step template with the two fixed pens/muffins examples |

The shared defaults are in
[`configs/gsm8k/experiment_0.yaml`](../../configs/gsm8k/experiment_0.yaml).
Notebook controls explicitly override budgets and execution order as described above.
Greedy temperatures are fixed at zero; `decoding.completion_temperature`
sets both MEDAL completion paths, defaulting to zero. A shared
`generation.temperature` is rejected. The implementation supports smaller equal length/steps/block values
for controlled tests, but the supplied experiment and notebooks use 256.
The shared `MCTSConfig()` defaults remain `Lc=20`, candidate-quota stopping,
and no forward/simulation cap for existing callers.

## 2. Implementation map and interfaces

Paths below are relative to the repository root.

| File | Responsibility |
|---|---|
| `scripts/experiment_0_baseline_test.py` | Thin CLI entry point |
| `configs/gsm8k/experiment_0.yaml` | E0 defaults, shared prompt, stage and budgets |
| `src/dlm_project/tasks/gsm8k/experiment_0.py` | Configuration resolution, one-load comparison, counters, timings, incremental artifacts and receipts |
| `src/dlm_project/tasks/gsm8k/e0_dataset.py` | Frozen disjoint selections, source identity and content validation |
| `src/dlm_project/tasks/gsm8k/e0_prompt.py` | Exact shared prompt rendering and legacy report compatibility |
| `src/dlm_project/tasks/gsm8k/e0_tokens.py` | EOS/EOT resolution and post-completion prefix scoring |
| `src/dlm_project/tasks/gsm8k/e0_report.py` | Offline validation, paired summaries, CSVs and plots |
| `src/dlm_project/decoding/entropy.py` | Entropy decoder using shared `decode_masked` machinery |
| `src/dlm_project/decoding/medal_no_search.py` | No-search control using the same configured completion method as searched MEDAL |
| `src/dlm_project/decoding/mcts_confidence.py` | Existing MEDAL adapter; optional search-only token exclusions |
| `src/dlm_project/decoding/mcts_decoding.py` | Existing search/completion engine; exclusions and additive shortlist metadata |
| `tests/decoding/test_experiment_0.py` | Decoder, budget, selection and EOS tests |
| `tests/tasks/test_e0_dataset.py` | Dataset and scored-prefix tests |
| `tests/workflows/test_experiment_0.py` | Runner, reports, partial runs, receipts and notebook tests |

Principal Python interfaces:

```python
resolve_settings(config, **overrides) -> dict
run_experiment(
    config, *, manifest_path=None, output_dir=None,
    run_manifest=None, model_factory=None
) -> (run_directory, summary)
generate_report(run_directory) -> summary
```

`model_factory` permits controlled-model integration tests. The entropy and
no-search adapters return the existing `DecodeResult` contract: `token_ids`,
`total_nfes`, `details`, and `trace`. The MEDAL adapter and `MCTSSearch` accept
`search_excluded_token_ids=()`; the empty default preserves existing callers.
The E0 runner supplies validated termination IDs only to the search path.
Both MEDAL adapters accept `completion_method="llada"` or `"medal"`; their
low-level default remains `"medal"` for existing callers. E0 explicitly selects
`"llada"` for both searched and no-search conditions.

A run resolves configuration and its frozen examples, loads the pinned model
once, validates termination tokens, and performs one separately counted warm-up
forward. It constructs and tokenizes each question's prompt once, caching those
inputs for reuse across conditions. Each prediction uses a fresh counter/tree,
completes the selected state, trims the scoring prefix, scores the answer, and
flushes a record. `evaluation.run_order: condition` finishes the entire selected
question list for confidence, then entropy, then no-search MEDAL, then each
search cap in the supplied order. The default `example` order runs all methods
per question. Neither order reuses search work. A streamed summary reports each
finished method's accuracy, parse rate, mean total NFEs, and mean runtime; these
metrics are also saved in the incremental summary's `observed_conditions`.
The offline report builder reads the saved artifacts.

## 3. Prompts, data and reproducibility

All conditions in development, evaluation, testing and optional quota runs
receive the same prompt text and chat-template token IDs. `build_e0_prompt`
replaces `{GSM8K_question}` literally with the selected question in this exact
`prompt.template`, ending with the question:

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

The model wrapper sends this as one user message through the pinned tokenizer's
chat template with `add_generation_prompt=True`; there is no separate system
message. No extra `Question:` block or instructions are appended. The two fixed
examples are part of every condition's prompt, including both greedy methods
and the no-search MEDAL control. The selected question's reference solution
enters scoring only.

`medal_decomposition: false` disables the repository's automatic demonstration
injection; the explicit decomposition template above is still active. Custom
questions and the legacy `prompt.instructions` configuration are rejected for
new E0 runs. Prompt and token-ID hashes verify identical inputs across conditions.

Protocol v4 sets all condition temperatures to zero by default and saves the
no-search control as `medal_no_search_t0`. Protocol v3 changed MEDAL completion
to standard LLaDA decoding at T=1. Protocol
v2 introduced this demonstration prompt in place of v1's plain instructions. Existing
v1 artifacts can still be rebuilt offline using their saved original prompt
configuration. Keep results from different prompt/completion protocols separate when
interpreting experiment comparisons. The output length and search settings are unchanged.

| Stage | Source | Frozen pool / default run | Intended use |
|---|---|---|---|
| `dev` | Training split | 5 / 5 | Inspect decoding, EOS boundaries, fallback and accounting |
| `evaluation` | Training split | 20 / 20 | Compare settings; optional explicitly labeled `Lc=7` run |
| `test` | Test split | Full unique shuffled pool / 25 | Assess frozen settings; extend the same prefix to 100 if affordable |

The first preparation resolves a 40-character dataset commit SHA, loads both
splits at that revision, shuffles once with seed 1, and saves a manifest.
Development and evaluation use disjoint training pools. Duplicate questions
are detected after whitespace normalization and case folding; training
questions also present in the test split are excluded. This is exact
normalized-text deduplication, not semantic deduplication.

The manifest stores ordered example IDs (`train:<row_id>` or `test:<row_id>`),
source split/row, question/reference content hashes, source counts, sampling
seed, revision and dataset fingerprints when available. Later runs validate
and reuse that file without loading datasets. `--num-samples` takes a prefix
of the relevant pool: development/evaluation cannot exceed 5/20 respectively;
test 100 contains the original test 25 in the same order. Every run saves a
manifest copy and its exact selection.

Keep the same manifest for all stages. Its default is
`<experiment_root>/manifests/gsm8k_e0.json`; the notebooks share the Drive copy.
Do not delete/rebuild it between stages. A conflicting revision or sampling
seed is an error. A custom immutable revision can be set before the first run.

Per-example generation seed:

```text
seed = (base_seed + 5 * source_row_id) % 2**63
completion_seed = derive_seed(seed, "final_completion")
```

All searched MEDAL conditions and the no-search MEDAL control use that same
completion-seed derivation. Each search starts a fresh tree; no search work is
reused between budgets. A root fallback produces the same completion as the
no-search control under identical model behavior and seeds. Reproducibility
metadata includes model/dataset revisions, Git commit and dirty status, and a
hash of Python source files. Uploaded source bundles may have no Git commit;
their source hash remains available. Seeds do not promise bitwise agreement
across different GPU/library environments.

## 4. Search, selection and completion

**Greedy decoders.** Both reuse the shared masked decoder. Its predictive
distribution is normalized over non-mask tokens. Confidence ranks positions by
their largest token probability; entropy ranks by
`H(p) = -sum(p * log(p))`. Both commit the argmax token at the chosen position.
Ties prefer earlier output positions and lower token IDs.

**Search.** `stop_mode="budget"` applies `max_model_calls=B` to the root analysis
and every immediate-child evaluation. The counter checks the cap before each
forward. Search continues after three candidates and stops at the cap or when
there are no expandable nodes. It does not spend extra forwards on candidate
ranking or rollout completion.

The original UCB, action ranking and backpropagation remain in place. MEDAL
uses native full-vocabulary probabilities, entropy
`H = -sum(p * log(p + epsilon))`, and the top-two probability margin:

```text
action_score(position, token) = p(token) * exp(-H) * sigmoid(gamma * margin)
IG(parent, child) = (parent_total_entropy - child_total_entropy) / parent_total_entropy
```

The existing IG guard returns zero when parent entropy is at most epsilon.
Backpropagation updates cached tree means. Filtering search actions does not
renormalize probabilities or change these statistics.

**Selection after search.** Distinct evaluated depth-`Lc` states are ranked by
cached mean tree reward, then earlier node ID. The top `C` are retained for
inspection; only the highest-ranked state is completed. Target-depth leaves
have their immediate IG as their cached mean reward. No new trajectory score
or candidate-scoring forward is introduced.

| Available states | Selection | `selection_source` | `shortlist_underfilled` | `fallback_decode_used` |
|---|---|---|---|---|
| At least `C` at target depth | Best target state | `target_depth` | False | False |
| One or two at target depth (`C=3`) | Best available target state | `target_depth` | True | False |
| No target state, but evaluated children | Deepest; then highest cached mean; then earliest node ID | `deepest_evaluated` | True | True |
| Root only | Root | `root` | True | True |

Unevaluated children are never selected. `candidate_count` counts distinct
target states; `shortlist_size` is at most `C`. The legacy
`details.fallback_used` retains its quota-shortfall meaning for compatibility.
E0's fallback rate uses `fallback_decode_used`, so an underfilled target-depth
shortlist does not inflate that rate.

**Completion.** Both searched and no-search MEDAL now use `complete_llada`,
a state adapter around the repository's standard LLaDA confidence decoder.
Each forward proposes an argmax token for every remaining masked position at
T=0 (no Gumbel noise), then commits the position with highest
native probability for its proposed token (`low_confidence` remasking).
It does not apply the search entropy/margin score to completion selection.

The shared machinery excludes the mask token, normalizes over eligible tokens,
and preserves the prompt and all MCTS commitments. Completion uses fresh
forwards, one per remaining mask, without reusing search logits as a free step.
All E0 conditions, including both MEDAL paths and the demo, default to T=0.
The no-search MEDAL control therefore matches confidence greedy decoding under
identical model outputs. EOS/EOT remain eligible throughout completion.

`decoding.completion_method: medal` selects the previous literal
`softmax(adjusted_score / T)` sampler, which requires explicitly setting
`decoding.completion_temperature` above zero. Invalid T=0/sampler combinations
are rejected before loading weights. The original
`complete_medal` function and legacy CLI/AB defaults retain that behavior.
Saved configuration and prediction records identify the selected completion
method; offline reports state which decoder generated the answers.

## 5. EOS/EOT policy and answer extraction

`resolve_termination_tokens` resolves distinct EOS and `<|eot_id|>` IDs from
the pinned tokenizer, combines declared tokenizer/model/generation-config
termination IDs, and validates vocabulary membership, range and non-mask
identity. IDs and token spellings are saved in `config.json`. Invalid or
unresolvable IDs fail explicitly.

Search excludes those IDs **before** each position's top-K selection and
refills from eligible tokens. Mask tokens are also ineligible actions. Entropy,
margins, native probabilities and rewards are unchanged for the same state.
The exclusion does not reach any final decoder.

Every final decoder permits EOS/EOT and fills all 256 output positions. A
terminator committed out of order cannot trigger early stopping. After
completion, only the generated suffix is inspected; scoring uses the prefix
before its first terminator:

| Output suffix | Scored IDs |
|---|---|
| No EOS/EOT | Entire suffix |
| EOS/EOT at position 0 | Empty prefix |
| Several EOS/EOT tokens | Prefix before the earliest one |

Full output IDs and text with special tokens remain available. The prefix is
decoded with special tokens skipped and passed to the existing GSM8K parser.
That parser requires a valid `####` answer line, selects the last marked line,
and normalizes its number with `Decimal`. It never guesses an answer from
reasoning text. Parsing failures are incorrect predictions; fallback-generated
answers use exactly the same scoring rule.

## 6. Run the experiment

### Colab notebooks

- [Development](../../notebooks/gsm8k/Experiment_0_Baseline_Test_Dev.ipynb):
  five training questions, plus controlled decoder/workflow checks before
  loading weights.
- [Evaluation](../../notebooks/gsm8k/Experiment_0_Baseline_Test_Evaluation.ipynb):
  twenty different training questions; optional `INIT_TOKENS=7`.
- [Testing](../../notebooks/gsm8k/Experiment_0_Baseline_Test_Testing.ipynb):
  twenty-five held-out questions; optionally `NUM_SAMPLES=100`.

Run top to bottom on the existing BF16-capable L4/A100 Colab workflow. The
notebooks verify the GPU, get source, install `requirements/llada-colab.txt`
and the editable package, mount Drive, configure E0, and cache/stage the pinned
model. They use one fresh comparison subprocess, stream progress, and display
the exact run's reports, raw/scored outputs, termination boundaries and traces.

`SOURCE_MODE="github"` checks out `adi` and displays its commit. The checked-out
branch must contain E0. Before publication, use `SOURCE_MODE="upload"` and upload
the delivered `dist/experiment_0_colab_source.zip`; extraction uses an isolated
runtime directory. The bundle contains source, configs, tests, documentation
and the three notebooks, without weights, credentials or `.git`.

Set `NUM_SAMPLES`, `INIT_TOKENS`, and `SEARCH_BUDGETS` in the controls cell.
All three E0 notebooks set `TEMPERATURE=0.0` and pass it through to both MEDAL
completion paths; confidence and entropy baselines also use zero. The demo
reads the same zero default from the shared configuration. Temperature is
recorded per condition and in every prediction; offline reports use the saved
temperature so earlier T=1 results retain their original labels.
Evaluation/testing set `SEARCH_BUDGETS=[256, 512, 1024]` and
`RUN_ORDER="condition"`. They run both greedy methods first, one at a time,
followed by no-search MEDAL and then the ascending search caps. All six methods
share one loaded model. Completed-method metrics appear before the next method
starts. If interrupted, `observed_summary.csv` retains those per-method results;
the paired report includes only questions with outcomes under every condition.
`RUN_QUOTA_COMPARISON=False` is the default. After changing controls, rerun the
configuration and run cells. Freeze settings on development/evaluation
questions before viewing final-test answers; testing remains `Lc=5` unless
changed beforehand.

For a single interactive MEDAL run, use
[`MEDAL_MCTS_Demo.ipynb`](../../notebooks/gsm8k/MEDAL_MCTS_Demo.ipynb).
Its existing cells use this E0 prompt, search and EOS policy, with `Lc=5` and
one `SEARCH_BUDGET=512` control (alternatives 1024/2048). A no-search MEDAL
decoder block runs before MCTS using the same loaded model, prompt and completion
seed; it displays separate full/scored outputs and metrics, costs 256 forwards
by default, and is excluded from searched MEDAL's reported totals. Live counters show
budget use, evaluated depth and candidates. The report includes E0's applicable
MEDAL metrics and separate pipeline/decoding timings; completion reruns reuse
the initialization. Its optional batch remains disabled by default. The
standalone `demo_medal.py` CLI retains its legacy defaults.

### Command line

Install the existing LLaDA profile in an environment with appropriate PyTorch:

```bash
python -m pip install -r requirements/llada-colab.txt
python -m pip install --no-deps -e .

python scripts/experiment_0_baseline_test.py --stage dev
python scripts/experiment_0_baseline_test.py --stage evaluation --run-order condition --search-budgets 256 512 1024
python scripts/experiment_0_baseline_test.py --stage evaluation --init-tokens 7 --run-order condition --search-budgets 256 512 1024
python scripts/experiment_0_baseline_test.py --stage test --num-samples 25 --run-order condition --search-budgets 256 512 1024
python scripts/experiment_0_baseline_test.py --stage test --num-samples 100 --run-order condition --search-budgets 256 512 1024
```

A cached-model invocation with explicit artifact locations:

```bash
python scripts/experiment_0_baseline_test.py \
  --stage dev --num-samples 5 --init-tokens 5 \
  --search-mode budget --search-budgets 512 1024 2048 \
  --model-path /path/to/pinned/LLaDA-8B-Instruct --local-files-only \
  --cache-dir /path/to/dataset-cache \
  --manifest /path/to/manifests/gsm8k_e0.json \
  --output-dir /path/to/e0-runs \
  --run-manifest /path/to/receipts/e0-dev.json
```

Use the repository caching workflow so local weights retain validated pinned
source metadata. An arbitrary checkpoint is rejected.

| Option | Meaning |
|---|---|
| `--config PATH` | YAML configuration; defaults to the E0 config |
| `--stage dev\|evaluation\|test` | Select a frozen pool |
| `--num-samples N` | Pool prefix length; default 5/20/25 from the stage |
| `--init-tokens N` | Target search depth (`Lc`); 5 by default |
| `--search-mode budget\|candidates` | Primary budget stopping or supplementary strict quota |
| `--search-budgets B ...` | Distinct positive actual-forward safety caps |
| `--run-order example\|condition` | All methods per question (default), or all selected questions per method |
| `--model-path PATH` | Pinned local snapshot or model repository |
| `--local-files-only` | Disable model download; this does not create an absent dataset manifest |
| `--cache-dir PATH` | Hugging Face dataset cache |
| `--dataset-revision SHA` | Immutable dataset commit; must match an existing manifest |
| `--manifest PATH` | Shared frozen dataset manifest |
| `--output-dir PATH` | Run parent; stage, mode/depth and timestamp are appended |
| `--run-manifest PATH` | Exact invocation receipt, atomically updated as the run progresses |
| `--report-only RUN_DIR` | Rebuild reports from saved files; no model or dataset loading |

CLI values override YAML. `data.num_samples: null` selects the stage default.
Each budget overrides `mcts.max_model_calls`; `max_simulations` must be null.
The run is a new timestamp, not a resume or overwrite of an earlier experiment.

### Supplementary candidate-quota comparison

```bash
python scripts/experiment_0_baseline_test.py --stage evaluation \
  --search-mode candidates --init-tokens 5 --run-order condition --search-budgets 256 512 1024
```

Use the same `Lc` as the associated primary run. This separately reported mode
stops at three distinct target-depth states while retaining the forward caps.
If a cap/frontier prevents the quota, the original strict failure is preserved:
the record is `search_incomplete`, no completion runs, and the prediction counts
as incorrect. Other conditions/questions continue. Its condition IDs use
`medal_candidates_b...`; artifacts are separated by mode. Primary budget
exhaustion instead selects an available state and produces a normally scored
answer.

## 7. Measurement and interpretation

`CountingModel` measures actual invocations. Successful records must satisfy:

```text
total_nfes = search_nfes + completion_nfes
completion_nfes = generation_length - selected_depth
search_nfes <= requested_search_budget
```

With defaults, each greedy/control output costs 256 forwards. A fully consumed
search at depth 5 costs `B + 251` total forwards: 507, 763 or 1275 for the
evaluation/testing caps (2299 for an explicit B=2048 run). At depth 7,
completion costs 249. These are accounting examples, not measured results.
Frontier exhaustion may use less than B; fallback below target depth needs more
completion forwards. The one warm-up forward is outside all condition totals.

Search, completion and combined decoding runtimes are CUDA-synchronized.
Combined decoding includes decoder bookkeeping; trace disk writes,
tokenization, scoring, model loading and report generation are outside that
interval. Search/completion timers and internal phase diagnostics are nested
measurements, so do not add all diagnostic phase timers together.
`setup.json` records the separate loading, warm-up and overhead costs.

| Metric | Definition / saved interpretation |
|---|---|
| Accuracy | Correct answers / terminal predictions; percent plus Wilson 95% interval |
| Parse rate | Fraction yielding a valid marked numeric answer |
| Search / completion / total NFEs | Actual phase and total invocations; per-example, means and aggregate totals |
| Budget utilization | Actual search NFEs / requested B; unavailable for no-search conditions |
| Runtime | Mean search/completion/combined seconds; combined median and p95, plus aggregate decode time |
| Selected depth | Number of output tokens committed before completion |
| Maximum / average search depth | Deepest evaluated search state / mean depth of evaluated child nodes |
| Candidates / shortlist | Distinct evaluated target states / retained top-C count |
| Fallback rate | Fraction completed from below target depth or root |
| Underfilled rate | Fraction with fewer than C target states, independently of fallback |
| Termination reason | Search cap, exhausted frontier, candidate quota, or completed no-search output |
| Gain per additional NFE | Accuracy gain in percentage points / additional mean total NFEs |

Gain is reported relative to **each** no-search baseline on the same paired
questions. A nonpositive denominator yields `null` in JSON, an empty CSV field,
and `—` in Markdown (unavailable). Plot error bars are marginal Wilson
intervals, not a significance test for differences. Fraction-valued fields
(`parse_rate`, fallback, utilization and JSON interval bounds) use 0–1;
`accuracy_percent` and gain fields use percent/percentage points.

Primary summaries use only questions with terminal records for every requested
condition. Supplementary quota failures are terminal incorrect predictions.
Interrupted/unattempted conditions remain partial, not fabricated incorrect
answers. `observed_summary.csv` separately includes all available terminal
records, which may have unequal sample counts. Always check `common_n`,
`requested_n`, run status and `complete` before interpreting a comparison.

## 8. Saved artifacts and exact receipts

Default run layout:

```text
<experiment_root>/LLaDA/GSM8K/experiment_0/<stage>/<mode>-lc<depth>/<timestamp>/
├── config.json             # Resolved settings, conditions, provenance, termination policy
├── manifest.json           # Full frozen dataset selection manifest
├── selection.json          # Ordered examples attempted by this run
├── setup.json              # Run status, loading/warm-up/overhead costs, active partial attempt
├── predictions.jsonl       # One flushed terminal example/condition record per line
├── summary.json            # Incremental progress summary
├── traces/*.jsonl          # Compact search/completion events or greedy commitment traces
└── report/
    ├── report.md
    ├── summary.json
    ├── plot_data.json
    ├── summary.csv          # Paired condition metrics
    ├── observed_summary.csv # All available terminal outcomes
    ├── accuracy_gains.csv
    ├── results.csv          # Flattened per-example results
    └── plots/
        ├── accuracy_vs_total_nfes.{png,pdf}
        └── accuracy_vs_runtime.{png,pdf}
```

Key `predictions.jsonl` fields:

| Group | Fields |
|---|---|
| Identity | `example_id`, `row_id`, `source_split`, `stage`, `content_hash`, model/dataset revisions, `protocol_version` |
| Pairing | `condition`, `method`, `completion_method`, `temperature`, `seed`, `prompt`, `prompt_hash`, `prompt_ids_hash` |
| Compute/time | `search_nfes`, `completion_nfes`, `total_nfes`, `requested_search_budget`, `budget_utilization`, `search_seconds`, `completion_seconds`, `decoding_seconds` |
| Search | `selected_depth`, `max_depth`, `average_search_depth`, `candidate_count`, `shortlist_size`, `shortlist_underfilled`, `selection_source`, `fallback_decode_used`, `stop_reason` |
| Outputs | `full_output_ids`, `raw_output`, `scoring_output_ids`, `scoring_text`, `first_terminator_position`, `first_terminator_id`, `tokens_after_terminator` |
| Scoring | `predicted_answer`, `reference_answer`, `parse_success`, `exact_match` |
| Inspection | `status`, `details`, `trace_file`; shortlist/reward/selected-state details are nested under `details` |

Compact traces omit full entropy/state arrays and repeated candidate payloads.
They retain action choices and search diagnostics; selected initialization and
candidate data remain in result details. Trace events are buffered during one
condition and written after decoding to keep disk I/O outside its timing.

`--run-manifest` writes a receipt containing `run_dir`, `status`,
`protocol_version`, `stage`, `search_mode` and `model_family`. Each notebook
allocates a unique receipt path, validates the returned identity/status, and
uses its exact directory. It never picks a directory by searching for the
latest timestamp.

Numerical/model errors stop the run explicitly as `failed`; Ctrl-C/SIGTERM
marks it `interrupted`. Completed records and partial diagnostics are
preserved, including the active attempt's actual forward count. Notebook Stop
requests graceful subprocess termination. A hard kill or runtime crash can
still lose an unfinished condition. Existing completed JSONL records remain
available; an incomplete trailing line is ignored with a report warning.
There is no automatic resume.

Offline rebuilding:

```bash
python scripts/experiment_0_baseline_test.py --report-only /path/to/exact/run
```

This checks manifest, record identity, paired prompts/seeds, NFE accounting,
output length, termination-prefix IDs and saved scores before rebuilding
tables/plots. It needs neither weights nor datasets and makes no downloads.
It does not re-tokenize saved text, so the saved token/text correspondence is
trusted. Preserve original artifacts for audit.

## 9. Validation and development acceptance

Install test dependencies, then run controlled and regression checks:

```bash
python -m pip install -r requirements/dev.txt

python -m pytest -q tests/decoding/test_experiment_0.py \
  tests/tasks/test_e0_dataset.py tests/workflows/test_experiment_0.py

python -m pytest -q tests/decoding tests/tasks/test_e0_dataset.py \
  tests/tasks/test_gsm8k_benchmark.py tests/workflows
```

Coverage includes greedy/entropy ordering and deterministic ties; exact
forward counts and prompt preservation; continuing beyond three candidates;
cap/frontier stopping; zero/one/two/many-candidate selection; root-control seed
equivalence; search-only EOS exclusions with unchanged entropy; output prefix
edge cases; strict matched-depth quota failures; frozen disjoint/scaled data;
one model load; partial accounting on errors/SIGTERM; valid notebooks and exact
receipts; and offline report rebuilding.

During implementation, the broader local regression pass completed with
**464 passed, 5 CUDA-only skipped**. The final E0-focused pass after graceful
termination handling completed with **24 passed, 1 CUDA-only skipped**.
These overlapping test counts must not be added. An isolated source-bundle
check also exercised the one-load integration and three notebook workflows
with Hub/dataset network access disabled.

The protocol-v2 prompt update subsequently passed **100 tests, with 1
CUDA-only skip**, covering E0 and the existing GSM8K benchmark. The checks
include identical demonstration prompts across conditions, the final question block,
literal question insertion, legacy prompt rendering, and updated notebook
receipts.

**The actual five-question LLaDA GPU development run remains pending.**
The implementation environment had no CUDA runtime and no approved Colab
browser access. Controlled-model results are not LLaDA accuracy/runtime
measurements.

Before scaling, run the development notebook and inspect the following:

1. All six conditions have five terminal records and the receipt is completed.
2. Each baseline/control has 256 NFEs; each searched output has
   `256 - selected_depth` completion NFEs and respects its search cap.
3. Search actions contain no resolved EOS/EOT IDs; full outputs and scored
   prefixes show the intended boundaries.
4. Underfilled shortlists and fallback answers are visible and scored normally.
5. Phase/combined runtime, candidate/depth metrics, parse rate and exact
   outputs look consistent; offline rebuilding reproduces the report.

Resolve implementation or configuration problems using the training pools.
Then run evaluation, freeze the final configuration, and run the held-out
testing notebook without tuning settings in response to test answers.
