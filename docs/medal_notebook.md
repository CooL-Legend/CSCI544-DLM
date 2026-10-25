# Matched MEDAL and AB-MEDAL notebooks

[MEDAL_MCTS_Demo.ipynb](../notebooks/gsm8k/MEDAL_MCTS_Demo.ipynb) and
[AB_MEDAL_MCTS_Colab.ipynb](../notebooks/gsm8k/AB_MEDAL_MCTS_Colab.ipynb)
share their executable cells. The sole code difference is the initialization
selector: `METHOD = "medal"` versus `METHOD = "ab_medal"`.
Both call the same development helpers, retaining the historical
`ab_development` module name for compatibility.

## Shared inputs and decoding

| Setting | Both notebooks |
|---|---|
| Initial question | First frozen development question (`SAMPLE_INDEX = 0`) |
| Stage pools | Existing 5 development / 20 evaluation / 200 test manifest |
| Model/tokenizer | Repository-pinned LLaDA-8B-Instruct, bfloat16 |
| Prompt | Exact shared pens/muffins two-shot template and chat tokenization |
| Seed | Shared per-question seed, `(1 + 5 * row_id) % 2**63` |
| Initial output | 256 masks |
| Default | Condition A: Lc=10, search budget=256, batch size=1 |
| Other conditions | B=(10,512), C=(10,1024) |
| Retained actions | K1=3, K2=5; identical probability/entropy/margin scoring |
| Reward | Immediate normalized information gain, including negative rewards |
| Stopping | Forward budget or frontier exhaustion; root forward counts |
| Final selection | Depth with >= C distinct states (fallback to most distinct, deeper on ties), first occurrence, empirical Q, top 3, best 1 |
| Fallback | Selected depth below Lc; valid root-only and short shortlists complete |
| Completion | LLaDA Confidence, temperature 0; fixed 256 output tokens |
| Scoring | EOS/EOT-prefix scoring and strict GSM8K answer parsing |

The frozen configuration, samples, prompt IDs and shared fingerprint are
validated without resampling. Both notebooks display the sample, prompt, seed
and hashes. Reference answers are used only for scoring. The complete template
and metric definitions are in [the shared protocol](experiments.md) and
[the implementation guide](ab_medal_walkthrough.md).

## Initialization is the only algorithmic change

MEDAL's single-sample section calls the existing `MCTSSearch`. It selects
eligible children using

\[
Q(s)+0.01\sqrt{\log(\max(1,N(p)))/N(s)}.
\]

Unvisited children receive an infinite exploration bonus. Existing deterministic
action/node tie rules remain intact. An expansion round evaluates the selected
parent's retained siblings in order, up to K2=5, before another UCB selection.
Each evaluation computes immediate IG and backs it up through the child and
its ancestors. A budget may end partway through a sibling round.

AB-MEDAL uses its existing Student-t GEN/CONT policy, one child per traversal,
and Bayesian updates in addition to ordinary visits and reward sums. Its full
algorithm is described in [the AB guide](ab_medal_walkthrough.md).

Neither initializer performs full rollouts or final-ranking forwards. Their
selected states, answers and measured metric values can differ because their
trees differ; input settings, decoder and metric definitions match.

## Identical walkthrough and artifacts

Run each notebook from the beginning:

1. Check out the repository, install the pinned dependency profile, and configure
   the existing Drive/model cache.
2. Validate frozen inputs, then load one model and perform a separate warm-up.
3. Display the exact prompt and run one initialization.
4. Inspect every deepest candidate ranked by empirical Q, the shortlist and
   selected commitments.
5. Complete the best state and score it.
6. Reconstruct metrics offline and display the tree, depth-allocation and
   root-subtree-allocation tables and plots.
7. Optionally enable `RUN_SERIAL_DEV` for all five development questions.
8. After serial validation, optionally enable `RUN_BATCH_STUDY` for sizes 1/2/4.

The same metric fields and JSON/JSONL/CSV artifact schemas apply to both methods.
MEDAL's GEN/CONT counts, forced-decision metrics and Bayesian diagnostics are
unavailable, rather than fabricated zeros. Its tree plot shows UCB edges without
GEN/CONT labels. Search, completion and total NFEs and timing use the same
accounting; successful completion costs exactly `256 - selected_depth` NFEs.

Outputs use separate `results/medal/...` and `results/ab_medal/...` directories
so one method cannot overwrite the other. Tree PNG names follow the method;
all other artifact names and offline reconstruction are shared. Interrupted
runs save partial evidence. The old notebook's independent no-search control,
custom-question defaults and alternate decoder section have been replaced by
this matched workflow. Historical low-level helpers and CLI interfaces remain.

## Optional batching

The shared coordinator batches independent questions, keeping one pending
forward per tree. MEDAL's cooperative adapter retains the selected parent
between child evaluations until that sibling round finishes. It uses the same
UCB, action, IG, backup and final-selection primitives as ordinary `MCTSSearch`,
and creates no Bayesian posteriors. Tests compare complete trees, selection
records, rewards, visits and outputs against the ordinary implementation.

Both notebooks run the same first four development questions for each batch
size. Saved-state padding checks and complete-run parity checks must pass before
throughput measurement. Discrete decisions, selected states and output tokens
must match serial results; IG/Q tolerances remain `rtol=1e-4, atol=1e-5`.
Three configurable repetitions rotate execution order after separate warm-up.
Reports separate per-question sequence NFEs from model invocations and use group
wall time for throughput, with occupancy and GPU memory measurements. OOMs and
interruptions retain previous results without silently resizing batches.

CPU stub tests verify notebook execution and serial/batched algorithm parity.
Real LLaDA numerical parity and throughput still require Colab CUDA validation.
Formal MEDAL-versus-AB comparisons must use matching execution conditions.

```bash
python -m pytest tests/decoding/test_medal_batch.py \
  tests/workflows/test_medal_development.py tests/workflows/test_ab_development.py
```
