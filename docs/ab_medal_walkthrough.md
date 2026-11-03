# AB-MEDAL implementation guide

The IG baseline retains `ab_medal_fixed_budget_v2`; the optional confidence
variants use `ab_medal_adalook_hybrid_v1`. Both use the same frozen inputs, final
candidate selection and LLaDA Confidence completion. The numbered algorithm
sections below describe the IG baseline unless explicitly stated otherwise.

The notebook defaults to a serial paired comparison on five development
questions, depth 10 and search budget 256. `demo` mode retains the single-question
walkthrough. Formal E1/E3 runners remain separate. Optional batching is an
implementation study, not an algorithmic treatment. MEDAL and AB reuse prompting,
completion, audit and report utilities.

Entry points:

- [AB-MEDAL notebook](../notebooks/gsm8k/AB_MEDAL_MCTS_Colab.ipynb).
- [Shared protocol and exact prompt](experiments.md).
- [Adaptive search](../src/dlm_project/decoding/ab_medal.py).
- [Development runner](../src/dlm_project/tasks/gsm8k/ab_development.py).
- [Paired comparison runner](../src/dlm_project/tasks/gsm8k/ab_comparison.py).
- [AB AdaLook adapter](../src/dlm_project/decoding/ab_adalook_rollout.py).
- [Offline reconstruction](../src/dlm_project/decoding/search_audit.py).
- [Batch executor](../src/dlm_project/decoding/ab_batch.py).

## Notebook comparison and AdaLook semantics

Settings in `AB_MEDAL_MCTS_Colab.ipynb`:

```python
NOTEBOOK_MODE = "comparison"   # or "demo"
STAGE = "dev"                  # dev: 5, evaluation: 20, test: 200
NUM_SAMPLES = 5
CONDITION = "A"                # Lc=10, budget=256; B: Lc=10, budget=512
COMPARISON_REWARD_MODES = ["ig", "adalook_adaptive"]
REWARD_MODE = "adalook_adaptive"  # demo only
RESUME_RUN = None
```

Use development questions for tuning; five questions are a diagnostic run.
Freeze settings before evaluation/test. Adding `adalook_one_step` separates the
confidence reward change from the extra adaptive computation. The default
AdaLook thresholds match MEDAL's starting values; they are not tuned results.
The reward modes share AB priors `(0, 1, 1, 0.1)`. Reward scales differ, so inspect
reward distributions and posterior updates; any prior sensitivity study should
be separate and development-only.

Every traversal still uses AB's Student-t GEN/CONT decisions and expands exactly
one real child. IG evaluates and backs up that child immediately. One-step mode
replaces IG with `alpha * log(action_probability) + log(confidence_mass)`, using
the existing epsilon floor, without extra forwards.

Adaptive mode retains the real child's base analysis while evaluating a detached
comparison hypothesis for every other retained action of the same parent. An
already evaluated real sibling contributes its cached **base** analysis, not its
previous refined reward. Otherwise a new comparison forward is charged. Negative
temporary IDs are scoped to a real-child group and never enter `search.nodes`,
posterior tables, remaining-action mutation or final candidate selection.
Groups retain the parent's action order for deterministic tie-breaking, rather
than giving the current real child priority in equal-score restarts.

The shared `ConfidenceRefinement` evaluator then applies the existing MEDAL
variance gate, confidence-based commitments, pruning and restart rules to this
group. It copies all hypothesis states. Only the current real child's resulting
score is returned to AB: exactly one tree backup and one update for each original
GEN/CONT posterior recipient. Other hypotheses receive no AB update. If the real
origin is pruned, it keeps its last valid score; a restart rooted at a different
origin cannot transfer that origin's reward to the real child.

`adalook_max_depth=1` and a zero variance threshold bypass comparison forwards and
give the one-step ablation. Other groups and each refinement round must fit in
full before the next configured checkpoint/budget; insufficient budget falls
back to the last valid score. Reaching a checkpoint finalizes the pending child
before the snapshot. Checkpoint schedules can change refinement allocation;
compare runs with the same schedule. With only one available candidate, adaptive
refinement stops rather than fabricating a variance comparison.

The evaluator is AdaLook-inspired, not the paper's block/ETE decoder. AB's
temporary sibling comparisons are a project-specific adaptation. They can
consume more computation than MEDAL's sibling groups, which are real expansions.

Accounting is explicit:

```text
search_nfes = root_analysis_nfes + real_child_evaluation_nfes
            + temporary_candidate_nfes + adalook_extra_nfes
adalook_total_extra_nfes = temporary_candidate_nfes + adalook_extra_nfes
total_nfes = search_nfes + completion_nfes
completion_nfes = 256 - selected_depth
```

`adalook_extra_nfes` retains MEDAL's rollout-only definition. Successful tree
forwards reconcile to `nfes_per_depth` and `nfes_per_root_subtree`; temporary work
is reported separately. Failed attempted forwards remain in cost accounting.
`adalook_trigger_rate` is the fraction of finalized groups reaching depth > 1;
stop reasons distinguish no progress, variance, pruning and budget boundaries.
All real rewards, original IG, one-step scores and posterior updates are saved.

Comparison outputs are under
`MyDrive/CSCI544-DLM/results/ab_medal/comparison/<stage>/<condition>/<run>/` in
Colab (under the local cache root outside Colab). They include:

- `config.json`, `manifest.json`, `selection.json`: inputs, variants and code fingerprint.
- `predictions.jsonl`: durable per-attempt results, including interruptions.
- `traces/` and `questions/<variant>/<question>/<attempt>/`: full diagnostic evidence.
- `summary.json`, `report/summary.csv`, `report/report.md`, `report/comparison.png`:
  accuracy, mean search/completion/total NFE and decoding time, plus AB diagnostics.
- `report/per_question.csv`, `report/paired_results.csv` and `.json`: latest results
  and wrong-to-right/right-to-wrong transitions on completed pairs only.

Setup/model loading, warm-up, tokenization, scoring and artifact writing are
excluded from decoding time. Incomplete summaries cover completed questions and
must not be treated as final matched-cohort comparisons. Set `RESUME_RUN` to the
printed run directory to skip completed variant/question pairs. Interrupted
questions restart with a new attempt directory. Configurations and source hashes
must match. A completed run can regenerate reports without a loaded model.

In `demo` mode, `REWARD_MODE` and `SAMPLE_INDEX` select one development question.
The existing serial-five and batching sections remain optional. The cooperative
executor uses the same request/submit state machine, including temporary forwards;
run its padding/search parity checks before collecting batched throughput results.

## 1. Prompting and the initial state

The runner first validates `configs/gsm8k/experiments_common.yaml` and the
checked-in sample manifest. The manifest has five development questions and
20 different evaluation questions from training, plus 200 test questions.
It is read offline, in its frozen order, without sampling or looking up a newer
dataset revision. A one-question run selects index zero of the development pool;
it does not redefine that pool.

The model and tokenizer use the repository-pinned LLaDA-8B-Instruct revision.
The shared configuration uses bfloat16. A local checkpoint must carry the same
model identity. The runner records model and dataset revisions, sample ID,
common-protocol fingerprint, template hash, rendered prompt hash, and token-ID
hash. Operational choices such as artifact/cache locations do not change the
experimental settings.

The exact user-message template is:

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

Only the current question is substituted. The wrapper passes this text as one
user message to the pinned tokenizer's existing chat template with
`add_generation_prompt=True`, then tokenizes with `add_special_tokens=False`.
There is no automatic demonstration insertion, separate decomposition generation,
or planning model call. Reference answers remain available only to final scoring.

For a prompt of length P, the root is a CPU int64 tensor of shape `[1, P+256]`:

```text
[prompt tokens, MASK, MASK, ... 256 output masks ...]
```

Prompt tokens are immutable. A tree edge commits one token at an arbitrary
output position; depth counts committed output positions, not reasoning steps.
The per-question seed is `(1 + 5 * row_id) % 2**63`, matching the shared protocol.
Every search owns an independent NumPy RNG.

## 2. Searching: selection

The root is analyzed once. Its forward counts against the search budget. A node
stores its CPU state, parent/action, depth, cached entropy/actions, children,
visits N, reward sum W, empirical Q, and whether its frontier is exhausted.
AB additionally stores independent `node`, `GEN`, and `CONT` posterior objects,
the untried retained actions, and decision history.

A traversal restarts at the root after each child evaluation. At a node:

- GEN is available if a retained action remains untried.
- CONT is available if at least one child has an open frontier.
- If only one is available, take it without drawing GEN/CONT scores.
- If both are available, draw from both predictive distributions. GEN wins a
  strict greater-than comparison; a tie chooses CONT.
- CONT draws a predictive score from every eligible child's `node` posterior.
  Choose the largest score; ties prefer the smaller node ID, then descend.
- GEN ends traversal at that parent and requests one new child evaluation.

Even forced CONT samples eligible child scores. Closed children are excluded.
No decision invokes the language model.

### Student-t predictive distribution

Each posterior has parameters `(m, kappa, nu, tau2)`, initialized independently
at `(0, 1, 1, 0.1)`. The underlying normal–inverse-chi-squared model produces:

\[
\widetilde R = m + \sqrt{\tau^2(1+1/\kappa)}\,t_\nu.
\]

These are Student-t **posterior-predictive reward draws**, including uncertainty
in a future reward. The implementation does not switch to a normal distribution
at high visit counts and does not sample only a posterior mean. `m` is a Bayesian
location parameter; it is distinct from the empirical tree value Q.

The `medal_equivalent` debugging policy uses MEDAL's UCB and buffered sibling
rounds. It supports exact tree comparisons under identical inputs. The normal
adaptive policy is expected to grow a different tree from UCB MEDAL.

## 3. Searching: expansion and action generation

A cached model analysis supplies probabilities over the full vocabulary at each
unresolved output position i. With epsilon `1e-8`:

\[
h_i=-\sum_v p_i(v)\log(p_i(v)+\epsilon),\qquad
\Delta_i=p_i^{(1)}-p_i^{(2)},
\]

\[
s(i,v)=p_i(v)\exp(-h_i)\,\sigma(10\Delta_i).
\]

Select up to K1=3 eligible tokens per position, then retain the K2=5 highest
scoring actions globally. Equal scores prefer earlier output positions and then
smaller token IDs. Search excludes MASK and the resolved EOS/EOT IDs from
eligible actions. These exclusions do not renormalize the probabilities used
for entropy and margins. This is the same implementation used by MEDAL.

GEN takes the next untried action in that cached ordering. It clones its parent's
state, commits exactly one token, assigns a new node ID, and increases depth by
one. A node at requested depth Lc has no expandable frontier. Cached action lists
are finite: adaptive branching chooses when to expand or descend within those
lists; it does not repeatedly generate new action proposals.

MEDAL expands retained siblings in a round before reselecting with UCB.
AB evaluates one sibling and backs up its reward before reselecting from the
root. That ordering is preserved even during cross-question batching.

## 4. Searching: simulation

"Simulation" is one model evaluation of the immediate child state. There are no
full completions or multi-step rollouts inside search. Parent entropy is cached.
Child entropy is the sum over its still-masked output positions:

\[
R_{IG}(p,s)=
\begin{cases}
(H(p)-H(s))/H(p), & H(p)>10^{-8},\\
0, & \text{otherwise}.
\end{cases}
\]

The reward includes the entropy removed by committing a position and changes to
other unresolved positions. It is not a correctness reward. Negative IG remains
negative; it is neither clipped nor discarded. Child analysis also caches the
next action list without another forward. No reference answer enters evaluation.

Model errors remain explicit failures. An unevaluated/failed node is excluded
from final candidate selection and evaluated-node statistics, but its attempted
forward remains in cost accounting.

## 5. Searching: backpropagation

Every successful child reward r updates ordinary tree statistics once, from
that child through the root:

\[
N\leftarrow N+1,\quad W\leftarrow W+r,\quad Q=W/N.
\]

There is no discount, gradient, learned verifier, or accumulated trajectory
reward. The same immediate edge reward is added to each ancestor.

Each affected Bayesian posterior uses its **old** parameters on the right:

\[
m'={\kappa m+r\over\kappa+1},\quad
\kappa'=\kappa+1,\quad \nu'=\nu+1,
\]

\[
\tau'^2={\nu\tau^2+{\kappa\over\kappa+1}(r-m)^2\over\nu+1}.
\]

For a traversal `root --CONT--> A --CONT--> P --GEN--> child`, update:

| Posterior | Why |
|---|---|
| `child.node` | Observed the new child's reward |
| `P.GEN` | Generation at P produced this reward |
| `A.node`, `P.node` | Their subtrees produced this reward |
| `root.CONT`, `A.CONT` | Continuing through these nodes produced this reward |

Do not update `P.CONT`, the new child's GEN/CONT priors, or `root.node` for this
observation. Each listed posterior receives the reward exactly once. The root's
ordinary visits/W/Q still update. For example, r=0.08 applied to the default
prior yields m=0.04, kappa=2, nu=2, tau2=0.0516.

## 6. Stopping, final candidate selection, and fallback

Supported shared configurations are:

| Configuration | Lc | Search budget |
|---|---:|---:|
| A (development default) | 10 | 256 |
| B | 10 | 512 |

Count root analysis and all child forwards. Stop at the forward budget or an
exhausted frontier; three candidates never stop a fixed-budget search.

1. Choose the deepest evaluated depth with at least three distinct states. If no
   depth qualifies, choose the most populated depth, preferring deeper on ties.
2. Collect every distinct state at that depth using exact token IDs/mask positions.
3. For duplicate states, keep the first evaluated occurrence and its final
   statistics; do not pool transposition statistics.
4. Rank by final empirical Q=W/N, breaking ties by ascending node ID.
5. Keep at most three; complete only the highest ranked state.

Ranking performs zero model forwards and uses neither UCB bonuses nor Bayesian
predictive draws. Selection reads accumulated W and N; a selected node may have
descendants, so its Q need not equal its immediate reward.

Apply exactly this rule below Lc. Fallback means selected depth < Lc, independent
of shortlist size. A root-only search has a one-item shortlist with Q=0 and
completes from all masks. A candidate reaching Lc counts as requested-depth
success, but the selected depth can still be below Lc under the shortlist rule.
Model errors do not become valid fallback runs.

Low-level callers can explicitly request legacy `stop_mode="candidates"` and
an optional cap. Historical comparison adapters retain this behavior, including
the quota requirement. Current notebook/CLI defaults use the shared budget mode.

## 7. Decoder and final scoring

Completion uses the same LLaDA Confidence implementation as E0, temperature 0,
with output length = steps = block length = 256. At each active step:

1. Forward the selected partially masked state.
2. Exclude MASK from completion proposals and compute legal-token probabilities.
3. Take the argmax token at every unresolved output position.
4. Commit the position whose proposed token has the highest probability, with
   earlier-position tie breaking.
5. Repeat until every output position is filled.

Prompt tokens and all search commitments stay fixed. EOS/EOT remain eligible
during completion. They do not stop generation; scoring trims the finished
output before its first terminator. The strict parser extracts the final marked
`####` answer. Malformed answers count as parsing failures and incorrect results.

For a successful question:

\[
N_{total}=N_{search}+N_{completion},\quad N_{search}\le B,\quad
N_{completion}=256-d_{selected}.
\]

At full budget 256 and depth 10: 256 search + 246 completion = 502 sequence NFEs.
At a root-only budget of 1: 1 + 256 = 257. There is one completion, no candidate
rollout, and no additional ranking forward.

## 8. Logs, offline reconstruction, and metric definitions

Each evaluated node has `node_id`, `parent_id`, depth, evaluation order, incoming
action, exact-state hash, immediate reward, final visits, reward sum, and Q.
Root evaluation order is zero and its immediate reward is null. Save prompt IDs
once per question. Parent/action links reconstruct every output state; hashes
verify exact full prompt-plus-output identities.

`search_trace_v1` is separate from the unchanged common-protocol fingerprint.
AB decisions have their own traversal ID and decision order; they are not
collapsed into one decision per child. The traces retain posterior recipients,
updates, and sampled decision scores. Forward attempts remain distinguishable
from successful node evaluations. Interrupted searches preserve partial evidence.

The shared calculator checks state hashes, parent/depth/order consistency,
ordinary reward backups, final empirical-Q ranking, and NFE allocation. It uses
only the standard library: no tokenizer, model, dataset download, or RNG draws.

| Group | Definition |
|---|---|
| Performance | Per-question correctness and parsing success; means give accuracy/parsing rate. n=1 is explicitly labeled. |
| Depth | Maximum evaluated depth; selected depth; mean evaluated-child depth excluding root. |
| Width | Evaluated children; parents with evaluated children; min/mean/max of those child counts. |
| Candidates | Distinct deepest states, distinct states at requested Lc, shortlist size. |
| State exploration | Unique evaluated states including root; duplicate evaluations = evaluated nodes minus unique states. Duplicate percentage divides by evaluated children. |
| Root diversity | Number of evaluated root-child subtrees; coverage divides by retained root actions. Deepest-candidate and shortlist coverage use first-occurrence representatives. |
| Reward | Count/percentage of evaluated children with IG < 0; IG min/mean/max; selected Q. Negative percentage divides by evaluated children. |
| Depth allocation | One search sequence NFE per evaluated state, including root at depth 0. |
| Subtree allocation | Descendant evaluations assigned to their first root-child ancestor, including that ancestor's evaluation. Root analysis is separate. |
| Adaptive decisions | GEN/CONT counts and percentages over recorded decisions; forced total and per-type counts/percentages with the same denominator. MEDAL values are unavailable. |
| Efficiency | Requested-depth success means at least one evaluated node at Lc. Fallback is selected depth < Lc. Budget utilization is search NFEs/B. |
| Cost | Search/completion/total sequence NFEs, physical batched invocations, and measured phase runtimes. |

Root-only branching and negative counts are zero; reward summaries and
percentages without observations are unavailable. Shortlist diversity reports
pairwise commitment-set Jaccard distance and common action-prefix length;
fewer than two candidates leave pairwise summaries unavailable.

For successful searches, depth counts sum to search NFEs, and root analysis plus
subtree counts also sum to search NFEs. Failed attempts are exposed separately
as uncompleted evaluation NFEs, not assigned fake node rewards.

Artifacts per question include `prompt.json`, `config.json`, `nodes.jsonl`,
`decisions.jsonl`, `forwards.jsonl`, `backups.jsonl`, `trace.jsonl`, `search_audit.json`, `tree.json`,
`selection.json`, `completion.json` when available, `metrics.json`, and
`metrics.csv`. Selection artifacts retain every deepest candidate's evidence
and the exact selected intermediate state. Allocation/tree PNGs are computed
from saved data after decoding.

E0 now emits the same common node records and stores prompt IDs under its
`prompts/` directory. Its offline report verifies audit evidence and exports
`compute_allocation.csv`. Historical traces recover topology/reward/allocation
where available; absent explicit hashes and selection evidence remain labeled
as unavailable. Old saved artifacts are not rewritten or silently rerun.

## 9. Optional batching and timing

Each question retains its own tree, posterior objects, seed, remaining budget,
and selected state. A cooperative request interface yields at most one pending
root/child per tree. The coordinator batches independent requests, processes
their results, then allows each tree's next traversal. No same-tree speculative
expansion or shared reward statistics are introduced.

```text
for each consecutive group of questions:
    create one independent AB tree per question
    while any tree has work:
        collect one pending request from each active tree
        forward the right-padded states together
        charge one sequence NFE to each participating tree
        apply each result and complete its reward/posterior backups
    select each tree's deepest state
    while any selected state has output masks:
        forward active completion states together
        commit one Confidence token per state
        remove completed rows
```

Padding is appended **after each real prompt-plus-output sequence**, preserving
original positions. Binary attention masks hide padding. Padding is never an
output mask, commitment, candidate identity, or prompt hash. One batched forward
with four active questions is one model invocation and four sequence NFEs.

Before measuring speedup, the study runs serial reference searches, compares
unpadded/padded-serial/batched logits and action predictions on saved root and
selected states, then compares complete runs at sizes 2 and 4. Discrete decisions,
selected states, and outputs must match. IG/Q use rtol=1e-4 and atol=1e-5.
Logit differences are recorded; attention masking alone is not accepted as
proof of equivalence. Failing configurations do not contribute speedup results.
Measured repetitions are checked again against the serial reference.

The study uses the same first four development questions, fixed seeds, and chosen
configuration for every size. Three configurable repetitions rotate execution
order following separate warm-up. Reports include median/spread, throughput,
speedup, occupancy, sequence NFEs, physical invocations, correctness/fallback,
and peak CUDA allocated/reserved memory. OOMs are recorded without silently
shrinking batches. Earlier results remain available.

CUDA synchronization bounds measured phases. Group wall time defines throughput;
per-question latencies overlap and must not be summed. Model loading, warm-up,
tokenization, scoring, offline analysis, plotting, and disk writes are separate.
The local CPU fixtures establish implementation behavior, not GPU speedup.
Future MEDAL-versus-AB algorithmic comparisons must use matched execution
conditions; serial remains the default until both batching paths pass real
LLaDA CUDA validation. Both paths have CPU stub parity tests.

## 10. Development commands and checks

Install the existing model-specific and development dependencies, then install
the package in editable mode. In a configured CUDA environment:

```bash
python scripts/run_ab_medal_single.py --condition A --num-samples 1
python scripts/run_ab_medal_single.py --condition A --num-samples 5
```

Use `--model-path` and `--local-files-only` for a validated cached checkpoint.
`--stage evaluation --num-samples 20` and `--stage test --num-samples 200` use
those existing frozen pools; select them only after development checks. This
runner remains a diagnostic workflow, not the future paired E1/E3 runner.

Offline reconstruction:

```bash
python scripts/run_ab_medal_single.py --from-result PATH_TO_QUESTION_ARTIFACTS
```

For a standard-library-only process, load `search_audit.py` with Python's
`importlib.util.spec_from_file_location` and call `load(question_directory)`.
The notebook provides the optional serial-five and batching sections, both off
by default. It requires the five-question serial section to complete before
starting the batch study.

Acceptance checks cover hand-computed trees and negative rewards, duplicate
states and first occurrences, root-only/short shortlists, no quota stop,
EOS/EOT policies, exact Confidence commitments, posterior recipients, fixed-budget
`medal_equivalent` parity, zero report/ranking forwards, and offline reconstruction.
Batch tests use unequal prompt lengths and stopping depths, independent seeds,
partial groups, failures/OOMs/interruptions, and actual multi-row model calls.
The functional notebook cells are exercised with stub models; real LLaDA
prediction parity and performance must be established on the target GPU.
