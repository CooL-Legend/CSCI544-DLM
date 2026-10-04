# Reading and running the MCTS decoder

Start with `src/dlm_project/decoding/mcts_decoding.py`. `MCTSSearch.step()` calls the four named stages in execution order. `run()` repeatedly calls `step()` on the **same tree**. Initialization is lazy: the first `step()` analyzes the root, then executes the first cycle.

This implementation adapts the confidence score and normalized entropy reward from *Diffusion Language Model Inference with Monte Carlo Tree Search* (MEDAL). It evaluates one committed token with one additional forward, then uses greedy completion after initialization. It is not a reproduction of full-rollout MEDAL evaluation. There are no training updates or reference answers in the search.

## States, actions and cached analysis

A node owns CPU token IDs of shape `[1, prompt_length + gen_length]`. Its depth is the number of output tokens committed along the path. An action is `(output_position, token_id)`; positions start at zero **after the prompt**. Tokens need not correspond one-to-one with Sudoku digits or cells. Prompt tokens and earlier output commitments stay fixed.

On Colab, the model already runs on CUDA. Forward passes, softmax, per-position entropy, top-token filtering and confidence factors run on the GPU. Only compact action statistics return to the CPU for global ordering, UCB, backups and display. Keeping tree states and traces on the CPU avoids retaining each branch in GPU memory. `model_device` in the saved inspection/prediction statistics reports the loaded model's device; `search_state_device` reports the tree storage separately.

`analyze_state()` transfers one state to the model's device and computes the full-vocabulary distribution at each remaining mask. The compact cache contains per-position entropy, retained actions, and a state fingerprint, all on CPU. Full logits and probabilities are released after analysis. Children are evaluated sequentially, using the same frozen model.

With natural logarithms, define:

```text
H_i = -sum_v p_i(v) log p_i(v)
E(state) = sum_{masked output i} H_i
confidence(i,v) = p_i(v) exp(-H_i) sigmoid(gamma * (p_i(top1) - p_i(top2)))
```

The margin uses the full vocabulary, including the mask token; mask tokens are excluded from commitments. Keep the best `K1` legal tokens per position, then the best `K2` actions globally. Equal confidence prefers smaller output position, then smaller token ID. Entropy is zero at committed positions in the saved vectors.

## One example across the four stages

This is a **controlled illustration**, not an HF prediction. Suppose the output has four masks and every masked position always has probabilities `p(A)=0.7`, `p(B)=0.2`, `p(C)=0.1`, `p(mask)=0`. Every position's entropy is about `0.801819` nats. Use `Lc=2`, `C=3`, `K1=K2=2`.

### 1. Selection — `select_node()`

The first cycle selects the unexpanded root (node 0):

```text
Root: [M0] [M1] [M2] [M3]    E = 3.207274
```

Later cycles traverse expandable children using:

```text
Q(child) = reward_sum / visits
UCB(child) = Q(child) + exploration_c * sqrt(log(parent.visits) / child.visits)
```

Unvisited children have priority; their infinite bonuses are represented by `unvisited: true` and `null` UCB fields in JSON. Ties use action confidence, position, token ID, then node ID. Closed nodes remain visible in sibling tables but are ineligible. Selection returns the entire node-ID path and each sibling comparison.

### 2. Expansion — `expand_node()`

The best global actions are `(0,A)` and `(1,A)`. They produce independent children, each with **one** committed token:

| Child | Action | State | Depth |
| --- | --- | --- | --- |
| 1 | `(0,A)` | `A [M1] [M2] [M3]` | 1 |
| 2 | `(1,A)` | `[M0] A [M2] [M3]` | 1 |

`K2=2` creates two children; it does not commit two tokens within each child. `Lc=2` is reached in a later cycle. The trace records confidence, probability, position, token representation and child ID for every expansion.

### 3. Evaluation — `evaluate_action()`

One forward observes each child's remaining-mask entropy. It makes no further token commitments:

```text
parent E = 3.207274
child E  = 2.405456
reward   = (parent E - child E) / parent E = 0.25
context_gain = parent E - H_at_committed_position - child E = 0
```

The reward here comes entirely from removing one uncertain position. `context_gain=0` reveals that other positions did not become more certain. With an actual model, their entropy can increase: context gain may be negative, and reward is negative if total remaining entropy exceeds the parent's total. No negative rewards are clipped. If parent entropy is at most epsilon, reward is zero.

The initial cycle costs **three model forwards**: root analysis and one evaluation for each child. An evaluation at full output depth still performs the contracted forward, with remaining-mask entropy zero.

### 4. Backpropagation — `backpropagate()`

After evaluating child 1, update child 1 and root with reward `0.25`. After child 2, update child 2 and root:

| Node | Visits | Reward sum | Mean reward |
| --- | --- | --- | --- |
| Root 0 | 2 | 0.50 | 0.25 |
| Child 1 | 1 | 0.25 | 0.25 |
| Child 2 | 1 | 0.25 | 0.25 |

Every update appears with before/after values in the trace. This changes search statistics, not model weights. Each evaluated child contributes its immediate entropy reward once to every ancestor.

## The next cycles and stopping

Cycle 2 selects child 1 by the stable tie rule. Its cached analysis supplies actions without another parent forward. Two evaluations produce `A A [M2] [M3]` (node 3) and `A [M1] A [M3]` (node 4). Both are depth 2, so they become closed candidates. Each edge reward is `1/3`; their root-relative gain is `1/2`.

Cycle 3 selects child 2, since child 1's frontier is exhausted. One child reproduces `A A [M2] [M3]`, which is deduplicated; the other produces `[M0] A A [M3]`. There are now three distinct depth-2 candidates after **seven forwards**. Equal final scores choose the earlier node ID, so node 3 is selected.

Search stops when:

1. A cycle has collected at least `C` distinct depth-`Lc` states. All retained siblings are evaluated unless the budget ends, so the count can exceed `C`.
2. The root-plus-child forward cap is reached, even in the middle of a sibling set.
3. No expandable frontier remains.

Final ranking always compares root-to-candidate normalized entropy reduction at the same depth; it does not use the last edge reward or UCB score. If the search stops with fewer than `C` candidates, it selects the best available state at the deepest evaluated depth and records `fallback_used`, `fallback_reason`, `selected_depth`, and `stop_reason`. A cap of one selects the analyzed root as a depth-zero fallback. A `result()` snapshot during unfinished inspection is provisional and does not mark a fallback.

Greedy completion of node 3 preserves its two `A` tokens. Its cached analysis supplies the next commitment at position 2; only the final position needs a new forward. Thus this four-token example uses **7 search + 1 completion forwards**. With 256 positions and selected depth 2, completion normally uses **253 forwards**, because the first of its 254 commitments uses the search cache.

## Direct use with the existing HF model

Run this inside the process that already holds your model. The Colab notebook instead launches the thin inspection/runner scripts so it never creates another model in the kernel.

```python
from dlm_project.decoding.mcts_decoding import MCTSConfig, MCTSSearch, complete_confidence

# model and tokenizer are the existing HF objects; prompt_ids is [1, P].
# If using the project wrapper: model = llada.model,
# tokenizer = llada.tokenizer, prompt_ids = llada.encode_prompt(prompt).
config = MCTSConfig(init_tokens=2, num_candidates=3,
                    top_k_tokens=2, top_k_actions=2, max_model_calls=32)
search = MCTSSearch(model, prompt_ids, mask_id=126336, gen_length=256,
                    config=config, tokenizer=tokenizer)

events = search.step()  # One full cycle; root + two children on first call.
for event in events:
    print(event["stage"], event)

result = search.run()   # Continue this SAME tree to termination.
print(result.stats)
completion = complete_confidence(
    model, result.selected.state, search.prompt_len, search.mask_id, config,
    initial_analysis=result.selected.analysis, tokenizer=tokenizer,
)
text = tokenizer.batch_decode(completion.token_ids[:, search.prompt_len:],
                              skip_special_tokens=True)[0]
```

For an end-to-end wrapper call:

```python
prediction = llada.predict_with_stats(
    prompt, gen_length=256, steps=256, block_length=256,
    temperature=0, remasking="low_confidence",
    decoding={"method": "mcts_entropy", "seed": 1,
              "mcts": {"init_tokens": 2, "num_candidates": 3}},
)
print(prediction["text"], prediction["stats"])
```

The configuration maps `Lc → init_tokens`, `C → num_candidates`, `K1 → top_k_tokens`, `K2 → top_k_actions`, `gamma → margin_gamma`. The seed belongs to `decoding`; temperature-zero search itself has no sampling. The runner uses `seed + puzzle_index` (modulo `2**63`) and records the actual per-puzzle seed. Model/device floating-point differences can still change close scores.

## Colab verification

After setup, run **Check the search on CUDA before loading LLaDA**. Its three small controlled-model tests compare CUDA and CPU results and verify actual GPU forward inputs, entropy rewards, fixed commitments and cached completion. This subprocess requires CUDA, so an unavailable GPU cannot be mistaken for a passed GPU check. These tests are skipped on local machines without CUDA.

Then run the notebook's one-puzzle configuration, the two inspection commands, full generation, and the saved-artifact checks. The notebook displays **actual HF token predictions**, all four cycle stages, raw generated text, Sudoku scores, and plots. It verifies the loaded HF model's device is CUDA. Its final verification cell checks that search forwards equal root plus evaluations, completion costs match cache reuse, every output position is committed exactly once across initialization/completion, and CUDA memory was measured.

CPU fake-model tests verify search logic, wrapper integration, artifact saving and notebook table rendering. They do not validate the 8B model on GPU or establish Sudoku quality. GPU validation must be run separately on Colab; notebook outputs are intentionally cleared in the repository.
