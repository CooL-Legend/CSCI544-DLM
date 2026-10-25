"""Cooperative MEDAL/AB-MEDAL questions, with actual batched forwards.

One outstanding root/child per tree. Search decisions are never speculative.
This executor does not select questions, construct prompts, or inspect answers.
"""

from dataclasses import dataclass, field
from time import perf_counter

import torch

from dlm_project.decoding.ab_medal import ABMEDALConfig, ABMEDALSearch
from dlm_project.decoding.common import commit_from_logits
from dlm_project.decoding.mcts_decoding import CompletionResult, MCTSConfig, model_device
from dlm_project.decoding.medal_batch import CooperativeMEDALSearch


def sync(model):
    device = model_device(model)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def pack_states(states, pad_token_id, device):
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError('A valid non-mask padding token is required.')
    lengths = [s.shape[1] for s in states]
    ids = torch.full((len(states), max(lengths)), pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    for index, (state, length) in enumerate(zip(states, lengths)):
        ids[index, :length] = state[0].to(device)
        attention[index, :length] = 1
    return ids, attention, lengths


def forward_states(model, states, pad_token_id):
    """Right-pad after real output positions, preserving every original position."""
    ids, attention, lengths = pack_states(states, pad_token_id, model_device(model))
    prediction = model(ids) if len(states) == 1 else model(ids, attention_mask=attention)
    return prediction, lengths


@dataclass
class BatchRun:
    searches: list
    results: list = field(default_factory=list)
    completions: list = field(default_factory=list)
    forwards: list = field(default_factory=list)
    search_seconds: float = 0.0
    completion_seconds: float = 0.0
    search_latencies: list = field(default_factory=list)
    completion_latencies: list = field(default_factory=list)
    status: str = 'running'
    error: str | None = None
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None

    @property
    def metrics(self):
        search_calls = sum(e['phase'] == 'search' for e in self.forwards)
        completion_calls = sum(e['phase'] == 'completion' for e in self.forwards)
        total = self.search_seconds + self.completion_seconds
        return dict(batch_size=len(self.searches), status=self.status, error=self.error,
                    search_model_invocations=search_calls, completion_model_invocations=completion_calls,
                    total_model_invocations=len(self.forwards),
                    search_nfes=sum(e['batch_size'] for e in self.forwards if e['phase'] == 'search'),
                    completion_nfes=sum(e['batch_size'] for e in self.forwards if e['phase'] == 'completion'),
                    total_nfes=sum(e['batch_size'] for e in self.forwards),
                    search_seconds=self.search_seconds, completion_seconds=self.completion_seconds,
                    decoding_seconds=total, questions_per_second=len(self.searches) / total if total and self.status == 'completed' else None,
                    mean_batch_occupancy=sum(e['batch_size'] for e in self.forwards) / len(self.forwards) if self.forwards else 0,
                    peak_allocated_bytes=self.peak_allocated_bytes, peak_reserved_bytes=self.peak_reserved_bytes,
                    timing_scope='group wall time; per-question latencies overlap')


@torch.inference_mode()
def run_group(model, prompts, mask_id, config, *, seeds, pad_token_id,
              search_excluded_token_ids=(), tokenizer=None):
    """Search a consecutive group, then complete its selected states together.

    Returns explicit error/interruption status with partial trees and forward
    attempts. Callers persist this result before deciding whether to propagate.
    """
    if not prompts or len(prompts) != len(seeds):
        raise ValueError('Supply one seed for each question.')
    if not isinstance(config, MCTSConfig) or config.stop_mode != 'budget':
        raise ValueError('Batched search requires a fixed-budget MCTSConfig.')
    if pad_token_id == mask_id:
        raise ValueError('Padding must not use the output mask token.')
    search_class = ABMEDALSearch if isinstance(config, ABMEDALConfig) else CooperativeMEDALSearch
    trees = [search_class(model, ids, mask_id, 256, config, seed=seed,
                          search_excluded_token_ids=search_excluded_token_ids, tokenizer=tokenizer)
             for ids, seed in zip(prompts, seeds)]
    return execute_trees(trees, pad_token_id=pad_token_id)


@torch.inference_mode()
def execute_trees(trees, *, pad_token_id):
    """Low-level executor also supports small synthetic generation lengths."""
    if not trees or any(t.model is not trees[0].model or t.config.stop_mode != 'budget' for t in trees):
        raise ValueError('Independent budget trees must share one loaded model.')
    if any(t.mask_id == pad_token_id for t in trees):
        raise ValueError('Padding token cannot be the mask token.')
    model, device = trees[0].model, model_device(trees[0].model)
    run = BatchRun(trees, results=[None] * len(trees), completions=[None] * len(trees),
                   search_latencies=[None] * len(trees), completion_latencies=[None] * len(trees))
    if device.type == 'cuda':
        sync(model)
        torch.cuda.reset_peak_memory_stats(device)
    phase = 'search'
    sync(model)
    started = perf_counter()
    states, completion_counts, traces = [], [0] * len(trees), [[] for _ in trees]

    def forward(active, current_states, requests=None):
        # Tensor preparation can itself fail (for example, OOM). Charge NFEs
        # only once a model invocation is actually about to be attempted.
        ids, attention, lengths = pack_states(current_states, pad_token_id, device)
        item = dict(invocation_id=len(run.forwards) + 1, phase=phase,
                    question_indices=active, batch_size=len(active), status='attempted')
        run.forwards.append(item)
        if requests is not None:
            for index, request in zip(active, requests):
                trees[index].record_external_forward(request, invocation_id=item['invocation_id'], batch_size=len(active))
        else:
            for index in active:
                completion_counts[index] += 1
                traces[index].append(dict(stage='forward', phase='completion',
                    model_calls=completion_counts[index], invocation_id=item['invocation_id'], batch_size=len(active)))
        output = model(ids) if len(active) == 1 else model(ids, attention_mask=attention)
        item.update(status='returned', sequence_lengths=lengths)
        return output

    try:
        while True:
            active, requests = [], []
            for index, tree in enumerate(trees):
                request = tree.next_evaluation()
                if request is not None:
                    active.append(index)
                    requests.append(request)
                elif run.search_latencies[index] is None:
                    sync(model)
                    run.search_latencies[index] = perf_counter() - started
            if not active:
                break
            prediction = forward(active, [r.node.state for r in requests], requests)
            for row, (index, request) in enumerate(zip(active, requests)):
                trees[index].submit_evaluation(request, logits=prediction.logits[row, :request.node.state.shape[1]])
                if trees[index].done and run.search_latencies[index] is None:
                    sync(model)
                    run.search_latencies[index] = perf_counter() - started
            del prediction
        run.results = [t.result() for t in trees]
        states = [r.selected.state.clone() for r in run.results]
        sync(model)
        run.search_seconds = perf_counter() - started
        for index, tree in enumerate(trees):
            tree.initialization_seconds = run.search_latencies[index]
        phase = 'completion'
        sync(model)
        started = perf_counter()
        while True:
            active = [i for i, (state, tree) in enumerate(zip(states, trees))
                      if (state[0, tree.prompt_len:] == tree.mask_id).any()]
            if not active:
                break
            prediction = forward(active, [states[i] for i in active])
            for row, index in enumerate(active):
                tree, state = trees[index], states[index]
                positions = torch.nonzero(state[0, tree.prompt_len:] == tree.mask_id).flatten() + tree.prompt_len
                actions, signals = commit_from_logits(state, prediction.logits[row, :state.shape[1]],
                    tree.prompt_len, tree.mask_id, positions, 1, selection_rule='token_probability', temperature=0.)
                traces[index].append(dict(stage='commit', block=0, step=completion_counts[index]-1,
                    model_calls=completion_counts[index], actions=actions, **signals,
                    remaining_masks=int((state[0, tree.prompt_len:] == tree.mask_id).sum())))
                if not (state[0, tree.prompt_len:] == tree.mask_id).any():
                    sync(model)
                    run.completion_latencies[index] = perf_counter() - started
                    run.completions[index] = CompletionResult(state.clone(), completion_counts[index], traces[index])
            del prediction
        for tree, result, completion in zip(trees, run.results, run.completions):
            expected = tree.root.state.shape[1] - tree.prompt_len - result.selected.depth
            if completion.model_calls != expected:
                raise RuntimeError('Completion sequence NFEs do not match remaining masks.')
            fixed = result.selected.state != tree.mask_id
            fixed[:, :tree.prompt_len] = True
            if not torch.equal(completion.token_ids[fixed], result.selected.state[fixed]):
                raise RuntimeError('Completion changed a fixed prompt or search token.')
        run.status = 'completed'
    except (Exception, KeyboardInterrupt) as error:
        run.status = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'oom' if isinstance(error, torch.cuda.OutOfMemoryError) else 'error'
        run.error = f'{type(error).__name__}: {error}'
    finally:
        sync(model)
        elapsed = perf_counter() - started
        if phase == 'search':
            run.search_seconds = elapsed
        else:
            run.completion_seconds = elapsed
        for index, tree in enumerate(trees):
            if tree.root.analysis is not None and run.results[index] is None:
                run.results[index] = tree.result()
        if device.type == 'cuda':
            run.peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
            run.peak_reserved_bytes = torch.cuda.max_memory_reserved(device)
    return run
