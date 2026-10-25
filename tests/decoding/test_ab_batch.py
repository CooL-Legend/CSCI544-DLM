"""Real batched tensors, independent trees and serial/padded numerical parity."""

from types import SimpleNamespace

import pytest
import torch

from dlm_project.decoding.ab_medal import ABMEDALConfig, ABMEDALSearch
from dlm_project.decoding.ab_batch import execute_trees
from dlm_project.decoding.mcts_decoding import complete_llada
from dlm_project.decoding import search_audit


class BatchModel:
    device = torch.device('cpu')
    config = SimpleNamespace(vocab_size=6, eos_token_id=4)

    def __init__(self, failure=None, fail_at=3, ignore_attention=False):
        self.calls = []
        self.failure, self.fail_at, self.ignore_attention = failure, fail_at, ignore_attention

    def __call__(self, ids, attention_mask=None):
        assert not torch.is_grad_enabled()
        self.calls.append((ids.clone(), None if attention_mask is None else attention_mask.clone()))
        if len(self.calls) == self.fail_at and self.failure is not None:
            raise self.failure
        valid = torch.ones_like(ids) if attention_mask is None or self.ignore_attention else attention_mask
        bias = ((ids * valid).sum(1) % 3).float() * .02
        logits = torch.tensor([-6., 1.2, .4, .1, -3., -4.]).expand(ids.shape[0], ids.shape[1], 6).clone()
        logits[:, :, 1] += bias[:, None] + torch.arange(ids.shape[1])[None] * .001
        return SimpleNamespace(logits=logits)


def trees(model, budgets=(1, 2, 9, 20)):
    return [ABMEDALSearch(model, torch.tensor([[3] * (i + 1)]), 0, 5,
             ABMEDALConfig(init_tokens=2, max_model_calls=budget), seed=i + 7,
             search_excluded_token_ids=(4, 5)) for i, budget in enumerate(budgets)]


@pytest.mark.parametrize('size', [1, 2, 4])
def test_batched_search_completion_equals_serial_and_charges_sequences(size):
    serial = trees(BatchModel())
    outputs = []
    for tree in serial:
        result = tree.run()
        outputs.append(complete_llada(tree.model, result.selected.state, tree.prompt_len, 0, temperature=0.).token_ids)
    batch_model = BatchModel()
    batched = trees(batch_model)
    runs = [execute_trees(batched[i:i+size], pad_token_id=4) for i in range(0, len(batched), size)]
    assert all(run.status == 'completed' for run in runs)
    completions = [c for run in runs for c in run.completions]
    for left, right, expected, completion in zip(serial, batched, outputs, completions):
        assert torch.equal(expected, completion.token_ids)
        assert left.rng.bit_generator.state == right.rng.bit_generator.state
        assert len(left.nodes) == len(right.nodes)
        for a, b in zip(left.nodes, right.nodes):
            assert torch.equal(a.state, b.state)
            assert (a.visits, a.reward_sum, a.mean_reward) == (b.visits, b.reward_sum, b.mean_reward)
        audit = search_audit.from_events(right.trace, prompt_ids=right.root.state[0, :right.prompt_len].tolist(),
            mask_id=0, gen_length=5, requested_depth=2, requested_budget=right.config.max_model_calls,
            search_nfes=right.model_calls, adaptive=True)
        assert sum(search_audit.metrics(audit)['nfes_per_depth'].values()) == right.model_calls
    metrics = [run.metrics for run in runs]
    assert sum(m['total_model_invocations'] for m in metrics) == len(batch_model.calls)
    assert sum(m['total_nfes'] for m in metrics) == sum(ids.shape[0] for ids, _ in batch_model.calls)
    if size > 1:
        assert any(ids.shape[0] > 1 for ids, _ in batch_model.calls)
        assert any(mask is not None and (mask == 0).any() for _, mask in batch_model.calls)
        assert sum(m['total_model_invocations'] for m in metrics) < sum(m['total_nfes'] for m in metrics)


@pytest.mark.parametrize('error,status', [(RuntimeError('broken'), 'error'),
    (KeyboardInterrupt('stop'), 'interrupted'), (torch.cuda.OutOfMemoryError('memory'), 'oom')])
def test_failure_keeps_actual_forward_attempts_and_no_speculative_updates(error, status):
    model = BatchModel(failure=error)
    batch = trees(model, budgets=(2, 2))
    run = execute_trees(batch, pad_token_id=4)
    assert run.status == status and run.error
    assert run.metrics['search_nfes'] == 4 and run.metrics['completion_nfes'] == 2
    assert run.metrics['total_model_invocations'] == 3
    assert all(t.root.visits == 1 for t in batch)
    assert run.completions == [None, None]
    assert run.forwards[-1]['status'] == 'attempted'


def test_request_cannot_be_consumed_twice_or_dispatched_twice():
    model = BatchModel()
    tree = trees(model, budgets=(2,))[0]
    with torch.inference_mode():
        request = tree.next_evaluation()
        assert tree.next_evaluation() is request
        tree.record_external_forward(request, invocation_id=1, batch_size=1)
        with pytest.raises(ValueError):
            tree.record_external_forward(request, invocation_id=2, batch_size=1)
        logits = model(request.node.state).logits[0]
        tree.submit_evaluation(request, logits=logits)
        with pytest.raises(ValueError):
            tree.submit_evaluation(request, logits=logits)


def test_tensor_preparation_failure_does_not_invent_model_invocations(monkeypatch):
    from dlm_project.decoding import ab_batch
    model = BatchModel()
    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError('padding allocation failed')
    monkeypatch.setattr(ab_batch, 'pack_states', fail)
    result = execute_trees(trees(model), pad_token_id=4)
    assert result.status == 'oom'
    assert result.metrics['total_nfes'] == result.metrics['total_model_invocations'] == 0
    assert not model.calls and all(tree.model_calls == 0 for tree in result.searches)
