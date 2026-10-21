"""Paper-rule oracle, deterministic edge cases, and batched EB decoding."""

import json
import math
from types import SimpleNamespace

import pytest
import torch

from dlm_project.decoding import confidence, eb_sampler
from tests.decoding.helpers import MASK, PROMPT


class StaticModel:
    device = torch.device("cpu")

    def __init__(self, probabilities=None, *, logits=None):
        self.logits = (torch.tensor(logits, dtype=torch.float32) if logits is not None
                       else torch.tensor(probabilities if probabilities is not None
                                         else [0., .7, .2, .1], dtype=torch.float32).log())
        self.seen, self.masks = [], []

    def __call__(self, ids, attention_mask=None):
        assert not torch.is_grad_enabled()
        self.seen.append(ids.clone())
        self.masks.append(None if attention_mask is None else attention_mask.clone())
        batch, length = ids.shape
        logits = self.logits
        if logits.ndim == 1:
            logits = logits.expand(batch, length, -1)
        else:
            if logits.ndim == 2:
                logits = logits.expand(batch, -1, -1)
            prefix = logits[:, :1].expand(-1, length - logits.shape[1], -1)
            logits = torch.cat((prefix, logits), dim=1)
        return SimpleNamespace(logits=logits.clone())


def oracle(entropies, errors, gamma):
    selected = []
    for position in sorted(range(len(errors)), key=lambda i: errors[i]):
        candidate = selected + [position]
        h = [entropies[i] for i in candidate]
        if sum(h) - max(h) > gamma:
            break
        selected = candidate
    return selected


@pytest.mark.parametrize("h,errors,gamma,expected", [
    ([.125, .25, .5, .75], [0, 1, 2, 3], .375, [0, 1, 2]),
    ([.5, .125, .75, .25], [.2, .1, .3, 0], .375, [3, 1, 0]),
    ([.5, .125, .75, .25], [0, 1, 2, 3], .625, [0, 1, 2]),
    ([.5, .125, .75], [0, 1, 2], 0, [0]),
    ([0., 0., .75, 0.], [0, 1, 2, 3], 0, [0, 1, 2, 3]),
    ([.5, .125, .75], [0, 1, 2], math.inf, [0, 1, 2]),
    ([9.], [0], 0, [0]),
    ([], [], 0, []),
    ([.5, .5, .5], [0, 0, 0], .5, [0, 1]),
])
def test_prefix_matches_algorithm_one(h, errors, gamma, expected):
    chosen = eb_sampler.select_entropy_bounded_prefix(torch.tensor(h, dtype=torch.float64),
                                                      torch.tensor(errors), gamma)
    assert chosen.tolist() == expected == oracle(h, errors, gamma)


def test_prefix_agrees_with_sequential_oracle_for_many_orders_and_bounds():
    generator = torch.Generator().manual_seed(77)
    for _ in range(100):
        # Binary fractions make exact threshold assertions independent of rounding.
        h = torch.randint(0, 16, (9,), generator=generator).double() / 8
        errors = torch.randint(0, 5, (9,), generator=generator)
        for gamma in [0, .125, .5, 1., 3., math.inf]:
            assert eb_sampler.select_entropy_bounded_prefix(h, errors, gamma).tolist() == oracle(
                h.tolist(), errors.tolist(), gamma)


@pytest.mark.parametrize("proxy,first_position", [("confidence", 0), ("entropy", 2), ("margin", 1)])
def test_error_proxies_use_unscaled_distributions(proxy, first_position):
    rows = [[0., .60, .39, .01, 0.], [0., .55, .15, .15, .15], [0., .5, .5, 0., 0.]]
    result = eb_sampler.decode(StaticModel(rows), PROMPT, MASK, gen_length=3,
                               gamma=0, error_proxy=proxy)
    assert result.trace[0]["actions"][0]["output_position"] == first_position
    assert result.generation_nfes == result.details["decoding_steps"] == 3


def test_temperature_changes_samples_but_not_unscaled_entropy_ranking_or_prefixes():
    kwargs = dict(gen_length=20, gamma=1.0, seed=17)
    greedy = eb_sampler.decode(StaticModel([0., .4, .35, .25]), PROMPT, MASK, **kwargs)
    sampled = eb_sampler.decode(StaticModel([0., .4, .35, .25]), PROMPT, MASK,
                                temperature=1.0, **kwargs)
    assert not torch.equal(greedy.token_ids, sampled.token_ids)
    for first, second in zip(greedy.trace, sampled.trace):
        assert first["tokens_unmasked"] == second["tokens_unmasked"]
        assert first["entropy_bounds_per_sample"] == second["entropy_bounds_per_sample"]
        assert [(a["output_position"], a["selection_score"], a["entropy"]) for a in first["actions"]] == [
            (a["output_position"], a["selection_score"], a["entropy"]) for a in second["actions"]]
    repeat = eb_sampler.decode(StaticModel([0., .4, .35, .25]), PROMPT, MASK,
                               temperature=1.0, **kwargs)
    assert sampled.trace == repeat.trace and torch.equal(sampled.token_ids, repeat.token_ids)


@pytest.mark.parametrize("gamma,expected_steps", [(0, 5), (math.inf, 1)])
def test_global_completion_ignores_compatibility_schedule_and_counts_calls(gamma, expected_steps):
    model = StaticModel()
    result = eb_sampler.decode(model, PROMPT, MASK, gen_length=5, steps=1,
                               block_length=2, gamma=gamma)
    assert result.total_nfes == len(model.seen) == expected_steps
    assert len(result.trace) == result.details["decoding_steps"] == expected_steps
    assert result.scoring_nfes == 0
    assert sum(result.details["tokens_unmasked_per_iteration"]) == 5
    assert result.details["average_tokens_unmasked_per_iteration"] == 5 / expected_steps
    assert torch.equal(result.token_ids[:, :2], PROMPT)
    assert not (result.token_ids[:, 2:] == MASK).any()
    # Infinity must be safe in predictions/traces written with allow_nan=False.
    json.dumps(result.details, allow_nan=False)
    json.dumps(result.trace, allow_nan=False)
    assert result.details["gamma"] == ("inf" if math.isinf(gamma) else 0)


def test_zero_gamma_can_commit_multiple_zero_entropy_positions():
    result = eb_sampler.decode(StaticModel([0., 1., 0., 0.]), PROMPT, MASK,
                               gen_length=4, gamma=0)
    assert result.total_nfes == 1
    assert result.details["tokens_unmasked_per_iteration"] == [4]
    assert all(action["entropy"] == 0 for action in result.trace[0]["actions"])


def test_batched_completion_attention_and_initial_commitments_are_independent():
    prompt = torch.tensor([[3, MASK], [2, MASK], [1, MASK]])
    initial = torch.cat((prompt, torch.tensor([[MASK] * 3, [MASK] * 3, [1, 2, 3]])), dim=1)
    original = initial.clone()
    attention = torch.tensor([[1, 0], [1, 1], [0, 1]])
    original_attention = attention.clone()
    model = StaticModel([[[0., 1., 0., 0.]] * 3, [[0., .5, .3, .2]] * 3,
                         [[0., .7, .2, .1]] * 3])
    result = eb_sampler.decode(model, prompt, MASK, gen_length=3, gamma=0,
                               initial_state=initial, attention_mask=attention)
    assert result.generation_nfes == len(model.seen) == 3
    assert result.details["state_evaluations"] == 9
    assert result.details["batch_size"] == 3
    assert result.details["decoding_steps_per_sample"] == [1, 3, 0]
    assert result.details["tokens_unmasked_per_iteration"] == [4, 1, 1]
    assert result.details["tokens_unmasked_per_iteration_per_sample"] == [[3, 0, 0], [1, 1, 1], [0, 0, 0]]
    assert result.details["average_tokens_unmasked_per_iteration"] == 2
    assert result.details["average_tokens_unmasked_per_iteration_per_sample"] == [3, 1, 0]
    for seen, mask in zip(model.seen, model.masks):
        assert seen.shape == initial.shape and torch.equal(seen[:, :2], prompt)
        assert torch.equal(seen[original != MASK], original[original != MASK])
        assert torch.equal(mask[:, :2], attention) and (mask[:, 2:] == 1).all()
    for before, after in zip(model.seen, [*model.seen[1:], result.token_ids]):
        fixed = before != MASK
        assert torch.equal(before[fixed], after[fixed])
    assert torch.equal(initial, original) and torch.equal(attention, original_attention)
    assert result.token_ids.data_ptr() != initial.data_ptr()
    assert not (result.token_ids[:, 2:] == MASK).any()
    assert len(result.trace[0]["actions"]) == 4
    assert all(action["batch_index"] == 1 for event in result.trace[1:] for action in event["actions"])


@pytest.mark.parametrize("remaining", [0, 1])
def test_completed_or_single_mask_initial_state(remaining):
    initial = torch.cat((PROMPT, torch.tensor([[1, 2, MASK if remaining else 3]])), dim=1)
    before = initial.clone()
    model = StaticModel()
    result = eb_sampler.decode(model, PROMPT, MASK, gen_length=3, initial_state=initial, gamma=0)
    assert result.generation_nfes == len(model.seen) == remaining
    assert torch.equal(initial, before)
    assert torch.equal(result.token_ids[:, :4], before[:, :4])
    assert result.details["tokens_unmasked_per_iteration"] == ([1] if remaining else [])
    assert result.details["average_tokens_unmasked_per_iteration_per_sample"] == [float(remaining)]


def test_mask_dominance_extreme_logits_and_zero_probabilities_are_stable():
    model = StaticModel(logits=[10000., 9999., 9998., -math.inf])
    result = eb_sampler.decode(model, PROMPT, MASK, gen_length=3, gamma=0)
    p = torch.tensor([9999., 9998., -math.inf]).softmax(-1)
    entropy = -torch.special.xlogy(p, p).sum().item()
    assert result.generation_nfes == 3
    assert (result.token_ids[:, 2:] == 1).all()
    assert result.trace[0]["actions"][0]["entropy"] == pytest.approx(entropy)


@pytest.mark.parametrize("overrides", [
    {"gamma": -1}, {"gamma": -math.inf}, {"gamma": math.nan}, {"gamma": True}, {"gamma": "inf"},
    {"error_proxy": "random"}, {"error_proxy": []}, {"gen_length": 0}, {"gen_length": True},
    {"steps": 0}, {"steps": False}, {"block_length": -1}, {"temperature": -1},
    {"temperature": math.nan}, {"temperature": math.inf}, {"seed": -1}, {"seed": 2**63},
    {"attention_mask": torch.ones(1, 3)}, {"attention_mask": torch.tensor([[1, 2]])},
    {"initial_state": torch.zeros(1, 3)},
])
def test_invalid_inputs_fail_before_forward(overrides):
    model = StaticModel()
    with pytest.raises(ValueError):
        eb_sampler.decode(model, PROMPT, MASK, **{"gen_length": 3, **overrides})
    assert model.seen == []


@pytest.mark.parametrize("prompt", [torch.zeros(0, 2, dtype=torch.long), torch.ones(2, 2),
                                    torch.ones(2, dtype=torch.long)])
def test_invalid_prompt_shapes_and_types(prompt):
    model = StaticModel()
    with pytest.raises(ValueError, match="prompt_ids"):
        eb_sampler.decode(model, prompt, MASK)
    assert not model.seen


@pytest.mark.parametrize("logits", [[0., math.nan, 0.], [0., math.inf, 0.],
                                   [-math.inf] * 3, [0., -math.inf, -math.inf]])
def test_nonfinite_or_empty_legal_distributions_raise(logits):
    with pytest.raises(ValueError, match="Non-finite"):
        eb_sampler.decode(StaticModel(logits=logits), PROMPT, MASK, gen_length=2)


def test_shared_helpers_still_reject_batches_for_existing_decoders():
    with pytest.raises(ValueError, match=r"\[1, P\]"):
        confidence.decode(StaticModel(), PROMPT.expand(2, -1), MASK,
                          gen_length=2, steps=2, block_length=2)
