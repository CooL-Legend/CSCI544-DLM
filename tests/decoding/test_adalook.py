"""Behavioral tests for reward, temporary rollout, compute budget and replay."""
from dataclasses import replace
import math

import pytest
import torch

from dlm_project.decoding import mcts_decoding as md, search_audit
from tests.decoding.helpers import FakeModel, MASK, PROMPT


def config(**kw):
    return md.MCTSConfig(**dict(init_tokens=2, top_k_tokens=1, top_k_actions=2,
        num_candidates=2, max_model_calls=15, stop_mode='budget',
        reward_mode='adalook_adaptive', adalook_confidence_threshold=0.6,
        adalook_variance_threshold=100., **kw))


def run(cfg, policy=None, checkpoints=()):
    model = FakeModel(policy)
    tree = md.MCTSSearch(model, PROMPT, MASK, 5, cfg, checkpoint_budgets=checkpoints)
    return tree, tree.run()


def partial_confidence(output):
    rows = [[0., .4, .35, .25] for _ in output]
    masked = torch.nonzero(output == MASK).flatten()
    if len(masked):
        rows[masked[0].item()] = [0., .7, .2, .1]
    return rows


@pytest.mark.parametrize('mode', ['ig', 'adalook_one_step', 'adalook_adaptive'])
def test_budget_and_auditable_tree_states(mode):
    tree, result = run(replace(config(), reward_mode=mode), partial_confidence)
    assert tree.model_calls == len(tree.model.seen) <= 15
    assert result.stats['adalook_extra_nfes'] == sum(e['stage'] == 'forward' and e['phase'] == 'adalook_rollout' for e in tree.trace)
    for node in tree.nodes:
        assert torch.equal(node.state[:, :2], PROMPT)
        assert int((node.state[:, 2:] != MASK).sum()) == node.depth
    audit = search_audit.from_events(tree.trace, prompt_ids=PROMPT[0].tolist(), mask_id=MASK,
        gen_length=5, requested_depth=2, requested_budget=15, search_nfes=tree.model_calls, shortlist_limit=2)
    metrics = search_audit.metrics(audit)
    assert metrics['uncompleted_evaluation_nfes'] == 0
    assert tree.root.visits == tree.simulations_completed
    if mode == 'adalook_adaptive':
        assert metrics['adalook_extra_nfes'] > 0
    else:
        assert metrics['adalook_extra_nfes'] == 0
    if mode != 'ig':
        assert metrics['ig_mean'] != metrics['reward_mean']


def test_one_step_uses_native_probability_mass_and_no_extra_forwards():
    cfg = replace(config(), reward_mode='adalook_one_step', max_model_calls=3)
    tree, _ = run(cfg)
    expected = .1 * math.log(.7) + math.log(4 * .7)
    for child in tree.root.children:
        assert child.immediate_reward == pytest.approx(expected)
        assert child.ig_reward == pytest.approx(.2)
    assert tree.model_calls == 3


def test_depth_one_exactly_matches_single_step_and_zero_gate_stops():
    one, r1 = run(replace(config(), reward_mode='adalook_one_step'))
    adaptive, r2 = run(replace(config(), adalook_max_depth=1))
    gate, r3 = run(replace(config(), adalook_variance_threshold=0.))
    for tree, result in [(adaptive, r2), (gate, r3)]:
        assert result.selected.node_id == r1.selected.node_id
        assert tree.model_calls == one.model_calls
        assert [n.mean_reward for n in tree.nodes] == [n.mean_reward for n in one.nodes]


@pytest.mark.parametrize('budget', range(1, 13))
def test_every_small_budget_and_checkpoint_is_a_valid_prefix(budget):
    cfg = replace(config(), max_model_calls=budget)
    tree, result = run(cfg, checkpoints=tuple(range(1, budget + 1)))
    assert tree.model_calls <= budget
    for b, checkpoint in tree.checkpoints.items():
        assert checkpoint['model_calls'] <= b
        assert all(n.analysis is not None for n in checkpoint['nodes'])
        assert sum(e['stage'] == 'forward' for e in checkpoint['trace']) <= b


def test_incomplete_round_does_not_privilege_first_candidate():
    tree, _ = run(replace(config(), max_model_calls=4), partial_confidence)
    # Root + two children leaves just one NFE, insufficient for two rollouts.
    first = next(e for e in tree.trace if e['stage'] == 'adalook_group')
    assert first['extra_nfes'] == 0 and first['stop_reason'] == 'round_budget'
    for child in tree.root.children:
        assert child.immediate_reward == pytest.approx(.1 * math.log(child.action.probability) + math.log(.7))


def test_extra_rounds_stop_at_checkpoint_without_future_evidence():
    tree, _ = run(replace(config(), adalook_max_depth=3), partial_confidence, checkpoints=(5, 9, 15))
    assert tree._counted.counts['adalook_rollout'] > 0
    for budget, checkpoint in tree.checkpoints.items():
        forwards = [e for e in checkpoint['trace'] if e['stage'] == 'forward']
        assert len(forwards) == checkpoint['model_calls'] <= budget
        assert all(e['search_model_calls'] <= budget for e in checkpoint['trace'])


def test_no_confident_tokens_is_finite_and_does_not_invent_commits():
    tree, _ = run(replace(config(), adalook_confidence_threshold=.95))
    assert all(math.isfinite(n.immediate_reward) for n in tree.nodes if n.immediate_reward is not None)
    assert tree._counted.counts['adalook_rollout'] == 0
    assert any(e.get('stop_reason') == 'no_progress' for e in tree.trace)


def test_all_uncertain_branches_restart_within_depth_cap():
    tree, _ = run(replace(config(), adalook_trigger_threshold=.9, adalook_max_depth=3))
    assert any(e.get('action') == 'all_retrigger' for e in tree.trace)
    assert max(g['rollout_depth'] for g in tree.trace if g['stage'] == 'adalook_group') <= 3
    assert tree.model_calls <= 15


def test_mixed_branches_stop_uncertain_candidates():
    def policy(output):
        if output[0] != MASK:
            return [0., .45, .3, .25]
        return [0., .85, .1, .05]
    tree, _ = run(replace(config(), adalook_trigger_threshold=.6), policy)
    assert any(e.get('action') == 'mixed_prune' for e in tree.trace)


@pytest.mark.parametrize('bad', [dict(adalook_max_depth=0), dict(adalook_alpha=float('nan')),
    dict(adalook_variance_threshold=-1), dict(adalook_min_masks=True), dict(adalook_confidence_threshold=1.1)])
def test_invalid_options(bad):
    with pytest.raises(ValueError):
        replace(config(), **bad)


def test_batched_and_serial_use_identical_adaptive_decisions():
    from dlm_project.decoding.medal_batch import CooperativeMEDALSearch
    from dlm_project.decoding.ab_batch import execute_trees
    from tests.decoding.test_ab_batch import BatchModel
    class PartialBatchModel(BatchModel):
        def __call__(self, ids, **kw):
            output = super().__call__(ids, **kw)
            output.logits[:] = torch.tensor([0., .4, .35, .25, 0., 0.]).log()
            for row in range(len(ids)):
                positions = torch.nonzero(ids[row] == 0).flatten()
                if len(positions):
                    output.logits[row, positions[0]] = torch.tensor([0., .7, .2, .1, 0., 0.]).log()
            return output
    cfg = config()
    prompts = [torch.tensor([[3, 2]]), torch.tensor([[3, 2, 1]])]
    expected = [md.MCTSSearch(PartialBatchModel(), p, 0, 5, cfg).run() for p in prompts]
    model = PartialBatchModel()
    trees = [CooperativeMEDALSearch(model, p, 0, 5, cfg) for p in prompts]
    group = execute_trees(trees, pad_token_id=4)
    assert group.status == 'completed', group.error
    for actual, serial in zip(group.results, expected):
        assert actual.stats['adalook_extra_nfes'] > 0
        assert torch.equal(actual.selected.state, serial.selected.state)
        assert actual.stats['search_model_calls'] == serial.stats['search_model_calls']
        assert actual.stats['adalook_extra_nfes'] == serial.stats['adalook_extra_nfes']
        assert [e for e in actual.trace if e['stage'] == 'adalook_group'] == [e for e in serial.trace if e['stage'] == 'adalook_group']
