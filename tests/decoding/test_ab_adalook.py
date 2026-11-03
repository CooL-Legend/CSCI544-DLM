"""AB real-tree isolation, posterior updates, lookahead budgets and replay."""

from collections import Counter
import math

import pytest
import torch

from dlm_project.decoding.ab_medal import ABMEDALConfig, ABMEDALSearch, GaussianPosterior
from dlm_project.decoding import search_audit
from tests.decoding.helpers import FakeModel, MASK, PROMPT
from tests.decoding.test_adalook import partial_confidence


def config(**overrides):
    settings = dict(init_tokens=2, top_k_tokens=1, top_k_actions=2, num_candidates=2,
                    max_model_calls=30, reward_mode='adalook_adaptive',
                    adalook_confidence_threshold=.6, adalook_variance_threshold=100.)
    settings.update(overrides)
    return ABMEDALConfig(**settings)


def run(cfg=None, checkpoints=()):
    tree = ABMEDALSearch(FakeModel(partial_confidence), PROMPT, MASK, 5,
                        cfg or config(), seed=7, checkpoint_budgets=checkpoints)
    tree.run()
    return tree


def audit(tree):
    return search_audit.from_events(tree.trace, prompt_ids=PROMPT[0].tolist(), mask_id=MASK,
        gen_length=5, requested_depth=tree.config.init_tokens,
        requested_budget=tree.config.max_model_calls, search_nfes=tree.model_calls,
        adaptive=True, shortlist_limit=2)


@pytest.mark.parametrize('mode', ['ig', 'adalook_one_step', 'adalook_adaptive'])
def test_counts_real_states_and_single_posterior_backup(mode):
    tree = run(config(reward_mode=mode))
    metrics = search_audit.metrics(audit(tree))
    assert tree.model_calls == len(tree.model.seen) <= 30
    assert sum(metrics['nfes_per_depth'].values()) + metrics['adalook_total_extra_nfes'] == tree.model_calls
    assert metrics['uncompleted_evaluation_nfes'] == 0
    assert tree.root.visits == tree.simulations_completed == len(tree.nodes) - 1
    assert set(tree.posteriors) == {n.node_id for n in tree.nodes}
    backups = [e for e in tree.trace if e['stage'] == 'backpropagation']
    assert Counter(e['child_id'] for e in backups) == Counter(n.node_id for n in tree.nodes[1:])
    expected = {}
    for event in backups:
        recipients = [(r['node_id'], r['model']) for r in event['posterior_updates']]
        assert len(recipients) == len(set(recipients))
        child = tree.nodes[event['child_id']]
        path, parent = [], child.parent
        while parent is not None:
            path.append(parent)
            parent = parent.parent
        path.reverse()
        oracle = {(child.node_id, 'node'), (child.parent.node_id, 'GEN')}
        oracle.update((n.node_id, 'node') for n in path[1:])
        oracle.update((n.node_id, 'CONT') for n in path[:-1])
        assert set(recipients) == oracle
        for key in oracle:
            expected.setdefault(key, GaussianPosterior()).update(event['reward'])
    for (node_id, name), posterior in expected.items():
        assert tree.posteriors[node_id][name] == posterior
    for node in tree.nodes:
        assert node.node_id >= 0
        assert torch.equal(node.state[:, :PROMPT.shape[1]], PROMPT)
        assert int((node.state[:, PROMPT.shape[1]:] != MASK).sum()) == node.depth
        if node.analysis is not None:
            assert len(tree.remaining_actions[node.node_id]) + len(node.children) == (
                len(node.analysis.actions) if node.depth < tree.config.init_tokens else 0)
    if mode == 'adalook_adaptive':
        assert metrics['temporary_candidate_nfes'] > 0 and metrics['adalook_extra_nfes'] > 0
        assert metrics['adalook_trigger_rate'] > 0
    else:
        assert metrics['adalook_total_extra_nfes'] == 0


@pytest.mark.parametrize('budget', range(1, 15))
def test_small_budgets_and_checkpoints_have_only_finalized_backups(budget):
    tree = run(config(max_model_calls=budget), checkpoints=tuple(range(1, budget + 1)))
    assert tree.model_calls <= budget
    assert search_audit.metrics(audit(tree))['uncompleted_evaluation_nfes'] == 0
    for b, checkpoint in tree.checkpoints.items():
        assert checkpoint['model_calls'] <= b
        assert all(n.node_id >= 0 and n.analysis is not None for n in checkpoint['nodes'])
        assert sum(e['stage'] == 'forward' for e in checkpoint['trace']) == checkpoint['model_calls']
        assert sum(e['stage'] == 'backpropagation' for e in checkpoint['trace']) == len(checkpoint['nodes']) - 1


def test_checkpoint_prefixes_equal_runs_with_the_same_earlier_boundaries():
    tree = run(config(), checkpoints=(5, 12, 30))
    for b, checkpoint in tree.checkpoints.items():
        standalone = run(config(max_model_calls=b), checkpoints=tuple(k for k in (5, 12, 30) if k <= b))
        assert checkpoint['model_calls'] == standalone.model_calls
        assert torch.equal(checkpoint['selected'].state, standalone.result().selected.state)
        assert [(n.visits, n.reward_sum) for n in checkpoint['nodes']] == [
            (n.visits, n.reward_sum) for n in standalone.nodes]


def test_depth_one_is_exact_one_step_ablation():
    one = run(config(reward_mode='adalook_one_step'))
    for settings in (config(adalook_max_depth=1), config(adalook_variance_threshold=0.)):
        adaptive = run(settings)
        assert adaptive.model_calls == one.model_calls
        assert adaptive.rng.bit_generator.state == one.rng.bit_generator.state
        assert [(n.visits, n.immediate_reward) for n in adaptive.nodes] == [
            (n.visits, n.immediate_reward) for n in one.nodes]
        assert adaptive.result().stats['adalook_total_extra_nfes'] == 0


def test_comparison_budget_fallback_and_reuse_of_real_sibling_analysis():
    short = run(config(max_model_calls=2))
    assert short.result().stats['adalook_total_extra_nfes'] == 0
    tree = run(config(init_tokens=1))
    groups = [e for e in tree.trace if e['stage'] == 'adalook_group']
    assert any(g['cached_comparisons'] > 0 for g in groups)
    assert len(tree.nodes) == 1 + len(tree.root.analysis.actions)


def test_no_new_decision_or_backup_while_comparison_is_pending():
    tree = ABMEDALSearch(FakeModel(partial_confidence), PROMPT, MASK, 5, config())
    with torch.inference_mode():
        for _ in range(2):  # root and first real child
            request = tree.next_evaluation()
            tree.record_external_forward(request, invocation_id=tree.model_calls + 1, batch_size=1)
            tree.submit_evaluation(request, logits=tree.model(request.node.state).logits[0])
        pending = tree.next_evaluation()
        assert pending.phase == 'adalook_comparison' and pending.node not in tree.nodes
        assert tree.root.visits == tree.simulations_completed == 0
        before = tree.rng.bit_generator.state.copy()
        assert tree.next_evaluation() is pending
        assert tree.rng.bit_generator.state == before
        assert not any(e['stage'] == 'backpropagation' for e in tree.trace)


def test_pruning_and_restart_scores_are_finite_and_update_only_real_children():
    tree = run(config(adalook_trigger_threshold=.9, adalook_max_depth=3))
    assert any(e.get('action') == 'all_retrigger' for e in tree.trace)
    assert all(math.isfinite(n.immediate_reward) for n in tree.nodes[1:])
    assert all(update['node_id'] >= 0 for e in tree.trace if e['stage'] == 'backpropagation'
               for update in e['posterior_updates'])


def test_cooperative_and_serial_search_share_the_same_state_machine():
    from dlm_project.decoding.ab_batch import execute_trees
    from tests.decoding.test_ab_batch import BatchModel

    class PartialBatchModel(BatchModel):
        def __call__(self, ids, **kw):
            output = super().__call__(ids, **kw)
            output.logits[:] = torch.tensor([0., .4, .35, .25, 0., 0.]).log()
            for row in range(len(ids)):
                positions = torch.nonzero(ids[row] == MASK).flatten()
                if len(positions):
                    output.logits[row, positions[0]] = torch.tensor([0., .7, .2, .1, 0., 0.]).log()
            return output

    prompts = [torch.tensor([[3, 2]]), torch.tensor([[3, 2, 1]])]
    serial = [ABMEDALSearch(PartialBatchModel(), p, MASK, 5, config(), seed=7) for p in prompts]
    for tree in serial:
        tree.run()
    model = PartialBatchModel()
    batch = [ABMEDALSearch(model, p, MASK, 5, config(), seed=7) for p in prompts]
    result = execute_trees(batch, pad_token_id=4)
    assert result.status == 'completed', result.error
    for a, b in zip(serial, batch):
        assert a.model_calls == b.model_calls
        assert a.rng.bit_generator.state == b.rng.bit_generator.state
        assert torch.equal(a.result().selected.state, b.result().selected.state)
        assert [(n.visits, n.reward_sum) for n in a.nodes] == [(n.visits, n.reward_sum) for n in b.nodes]
        assert a.result().stats['adalook_extra_nfes'] > 0
        assert a.result().stats['adalook_extra_nfes'] == b.result().stats['adalook_extra_nfes']
