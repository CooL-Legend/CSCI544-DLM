"""Verify checkpoint prefix equivalence between continuous and independent search runs."""
import pytest
from dlm_project.decoding.mcts_decoding import MCTSConfig, MCTSSearch
from dlm_project.decoding.ab_medal import ABMEDALConfig, ABMEDALSearch
from dlm_project.decoding import search_audit
from tests.decoding.helpers import FakeModel, MASK, PROMPT


@pytest.mark.parametrize("search_cls,config_cls", [
    (MCTSSearch, MCTSConfig),
    (ABMEDALSearch, ABMEDALConfig),
])
@pytest.mark.parametrize("b_check,b_max", [(2, 6), (4, 8), (5, 10)])
def test_checkpoint_prefix_matches_independent_run(search_cls, config_cls, b_check, b_max):
    # Standalone run capped at b_check
    cfg_standalone = config_cls(init_tokens=4, top_k_tokens=2, top_k_actions=2, num_candidates=3,
                                stop_mode="budget", max_model_calls=b_check)
    s_standalone = search_cls(FakeModel(), PROMPT, MASK, 6, cfg_standalone, seed=42)
    r_standalone = s_standalone.run()

    # Continuous run capped at b_max with checkpoint at b_check
    cfg_continuous = config_cls(init_tokens=4, top_k_tokens=2, top_k_actions=2, num_candidates=3,
                                stop_mode="budget", max_model_calls=b_max)
    s_continuous = search_cls(FakeModel(), PROMPT, MASK, 6, cfg_continuous, seed=42,
                              checkpoint_budgets=(b_check, b_max))
    s_continuous.run()

    assert b_check in s_continuous.checkpoints, f"Checkpoint {b_check} not captured."
    chk = s_continuous.checkpoints[b_check]

    # Verify evaluated nodes count
    standalone_nodes = [n for n in s_standalone.nodes if n.evaluation_order is not None]
    chk_nodes = chk["nodes"]
    assert len(chk_nodes) == len(standalone_nodes)

    # Verify per-node visits and Q values
    for n_s, n_c in zip(standalone_nodes, chk_nodes):
        assert n_s.node_id == n_c.node_id
        assert n_s.depth == n_c.depth
        assert n_s.visits == n_c.visits
        assert abs(n_s.reward_sum - n_c.reward_sum) < 1e-10
        assert abs(n_s.mean_reward - n_c.mean_reward) < 1e-10

    # Verify selection outcomes
    assert chk["selected_node_id"] == r_standalone.selected.node_id
    assert chk["selected_depth"] == r_standalone.selected.depth
    assert chk["shortlist_node_ids"] == [row["node_id"] for row in r_standalone.stats["candidate_shortlist"]]

    # Verify search audit reconstruction equivalence
    audit_standalone = search_audit.from_events(s_standalone.trace, mask_id=MASK, gen_length=6,
                                                requested_depth=4, requested_budget=b_check,
                                                search_nfes=b_check,
                                                adaptive=(search_cls == ABMEDALSearch))
    metrics_standalone = search_audit.metrics(audit_standalone)

    chk_trace = chk["trace"] + [chk["selection"]]
    audit_chk = search_audit.from_events(chk_trace, mask_id=MASK, gen_length=6,
                                         requested_depth=4, requested_budget=b_check,
                                         search_nfes=b_check,
                                         adaptive=(search_cls == ABMEDALSearch))
    metrics_chk = search_audit.metrics(audit_chk)

    for key, val_expected in metrics_standalone.items():
        val_actual = metrics_chk.get(key)
        assert val_actual == val_expected, f"Metric {key} mismatch: {val_actual} != {val_expected}"
