"""Interrupted trajectories must be charged but never backed up."""
import pytest
from dlm_project.decoding.mcts_decoding import MCTSConfig,MCTSSearch,SearchIncompleteError
from dlm_project.experiments.paths import experiment_name
from tests.decoding.helpers import FakeModel,MASK,PROMPT


def tree(**kw):
    return MCTSSearch(FakeModel(),PROMPT,MASK,4,MCTSConfig(**{
        'init_tokens':2,'top_k_tokens':2,'top_k_actions':2,'num_candidates':3,**kw}))


@pytest.mark.parametrize('cap,depth,children',[(1,0,0),(2,0,0),(3,0,0),(4,1,1),(5,1,1),(8,1,2),(9,2,3)])
def test_hard_cap_and_deepest_fully_evaluated_fallback(cap,depth,children):
    search=tree(stop_mode='budget',max_model_calls=cap)
    r=search.run()
    assert r.stats['search_model_calls']==len(search.model.seen)==cap
    assert r.stats['selected_depth']==depth
    assert r.stats['evaluated_children']==search.root.visits==children
    assert r.stats['fallback_used'] and r.stats['stop_reason']=='model_call_budget'
    assert sum(e['stage']=='backpropagation' for e in r.trace)==children
    assert r.stats['search_model_calls']==sum(r.stats[f'{p}_model_calls'] for p in
                                            ('tree_analysis','simulation','candidate_evaluation'))
    if cap in (2,3,5,8):
        aborted=search.nodes[-1]
        assert aborted.rollout is None and aborted.analysis is None and aborted.visits==0
        assert any(e['stage']=='rollout_interrupted' for e in r.trace)


@pytest.mark.parametrize('cap',[1,2,7])
def test_candidate_mode_fails_instead_of_silently_completing(cap):
    search=tree(num_candidates=1,top_k_actions=1,max_model_calls=cap)
    with pytest.raises(SearchIncompleteError) as error: search.run()
    assert error.value.stats['fallback_used']
    assert len(search.model.seen)==cap
    assert search.trace[-1]['stage']=='termination'
    assert search.root.visits==sum(e['stage']=='backpropagation' for e in search.trace)


def test_cap_during_fresh_candidate_rollout_does_not_backup_partial_reward():
    search=tree(num_candidates=1,top_k_actions=1,max_model_calls=7,stop_mode='budget')
    # budget mode explores until cap; use quota mode to isolate terminal evaluation.
    strict=tree(num_candidates=1,top_k_actions=1,max_model_calls=7)
    with pytest.raises(SearchIncompleteError): strict.run()
    assert strict.root.visits==2
    assert strict.result().stats['candidate_count']==1
    assert strict.result().stats['evaluated_candidates']==0
    assert strict.result().stats['candidate_evaluation_model_calls']==1


def test_exhausted_frontier_fails_strict_but_budget_variant_can_complete():
    with pytest.raises(SearchIncompleteError,match='frontier_exhausted'):
        tree(top_k_actions=1).run()
    r=tree(top_k_actions=1,stop_mode='budget',max_model_calls=100).run()
    assert r.stats['stop_reason']=='frontier_exhausted'
    assert r.stats['selected_depth']==2 and r.stats['fallback_used']


def test_versioned_names_and_uncapped_defaults():
    generation=dict(gen_length=256,steps=256,block_length=256,temperature=1,remasking='low_confidence')
    name=experiment_name(generation,{'method':'mcts_entropy'})
    assert 'budgetnone' in name and 'medal-v1' in name
    assert MCTSConfig().max_model_calls is None
