"""Immediate-child evaluations respect forward and candidate budgets."""
import pytest
from dlm_project.decoding.mcts_decoding import MCTSConfig,MCTSSearch,SearchIncompleteError
from dlm_project.experiments.paths import experiment_name
from tests.decoding.helpers import FakeModel,MASK,PROMPT


def tree(**kw):
    return MCTSSearch(FakeModel(),PROMPT,MASK,4,MCTSConfig(**{
        'init_tokens':2,'top_k_tokens':2,'top_k_actions':2,'num_candidates':3,**kw}))


@pytest.mark.parametrize('cap,depth,children',[(1,0,0),(2,1,1),(3,1,2),(4,2,3),(5,2,4),(6,2,5)])
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
    assert r.stats['simulations_interrupted']==0
    assert all(node.analysis is not None for node in search.nodes)


@pytest.mark.parametrize('cap',[1,2])
def test_candidate_mode_fails_instead_of_silently_completing(cap):
    search=tree(num_candidates=1,top_k_actions=1,max_model_calls=cap)
    with pytest.raises(SearchIncompleteError) as error: search.run()
    assert error.value.stats['fallback_used']
    assert len(search.model.seen)==cap
    assert search.trace[-1]['stage']=='termination'
    assert search.root.visits==sum(e['stage']=='backpropagation' for e in search.trace)


def test_candidate_or_budget_stops_at_first_candidate_or_accepts_cap_fallback():
    completed=tree(num_candidates=1,top_k_actions=1,stop_mode='candidates_or_budget',
                   max_model_calls=100)
    result=completed.run()
    assert result.stats['stop_reason']=='candidate_quota'
    assert result.stats['candidate_count']==1 and not result.stats['fallback_used']
    assert completed.model_calls==3

    capped=tree(num_candidates=1,top_k_actions=1,stop_mode='candidates_or_budget',
                max_model_calls=2)
    result=capped.run()
    assert result.stats['stop_reason']=='model_call_budget'
    assert result.stats['candidate_count']==0 and result.stats['fallback_used']
    assert result.stats['selected_depth']==1 and capped.model_calls==2


def test_candidate_quota_at_exact_forward_cap_requires_no_extra_evaluation():
    strict=tree(num_candidates=1,top_k_actions=1,max_model_calls=3)
    result=strict.run()
    assert strict.root.visits==2
    assert strict.result().stats['candidate_count']==1
    assert result.stats['evaluated_candidates']==1
    assert result.stats['candidate_evaluation_model_calls']==0
    assert result.stats['stop_reason']=='candidate_quota' and not result.stats['fallback_used']
    assert len(strict.model.seen)==3


def test_exhausted_frontier_fails_strict_but_budget_variant_can_complete():
    with pytest.raises(SearchIncompleteError,match='frontier_exhausted'):
        tree(top_k_actions=1).run()
    r=tree(top_k_actions=1,stop_mode='budget',max_model_calls=100).run()
    assert r.stats['stop_reason']=='frontier_exhausted'
    assert r.stats['selected_depth']==2 and r.stats['fallback_used']


def test_versioned_names_and_uncapped_defaults():
    generation=dict(gen_length=256,steps=256,block_length=256,temperature=1,remasking='low_confidence')
    name=experiment_name(generation,{'method':'mcts_entropy'})
    assert 'budgetnone' in name and 'medal-v2' in name
    hybrid=experiment_name(generation,{'method':'mcts_entropy','mcts':{
        'stop_mode':'candidates_or_budget','max_model_calls':2}})
    assert '-stop-candidates_or_budget' in hybrid
    assert MCTSConfig().max_model_calls is None
