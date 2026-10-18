"""Public MEDAL adapter preserves commitments and charges every branch."""
import pytest
import torch
from dlm_project.decoding import mcts_confidence
from dlm_project.decoding.mcts_decoding import MCTSConfig, derive_seed,SearchIncompleteError
from tests.decoding.helpers import FakeModel,FakeTokenizer,MASK,PROMPT


def options(**kw):
    return {'init_tokens':2,'num_candidates':1,'top_k_tokens':2,'top_k_actions':2,**kw}


@pytest.mark.parametrize('cap,depth',[(1,0),(2,1),(6,2)])
def test_explicit_budget_completion_preserves_committed_tokens_and_costs(cap,depth):
    model=FakeModel();sink=[]
    r=mcts_confidence.decode(model,PROMPT,MASK,gen_length=6,steps=6,block_length=6,
        mcts=options(stop_mode='budget',max_model_calls=cap),tokenizer=FakeTokenizer(),on_event=sink.append)
    assert r.details['selected_depth']==depth
    assert r.details['search_model_calls']==cap
    assert r.details['completion_model_calls']==6-depth
    assert r.total_nfes==len(model.seen)==cap+6-depth
    assert r.trace==sink
    assert torch.equal(r.token_ids[:,:2],PROMPT)
    for commit in r.details['selected_initialization']['committed']:
        assert r.token_ids[0,2+commit['output_position']]==commit['token_id']
    assert (r.token_ids[:,2:]!=MASK).all()
    assert r.details['completion_seed']==derive_seed(1,'final_completion')


def test_uncapped_candidate_search_single_forward_cost_and_global_completion():
    model=FakeModel()
    r=mcts_confidence.decode(model,PROMPT,MASK,gen_length=4,steps=4,block_length=4,
                             mcts=options(num_candidates=3),tokenizer=FakeTokenizer())
    assert r.details['candidate_count']==r.details['evaluated_candidates']==3
    assert r.details['search_model_calls']==7
    assert r.details['completion_model_calls']==2
    assert r.total_nfes==len(model.seen)==9
    assert r.details['implementation_version']=='medal_reconstruction_v2'
    assert not r.details['fallback_used']
    assert r.trace[-1]['generation_nfes']==9
    assert r.details['candidate_evaluations'][0]['remaining_masks']==2
    assert r.details['candidate_evaluation_model_calls']==0
    assert all(e.get('phase')=='completion' for e in r.trace if e['stage']=='rollout_step')


def test_dlm_completion_accepts_greedy_temperature_and_preserves_search_state():
    model=FakeModel();sink=[]
    r=mcts_confidence.decode(model,PROMPT,MASK,gen_length=4,steps=4,block_length=4,
        temperature=0,seed=1,mcts=options(num_candidates=3,completion_policy='dlm'),
        tokenizer=FakeTokenizer(),on_event=sink.append)
    assert r.details['completion_policy']=='dlm'
    assert r.details['completion_model_calls']==2
    assert r.total_nfes==len(model.seen)==9
    commits=[e for e in sink if e['stage']=='commit' and e.get('phase')=='completion']
    assert len(commits)==2 and commits[-1]['generation_nfes']==9
    assert all(e['completion_model_calls'] in (1,2) for e in commits)
    assert torch.equal(r.token_ids[:,:2],PROMPT)
    for commit in r.details['selected_initialization']['committed']:
        assert r.token_ids[0,2+commit['output_position']]==commit['token_id']


def test_seed_reproducibility_and_phase_separation():
    kw=dict(gen_length=24,steps=24,block_length=24,
            mcts=options(stop_mode='budget',max_model_calls=1))
    a=mcts_confidence.decode(FakeModel(),PROMPT,MASK,seed=1,**kw)
    b=mcts_confidence.decode(FakeModel(),PROMPT,MASK,seed=1,**kw)
    c=mcts_confidence.decode(FakeModel(),PROMPT,MASK,seed=2,**kw)
    assert torch.equal(a.token_ids,b.token_ids) and a.trace==b.trace
    assert not torch.equal(a.token_ids,c.token_ids)
    assert len({derive_seed(1,p) for p in ['simulation','candidate_evaluation','final_completion']})==3


@pytest.mark.parametrize('changes',[{'mcts':options(init_tokens=4)},{'block_length':2},
                                   {'temperature':0},{'mcts':options(max_model_calls=0)},
                                   {'mcts':options(completion_policy='invalid')}])
def test_invalid_settings_fail_before_forward(changes):
    model=FakeModel()
    kw=dict(gen_length=4,steps=4,block_length=4,mcts=options())
    kw.update(changes)
    with pytest.raises(ValueError): mcts_confidence.decode(model,PROMPT,MASK,**kw)
    assert not model.seen


def test_default_settings_shared_across_all_mcts_paths():
    assert mcts_confidence.resolve_mcts_config(gen_length=256)==MCTSConfig()
    assert MCTSConfig().init_tokens==20 and MCTSConfig().max_model_calls is None


@pytest.mark.parametrize('family',['llada','illada','llada_bgpo'])
def test_registered_model_mask_ids_are_excluded_from_all_commitments(family):
    from dlm_project.models.profiles import get_profile
    mask=get_profile(family)['mask_id']
    probabilities=torch.zeros(max(mask+1,6))
    probabilities[1:4]=torch.tensor([.7,.2,.1])
    model=FakeModel(lambda _:probabilities)
    prompt=torch.tensor([[3,mask]])
    r=mcts_confidence.decode(model,prompt,mask,gen_length=4,steps=4,block_length=4,
                             mcts=options(top_k_tokens=1,top_k_actions=1))
    assert torch.equal(r.token_ids[:,:2],prompt)
    assert (r.token_ids[:,2:]!=mask).all()
    assert r.total_nfes==len(model.seen)==5
