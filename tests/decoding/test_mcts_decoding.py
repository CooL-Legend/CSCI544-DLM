"""Immediate-child rewards, filtering, tree statistics, and final sampling."""
import json
import pytest
import torch
from dlm_project.decoding import mcts_decoding as md
from tests.decoding.helpers import FakeModel, FakeTokenizer, MASK, PROMPT


def settings(**kwargs):
    return md.MCTSConfig(**{'init_tokens':2,'num_candidates':3,'top_k_tokens':2,'top_k_actions':2,**kwargs})


def search(model=None, **kwargs):
    return md.MCTSSearch(model or FakeModel(), PROMPT, MASK, 4, settings(**kwargs), tokenizer=FakeTokenizer())


def test_epsilon_entropy_and_confidence_filtering():
    rows = [[.6,.2,.1,.1],[0,.55,.225,.225],[0,.6,.39,.01],[0,.25,.25,.5]]
    model=FakeModel(lambda _: rows)
    state=torch.cat([PROMPT, torch.zeros((1,4),dtype=torch.long)],dim=1)
    config=settings(epsilon=.001,top_k_tokens=2,top_k_actions=5)
    analysis=md.analyze_state(model,state,2,MASK,config)
    p=torch.tensor(rows)
    h=-(p*torch.log(p+.001)).sum(-1)
    gaps=p.topk(2,-1).values
    factors=torch.exp(-h)*torch.sigmoid(10*(gaps[:,0]-gaps[:,1]))
    expected=sorted([(float(p[i,v]*factors[i]),i,v) for i in range(4)
                     for v in sorted(range(1,4),key=lambda v:(-float(p[i,v]),v))[:2]],
                    key=lambda a:(-a[0],a[1],a[2]))[:5]
    assert analysis.entropies.tolist()==pytest.approx(h.tolist())
    assert [(a.output_position,a.token_id) for a in analysis.actions]==[(i,v) for _,i,v in expected]
    assert all(a.token_id!=MASK for a in analysis.actions)
    assert len(model.seen)==1


def test_uct_and_backups_are_per_evaluated_child():
    tree=search()
    tree.step()
    assert tree.root.visits==2
    assert len(tree.model.seen)==3  # Root + one forward per immediate child.
    assert [n.visits for n in tree.root.children]==[1,1]
    tree.root.children[0].visits=0
    chosen,path=md.select_node(tree.root,.01)
    assert chosen is tree.root.children[0]
    assert path[0]['siblings'][0]['ucb'] is None
    json.dumps(path,allow_nan=False)
    leaf=md.Node(99,PROMPT,tree.root)
    before=tree.root.reward_sum
    updates=md.backpropagate(leaf,-.5)
    assert [u['node_id'] for u in updates]==[99,0]
    assert leaf.visits==1 and tree.root.reward_sum==pytest.approx(before-.5)


def test_simulations_preserve_partial_tree_states_and_prompt_masks(monkeypatch):
    monkeypatch.setattr(md, 'rollout', lambda *a, **kw: pytest.fail('Search performed a rollout'))
    tree=search()
    tree.step()
    root=tree.root.state.clone()
    for child in tree.root.children:
        assert child.state.data_ptr()!=tree.root.state.data_ptr()
        assert (child.state[:,2:]!=MASK).sum()==1
        assert torch.equal(child.state[:,:2],PROMPT)
        assert child.analysis.state_key==tuple(child.state[0].tolist())
        assert child.immediate_reward==pytest.approx(.25)
        assert child.mean_reward==child.immediate_reward
    assert torch.equal(tree.root.state,root)
    assert all(torch.equal(state[:,:2],PROMPT) for state in tree.model.seen)
    assert [(state[:,2:]!=MASK).sum().item() for state in tree.model.seen]==[0,1,1]
    assert not any(event['stage']=='rollout_step' for event in tree.trace)


def test_reward_depends_only_on_immediate_child_not_later_trajectory():
    def policy(output):
        if (output!=MASK).sum() < 2:
            return [0,.9,.05,.05]
        return [0,.99,.005,.005] if output[0]==1 else [0,1/3,1/3,1/3]
    model=FakeModel(policy)
    root=md.Node(0,torch.cat([PROMPT,torch.zeros((1,4),dtype=torch.long)],1))
    root.analysis=md.analyze_state(model,root.state,2,MASK,settings())
    left=md.expand_node(root,md.Action(0,1,.5),1,2,MASK,2)
    right=md.expand_node(root,md.Action(0,2,.4),2,2,MASK,2)
    a=md.evaluate_action(model,root,left,2,MASK,settings())
    b=md.evaluate_action(model,root,right,2,MASK,settings())
    assert a['immediate_gain']==pytest.approx(b['immediate_gain'])
    assert a['reward']==pytest.approx(b['reward'])
    assert a['reward']==a['immediate_gain']==pytest.approx(.25)
    assert a['evaluation_model_calls']==b['evaluation_model_calls']==1
    assert len(model.seen)==3
    assert md.information_gain(0,20,1e-8)==0


@pytest.mark.parametrize('probabilities', [[0,.99,.005,.005], [0,1.,0.,0.]])
def test_negative_and_zero_parent_entropy_rewards(probabilities):
    model=FakeModel(lambda output: probabilities if (output==MASK).all() else [0,1/3,1/3,1/3])
    tree=search(model)
    events=tree.step()
    evaluations=[event for event in events if event['stage']=='evaluation']
    assert len(model.seen)==3
    for event in evaluations:
        expected=md.information_gain(event['entropy_before'],event['entropy_after'],settings().epsilon)
        assert event['reward']==pytest.approx(expected)
        if probabilities[1]<1:
            assert event['reward']<0
        else:
            assert event['reward']==0


def test_candidates_are_distinct_closed_and_ranked_without_extra_forwards():
    tree=search()
    result=tree.run()
    assert len(result.candidates)==3
    assert len({tuple(n.state[0].tolist()) for n in result.candidates})==3
    assert all(n.closed and not n.children for n in result.candidates)
    assert result.stats['evaluated_candidates']==3
    assert result.stats['candidate_evaluation_model_calls']==0
    assert result.stats['selected_score']==max(n.mean_reward for n in result.candidates)
    assert result.selected is result.candidates[0]  # Static entropies tie; collection order wins.
    assert all(n.mean_reward==n.immediate_reward for n in result.candidates)
    assert all(e['model_calls']==0 for e in result.stats['candidate_evaluations'])
    assert (result.selected.state[:,2:]==MASK).sum()==2
    assert sum(e.get('duplicate',False) for e in result.trace)==1
    before=len(tree.model.seen)
    tree.result();tree.result()
    assert len(tree.model.seen)==before


def test_candidate_ranking_uses_cached_immediate_rewards(monkeypatch):
    monkeypatch.setattr(md, 'rollout', lambda *a, **kw: pytest.fail('Search performed a rollout'))
    def policy(output):
        if (output!=MASK).sum()<2:
            return [0,.7,.2,.1]
        return [0,.99,.005,.005] if output[2]!=MASK else [0,.7,.2,.1]
    tree=search(FakeModel(policy))
    result=tree.run()
    assert len(tree.model.seen)==1+result.stats['mcts_simulations']
    assert result.selected.state[0,4]!=MASK
    expected=md.information_gain(result.selected.parent.analysis.total_entropy,
                                 result.selected.analysis.total_entropy,settings().epsilon)
    assert result.stats['selected_score']==pytest.approx(expected)
    assert result.stats['selected_score']>result.candidates[0].mean_reward


def test_native_sampling_uses_tokens_outside_tree_top_k():
    state=torch.cat([PROMPT,torch.zeros((1,24),dtype=torch.long)],1)
    r=md.rollout(FakeModel(lambda _:[0,.4,.3,.3]),state,2,MASK,
                 settings(top_k_tokens=1,top_k_actions=1),seed=1)
    assert set(r.token_ids[0,2:].tolist())=={1,2,3}
    assert r.model_calls==24


def test_literal_softmax_covers_all_legal_actions_not_log_scores(monkeypatch):
    weights=[]
    def draw(p,*a,**kw):
        weights.append(p.clone())
        return p.argmax().reshape(1)
    monkeypatch.setattr(torch,'multinomial',draw)
    state=torch.cat([PROMPT,torch.zeros((1,2),dtype=torch.long)],1)
    rows=torch.tensor([[0,.9,.05,.05],[0,.4,.3,.3]])
    r=md.complete_medal(FakeModel(lambda _:rows.tolist()),state,2,MASK,settings(),seed=2)
    h=-(rows*torch.log(rows+1e-8)).sum(-1)
    gap=rows.topk(2,-1).values
    s=rows* (torch.exp(-h)*torch.sigmoid(10*(gap[:,0]-gap[:,1])))[:,None]
    s[:,0]=-torch.inf
    expected=torch.softmax(s.flatten(),0)
    assert torch.allclose(weights[0],expected)
    assert weights[0][1]/weights[0][2]<3  # exp(s), not probabilities proportional to s.
    assert weights[0][4]==0 and weights[0][0]==0
    assert r.model_calls==2 and (r.token_ids[:,2:]!=MASK).all()


def test_step_run_equivalence_and_json_events():
    streamed=[]
    stepped=search();stepped.on_event=streamed.append
    while not stepped.done: stepped.step()
    direct=search().run()
    result=stepped.result()
    assert result.trace==streamed==direct.trace
    assert torch.equal(result.selected.state,direct.selected.state)
    assert stepped.model_calls==len(stepped.model.seen)==7
    assert stepped.root.visits==result.stats['evaluated_children']==6
    assert stepped.model_calls==sum(result.stats[f'{p}_model_calls'] for p in
                                  ('tree_analysis','simulation','candidate_evaluation'))
    assert stepped.step()==[]
    for e in result.trace: json.dumps(e,allow_nan=False)


@pytest.mark.parametrize('kwargs',[{'init_tokens':0},{'top_k_tokens':True},{'max_model_calls':0},
                                  {'max_model_calls':False},{'stop_mode':'budget'},
                                  {'epsilon':0},{'exploration_c':-1},{'margin_gamma':float('nan')}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError): md.MCTSConfig(**kwargs)


def test_full_length_initialization_nonfinite_and_illegal_commitments():
    with pytest.raises(ValueError,match='init_tokens <'):
        search(init_tokens=4)
    with pytest.raises(ValueError,match='Non-finite'):
        search(FakeModel(lambda _:[float('nan')]*4)).step()
    state=torch.tensor([[3,MASK,1,2,3,1]])
    assert md.analyze_state(FakeModel(),state,2,MASK,settings()).total_entropy==0
    assert md.complete_medal(FakeModel(),state,2,MASK,settings()).model_calls==0
    for action in [md.Action(-1,1,.5),md.Action(4,1,.5),md.Action(0,MASK,.5),md.Action(0,2,.5)]:
        with pytest.raises(ValueError): md.expand_node(md.Node(0,state),action,1,2,MASK,2)


def test_uct_exploration_and_stable_position_token_ties():
    root=md.Node(0,PROMPT,visits=10,expanded=True)
    first=md.Node(1,PROMPT,root,md.Action(2,2,.4),visits=5,reward_sum=.5)
    second=md.Node(2,PROMPT,root,md.Action(0,1,.3),visits=1,reward_sum=.05)
    root.children=[first,second]
    assert md.select_node(root,.2)[0] is second
    second.visits=5;second.reward_sum=.5;second.action=md.Action(0,1,.4)
    assert md.select_node(root,.2)[0] is second
    first.action=md.Action(0,2,.4)
    assert md.select_node(root,.2)[0] is second
    second.action=md.Action(0,2,.4)
    assert md.select_node(root,.2)[0] is first
