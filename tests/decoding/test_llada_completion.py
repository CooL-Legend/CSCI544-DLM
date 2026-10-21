"""Standard LLaDA completion can replace adjusted-score sampling without changing search."""

import pytest
import torch

from dlm_project.decoding import confidence, medal_demo, medal_no_search, mcts_confidence
from dlm_project.decoding.mcts_decoding import MCTSConfig, complete_llada, derive_seed
from tests.decoding.helpers import FakeModel, MASK, PROMPT


def config(cap=1):
    return MCTSConfig(init_tokens=2, num_candidates=3, top_k_tokens=2, top_k_actions=3,
                      stop_mode='budget', max_model_calls=cap)


@pytest.mark.parametrize('temperature', [0.0, 1.0])
@pytest.mark.parametrize('partial', [False, True])
def test_llada_completion_matches_existing_confidence_decoder_and_preserves_state(temperature, partial):
    state = torch.cat([PROMPT, torch.zeros((1, 8), dtype=torch.long)], dim=1)
    if partial:
        state[0, 3], state[0, 7] = 3, 2
    saved = state.clone()
    model, events = FakeModel(), []
    output = complete_llada(model, state, 2, MASK, config(), seed=7, temperature=temperature, on_event=events.append)
    baseline = confidence.decode(FakeModel(), PROMPT, MASK, gen_length=8, steps=8, block_length=8,
                                 temperature=temperature, seed=7, initial_state=state)
    remaining = 6 if partial else 8
    assert output.model_calls == len(model.seen) == remaining
    assert torch.equal(output.token_ids, baseline.token_ids)
    assert torch.equal(state, saved)
    assert torch.equal(output.token_ids[saved != MASK], saved[saved != MASK])
    assert torch.equal(output.token_ids[:, :2], PROMPT)
    assert not (output.token_ids[:, 2:] == MASK).any()
    assert [event['stage'] for event in events] == ['forward', 'commit'] * remaining
    assert [event['model_calls'] for event in events if event['stage'] == 'forward'] == list(range(1, remaining + 1))
    assert output.trace == events


def test_standard_completion_ranks_native_token_confidence_and_permits_eos_eot():
    # Position 0 has higher argmax confidence; position 1 has lower entropy.
    rows = [[0, .6, .15, .15, .1], [0, .55, .45, 0, 0],
            [0, .25, .25, .25, .25], [0, .25, .25, .25, .25]]
    state = torch.cat([PROMPT, torch.zeros((1, 4), dtype=torch.long)], dim=1)
    result = complete_llada(FakeModel(lambda _: rows), state, 2, MASK, config(), temperature=0)
    commits = [event for event in result.trace if event['stage'] == 'commit']
    assert commits[0]['actions'][0]['output_position'] == 0
    assert commits[0]['actions'][0]['token_id'] == 1
    model = FakeModel(lambda _: [[0, 1, 0, 0], [0, 0, 1, 0]])
    result = medal_no_search.decode(model, PROMPT, MASK, gen_length=2, steps=2, block_length=2,
                                    temperature=0, completion_method='llada',
                                    mcts=MCTSConfig(init_tokens=1, stop_mode='budget', max_model_calls=1))
    assert result.token_ids[0, 2:].tolist() == [1, 2]
    assert result.total_nfes == len(model.seen) == 2


@pytest.mark.parametrize('temperature', [0.0, 1.0])
def test_llada_root_fallback_matches_no_search_and_demo_with_shared_completion_seed(temperature):
    options = dict(gen_length=16, steps=16, block_length=16, seed=1,
                   temperature=temperature, mcts=config(), completion_method='llada')
    control_model, search_model = FakeModel(), FakeModel()
    control = medal_no_search.decode(control_model, PROMPT, MASK, **options)
    searched = mcts_confidence.decode(search_model, PROMPT, MASK, **options, search_excluded_token_ids=(1, 2))
    initial = medal_demo.run_initialization(FakeModel(), PROMPT, MASK, gen_length=16,
                                           mcts=config(), search_excluded_token_ids=(1, 2))
    demo = medal_demo.run_completion(initial, method='llada', seed=1, temperature=temperature)
    assert torch.equal(control.token_ids, searched.token_ids)
    assert torch.equal(demo.token_ids, control.token_ids)
    assert control.total_nfes == len(control_model.seen) == demo.model_calls == 16
    assert searched.total_nfes == len(search_model.seen) == 17
    assert control.details['completion_method'] == searched.details['completion_method'] == 'llada'
    assert control.details['completion_seed'] == searched.details['completion_seed'] == derive_seed(1, 'final_completion')


def test_switching_completion_does_not_change_search_or_initial_commitments():
    options = dict(gen_length=8, steps=8, block_length=8, mcts=config(12), search_excluded_token_ids=(1, 2))
    legacy = mcts_confidence.decode(FakeModel(), PROMPT, MASK, completion_method='medal', **options)
    model, events = FakeModel(), []
    llada = mcts_confidence.decode(model, PROMPT, MASK, completion_method='llada', on_event=events.append, **options)
    assert llada.details['selected_initialization'] == legacy.details['selected_initialization']
    assert [event for event in llada.trace if event.get('phase') != 'completion'] == [
        event for event in legacy.trace if event.get('phase') != 'completion']
    depth = llada.details['selected_depth']
    assert llada.total_nfes == len(model.seen) == llada.details['search_model_calls'] + 8 - depth
    assert llada.details['decoding_steps'] == llada.details['completion_model_calls'] == 8 - depth
    assert llada.trace == events and llada.trace[-1]['generation_nfes'] == llada.total_nfes
    for token in llada.details['selected_initialization']['committed']:
        assert llada.token_ids[0, 2 + token['output_position']] == token['token_id']


@pytest.mark.parametrize('decoder', [mcts_confidence.decode, medal_no_search.decode])
def test_unknown_completion_method_fails_before_forward(decoder):
    model = FakeModel()
    with pytest.raises(ValueError, match='completion_method'):
        decoder(model, PROMPT, MASK, gen_length=4, steps=4, block_length=4,
                mcts=config(), completion_method='unknown')
    assert model.seen == []
