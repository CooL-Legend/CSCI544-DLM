"""Controlled tests of position ranking, sampling schedules, and answer voting."""

import pytest
import torch

from dlm_project.decoding import best_of_5, confidence, random_mdm, top_k_margin
from dlm_project.decoding.decoder_result import DecodeResult
from dlm_project.tasks.gsm8k.evaluation import extract_answer, score_answer
from tests.decoding.helpers import FakeModel, FakeTokenizer, MASK, PROMPT


@pytest.mark.parametrize("decoder", [random_mdm, confidence, top_k_margin])
def test_seeded_sampling_schedule_preserves_prompt_and_future_blocks(decoder):
    kwargs = dict(gen_length=6, steps=4, block_length=3, temperature=1.0, seed=17)
    model = FakeModel()
    result = decoder.decode(model, PROMPT, MASK, **kwargs)
    again = decoder.decode(FakeModel(), PROMPT, MASK, **kwargs)
    assert torch.equal(result.token_ids, again.token_ids)
    assert result.trace == again.trace
    assert torch.equal(result.token_ids[:, :2], PROMPT)
    assert all(torch.equal(state[:, :2], PROMPT) for state in model.seen)
    assert result.total_nfes == result.generation_nfes == len(model.seen) == 4
    assert [len(event["actions"]) for event in result.trace] == [2, 1, 2, 1]
    for event in result.trace:
        assert all(action["output_position"] // 3 == event["block"] for action in event["actions"])
    assert (model.seen[1][0, 5:] == MASK).all()
    assert not (result.token_ids[:, 2:] == MASK).any()
    assert result.details["remaining_masks"] == 0


def test_margin_and_token_confidence_rank_different_positions():
    # Position 0 has the larger top probability; position 1 has the larger margin.
    rows = [[0., .60, .39, .01], [0., .55, .225, .225]]
    kwargs = dict(gen_length=2, steps=2, block_length=2, temperature=0)
    probability = confidence.decode(FakeModel(lambda _: rows), PROMPT, MASK, **kwargs)
    margin = top_k_margin.decode(FakeModel(lambda _: rows), PROMPT, MASK, **kwargs)
    assert probability.trace[0]["actions"][0]["output_position"] == 0
    assert margin.trace[0]["actions"][0]["output_position"] == 1


def test_margin_uses_scheduled_k_and_resolves_position_ties():
    rows = [[0., .9, .05, .05], [0., .4, .35, .25],
            [0., .8, .1, .1], [0., .4, .3, .3]]
    result = top_k_margin.decode(FakeModel(lambda _: rows), PROMPT, MASK,
                                 gen_length=4, steps=2, block_length=4, temperature=0)
    assert [[a["output_position"] for a in event["actions"]] for event in result.trace] == [[0, 2], [3, 1]]
    tied = top_k_margin.decode(FakeModel(), PROMPT, MASK, gen_length=4, steps=2,
                               block_length=4, temperature=0)
    assert [action["output_position"] for action in tied.trace[0]["actions"]] == [0, 1]


@pytest.mark.parametrize("decoder", [random_mdm, confidence, top_k_margin])
def test_mask_dominant_distribution_still_completes(decoder):
    result = decoder.decode(FakeModel(lambda _: [.99, .005, .003, .002]), PROMPT, MASK,
                            gen_length=4, steps=4, block_length=2)
    assert torch.equal(result.token_ids[:, :2], PROMPT)
    assert not (result.token_ids[:, 2:] == MASK).any()


def test_random_seed_changes_position_order_and_sampled_tokens():
    first = random_mdm.decode(FakeModel(), PROMPT, MASK, gen_length=32, steps=32,
                              block_length=32, seed=1)
    second = random_mdm.decode(FakeModel(), PROMPT, MASK, gen_length=32, steps=32,
                               block_length=32, seed=2)
    assert first.trace[0]["actions"] != second.trace[0]["actions"]
    assert not torch.equal(first.token_ids, second.token_ids)


@pytest.mark.parametrize("overrides", [
    {"steps": 0}, {"steps": 5}, {"block_length": 3}, {"gen_length": True},
    {"temperature": float("nan")}, {"temperature": float("inf")},
    {"temperature": -1}, {"seed": -1}, {"seed": 2**63},
])
def test_invalid_generation_fails_before_model_forward(overrides):
    model = FakeModel()
    kwargs = {"gen_length": 4, "steps": 4, "block_length": 2, **overrides}
    with pytest.raises(ValueError):
        confidence.decode(model, PROMPT, MASK, **kwargs)
    assert model.seen == []


def test_nonfinite_probabilities_raise():
    with pytest.raises(ValueError, match="Non-finite"):
        confidence.decode(FakeModel(lambda _: [0., float("nan"), .2, .1]), PROMPT, MASK,
                          gen_length=2, steps=2, block_length=2)


@pytest.mark.parametrize("texts,answer,winner,votes", [
    (["#### 2", "#### 1.0", "#### 1", "no answer", "#### 1.00"], "1", 1, {"2": 1, "1": 3}),
    (["bad", "#### 4", "#### 5", "#### 5", "#### 4"], "4", 1, {"4": 2, "5": 2}),
    (["#### $1,200.00", "#### 1200", "#### 1,200.0", "#### 2", "bad"], "1200", 0, {"1200": 3, "2": 1}),
    (["bad", "#### unknown", "Reasoning 1", "#### NaN", ""], None, 0, {}),
])
def test_best_of_5_normalized_voting_and_ties(monkeypatch, texts, answer, winner, votes):
    calls = []

    def candidate_decode(model, prompt_ids, mask_id, **kwargs):
        index = len(calls)
        calls.append(kwargs)
        return DecodeResult(torch.cat((prompt_ids, torch.tensor([[index + 1]])), dim=1), 1)

    class Tokenizer:
        def batch_decode(self, ids, **kwargs):
            return [texts[int(ids[0, 0]) - 1]]

    monkeypatch.setattr(best_of_5, "confidence_decode", candidate_decode)
    result = best_of_5.decode(FakeModel(), PROMPT, MASK, tokenizer=Tokenizer(),
                              answer_extractor=extract_answer, gen_length=1, steps=1,
                              block_length=1, temperature=1, seed=2**63 - 2)
    assert result.details["selected_answer"] == answer
    assert result.details["selected_candidate_index"] == winner
    assert result.details["votes"] == votes
    assert result.details["all_candidates_invalid"] == (answer is None)
    assert [call["seed"] for call in calls] == [2**63 - 2, 2**63 - 1, 0, 1, 2]
    assert all(call["temperature"] == 1 for call in calls)
    assert [candidate["raw_output"] for candidate in result.details["candidates"]] == texts
    assert result.total_nfes == result.generation_nfes == 5
    assert result.scoring_nfes == 0
    if answer is None:
        assert score_answer(texts[winner], "#### 1")["exact_match"] is False


def test_best_of_5_charges_real_forwards_and_pairs_first_confidence_sample():
    kwargs = dict(gen_length=4, steps=4, block_length=2, temperature=1, seed=17)
    first = confidence.decode(FakeModel(), PROMPT, MASK, **kwargs)
    model = FakeModel()
    # No valid numbers in this tokenizer's output: all-invalid fallback is candidate 0.
    result = best_of_5.decode(model, PROMPT, MASK, tokenizer=FakeTokenizer(),
                              answer_extractor=extract_answer, **kwargs)
    assert len(model.seen) == result.total_nfes == 20
    assert result.scoring_nfes == 0
    assert torch.equal(result.token_ids, first.token_ids)
    assert result.details["candidate_seeds"] == [17, 18, 19, 20, 21]
