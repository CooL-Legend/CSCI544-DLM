"""Controlled checks for search-free decoding and reference-free self-selection."""

import pytest
import torch

from dlm_project.decoding import best_of_five, confidence_guided, random_mdm, topk_margin
from tests.decoding.helpers import FakeModel, MASK, PROMPT


def test_random_mdm_uses_original_sampler_with_seed_and_preserved_prompt():
    model = FakeModel()
    first = random_mdm.decode(model, PROMPT, MASK, gen_length=4, steps=4,
                              block_length=4, seed=17)
    second = random_mdm.decode(FakeModel(), PROMPT, MASK, gen_length=4, steps=4,
                               block_length=4, seed=17)
    assert first.generation_nfes == first.total_nfes == len(model.seen) == 4
    assert first.scoring_nfes == 0
    assert torch.equal(first.token_ids, second.token_ids)
    assert torch.equal(first.token_ids[:, :2], PROMPT)
    assert all(torch.equal(state[:, :2], PROMPT) for state in model.seen)


def test_topk_margin_commits_highest_two_positions_per_forward():
    rows = [[0., .9, .05, .05], [0., .4, .35, .25],
            [0., .8, .1, .1], [0., .4, .3, .3]]
    model = FakeModel(lambda output: rows)
    result = topk_margin.decode(model, PROMPT, MASK, gen_length=4,
                                steps=4, top_k_positions=2)
    assert result.generation_nfes == len(model.seen) == 2
    assert [[a["output_position"] for a in event["actions"]] for event in result.trace] == [[0, 2], [3, 1]]
    assert result.details["mean_selected_margin"] == pytest.approx((.85 + .7 + .1 + .05) / 4, abs=1e-6)
    assert (result.token_ids[0, 2:] == 1).all()
    assert torch.equal(result.token_ids[:, :2], PROMPT)
    assert torch.equal(model.seen[0][:, 2:], torch.zeros(1, 4, dtype=torch.long))
    assert model.seen[1][0, 2] == model.seen[1][0, 4] == 1
    with pytest.raises(ValueError, match="cover"):
        topk_margin.decode(model, PROMPT, MASK, gen_length=4, steps=1, top_k_positions=2)


def test_confidence_guided_reuses_existing_search_free_scorer():
    model = FakeModel()
    result = confidence_guided.decode(model, PROMPT, MASK, gen_length=4, margin_gamma=10)
    assert result.generation_nfes == result.total_nfes == len(model.seen) == 4
    assert result.scoring_nfes == 0
    assert len(result.trace) == 4
    assert all(event["stage"] == "completion" for event in result.trace)
    assert torch.equal(result.token_ids[:, :2], PROMPT)
    assert all(torch.equal(state[:, :2], PROMPT) for state in model.seen)


def test_best_of_five_charges_all_generations_and_masked_scoring():
    model = FakeModel()
    result = best_of_five.decode(model, PROMPT, MASK, gen_length=4, steps=4,
                                 block_length=4, score_block_size=2, seed=17)
    assert result.generation_nfes == 20
    assert result.scoring_nfes == 10
    assert result.total_nfes == len(model.seen) == 30
    assert result.details["candidate_seeds"] == [17, 18, 19, 20, 21]
    assert result.details["candidate_count"] == 5
    assert result.details["unique_candidate_count"] == 1
    assert result.details["selected_candidate_index"] == 0
    assert len([e for e in result.trace if e["stage"] == "score"]) == 5
    assert torch.equal(result.token_ids[:, :2], PROMPT)
    # Scoring forwards mask only output positions, never prompt tokens.
    assert all(torch.equal(state[:, :2], PROMPT) for state in model.seen[20:])
    assert [int((state[0, 2:] == MASK).sum()) for state in model.seen[20:]] == [2] * 10


def test_block_score_is_sensitive_to_generated_token_without_reference():
    model = FakeModel(lambda output: [0., .7, .2, .1])
    good = torch.tensor([[3, MASK, 1, 1, 1, 1]])
    bad = torch.tensor([[3, MASK, 2, 2, 2, 2]])
    good_score, calls1 = best_of_five.masked_block_score(model, good, 2, MASK, 2)
    bad_score, calls2 = best_of_five.masked_block_score(model, bad, 2, MASK, 2)
    assert good_score > bad_score
    assert (calls1, calls2) == (2, 2)
    assert all(torch.equal(state[:, :2], PROMPT) for state in model.seen)


@pytest.mark.parametrize("invalid", [0, -1, 5, True])
def test_invalid_selection_settings_are_rejected(invalid):
    with pytest.raises(ValueError):
        topk_margin.decode(FakeModel(), PROMPT, MASK, gen_length=4, steps=4,
                           top_k_positions=invalid)
    with pytest.raises(ValueError):
        best_of_five.decode(FakeModel(), PROMPT, MASK, gen_length=4, steps=4,
                            block_length=4, score_block_size=invalid)
