"""Five confidence-decoded candidates, selected by final-answer majority vote."""

from collections import Counter

import torch

from dlm_project.decoding.confidence import decode as confidence_decode
from dlm_project.decoding.common import validate_generation
from dlm_project.decoding.decoder_result import DecodeResult


@torch.inference_mode()
def decode(model, prompt_ids, mask_id, *, tokenizer, answer_extractor,
           gen_length=256, steps=256, block_length=128, temperature=1.0, seed=1):
    """Vote on normalized answers, resolving ties by earliest valid candidate.

    answer_extractor(text) returns a normalized answer string or None. Neither
    generation nor voting accepts a reference answer. Selection adds no model
    forwards; all five generations are included in generation_nfes.
    """
    validate_generation(gen_length=gen_length, steps=steps, block_length=block_length,
                        temperature=temperature, seed=seed)
    if not callable(answer_extractor):
        raise ValueError("answer_extractor must be callable.")
    outputs, candidates, trace = [], [], []
    calls = 0
    for index in range(5):
        candidate_seed = (seed + index) % 2**63
        result = confidence_decode(model, prompt_ids, mask_id, gen_length=gen_length,
                                   steps=steps, block_length=block_length,
                                   temperature=temperature, seed=candidate_seed)
        text = tokenizer.batch_decode(result.token_ids[:, prompt_ids.shape[1]:],
                                      skip_special_tokens=True)[0]
        answer = answer_extractor(text)
        outputs.append(result.token_ids)
        candidates.append({"index": index, "seed": candidate_seed, "raw_output": text,
                           "predicted_answer": answer, "parse_success": answer is not None,
                           "generation_nfes": result.generation_nfes})
        calls += result.generation_nfes
        trace.extend({**event, "candidate_index": index} for event in result.trace)
    votes = Counter(candidate["predicted_answer"] for candidate in candidates
                    if candidate["parse_success"])
    # Counter preserves first-occurrence order, making ties deterministic.
    winning_answer = max(votes, key=votes.get) if votes else None
    winner = next((candidate["index"] for candidate in candidates
                   if winning_answer is not None and candidate["predicted_answer"] == winning_answer), 0)
    details = {"selection_rule": "answer_majority_vote", "candidate_count": 5,
               "candidate_seeds": [candidate["seed"] for candidate in candidates],
               "candidates": candidates, "votes": dict(votes),
               "selected_candidate_index": winner, "selected_answer": winning_answer,
               "valid_candidate_count": sum(candidate["parse_success"] for candidate in candidates),
               "all_candidates_invalid": not bool(votes),
               "unique_candidate_count": len({candidate["raw_output"] for candidate in candidates}),
               "remaining_masks": 0, "seed": seed}
    trace.append({"stage": "selection", "votes": dict(votes),
                  "selected_candidate_index": winner, "generation_nfes": calls})
    return DecodeResult(outputs[winner], calls, details=details, trace=trace)
