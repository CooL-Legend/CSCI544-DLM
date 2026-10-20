"""Frozen stage partitions and completed-output termination semantics."""

import copy
import json
from types import SimpleNamespace

import pytest

from dlm_project.tasks.gsm8k import e0_dataset as data
from dlm_project.tasks.gsm8k.e0_tokens import decode_output, resolve_termination_tokens


def manifest():
    train = [{"question": f"Training problem {i}?", "answer": "REFERENCE_ONLY\n#### 1"} for i in range(40)]
    test = [{"question": f"Test problem {i}?", "answer": "REFERENCE_ONLY\n#### 1"} for i in range(110)]
    train.append({"question": " Test  problem 0? ", "answer": "#### 1"})
    return data.build_manifest(train, test, revision="a" * 40)


class Tokenizer:
    eos_token_id = 1
    def get_vocab(self):
        return {"[MASK]": 0, "<|endoftext|>": 1, "<|eot_id|>": 2, "word": 3, "number": 4, "other": 5}
    def convert_ids_to_tokens(self, token):
        return next(key for key, value in self.get_vocab().items() if value == token)
    def decode(self, ids, skip_special_tokens=False):
        if skip_special_tokens:
            return "#### 1" if ids else ""
        return " ".join(self.convert_ids_to_tokens(token) for token in ids)


def test_manifest_disjoint_reproducible_and_test_scaling_preserves_prefix(tmp_path):
    frozen = manifest()
    assert frozen == manifest()
    pools = {stage: data.select_examples(frozen, stage) for stage in data.STAGE_COUNTS}
    assert [len(rows) for rows in pools.values()] == [5, 20, 25]
    assert {row["source_split"] for row in pools["evaluation"]} == {"train"}
    assert {row["source_split"] for row in pools["test"]} == {"test"}
    all_ids = [row["example_id"] for rows in pools.values() for row in rows]
    assert len(set(all_ids)) == len(all_ids)
    assert data.select_examples(frozen, "test", 100)[:25] == pools["test"]
    assert not any(row["row_id"] == 40 for row in frozen["examples"] if row["source_split"] == "train")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(frozen))
    assert data.get_manifest(path) == frozen
    with pytest.raises(ValueError, match="revision"):
        data.get_manifest(path, revision="b" * 40)
    bad = copy.deepcopy(frozen)
    bad["examples"][0]["question"] = "tampered"
    with pytest.raises(ValueError):
        data.validate_manifest(bad)
    with pytest.raises(ValueError):
        data.select_examples(frozen, "dev", 6)


def test_token_resolution_validates_both_tokens_and_config_ids():
    tokenizer = Tokenizer()
    model = SimpleNamespace(config=SimpleNamespace(eos_token_id=[1, 2], vocab_size=6))
    resolved = resolve_termination_tokens(tokenizer, model, 0)
    assert resolved["token_ids"] == [1, 2]
    assert resolved["tokens"] == {"1": "<|endoftext|>", "2": "<|eot_id|>"}
    with pytest.raises(ValueError):
        resolve_termination_tokens(tokenizer, model, 2)
    model.config.eos_token_id = 99
    with pytest.raises(ValueError):
        resolve_termination_tokens(tokenizer, model, 0)
    tokenizer.get_vocab = lambda: {"[MASK]": 0, "<|endoftext|>": 1, "word": 3}
    with pytest.raises(ValueError, match="distinct EOS"):
        resolve_termination_tokens(tokenizer, SimpleNamespace(), 0)


@pytest.mark.parametrize("ids,first,prefix", [([3, 4], None, [3, 4]), ([1, 3], 0, []),
                                            ([3, 2, 4, 1], 1, [3]), ([3, 4, 1], 2, [3, 4])])
def test_prefix_after_complete_generation(ids, first, prefix):
    result = decode_output(Tokenizer(), ids, [1, 2])
    assert result["full_output_ids"] == ids
    assert result["first_terminator_position"] == first
    assert result["scoring_output_ids"] == prefix
    assert result["scoring_text"] == ("#### 1" if prefix else "")
