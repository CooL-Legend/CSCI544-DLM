"""Tests for unified Experiment 1 runner, checkpointing, and caching."""
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

from dlm_project.tasks.gsm8k import experiment_1 as e1
from dlm_project.tasks.gsm8k import ab_development as dev
from tests.decoding.test_ab_batch import BatchModel


class Tokenizer:
    eos_token_id = pad_token_id = 4

    def get_vocab(self):
        return {"[MASK]": 0, "one": 1, "two": 2, "three": 3, "<|endoftext|>": 4, "<|eot_id|>": 5}

    def convert_ids_to_tokens(self, i):
        return next(k for k, v in self.get_vocab().items() if v == i)

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(self.convert_ids_to_tokens(i) for i in ids if not skip_special_tokens or i not in (0, 4, 5))


def mock_wrapper(model=None):
    ctx = dev.context()
    return SimpleNamespace(
        model=model or BatchModel(),
        tokenizer=Tokenizer(),
        mask_id=0,
        identity={k: ctx["common"]["model"][k] for k in ("family", "repo_id", "revision")},
        encode_prompt=lambda text: torch.tensor([[3] * (2 + len(text) % 4)]),
    )


def test_e1_context_validation():
    ctx = e1.context(stage="dev", methods=("medal", "ab_medal"), num_samples=2)
    assert ctx["stage"] == "dev"
    assert len(ctx["examples"]) == 2
    assert ctx["budgets"] == list(e1.CHECKPOINT_BUDGETS)
    assert ctx["max_depth"] == 10


@pytest.mark.parametrize("method", ["medal", "ab_medal"])
def test_e1_run_question_checkpoints_and_artifacts(tmp_path, method):
    w = mock_wrapper()
    ctx = e1.context(stage="dev", methods=(method,), num_samples=1)
    prepared = e1.prepare(w, ctx)[0]
    budgets = (2, 4, 6)

    results = e1.run_question_e1(
        w, prepared, ctx, method, tmp_path, budgets=budgets, max_depth=4, num_candidates=2
    )

    assert len(results) == 3
    q_dir = tmp_path / method / prepared["example"]["example_id"]
    assert (q_dir / "trace.jsonl").exists()
    assert (q_dir / "nodes.jsonl").exists()
    assert (q_dir / "search_audit.json").exists()
    assert (q_dir / "decisions.jsonl").exists()

    for r in results:
        b = r["search_budget"]
        assert b in budgets
        chk_dir = q_dir / "checkpoints" / str(b)
        assert (chk_dir / "tree.json").exists()
        assert (chk_dir / "selection.json").exists()
        assert (chk_dir / "completion.json").exists()
        assert (chk_dir / "metrics.json").exists()
        assert r["search_nfes"] == b
        assert r["completion_nfes"] == 256 - r["selected_depth"]
        assert r["total_nfes"] == b + (256 - r["selected_depth"])


def test_e1_completion_caching(tmp_path):
    w = mock_wrapper()
    ctx = e1.context(stage="dev", methods=("medal",), num_samples=1)
    prepared = e1.prepare(w, ctx)[0]

    cache = {}
    state = torch.full((1, 260), 0, dtype=torch.long)
    comp1, dec1, score1, t1, cached1 = e1.complete_selected_state(w, state, prepared, ctx, completion_cache=cache)
    assert not cached1

    comp2, dec2, score2, t2, cached2 = e1.complete_selected_state(w, state, prepared, ctx, completion_cache=cache)
    assert cached2
    assert t2 == 0.0
    assert dec1 == dec2
    assert score1 == score2


def test_e1_run_stage_csv_outputs(tmp_path):
    w = mock_wrapper()
    ctx = e1.context(stage="dev", methods=("medal", "ab_medal"), num_samples=2)
    budgets = (2, 4)

    q_rows, agg_rows = e1.run_e1_stage(
        w, ctx, tmp_path, methods=("medal", "ab_medal"), budgets=budgets, max_depth=4, num_candidates=2
    )

    assert len(q_rows) == 2 * 2 * 2  # 2 questions * 2 methods * 2 budgets = 8 outcomes
    assert len(agg_rows) == 4  # 2 methods * 2 budgets = 4 aggregate rows

    summaries_dir = tmp_path / "summaries"
    assert (summaries_dir / "e1_dev_per_question.csv").exists()
    assert (summaries_dir / "e1_dev_aggregate.csv").exists()
    assert (tmp_path / "run_manifest.json").exists()

    for row in agg_rows:
        assert 0.0 <= row["accuracy"] <= 1.0
        assert 0.0 <= row["ci_95_low"] <= row["ci_95_high"] <= 1.0
