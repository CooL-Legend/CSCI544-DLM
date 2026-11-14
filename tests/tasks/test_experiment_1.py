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
        assert r["nfe_utilization"] == 1.0
        assert r["simulations_completed"] >= 1
        assert r["simulation_cap_bound"] is False
        assert r["reward_mode"] == "normalized_ig"


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
        assert row["mean_nfe_utilization"] == 1.0
        assert row["mean_simulations"] >= 1.0
        assert row["sim_cap_bind_rate"] == 0.0


def test_e1_incremental_saving_and_resumption(tmp_path):
    import json
    w = mock_wrapper()
    ctx = e1.context(stage="dev", methods=("medal",), num_samples=2)
    budgets = (2, 4)

    # First run: writes checkpoints and summaries incrementally
    q_rows_1, agg_rows_1 = e1.run_e1_stage(
        w, ctx, tmp_path, methods=("medal",), budgets=budgets, max_depth=4, num_candidates=2, resume=True
    )
    assert len(q_rows_1) == 4
    assert len(agg_rows_1) == 2

    # Check that per-question CSV, aggregate CSV, and manifest exist
    summaries_dir = tmp_path / "summaries"
    assert (summaries_dir / "e1_dev_per_question.csv").exists()
    assert (summaries_dir / "e1_dev_aggregate.csv").exists()
    manifest_data = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest_data["status"] == "completed"

    # Reset calls count on mock model
    calls_before = len(w.model.calls)
    assert calls_before > 0

    # Second run with resume=True: verify is_question_complete skips forward passes
    q_rows_2, agg_rows_2 = e1.run_e1_stage(
        w, ctx, tmp_path, methods=("medal",), budgets=budgets, max_depth=4, num_candidates=2, resume=True
    )
    assert len(q_rows_2) == 4
    assert len(w.model.calls) == calls_before  # 0 new model calls performed!
    assert [r["exact_match"] for r in q_rows_1] == [r["exact_match"] for r in q_rows_2]


def test_e1_batch_size_parity(tmp_path):
    import json
    w_serial = mock_wrapper()
    w_batch = mock_wrapper()
    ctx_serial = e1.context(stage="dev", methods=("medal", "ab_medal"), num_samples=2)
    ctx_batch = e1.context(stage="dev", methods=("medal", "ab_medal"), num_samples=2)
    budgets = (3, 6)

    serial_dir = tmp_path / "serial"
    batch_dir = tmp_path / "batch"

    q_serial, agg_serial = e1.run_e1_stage(
        w_serial, ctx_serial, serial_dir, methods=("medal", "ab_medal"),
        budgets=budgets, max_depth=4, num_candidates=2, batch_size=1
    )
    q_batch, agg_batch = e1.run_e1_stage(
        w_batch, ctx_batch, batch_dir, methods=("medal", "ab_medal"),
        budgets=budgets, max_depth=4, num_candidates=2, batch_size=2
    )

    assert len(q_serial) == len(q_batch) == 8
    # Batch execution must use strictly fewer model invocations than serial
    assert len(w_batch.model.calls) < len(w_serial.model.calls)

    deterministic_keys = (
        "method", "search_budget", "example_id", "row_id",
        "scoring_text", "predicted_answer", "reference_answer",
        "exact_match", "parse_success", "search_nfes", "completion_nfes",
        "total_nfes", "max_depth_reached", "selected_depth",
        "selection_depth_gap", "selected_depth_candidate_count",
        "shortlist_node_ids", "selected_node_id", "selected_q",
        "selected_state_hash", "reached_target_depth",
        "selection_below_target", "fallback_used", "stop_reason", "was_cached",
    )
    for r_s, r_b in zip(q_serial, q_batch):
        for k in deterministic_keys:
            assert r_s[k] == r_b[k], f"Mismatch on {k} for {r_s['example_id']} {r_s['method']}@{r_s['search_budget']}"

        chk_s = Path(r_s["artifact_directory"])
        chk_b = Path(r_b["artifact_directory"])
        comp_s = json.loads((chk_s / "completion.json").read_text())
        comp_b = json.loads((chk_b / "completion.json").read_text())
        assert comp_s["token_ids"] == comp_b["token_ids"]
        tree_s = json.loads((chk_s / "tree.json").read_text())
        tree_b = json.loads((chk_b / "tree.json").read_text())
        assert tree_s["nodes"] == tree_b["nodes"]


def test_e1_extended_test_500_manifest_support():
    ctx_200 = e1.context(stage="test", methods=("medal", "ab_medal"), num_samples=200)
    ctx_500 = e1.context(stage="test", methods=("medal", "ab_medal"), num_samples=500)
    assert len(ctx_200["examples"]) == 200
    assert len(ctx_500["examples"]) == 500
    # First 200 examples in the 500-example manifest must be 100% identical to the 200-example primary set
    assert ctx_500["examples"][:200] == ctx_200["examples"]
    assert len({ex["example_id"] for ex in ctx_500["examples"]}) == 500


def test_e1_context_e2_eval_slicing():
    ctx_default = e1.context(stage="evaluation", methods=("medal", "ab_medal"))
    ctx_20 = e1.context(stage="evaluation", methods=("medal", "ab_medal"), sample_slice=(0, 20))
    ctx_50 = e1.context(stage="evaluation", methods=("medal", "ab_medal"), sample_slice=(20, 70))
    ctx_70 = e1.context(stage="evaluation", methods=("medal", "ab_medal"), sample_slice=(0, 70))

    assert len(ctx_default["examples"]) == 20
    assert len(ctx_20["examples"]) == 20
    assert len(ctx_50["examples"]) == 50
    assert len(ctx_70["examples"]) == 70
    assert ctx_20["examples"] == ctx_default["examples"]
    assert ctx_70["examples"][:20] == ctx_20["examples"]
    assert ctx_70["examples"][20:] == ctx_50["examples"]
    assert len({ex["example_id"] for ex in ctx_70["examples"]}) == 70


def test_e1_test_script_and_gpu_profiler_and_notebooks():
    import ast
    import json
    from dlm_project.paths import PROJECT_ROOT
    from scripts.run_e1_test import compute_paired_comparison, mcnemar_exact_pvalue
    from scripts.profile_gpu_batch_size import estimate_vram_and_compute, inspect_manifest_token_lengths

    assert mcnemar_exact_pvalue(0, 0) == 1.0
    assert 0.0 < mcnemar_exact_pvalue(8, 1) < 0.05

    mock_rows = [
        {"example_id": "test:1", "method": "medal", "search_budget": 256, "exact_match": False, "total_nfes": 502},
        {"example_id": "test:1", "method": "ab_medal", "search_budget": 256, "exact_match": True, "total_nfes": 502},
        {"example_id": "test:2", "method": "medal", "search_budget": 256, "exact_match": True, "total_nfes": 502},
        {"example_id": "test:2", "method": "ab_medal", "search_budget": 256, "exact_match": True, "total_nfes": 502},
    ]
    paired = compute_paired_comparison(mock_rows, budgets=(256,))
    assert len(paired) == 1
    assert paired[0]["paired_questions"] == 2
    assert paired[0]["medal_accuracy"] == 0.5
    assert paired[0]["ab_medal_accuracy"] == 1.0
    assert paired[0]["ab_medal_only_correct"] == 1
    assert paired[0]["medal_only_correct"] == 0

    token_stats = inspect_manifest_token_lengths(num_samples=200)
    assert token_stats["num_samples"] == 200
    assert token_stats["max_total_tokens"] >= token_stats["p95_total_tokens"] >= token_stats["mean_total_tokens"]

    vram_rows = estimate_vram_and_compute(batch_sizes=(1, 2, 4, 8, 16), max_seq_len=850, max_budget=512)
    assert len(vram_rows) == 5
    assert vram_rows[0]["est_peak_reserved_gib"] < vram_rows[1]["est_peak_reserved_gib"] < vram_rows[2]["est_peak_reserved_gib"]
    assert vram_rows[2]["fits_l4_22_5gib"] is True

    for nb_rel in (
        "notebooks/gsm8k/Experiment_1_Dev.ipynb",
        "notebooks/gsm8k/Experiment_1_Eval.ipynb",
        "notebooks/gsm8k/Experiment_1_Test.ipynb",
        "notebooks/gsm8k/Profile_GPU_Batch_Size.ipynb",
    ):
        nb_path = PROJECT_ROOT / nb_rel
        assert nb_path.is_file(), f"Missing notebook: {nb_path}"
        nb_data = json.loads(nb_path.read_text())
        code_cells = ["".join(c["source"]) for c in nb_data["cells"] if c["cell_type"] == "code"]
        assert len(code_cells) >= 4
        for code in code_cells:
            ast.parse(code)

    for dev_eval_rel in ("notebooks/gsm8k/Experiment_1_Dev.ipynb", "notebooks/gsm8k/Experiment_1_Eval.ipynb"):
        nb_src = "\n".join(
            "".join(c["source"])
            for c in json.loads((PROJECT_ROOT / dev_eval_rel).read_text())["cells"]
            if c["cell_type"] == "code"
        )
        assert "BATCH_SIZE = 4" in nb_src
        assert "batch_size=BATCH_SIZE" in nb_src

    e1_eval_src = "\n".join(
        "".join(c["source"])
        for c in json.loads((PROJECT_ROOT / "notebooks/gsm8k/Experiment_1_Eval.ipynb").read_text())["cells"]
        if c["cell_type"] == "code"
    )
    assert "sample_slice=(0, 20)" in e1_eval_src
    assert "sample_slice=(20, 70)" in e1_eval_src
    assert "compute_paired_comparison" in e1_eval_src
    assert "mean_nfe_utilization" in e1_eval_src
    assert "mean_simulations" in e1_eval_src

    test_nb_src = "\n".join(
        "".join(c["source"])
        for c in json.loads((PROJECT_ROOT / "notebooks/gsm8k/Experiment_1_Test.ipynb").read_text())["cells"]
        if c["cell_type"] == "code"
    )
    assert 'STAGE = "test"' in test_nb_src
    assert "NUM_SAMPLES = 2" in test_nb_src
    assert "BUDGETS = [256, 512]" in test_nb_src
    assert "BATCH_SIZE = 4" in test_nb_src
    assert "profile_resources=PROFILE_RESOURCES" in test_nb_src



