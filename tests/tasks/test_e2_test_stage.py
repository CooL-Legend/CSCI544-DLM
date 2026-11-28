"""E2 SWD and CMER use the same frozen test questions and prompts as E1 on CARC."""

import csv
import json
from types import SimpleNamespace

import pytest

from scripts import run_e2 as cli
from dlm_project.tasks.gsm8k import experiment_1 as e1, experiment_2 as e2
from dlm_project.tasks.gsm8k.experiment_protocol import build_prompt, example_seed
from tests.tasks.test_experiment_1 import mock_wrapper


@pytest.mark.parametrize("count", [None, 2, 200, 201, 500])
@pytest.mark.parametrize("reward_mode", ["swd", "cmer"])
def test_test_pool_matches_e1_content_order_and_prompts(count, reward_mode):
    reference = e1.context(stage="test", num_samples=count)
    actual = e2.context(stage="test", num_samples=count, reward_mode=reward_mode)
    assert actual["stage"] == actual["norm_stage"] == "test"
    assert actual["examples"] == reference["examples"]
    assert actual["manifest_hash"] == reference["manifest_hash"]
    assert actual["manifest"] == reference["manifest"]
    assert actual["common"] == reference["common"]
    assert actual["prompt_template_hash"] == reference["prompt_template_hash"]
    assert len(actual["examples"]) == (count if count is not None else 200)
    assert len({row["example_id"] for row in actual["examples"]}) == len(actual["examples"])
    for row, old in zip(actual["examples"], reference["examples"]):
        assert build_prompt(row["question"], actual["common"]["prompt"]) == build_prompt(
            old["question"], reference["common"]["prompt"])
        assert example_seed(row["row_id"]) == example_seed(old["row_id"])
    assert actual["reward_mode"] == reward_mode
    assert actual["reward_lambda"] == actual["reward_beta"] == 1.0
    assert actual["sim_cap_multiplier"] == 4
    assert actual["budgets"] == [256, 512]


def test_cmer_single_run_preserves_notebook_200_plus_300_order_and_batch_boundaries():
    full = e2.context(stage="test", num_samples=500, reward_mode="cmer")["examples"]
    primary = e2.context(stage="test", sample_slice=(0, 200), reward_mode="cmer")["examples"]
    extended = e2.context(stage="test", sample_slice=(200, 500), reward_mode="cmer")["examples"]
    assert full == primary + extended
    batches = lambda rows: [rows[i:i + 8] for i in range(0, len(rows), 8)]
    assert batches(full) == batches(primary) + batches(extended)


def test_existing_eval_splits_and_prefix_limit_remain_distinct_from_test():
    full = e2.context(stage="evaluation")
    canonical = e2.context(stage="evaluation", sample_slice=(0, 20))
    extended = e2.context(stage="eval", sample_slice=(20, 70))
    limited = e2.context(stage="eval", sample_slice=(20, 70), num_samples=2)
    assert len(e2.context(stage="dev")["examples"]) == 5
    assert len(full["examples"]) == 70
    assert canonical["examples"] + extended["examples"] == full["examples"]
    assert limited["examples"] == full["examples"][20:22]
    assert all(row["source_split"] == "train" for row in full["examples"])
    assert all(row["source_split"] == "test" for row in e2.context(stage="test", num_samples=500)["examples"])


@pytest.mark.parametrize("options", [
    {"stage": "test", "num_samples": 0},
    {"stage": "test", "num_samples": -1},
    {"stage": "test", "num_samples": 501},
    {"stage": "test", "num_samples": True},
    {"stage": "test", "num_samples": 2.5},
    {"stage": "evaluation", "num_samples": 500},
    {"stage": "dev", "num_samples": 6},
    {"stage": "evaluation", "sample_slice": (20, 70), "num_samples": 51},
    {"stage": "test", "sample_slice": (-1, 2)},
    {"stage": "test", "sample_slice": (0, 501)},
    {"stage": "test", "sample_slice": (2, 2)},
    {"stage": "test", "sample_slice": (True, 2)},
    {"stage": "test", "sample_slice": (0, 1, 2)},
    {"stage": "unknown"},
])
def test_invalid_sample_requests_fail_instead_of_silently_truncating(options):
    with pytest.raises(ValueError):
        e2.context(**options)


def test_test_slice_can_cross_the_canonical_200_boundary():
    reference = e1.context(stage="test", num_samples=500)
    actual = e2.context(stage="test", sample_slice=(190, 210))
    assert actual["examples"] == reference["examples"][190:210]


def _fake_checkpoint_rows(ctx, methods, budgets):
    return [
        {"method": method, "search_budget": budget, "example_id": ex["example_id"],
         "exact_match": True, "parse_success": True, "search_nfes": budget,
         "completion_nfes": 250, "total_nfes": budget + 250,
         "search_wall_seconds": 1.0, "completion_wall_seconds": 1.0,
         "total_wall_seconds": 2.0, "selected_depth": 6, "selected_q": 0.1,
         "fallback_used": True}
        for ex in ctx["examples"] for method in methods for budget in budgets
    ]


@pytest.mark.parametrize("reward_mode", ["swd", "cmer"])
def test_cli_forwards_500_test_settings_and_reports_test_artifacts(tmp_path, monkeypatch, reward_mode):
    loads, runs, offline, reporting = [], [], [], []
    wrapper = SimpleNamespace()

    def load_model(ctx, **kwargs):
        loads.append((ctx, kwargs))
        return wrapper

    def run_stage(actual_wrapper, ctx, output_dir, **kwargs):
        assert actual_wrapper is wrapper
        runs.append((ctx, output_dir, kwargs))
        rows = _fake_checkpoint_rows(ctx, kwargs["methods"], kwargs["budgets"])
        aggregate = e2.save_e2_progress(output_dir, output_dir / "summaries", ctx["stage"],
                                        rows, kwargs["methods"], kwargs["budgets"])
        return rows, aggregate

    monkeypatch.setattr(cli, "load_model", load_model)
    monkeypatch.setattr(cli, "run_e2_stage", run_stage)
    monkeypatch.setattr(cli, "finish_reporting", lambda directory, **kwargs: reporting.append((directory, kwargs)))
    monkeypatch.setattr("dlm_project.tasks.gsm8k.e3_analysis.analyze_e3_stage", lambda **kwargs: offline.append(kwargs))
    monkeypatch.setattr(e2.os, "sync", lambda: None, raising=False)
    assert cli.main([
        "--stage", "test", "--num-samples", "500", "--batch-size", "8",
        "--methods", "medal", "ab_medal", "--budgets", "256", "512",
        "--reward-mode", reward_mode, "--reward-lambda", "1.0",
        "--sim-cap-multiplier", "4", "--max-depth", "10", "--num-candidates", "3",
        "--device", "cpu", "--model-path", "/prepared/llada", "--local-files-only",
        "--run-e3", "--profile-resources", "--output-dir", str(tmp_path),
    ]) == 0

    assert len(loads) == len(runs) == len(offline) == len(reporting) == 1
    ctx, directory, kwargs = runs[0]
    assert len(ctx["examples"]) == 500
    assert ctx["examples"] == e1.context(stage="test", num_samples=500)["examples"]
    assert ctx["reward_mode"] == reward_mode
    assert ctx["reward_beta"] == ctx["reward_lambda"] == 1.0
    assert ctx["sim_cap_multiplier"] == 4
    assert loads[0][1] == {"model_path": "/prepared/llada", "device": "cpu", "local_files_only": True}
    assert kwargs == {"methods": ("medal", "ab_medal"), "budgets": (256, 512),
                      "max_depth": 10, "num_candidates": 3, "resume": True,
                      "batch_size": 8, "profile_resources": True}
    assert offline[0] == {"results_dir": tmp_path / "test", "output_dir": tmp_path / "summaries",
                          "stage": "test", "methods": ("medal", "ab_medal"), "budgets": (256, 512)}
    summaries = tmp_path / "summaries"
    with (summaries / "e2_test_per_question.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2000
    assert len({(row["example_id"], row["method"], row["search_budget"]) for row in rows}) == 2000
    with (summaries / "e2_test_aggregate.csv").open(newline="") as stream:
        assert [int(row["sample_count"]) for row in csv.DictReader(stream)] == [500] * 4
    with (summaries / "e2_test_paired_comparison.csv").open(newline="") as stream:
        assert [int(row["paired_questions"]) for row in csv.DictReader(stream)] == [500, 500]
    assert (summaries / "e2_paired_comparison.csv").read_bytes() == (summaries / "e2_test_paired_comparison.csv").read_bytes()


def test_cli_eval_num_samples_limits_the_requested_split(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(cli, "load_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "run_e2_stage", lambda wrapper, ctx, output_dir, **kwargs: (seen.append(ctx) or [], []))
    assert cli.main(["--stage", "eval", "--eval-split", "extended_50", "--num-samples", "2",
                     "--device", "cpu", "--output-dir", str(tmp_path)]) == 0
    assert seen[0]["examples"] == e2.context(stage="eval")["examples"][20:22]


def test_cli_rejects_eval_split_for_test_before_model_loading(monkeypatch):
    monkeypatch.setattr(cli, "load_model", lambda *args, **kwargs: pytest.fail("Unexpected model load"))
    with pytest.raises(SystemExit, match="2"):
        cli.main(["--stage", "test", "--eval-split", "extended_50", "--device", "cpu"])


@pytest.mark.parametrize("reward_mode", ["swd", "cmer"])
def test_test_stage_saves_trees_and_resumes_without_forwards(tmp_path, monkeypatch, reward_mode):
    wrapper = mock_wrapper()
    ctx = e2.context(stage="test", num_samples=2, reward_mode=reward_mode)
    ctx["common"]["generation"] = {"gen_length": 8, "steps": 8, "block_length": 8}
    monkeypatch.setattr(e2.os, "sync", lambda: None, raising=False)
    options = {"methods": ("medal", "ab_medal"), "budgets": (4, 8),
               "max_depth": 4, "num_candidates": 2, "batch_size": 2}
    rows, aggregate = e2.run_e2_stage(wrapper, ctx, tmp_path, **options)
    assert len(rows) == 8 and len(aggregate) == 4
    assert all(row["reward_mode"] == reward_mode and row["reward_beta"] == 1.0 for row in rows)
    for row in rows:
        question_dir = tmp_path / "test" / row["method"] / row["example_id"]
        assert (question_dir / "trace.jsonl").stat().st_size > 0
        assert (question_dir / "nodes.jsonl").stat().st_size > 0
        tree = json.loads((question_dir / "checkpoints" / str(row["search_budget"]) / "tree.json").read_text())
        nodes = {node["node_id"]: node for node in tree["nodes"]}
        assert row["selected_node_id"] in nodes
        assert sum(node["parent_id"] is None for node in nodes.values()) == 1
        assert all(node["parent_id"] is None or node["parent_id"] in nodes for node in nodes.values())
        assert all(node["state"] for node in nodes.values())
    for example in ctx["examples"]:
        for method in options["methods"]:
            assert e2.is_question_complete(tmp_path / "test", method, example["example_id"], options["budgets"])
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["stage"] == "test" and manifest["status"] == "completed"
    assert manifest["reward_mode"] == reward_mode and manifest["reward_beta"] == 1.0
    assert (tmp_path / "summaries/e2_test_aggregate.csv").is_file()
    assert (tmp_path / "summaries/e2_test_per_question.csv").is_file()
    monkeypatch.setattr(type(wrapper.model), "__call__", lambda *args, **kwargs: pytest.fail("Resumed completed example performed inference"))
    resumed, _ = e2.run_e2_stage(wrapper, ctx, tmp_path, **options)
    # JSON artifacts stringify integer keys in per-depth metrics.
    assert resumed == json.loads(json.dumps(rows))
