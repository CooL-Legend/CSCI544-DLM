"""Artifacts, incomplete-search handling, pairing and offline reports on CPU."""

from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts import compare_medal_ab_medal as compare
from scripts import run_ab_medal_single as single
from dlm_project.decoding.ab_medal import ABMEDALConfig, tree_record
from dlm_project.decoding.mcts_decoding import MCTSConfig
from tests.decoding.helpers import FakeModel, FakeTokenizer, MASK, PROMPT
from tests.decoding.test_ab_medal import finite_tree


def wrapper():
    return SimpleNamespace(model=FakeModel(), tokenizer=FakeTokenizer(), mask_id=MASK,
                           identity={"test_model": True}, encode_prompt=lambda text: PROMPT.clone())


def settings(**kwargs):
    return ABMEDALConfig(**{"init_tokens": 2, "num_candidates": 3,
                            "top_k_tokens": 2, "top_k_actions": 2, **kwargs})


def attempt(tmp_path, **kwargs):
    instance = kwargs.pop("instance", wrapper())
    config = kwargs.pop("config", settings(search_policy="medal_equivalent"))
    report = single.run_attempt(instance, PROMPT, config, tmp_path, gen_length=4, print_history=False,
                                plot_max_nodes=20, plot_max_depth=3, **kwargs)
    return instance, report


def test_artifacts_exact_nfe_scoring_and_exclusive_times(tmp_path):
    instance, report = attempt(tmp_path, reference="#### 27", prompt_seconds=.2, setup_seconds=.4)
    assert report["status"] == "complete" and report["target_reached"]
    assert report["initialization_nfe"] == report["initialization_model_calls"] == 7
    assert report["decoding_nfe"] == report["decoding_model_calls"] == 2
    assert report["total_nfe"] == report["total_model_calls"] == len(instance.model.seen) == 9
    assert report["accuracy"] == 0  # Unparseable generated answers are incorrect.
    assert report["score"]["parse_success"] is False
    assert report["full_target_dlm_gate"] is False
    assert report["decoding_performed"]
    stages = [row for row in report["stages"] if row["stage"] == "Initialization"]
    assert sum(row["nfe"] for row in stages) == 7
    assert sum(row["wall_seconds"] for row in stages) == pytest.approx(report["initialization_wall_seconds"])
    assert all(row["wall_seconds"] >= 0 for row in stages)
    assert report["total_wall_seconds"] == pytest.approx(.2 + report["initialization_wall_seconds"] + report["decoding_wall_seconds"])
    assert report["setup_wall_seconds"] == .4
    assert set(p.name for p in tmp_path.iterdir()) >= {
        "config.json", "trace.jsonl", "tree.json", "history.json", "metrics.json", "completion.json", "ab_medal_tree.png"}
    tree = json.loads((tmp_path / "tree.json").read_text())
    assert len(tree["nodes"]) == 7
    assert sum(n["candidate"] for n in tree["nodes"]) == 3
    assert all("posteriors" in n for n in tree["nodes"])
    trace = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert sum(row["stage"] == "forward" for row in trace) == 9
    assert (tmp_path / "ab_medal_tree.png").read_bytes().startswith(b"\x89PNG")


@pytest.mark.parametrize("cap", [1, 4, 6])
def test_incomplete_search_never_decodes_or_scores(tmp_path, monkeypatch, cap):
    monkeypatch.setattr(single, "run_completion", lambda *a, **kw: pytest.fail("Incomplete search decoded"))
    instance, report = attempt(tmp_path, config=settings(max_model_calls=cap), reference="#### 27")
    assert report["status"] == "incomplete" and not report["target_reached"]
    assert not report["decoding_performed"] and report["decoding_nfe"] == 0
    assert report["score"] is None and report["accuracy"] is None
    assert report["initialization_nfe"] == len(instance.model.seen) == cap
    assert report["search"]["stop_reason"] == "model_call_budget"
    assert (tmp_path / "metrics.json").is_file() and (tmp_path / "tree.json").is_file()
    assert not (tmp_path / "completion.json").exists()


def test_missing_reference_leaves_accuracy_unavailable(tmp_path):
    _, report = attempt(tmp_path)
    assert report["status"] == "complete" and report["accuracy"] is None
    assert report["score"]["reference_available"] is False


@pytest.mark.parametrize("failure,status", [(RuntimeError("controlled model error"), "error"),
                                           (KeyboardInterrupt("controlled stop"), "interrupted")])
def test_failed_first_forward_still_saves_diagnostics(tmp_path, failure, status):
    class BrokenModel(FakeModel):
        def __call__(self, ids, **kwargs):
            raise failure
    instance = wrapper()
    instance.model = BrokenModel()
    _, report = attempt(tmp_path, instance=instance)
    assert report["status"] == status and not report["decoding_performed"]
    assert report["initialization_model_calls"] == report["initialization_nfe"] == 1
    assert report["total_model_calls"] == 1 and report["accuracy"] is None
    assert (tmp_path / "metrics.json").is_file() and (tmp_path / "trace.jsonl").is_file()
    assert (tmp_path / "tree.json").is_file() and (tmp_path / "ab_medal_tree.png").is_file()


def test_full_depth_synthetic_run_does_not_claim_real_dlm_gate(tmp_path, monkeypatch):
    tree = finite_tree(monkeypatch, "shared")
    instance = wrapper()
    report = single.run_attempt(instance, PROMPT, tree.config, tmp_path,
                                gen_length=21, print_history=False, plot_max_nodes=100, plot_max_depth=20)
    assert report["target_reached"] and report["structure"]["max_evaluated_depth"] == 20
    assert report["initialization_nfe"] == 23
    assert report["decoding_nfe"] == 1
    assert report["full_target_dlm_gate"] is False


def test_history_and_generated_answers_are_seeded_and_baseline_parity_holds(tmp_path):
    original_config = MCTSConfig(init_tokens=2, num_candidates=3, top_k_tokens=2, top_k_actions=2)
    _, original = attempt(tmp_path / "medal", config=original_config, method="medal", seed=7)
    _, adaptive = attempt(tmp_path / "equivalent", seed=7)
    _, repeated = attempt(tmp_path / "repeated", seed=7)
    assert original["response"] == adaptive["response"] == repeated["response"]
    assert original["total_nfe"] == adaptive["total_nfe"] == 9
    assert adaptive["search"]["selected_node_id"] == original["search"]["selected_node_id"]
    assert (tmp_path / "equivalent" / "history.json").read_text() == (tmp_path / "repeated" / "history.json").read_text()


def test_tree_plot_truncation_is_visible_and_full_json_is_untouched(tmp_path, monkeypatch):
    tree = finite_tree(monkeypatch, "separated")
    tree.run()
    record = tree_record(tree)
    before = len(tree.model.seen)
    full = single.plot_tree(record, tmp_path / "full.png", max_nodes=100, max_depth=20)
    clipped = single.plot_tree(record, tmp_path / "clipped.png", max_nodes=9, max_depth=3)
    assert full["shown_nodes"] == 61 and full["omitted_nodes"] == 0
    assert clipped["shown_nodes"] <= 9 and clipped["omitted_nodes"] >= 52
    assert len(record["nodes"]) == 61 and len(tree.model.seen) == before


def paired_row(method, example, complete=True, correct=True):
    return {
        "method": method, "condition": "budget_128", "example_id": f"test:{example}", "base_seed": 1, "seed": 1 + 5 * example,
        "status": "complete" if complete else "incomplete", "target_reached": complete,
        "score": {"exact_match": correct} if complete else None, "accuracy": float(correct) if complete else None,
        "total_nfe": 200 if complete else 128, "initialization_nfe": 100 if complete else 128,
        "decoding_nfe": 100 if complete else 0, "total_model_calls": 200 if complete else 128,
        "initialization_wall_seconds": 2, "decoding_wall_seconds": 3 if complete else 0,
        "total_wall_seconds": 5 if complete else 2, "stages": [],
        "structure": {"expanded_parents": 40, "evaluated_children": 99, "unique_states": 100,
                      "mean_evaluated_depth": 10, "max_evaluated_depth": 20,
                      "mean_branching_factor": 2.475, "candidate_diversity": {"mean_jaccard_distance": .5 if complete else None},
                      "depth_histogram": {"1": 3, "20": 3}, "branching_factor_distribution": {"1": 10, "3": 30}},
    }


def test_primary_accuracy_uses_paired_intersection_with_coverage():
    rows = [paired_row("medal", 0, correct=True), paired_row("ab_medal", 0, correct=False),
            paired_row("medal", 1, complete=False), paired_row("ab_medal", 1, correct=True)]
    summary = compare.summarize_attempts(rows, expected_pairs=2)
    medal, adaptive = summary["groups"]
    assert medal["paired_n"] == adaptive["paired_n"] == 1
    assert medal["paired_coverage"] == adaptive["paired_coverage"] == .5
    assert medal["paired_accuracy"] == 1 and adaptive["paired_accuracy"] == 0
    assert medal["target_coverage"] == .5 and adaptive["target_coverage"] == 1
    assert adaptive["conditional_accuracy"] == .5
    assert medal["mean_total_nfe"] == 164  # Includes the incomplete attempt's cost.
    assert medal["paired_mean_total_nfe"] == 200
    empty = compare.summarize_attempts([paired_row("medal", 0, False), paired_row("ab_medal", 0, False)])
    assert all(g["paired_accuracy"] is None and g["conditional_accuracy"] is None for g in empty["groups"])


def test_saved_result_replotting_adds_zero_forwards(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "load_inputs", lambda *a, **kw: pytest.fail("Offline report loaded a model"))
    source = tmp_path / "source"
    source.mkdir()
    rows = [paired_row("medal", 0), paired_row("ab_medal", 0, False)]
    (source / "attempts.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    assert compare.main(["--from-results", str(source), "--output-dir", str(tmp_path / "report")]) == 0
    report_dir = next((tmp_path / "report").iterdir())
    assert set(p.name for p in report_dir.glob("*.png")) >= {
        "accuracy_vs_nfe.png", "accuracy_vs_time.png", "expanded_nodes.png", "depth.png", "branching.png",
        "target_coverage.png", "initialization_cost.png", "structural_distributions.png"}
    assert json.loads((report_dir / "summary.json").read_text())["groups"][0]["paired_n"] == 0
    assert (report_dir / "attempts.csv").is_file()


def comparison_args(tmp_path):
    return Namespace(seeds=[1], budgets=[4, 7], experiment="all", target_max_nfe=None,
                     Lc=2, C=3, K1=2, K2=2, gen_length=4, temperature=1,
                     prior_m=0, prior_kappa=1, prior_nu=1, prior_tau2=.1,
                     search_policy="medal_equivalent", plot_max_nodes=12, plot_max_depth=3,
                     num_examples=2, start_index=0, output_dir=tmp_path,
                     demonstrations_path=tmp_path / "demos.json", cache_dir=None, quiet_history=True)


def test_comparison_shares_loaded_model_prompts_and_seeds_and_retains_partial_results(tmp_path, monkeypatch):
    instance, loads, prompts, attempts = wrapper(), [], [], []
    examples = [{"row_id": i, "example_id": f"test:{i}", "question": f"q{i}", "reference_solution": "#### 27"} for i in range(2)]
    def load(args):
        loads.append(args)
        return instance, {"examples": examples, "dataset": {"revision": "a" * 40}}, {}
    def prompt(wrapper, question, *a, **kw):
        prompts.append(question)
        return question, PROMPT.clone(), {"prompt_wall_seconds": .1, "setup_wall_seconds": .2}
    real_attempt = single.run_attempt
    def record_attempt(wrapper, ids, config, directory, **kwargs):
        attempts.append((id(wrapper.model), ids, kwargs["seed"], config.max_model_calls))
        if len(attempts) == 5:
            raise KeyboardInterrupt("controlled stop")
        return real_attempt(wrapper, ids, config, directory, **kwargs)
    monkeypatch.setattr(compare, "load_inputs", load)
    monkeypatch.setattr(compare, "prepare_demo_prompt", prompt)
    monkeypatch.setattr(compare, "run_attempt", record_attempt)
    directory, summary = compare.run_comparison(comparison_args(tmp_path))
    assert len(loads) == 1 and len({a[0] for a in attempts}) == 1
    assert prompts == ["q0"]  # One prepared tensor reused across all conditions.
    assert attempts[0][1] is attempts[1][1] and attempts[0][2] == attempts[1][2] == 1
    assert [a[3] for a in attempts[:4]] == [None, None, 4, 4]
    rows = [json.loads(line) for line in (directory / "attempts.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and summary["status"] == "interrupted"
    assert [r["status"] for r in rows] == ["complete", "complete", "incomplete", "incomplete"]
    assert all(r["decoding_nfe"] == 0 and r["accuracy"] is None for r in rows[2:])
    assert (directory / "summary.json").is_file()


def test_default_cuda_failure_happens_before_loading_or_fetching(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(single, "get_manifest", lambda *a, **kw: pytest.fail("Downloaded before checking CUDA"))
    with pytest.raises(RuntimeError, match="Colab GPU"):
        single.main([])


@pytest.mark.parametrize("script", ["run_ab_medal_single.py", "compare_medal_ab_medal.py"])
def test_script_help_outside_checkout(tmp_path, script):
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run([sys.executable, "-I", str(root / "scripts" / script), "--help"],
                               cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert "--Lc" in completed.stdout and "--prior-tau2" in completed.stdout
