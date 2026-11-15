"""Verify the copied notebook runs historical IG-KL with E2's batch size."""

import ast
import csv
from importlib.machinery import ModuleSpec
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import nbformat
import pytest

from dlm_project.tasks.gsm8k import ig_kl_evaluation as evaluation


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks/gsm8k/Experiment_2_Eval_IG_KL.ipynb"


def notebook_sources():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    return notebook, [cell.source for cell in notebook.cells if cell.cell_type == "code"]


def literal_assignments(sources):
    literals = {}
    for source in sources:
        for statement in ast.parse(source).body:
            if isinstance(statement, ast.Assign):
                try:
                    value = ast.literal_eval(statement.value)
                except (ValueError, TypeError):
                    continue
                for target in statement.targets:
                    if isinstance(target, ast.Name):
                        literals[target.id] = value
    return literals


def mock_display(monkeypatch):
    """UI is external to these offline execution checks."""
    ipython = ModuleType("IPython")
    ipython.__spec__ = ModuleSpec("IPython", loader=None)
    ipython.get_ipython = lambda: None
    ipython.version_info = (9, 0, 0)
    display_module = ModuleType("IPython.display")
    display_module.display = lambda *args, **kwargs: None
    display_module.Image = lambda *args, **kwargs: None
    display_module.Markdown = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)


def test_notebook_is_valid_compiles_and_has_no_saved_results():
    notebook, sources = notebook_sources()
    nbformat.validate(notebook)
    for i, source in enumerate(sources):
        compile(source, f"{NOTEBOOK}:cell_{i}", "exec")
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None
            assert cell.outputs == []


def test_notebook_uses_ig_kl_defaults_original_batch_size_and_its_own_branch():
    notebook, sources = notebook_sources()
    literals = literal_assignments(sources)
    original = nbformat.read(ROOT / "notebooks/gsm8k/Experiment_2_Eval.ipynb", as_version=4)
    original_literals = literal_assignments(cell.source for cell in original.cells if cell.cell_type == "code")
    assert literals["BRANCH"] == "sean-1114"
    assert literals["REWARD_MODE"] == "ig_kl"
    assert literals["KL_WEIGHT"] == 0.05
    assert literals["KL_TOP_K"] == 32
    assert literals["BATCH_SIZE"] == original_literals["BATCH_SIZE"] == 2
    assert tuple(literals["METHODS"]) == ("medal", "ab_medal")
    assert tuple(literals["BUDGETS"]) == (256, 512)
    assert literals["RESUME"] is True
    joined = "\n".join(sources)
    assert "run_e1_stage(" in joined
    assert "run_e2_stage(" not in joined
    assert "E2MEDALSearch" not in joined and "E2ABMEDALSearch" not in joined
    assert "dlm_project.tasks.gsm8k.ig_kl_evaluation" in joined
    badge = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    assert "blob/sean-1114/notebooks/gsm8k/Experiment_2_Eval_IG_KL.ipynb" in badge


def test_reward_settings_reach_every_notebook_context_call():
    _, sources = notebook_sources()
    calls = [node for source in sources for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "context"]
    assert len(calls) >= 3  # model setup, canonical 20, additional 50
    for call in calls:
        arguments = {keyword.arg: keyword.value for keyword in call.keywords}
        for name, variable in (("reward_mode", "REWARD_MODE"),
                               ("kl_weight", "KL_WEIGHT"), ("kl_top_k", "KL_TOP_K")):
            assert isinstance(arguments[name], ast.Name)
            assert arguments[name].id == variable


def test_actual_notebook_cells_run_twenty_plus_fifty_with_one_model(tmp_path, monkeypatch):
    """Execute controls, both stages, comparison CSVs and combined E3 routing."""
    _, sources = notebook_sources()
    loads, runs, e3_calls = [], [], []
    wrapper = SimpleNamespace(identity={"family": "llada", "repo_id": "fixture", "revision": "fixture"})
    model_dir = tmp_path / "existing-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    monkeypatch.setenv("MODEL_PATH", str(model_dir))

    def load_model(ctx, **kwargs):
        loads.append((ctx, kwargs))
        return wrapper

    def run_stage(model, ctx, output_dir, **kwargs):
        assert model is wrapper
        rows = []
        for example in ctx["examples"]:
            for method in kwargs["methods"]:
                (output_dir / "evaluation" / method / example["example_id"]).mkdir(parents=True)
                for budget in kwargs["budgets"]:
                    rows.append({
                        "example_id": example["example_id"], "method": method,
                        "search_budget": budget, "exact_match": method == "ab_medal",
                        "total_nfes": budget + 246, "selected_ig_reward": 0.2,
                        "selected_action_history_kl": 0.1, "selected_kl_penalty": 0.005,
                        "kl_extra_model_calls": 0,
                    })
                    checkpoint = output_dir / "evaluation" / method / example["example_id"] / "checkpoints" / str(budget)
                    checkpoint.mkdir(parents=True)
                    for filename in ("metrics.json", "tree.json", "selection.json", "completion.json"):
                        (checkpoint / filename).write_text(json.dumps(rows[-1]))
        runs.append((ctx, output_dir, kwargs, rows))
        return rows, []

    def analyze_e3_stage(**kwargs):
        e3_calls.append(kwargs)
        for method in kwargs["methods"]:
            directories = list((kwargs["results_dir"] / method).iterdir())
            assert len(directories) == 70
            assert all(directory.is_dir() for directory in directories)
            assert all(directory.resolve().exists() for directory in directories)
        return [row for _, _, _, rows in runs for row in rows], []

    monkeypatch.setattr(evaluation, "load_model", load_model)
    monkeypatch.setattr(evaluation, "run_e1_stage", run_stage)
    monkeypatch.setattr("dlm_project.tasks.gsm8k.e3_analysis.analyze_e3_stage", analyze_e3_stage)

    # Display is an external UI dependency; all experiment and CSV logic is real.
    mock_display(monkeypatch)

    scope = {"IN_COLAB": False, "REPO_DIR": tmp_path}
    selected = [source for source in sources if any(marker in source for marker in (
        "OUTPUT_BASE =", "base_ctx = context(", "q_rows_20, agg_rows_20 = run_e1_stage(",
        "q_rows_50, agg_rows_50 = run_e1_stage(", "paired_combined = compute_paired_comparison(",
        "combined_results_dir =",
    ))]
    assert len(selected) == 6
    for source in selected:
        exec(source, scope)

    assert len(loads) == 1
    assert len(loads[0][0]["examples"]) == 70
    assert loads[0][1] == {"model_path": str(model_dir), "device": "cuda", "local_files_only": True}
    assert [len(ctx["examples"]) for ctx, _, _, _ in runs] == [20, 50]
    assert [path.name for _, path, _, _ in runs] == ["canonical_20", "extended_50"]
    all_ids = [ex["example_id"] for ctx, _, _, _ in runs for ex in ctx["examples"]]
    assert len(all_ids) == len(set(all_ids)) == 70
    output_root = tmp_path / "results/e2_eval_ig_kl/lambda_0.05_topk_32/batch_2"
    for ctx, path, kwargs, _ in runs:
        assert path.parent == output_root
        assert kwargs == {
            "methods": ("medal", "ab_medal"), "budgets": (256, 512),
            "max_depth": 10, "num_candidates": 3, "resume": True, "batch_size": 2,
        }
        assert ctx["reward"] == evaluation.context()["reward"]
    assert len(scope["combined_q_rows"]) == 280
    assert [row["paired_questions"] for row in scope["paired_combined"]] == [70, 70]
    paired_file = output_root / "summaries/e2_ig_kl_combined_70_paired_comparison.csv"
    with paired_file.open(newline="") as stream:
        assert [int(row["paired_questions"]) for row in csv.DictReader(stream)] == [70, 70]
    assert len(e3_calls) == 1
    assert e3_calls[0]["results_dir"] == output_root / "combined_70/evaluation"


def test_combined_report_rejects_partial_results(tmp_path, monkeypatch):
    _, sources = notebook_sources()
    source = next(source for source in sources if "paired_combined = compute_paired_comparison(" in source)
    canonical = evaluation.context(sample_slice=(0, 20))
    extended = evaluation.context(sample_slice=(20, 70))
    partial_rows = [
        {"example_id": ex["example_id"], "method": method, "search_budget": budget,
         "exact_match": False, "total_nfes": 256}
        for ex in canonical["examples"] for method in ("medal", "ab_medal") for budget in (256, 512)
    ]
    mock_display(monkeypatch)
    scope = {"OUTPUT_DIR": tmp_path, "ctx_20": canonical, "ctx_50": extended,
             "METHODS": ("medal", "ab_medal"), "BUDGETS": (256, 512),
             "q_rows_20": partial_rows, "q_rows_50": []}
    with pytest.raises(AssertionError, match="all 70 questions"):
        exec(source, scope)
    assert not (tmp_path / "summaries/e2_ig_kl_combined_70_paired_comparison.csv").exists()


def test_combined_e3_rejects_missing_checkpoint_artifacts(tmp_path, monkeypatch):
    _, sources = notebook_sources()
    source = next(source for source in sources if "combined_results_dir =" in source)
    mock_display(monkeypatch)
    calls = []
    monkeypatch.setattr("dlm_project.tasks.gsm8k.e3_analysis.analyze_e3_stage",
                        lambda **kwargs: calls.append(kwargs))
    scope = {"OUTPUT_DIR": tmp_path, "ctx_20": evaluation.context(sample_slice=(0, 20)),
             "ctx_50": evaluation.context(sample_slice=(20, 70)),
             "METHODS": ("medal", "ab_medal"), "BUDGETS": (256, 512)}
    with pytest.raises(RuntimeError, match="Finish both evaluation parts"):
        exec(source, scope)
    assert calls == []
