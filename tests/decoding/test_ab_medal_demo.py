"""The staged AB notebook shares MEDAL reports without changing AB decisions."""

import json
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
import torch

from dlm_project.decoding import ab_medal_demo as ab
from dlm_project.decoding import medal_demo as demo
from dlm_project.decoding.ab_medal import ABMEDALConfig, ABMEDALSearch
from dlm_project.experiments.manifests import digest
from tests.decoding.helpers import FakeModel, FakeTokenizer, MASK, PROMPT
from tests.decoding.test_ab_medal import finite_tree


@pytest.fixture
def demonstrations_path(tmp_path):
    examples = []
    for index in range(2):
        content = {"question": f"Training example {index}?", "reference_solution": "1 + 1 = 2.\n#### 2"}
        examples.append({"row_id": index, **content, "content_hash": digest(content)})
    path = tmp_path / "gsm8k_decomposition_train.json"
    path.write_text(json.dumps({
        "version": "medal_inline_v1",
        "dataset": {"repo_id": "openai/gsm8k", "subset": "main", "split": "train", "revision": "a" * 40},
        "examples": examples,
    }))
    return path


def settings(**kwargs):
    return ABMEDALConfig(**{"init_tokens": 2, "num_candidates": 3,
                            "top_k_tokens": 2, "top_k_actions": 2, **kwargs})


def initialize(**kwargs):
    return ab.run_initialization(FakeModel(), PROMPT, MASK, gen_length=4,
                                 mcts=settings(**kwargs), seed=7, tokenizer=FakeTokenizer())


def wrapper(model=None):
    return SimpleNamespace(model=model or FakeModel(), tokenizer=FakeTokenizer(), mask_id=MASK,
                           identity={"test_model": True}, encode_prompt=lambda text: PROMPT.clone())


def test_staging_preserves_gaussian_search_and_cached_reports(capsys):
    initial = initialize()
    direct = ABMEDALSearch(FakeModel(), PROMPT, MASK, 4, settings(), tokenizer=FakeTokenizer(), seed=7)
    result = direct.run()
    assert isinstance(initial, demo.InitializationResult)
    assert initial.search.history == direct.history
    assert initial.result.trace == result.trace
    assert initial.result.selected.node_id == result.selected.node_id
    assert torch.equal(initial.state, result.selected.state)
    assert initial.search.rng.bit_generator.state == direct.rng.bit_generator.state
    assert len(initial.search.model.seen) == direct.model_calls == 7

    rng_before = json.dumps(initial.search.rng.bit_generator.state)
    rows = demo.candidate_summaries(initial)
    demo.show_candidates(initial)
    report = demo.show_metrics(initial, prompt_seconds=.2, setup_seconds=.4)
    assert rows[0]["selected"] and len(rows) == 3
    assert all(len(row["tokens"]) == 2 for row in rows)
    assert report["total_nfe"] == 7
    assert sum(row["nfe"] for row in report["stages"]) == 7
    assert sum(row["wall_seconds"] for row in report["stages"]) == pytest.approx(report["total_wall_seconds"])
    assert all(row["wall_seconds"] >= 0 for row in report["stages"])
    assert json.dumps(initial.search.rng.bit_generator.state) == rng_before
    assert len(initial.search.model.seen) == 7
    output = capsys.readouterr().out
    assert "AB initialization complete" in output
    assert "C1 [selected]" in output and "root-to-candidate reduction" in output
    assert "Selection" in output and "Backpropagation" in output and "Total NFE: 7" in output


def test_completion_reuses_state_seed_checks_and_counts_only_current_trial():
    initial = initialize()
    saved = initial.state.clone()
    first = ab.run_completion(initial, seed=7)
    second = ab.run_completion(initial, seed=7)
    assert torch.equal(first.token_ids, second.token_ids)
    assert torch.equal(saved, initial.state)
    assert torch.equal(second.token_ids[:, :PROMPT.shape[1]], PROMPT)
    assert first.model_calls == second.model_calls == 2
    assert len(initial.search.model.seen) == 11
    assert demo.show_metrics(initial, second, display=False)["total_nfe"] == 9

    def corrupt_prompt(model, state, prompt_len, mask_id, config, **kwargs):
        state[state == mask_id] = 1
        return SimpleNamespace(token_ids=state)

    with pytest.raises(ValueError, match="prompt or an MCTS commitment"):
        ab.run_completion(initial, method=corrupt_prompt)
    assert torch.equal(saved, initial.state)


@pytest.mark.parametrize("cap,candidates", [(1, 0), (2, 0), (4, 1)])
def test_incomplete_search_is_inspectable_and_never_calls_sampler(cap, candidates, monkeypatch, capsys):
    initial = initialize(max_model_calls=cap)
    assert initial.result.stats["target_reached"] is False
    assert initial.result.stats["stop_reason"] == "model_call_budget"
    assert len(initial.result.candidates) == candidates
    monkeypatch.setattr(demo, "run_completion", lambda *a, **kw: pytest.fail("Incomplete search decoded"))
    assert ab.run_completion(initial) is None
    demo.show_candidates(initial)
    report = demo.show_metrics(initial, display=False)
    assert report["total_nfe"] == len(initial.search.model.seen) == cap
    assert report["decoding_nfe"] == 0 and report["score"] is report["accuracy"] is None
    assert "Completion and scoring skipped" in capsys.readouterr().out


def execute_notebook(demonstrations_path, tmp_path, *, question=demo.DEFAULT_QUESTION,
                     reference=None, use_file=False, cap=None, monkeypatch=None):
    notebook = nbformat.read(Path(__file__).resolve().parents[2] /
                             "notebooks/gsm8k/AB_MEDAL_MCTS_Colab.ipynb", as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None
            compile(cell.source, f"cell-{index}", "exec")
    instance, loads = wrapper(), []
    scope = {name: getattr(demo, name) for name in (
        "DEFAULT_QUESTION", "prepare_demo_prompt", "show_prompt", "show_candidates", "show_metrics", "score_answer")}
    if cap is not None:
        scope["score_answer"] = lambda *a, **kw: pytest.fail("Incomplete search scored")
    prompt_file = tmp_path / "question.txt"
    if use_file:
        prompt_file.write_text(question, encoding="utf-8")
    scope.update(
        Path=Path, MODEL_PATH="/drive/model", DTYPE="bfloat16", QUESTION=question,
        PROMPT_FILE=prompt_file if use_file else None, REFERENCE_SOLUTION=reference,
        DRIVE_ROOT=tmp_path, DEMONSTRATIONS_PATH=demonstrations_path,
        os=SimpleNamespace(environ={"HF_DATASETS_CACHE": str(tmp_path / "cache")}),
        GEN_LENGTH=4, SEED=7, TEMPERATURE=1.0, MCTS_SETTINGS=settings(max_model_calls=cap),
        load_demo_model=lambda **kw: loads.append(kw) or instance,
        run_initialization=ab.run_initialization, run_completion=ab.run_completion,
        export_diagnostics=lambda *a, **kw: pytest.fail("Disabled diagnostics ran"),
        run_pilot=lambda *a, **kw: pytest.fail("Disabled pilot ran"),
    )
    start = next(i for i, c in enumerate(notebook.cells) if c.source.startswith("llada = load_demo_model"))
    initialization_calls = None
    for cell in notebook.cells[start:]:
        if cell.cell_type == "code":
            if cell.source.startswith("DECODING_METHOD ="):
                initialization_calls = len(instance.model.seen)
                assert scope["completion"] is None and scope["score"] is None
            exec(cell.source, scope)
    assert len(loads) == 1
    assert scope["RUN_DIAGNOSTICS"] is scope["RUN_PILOT"] is False
    assert initialization_calls == scope["metrics"]["initialization_nfe"]
    expected, _ = demo.prepare_prompt(question, demonstrations_path)
    assert scope["prompt"] == expected
    assert scope["decomposition"]["planning_model_calls"] == 0
    return scope, instance


@pytest.mark.parametrize("question,reference,use_file,expected_reference", [
    (demo.DEFAULT_QUESTION, None, False, "27"),
    ("Custom question?", None, False, None),
    ("Custom question?", "#### 42", False, "42"),
    ("Question from file?", "#### 8", True, "8"),
])
def test_actual_notebook_cells_share_prompting_and_scoring(demonstrations_path, tmp_path, monkeypatch,
                                                          question, reference, use_file, expected_reference):
    from dlm_project.tasks.gsm8k import benchmark_dataset
    monkeypatch.setattr(benchmark_dataset, "get_manifest", lambda *a, **kw: pytest.fail("Single demo loaded test set"))
    scope, instance = execute_notebook(demonstrations_path, tmp_path, question=question,
                                       reference=reference, use_file=use_file)
    assert scope["score"]["reference_answer"] == expected_reference
    assert scope["completion"].model_calls == 2
    assert scope["metrics"]["total_nfe"] == len(instance.model.seen) == 9


def test_actual_notebook_cells_skip_completion_and_scoring_when_capped(demonstrations_path, tmp_path):
    scope, instance = execute_notebook(demonstrations_path, tmp_path, cap=4)
    assert scope["completion"] is scope["response"] is scope["score"] is None
    assert scope["metrics"]["total_nfe"] == len(instance.model.seen) == 4
    assert scope["metrics"]["accuracy"] is None


@pytest.mark.parametrize("complete", [True, False])
def test_export_is_cached_and_contains_only_supplied_completion(tmp_path, complete):
    initial = initialize(**({} if complete else {"max_model_calls": 4}))
    instance = wrapper(initial.search.model)
    completion = None
    score = None
    if complete:
        ab.run_completion(initial, seed=7)  # An earlier trial is excluded from export.
        completion = ab.run_completion(initial, seed=7)
        score = demo.score_answer("#### 27", "#### 27")
    calls = len(instance.model.seen)
    rng_before = json.dumps(initial.search.rng.bit_generator.state)
    directory, report = ab.export_diagnostics(
        instance, initial, tmp_path, completion=completion, prompt="prompt", score=score,
        prompt_metadata={"prompt_wall_seconds": .2, "setup_wall_seconds": .4},
    )
    assert len(instance.model.seen) == calls
    assert json.dumps(initial.search.rng.bit_generator.state) == rng_before
    assert report["total_nfe"] == (9 if complete else 4)
    assert report["status"] == ("complete" if complete else "incomplete")
    assert report["accuracy"] == (1.0 if complete else None)
    assert (directory / "completion.json").exists() is complete
    assert (directory / "ab_medal_tree.png").read_bytes().startswith(b"\x89PNG")
    tree = json.loads((directory / "tree.json").read_text())
    assert "posteriors" in tree["nodes"][0]
    history = json.loads((directory / "history.json").read_text())
    assert history and history[0]["posterior_updates"]
    trace = [json.loads(line) for line in (directory / "trace.jsonl").read_text().splitlines()]
    assert sum(event["stage"] == "forward" for event in trace) == report["total_nfe"]
    assert json.loads((directory / "metrics.json").read_text())["total_nfe"] == report["total_nfe"]


def test_pilot_gate_precedes_manifest_loading(demonstrations_path, tmp_path, monkeypatch):
    from dlm_project.tasks.gsm8k import benchmark_dataset
    monkeypatch.setattr(benchmark_dataset, "get_manifest", lambda *a, **kw: pytest.fail("Failed gate loaded test set"))
    with pytest.raises(RuntimeError, match="full-target Gaussian DLM"):
        ab.run_pilot(wrapper(), initialize(), demonstrations_path, tmp_path / "test.json", tmp_path / "pilot")


@pytest.mark.parametrize("failure", [None, "interrupt", "error"])
def test_pilot_pins_revision_uses_ab_runner_and_keeps_partial_summary(
        demonstrations_path, tmp_path, monkeypatch, failure):
    from dlm_project.tasks.gsm8k import benchmark_dataset
    from dlm_project.tasks.gsm8k.benchmark import example_seed
    from scripts import run_ab_medal_single as single

    tree = finite_tree(monkeypatch, "shared")
    initial = demo.InitializationResult(tree, tree.run(), tree.initialization_seconds)
    instance = wrapper(tree.model)
    assert not ab.full_target_dlm_gate(instance, initial)
    instance.identity = {"family": "llada"}
    assert ab.full_target_dlm_gate(instance, initial)
    examples = [{"row_id": i, "example_id": f"test:{i}", "question": f"Pilot question {i}?",
                 "reference_solution": "#### 27"} for i in range(5)]
    manifest = {"dataset": {"revision": "a" * 40}, "examples": examples}
    manifest_calls, attempts = [], []
    monkeypatch.setattr(benchmark_dataset, "get_manifest",
                        lambda *a, **kw: manifest_calls.append((a, kw)) or manifest)

    def attempt(wrapper, ids, config, directory, **kwargs):
        attempts.append((wrapper, ids, config, kwargs))
        index = kwargs["metadata"]["row_id"]
        if index == 1 and failure:
            if failure == "interrupt":
                raise KeyboardInterrupt("controlled stop")
            raise RuntimeError("controlled failure")
        complete = index % 2 == 0
        return {"status": "complete" if complete else "incomplete", "error": None,
                "target_reached": complete, "accuracy": 1.0 if complete else None,
                "total_nfe": 24 if complete else 4, "total_wall_seconds": 1.0}

    monkeypatch.setattr(single, "run_attempt", attempt)
    directory, summary = ab.run_pilot(
        instance, initial, demonstrations_path, tmp_path / "test.json", tmp_path / "pilot",
        gen_length=21, seed=7,
    )
    assert manifest_calls[0][1]["revision"] == "a" * 40
    for index, (loaded, ids, config, kwargs) in enumerate(attempts):
        assert loaded is instance and config is tree.config and torch.equal(ids, PROMPT)
        assert kwargs["seed"] == example_seed(7, index)
        assert kwargs["reference"] == examples[index]["reference_solution"]
        assert kwargs["print_history"] is False
    expected_count = 1 if failure else 5
    assert summary["requested"] == 5 and summary["processed"] == expected_count
    assert summary["status"] == ({"interrupt": "interrupted", "error": "error"}.get(failure, "complete"))
    assert summary["conditional_accuracy"] == 1.0
    assert summary["target_coverage"] == (1 / 5 if failure else 3 / 5)
    assert len((directory / "attempts.jsonl").read_text().splitlines()) == expected_count
    assert json.loads((directory / "summary.json").read_text()) == summary
