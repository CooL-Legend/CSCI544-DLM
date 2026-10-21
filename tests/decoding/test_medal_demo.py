"""The interactive demo reuses one initialization and measures actual forwards."""

import json
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
import torch

from dlm_project.decoding import medal_demo as demo
from dlm_project.decoding.mcts_decoding import MCTSConfig
from dlm_project.decoding.decoder_result import CountingModel
from dlm_project.decoding.medal_no_search import decode as decode_medal_no_search
from dlm_project.experiments.config import load_config
from dlm_project.experiments.manifests import digest
from dlm_project.models.profiles import get_profile
from dlm_project.tasks.gsm8k.e0_tokens import decode_output, resolve_termination_tokens
from tests.decoding.helpers import FakeModel, FakeTokenizer, MASK, PROMPT
from tests.tasks.test_e0_dataset import Tokenizer as E0Tokenizer


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


def initialize(model=None):
    return demo.run_initialization(
        model or FakeModel(), PROMPT, MASK, gen_length=4, seed=7,
        mcts=MCTSConfig(init_tokens=2, num_candidates=3, top_k_tokens=2, top_k_actions=2),
        tokenizer=FakeTokenizer(),
    )


def test_cached_candidate_reports_cost_no_forwards_and_explain_entropy(capsys):
    initial = initialize()
    model = initial.search.model
    calls = len(model.seen)
    rows = demo.candidate_summaries(initial)
    assert len(rows) == 3
    assert rows[0]["selected"]
    assert [row["value"] for row in rows] == sorted((n.mean_reward for n in initial.result.candidates), reverse=True)
    root_entropy = initial.search.root.analysis.total_entropy
    for row in rows:
        assert len(row["tokens"]) == 2
        assert row["entropy_reduction"] == pytest.approx(root_entropy - row["entropy"])
        assert row["relative_entropy_reduction"] == pytest.approx(0.5)
        assert row["value"] == pytest.approx(1 / 3)  # Last edge, not the root-relative 1/2.
        assert all(token["token_id"] != MASK for token in row["tokens"])
    demo.show_candidates(initial)
    demo.show_metrics(initial)
    assert len(model.seen) == calls == 7
    report = capsys.readouterr().out
    assert "C1 [selected]" in report and "C2:" in report and "C3:" in report
    assert "root-to-candidate reduction" in report
    assert "Selection" in report and "Backpropagation" in report
    assert "Total NFE: 7" in report


def test_medal_completion_reuses_initialization_and_is_repeatable(capsys):
    initial = initialize()
    saved = initial.state.clone()
    calls = len(initial.search.model.seen)
    first = demo.run_completion(initial, seed=7)
    second = demo.run_completion(initial, seed=7)
    assert first.model_calls == second.model_calls == 2
    assert len(initial.search.model.seen) == calls + 4
    assert torch.equal(initial.state, saved)
    assert torch.equal(first.token_ids, second.token_ids)
    assert torch.equal(first.token_ids[:, :2], PROMPT)
    assert not (first.token_ids[:, 2:] == MASK).any()
    demo.show_metrics(initial, second)
    assert len(initial.search.model.seen) == calls + 4
    assert "Total NFE: 9" in capsys.readouterr().out  # One search plus this completion.


def test_custom_module_sampler_gets_a_copy_and_cannot_misreport_nfe(tmp_path, monkeypatch):
    (tmp_path / "custom_sampler.py").write_text('''from types import SimpleNamespace

def decode(model, state, prompt_len, mask_id, config, *, seed, temperature, tokenizer):
    model(state.to(model.device))
    output = state[:, prompt_len:]
    output[output == mask_id] = 3
    return SimpleNamespace(token_ids=state, model_calls=999)
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    initial = initialize()
    saved = initial.state.clone()
    before = len(initial.search.model.seen)
    completion = demo.run_completion(initial, method="custom_sampler:decode")
    assert completion.model_calls == 1
    assert len(initial.search.model.seen) == before + 1
    assert torch.equal(initial.state, saved)
    assert torch.equal(completion.token_ids[:, :2], PROMPT)
    assert completion.trace == []


@pytest.mark.parametrize("failure", ["prompt", "commitment", "masks", "shape"])
def test_custom_sampler_output_checks_stay_inside_helper(failure):
    initial = initialize()
    saved = initial.state.clone()

    def sampler(model, state, prompt_len, mask_id, config, **kwargs):
        if failure == "shape":
            return SimpleNamespace(token_ids=state[:, :-1])
        if failure == "masks":
            return SimpleNamespace(token_ids=state)
        output = state[:, prompt_len:]
        output[output == mask_id] = 1
        if failure == "prompt":
            state[0, 1] = 2  # A mask in the prompt must also remain untouched.
        else:
            position = torch.nonzero(saved[0, prompt_len:] != mask_id)[0].item()
            state[0, prompt_len + position] = 3
        return SimpleNamespace(token_ids=state)

    with pytest.raises(ValueError, match="sampler"):
        demo.run_completion(initial, method=sampler)
    assert torch.equal(initial.state, saved)


def test_entropy_increases_are_not_clamped():
    model = FakeModel(lambda output: [0, .999, .0005, .0005] if (output == MASK).all()
                      else [0, 1 / 3, 1 / 3, 1 / 3])
    initial = initialize(model)
    assert all(row["entropy_reduction"] < 0 for row in demo.candidate_summaries(initial))


def test_prompt_reuses_frozen_training_examples_offline(demonstrations_path, monkeypatch, capsys):
    import datasets
    import huggingface_hub

    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **kw: pytest.fail("Unexpected dataset load"))
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: pytest.fail("Unexpected network access"))
    prompt, metadata = demo.prepare_prompt("A new question?", demonstrations_path)
    assert "Training example 0" in prompt and "Training example 1" in prompt
    assert "A new question?" in prompt
    assert metadata["planning_model_calls"] == 0
    assert metadata["demonstrations_dataset"]["split"] == "train"
    demo.show_prompt("A new question?")
    assert "Planning NFE: 0" in capsys.readouterr().out


def test_cli_runs_one_question_without_an_evaluation(demonstrations_path, tmp_path, monkeypatch, capsys):
    model, tokenizer = FakeModel(), FakeTokenizer()
    wrapper = SimpleNamespace(model=model, tokenizer=tokenizer, mask_id=MASK,
                              encode_prompt=lambda text: tokenizer(text).input_ids)
    loads = []
    monkeypatch.setattr(demo, "load_demo_model", lambda **settings: loads.append(settings) or wrapper)
    question_file = tmp_path / "question.txt"
    question_file.write_text("A custom question loaded from disk?")
    monkeypatch.setattr("sys.argv", [
        "demo_medal.py", "--model-path", "/drive/model", "--local-files-only",
        "--prompt-file", str(question_file), "--demonstrations-path", str(demonstrations_path),
        "--gen-length", "4", "--init-tokens", "2", "--top-k-tokens", "2", "--top-k-actions", "2",
    ])
    demo.main()
    assert len(loads) == 1 and loads[0]["model_path"] == "/drive/model"
    assert loads[0]["local_files_only"]
    assert len(model.seen) == 9
    output = capsys.readouterr().out
    assert "A custom question loaded from disk?" in output
    assert "Generated response:" in output and "Total NFE: 9" in output
    assert "accuracy" not in output.lower()


@pytest.mark.parametrize("question", ["A custom question?", demo.DEFAULT_QUESTION])
@pytest.mark.parametrize("budget", [1, 12])
def test_notebook_action_cells_run_with_one_model(demonstrations_path, tmp_path, question, budget):
    root = Path(__file__).resolve().parents[2]
    notebook = nbformat.read(root / "notebooks/gsm8k/MEDAL_MCTS_Demo.ipynb", as_version=4)
    nbformat.validate(notebook)
    assert [cell.id for cell in notebook.cells] == [
        "0fb94d89", "5a532510", "05cd374e", "2bda1f2f", "b0cdc487", "422b92ec", "9b56c23c",
        "3e50fa7b", "8e2fb8a6", "1a98942a", "cc6e8808", "46d5d403",
        "medal-no-search-heading", "medal-no-search-decode", "medal-no-search-output", "medal-no-search-metrics",
        "292fe0aa", "ec0e6744",
        "5a5fdb7d", "02b7b325", "f0d6e2ee", "9faf852b", "e71f242e", "2b576fd9", "score-answer",
        "7741b4d8", "3ae77c20", "runtime-explanation", "runtime-costs", "batch-description",
        "batch-settings", "run-batch", "batch-summary",
    ]
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            assert cell.outputs == [] and cell.execution_count is None
            compile(cell.source, f"cell-{index}", "exec")
    # Execute the actual action cells with a controlled model, not copied orchestration.
    model, tokenizer = FakeModel(lambda _: [0, .45, .35, .1, .06, .04]), E0Tokenizer()
    prompts = []
    wrapper = SimpleNamespace(model=model, tokenizer=tokenizer, mask_id=MASK,
                              encode_prompt=lambda text: prompts.append(text) or PROMPT.clone())
    loads = []
    scope = {name: getattr(demo, name) for name in (
        "prepare_demo_prompt", "show_prompt", "run_initialization", "show_candidates", "show_metrics", "run_completion",
        "score_answer", "show_no_search_metrics")}
    scope.update(Path=Path, REPO_DIR=root, get_profile=get_profile, load_config=load_config,
                 MCTSConfig=MCTSConfig, DEFAULT_QUESTION=demo.DEFAULT_QUESTION, DRIVE_ROOT=tmp_path,
                 run_demo_batch=lambda *a, **kw: pytest.fail("Disabled batch must not run"),
                 DEMONSTRATIONS_PATH=demonstrations_path, os=SimpleNamespace(environ={"HF_DATASETS_CACHE": "cache"}),
                 resolve_termination_tokens=resolve_termination_tokens,
                 decode_medal_no_search=decode_medal_no_search, CountingModel=CountingModel, decode_output=decode_output,
                 load_demo_model=lambda **kw: loads.append(kw) or wrapper)
    exec(next(cell.source for cell in notebook.cells if cell.id == "3e50fa7b"), scope)
    assert scope["SEARCH_BUDGET"] == 512 and scope["MCTS_SETTINGS"].init_tokens == 5
    assert scope["COMPLETION_METHOD"] == "llada"
    assert scope["TEMPERATURE"] == 0.0
    assert scope["MCTS_SETTINGS"].stop_mode == "budget" and scope["MCTS_SETTINGS"].max_simulations is None
    scope.update(MODEL_PATH="/drive/model", QUESTION=question, GEN_LENGTH=8, SEED=7,
                 MCTS_SETTINGS=MCTSConfig(init_tokens=2, num_candidates=3, top_k_tokens=2, top_k_actions=3,
                                          stop_mode="budget", max_model_calls=budget))
    start = next(i for i, cell in enumerate(notebook.cells) if cell.source.startswith("llada = load_demo_model"))
    for cell in notebook.cells[start:]:
        if cell.cell_type == "code":
            assert len(cell.source.splitlines()) <= 5
            if cell.id == "ec0e6744":
                assert "initialization" not in scope
                assert scope["no_search"].total_nfes == len(model.seen) == scope["GEN_LENGTH"]
                assert scope["no_search_metrics"]["search_nfes"] == 0
            exec(cell.source, scope)
    assert len(loads) == 1
    assert len(model.seen) == scope["no_search"].total_nfes + scope["metrics"]["total_nfe"]
    assert scope["metrics"]["total_nfe"] == scope["metrics"]["total_nfes"]
    assert scope["initialization"].result.stats["search_model_calls"] == budget
    assert scope["completion"].model_calls == 8 - scope["metrics"]["selected_depth"]
    assert scope["prompt"] == prompts[0] and scope["prompt"].endswith(f"Problem: {question}")
    assert "Example 1:" in scope["prompt"] and "Example 2:" in scope["prompt"]
    assert scope["response"] == scope["completion"].decoded_output["scoring_text"]
    assert scope["completion"].method == scope["no_search"].details["completion_method"] == "llada"
    assert scope["no_search"].details["temperature"] == 0.0
    assert all(node.action.token_id not in scope["TERMINATION_IDS"] for node in scope["initialization"].search.nodes[1:])
    assert scope["metrics"]["fallback_decode_used"] == (budget == 1)
    if budget == 1:
        assert torch.equal(scope["no_search"].token_ids, scope["completion"].token_ids)
    assert scope["no_search_score"]["reference_answer"] == scope["score"]["reference_answer"]
    assert scope["score"]["reference_answer"] == ("27" if question == demo.DEFAULT_QUESTION else None)
    assert scope["RUN_BATCH"] is False


def test_prompt_timing_separates_setup_and_adds_no_forwards(demonstrations_path, monkeypatch):
    model = FakeModel()
    wrapper = SimpleNamespace(model=model, encode_prompt=lambda text: PROMPT.clone())
    clock = iter([10.0, 12.0, 15.0, 18.0])
    monkeypatch.setattr(demo, "perf_counter", lambda: next(clock))
    prompt, ids, metadata = demo.prepare_demo_prompt(wrapper, "A new question?", demonstrations_path)
    assert "A new question?" in prompt and torch.equal(ids, PROMPT)
    assert metadata["setup_wall_seconds"] == 3.0
    assert metadata["prompt_wall_seconds"] == 5.0
    assert metadata["planning_model_calls"] == 0 and model.seen == []


def test_stage_report_charges_nested_filtering_once_and_reconciles_totals():
    initial = initialize()
    completion = demo.run_completion(initial)
    initial.wall_seconds, completion.wall_seconds = 20.0, 3.0
    initial.result.stats.update(tree_analysis_latency_seconds=1.0, simulation_latency_seconds=7.0,
                                candidate_filtering_latency_seconds=2.0, selection_latency_seconds=.5,
                                expansion_latency_seconds=.25, backpropagation_latency_seconds=.75,
                                candidate_evaluation_latency_seconds=.5)
    report = demo.show_metrics(initial, completion, prompt_seconds=2.0, setup_seconds=99.0, display=False)
    stages = {row["substage"]: row for row in report["stages"]}
    assert stages["Candidate generation / filtering"]["wall_seconds"] == 3.0
    assert stages["Child evaluation / r_IG"]["wall_seconds"] == 5.0
    assert stages["Bookkeeping / trace overhead"]["wall_seconds"] == 10.0
    assert stages["Full rollout (disabled)"]["wall_seconds"] == stages["Full rollout (disabled)"]["nfe"] == 0
    assert sum(row["nfe"] for row in report["stages"]) == report["total_nfe"] == 9
    assert sum(row["wall_seconds"] for row in report["stages"]) == report["total_wall_seconds"] == 25.0
    assert report["initialization_nfe"] == 7 and report["decoding_nfe"] == 2
    assert report["model_evaluations"] == len(initial.search.model.seen) == 9
    assert report["tree"]["expanded_parents"] == 3
    assert report["tree"]["evaluated_children"] == 6
    assert report["tree"]["unique_tree_states"] == 6
    assert report["tree"]["average_search_depth"] == pytest.approx(10 / 6)
    assert report["tree"]["max_evaluated_depth"] == 2
    assert report["tree"]["realized_branching_factor"] == 2
    assert demo.show_metrics(initial, completion, prompt_seconds=2.0, setup_seconds=99.0, display=False) == report
    assert len(initial.search.model.seen) == 9
    json.dumps(report, allow_nan=False)


def test_budget_root_only_has_zero_tree_depth_and_nonnegative_exclusive_times():
    initial = demo.run_initialization(FakeModel(), PROMPT, MASK, gen_length=4,
                                     mcts=MCTSConfig(init_tokens=2, stop_mode="budget", max_model_calls=1))
    report = demo.show_metrics(initial, display=False)
    assert report["total_nfe"] == 1
    assert report["tree"]["max_evaluated_depth"] == report["tree"]["expanded_parents"] == 0
    assert report["tree"]["average_search_depth"] == report["tree"]["realized_branching_factor"] == 0
    assert report["tree"]["stop_reason"] == "model_call_budget"
    assert all(row["wall_seconds"] >= 0 for row in report["stages"])
    assert sum(row["wall_seconds"] for row in report["stages"]) == pytest.approx(initial.wall_seconds)


@pytest.mark.parametrize("text,reference,accuracy", [
    ("#### 27.0", "#### 27", 1.0), ("#### 28", "#### 27", 0.0),
    ("Answer: 27", "#### 27", 0.0), ("Answer: 27", None, None),
])
def test_report_scores_only_supplied_final_answers_without_inference(text, reference, accuracy):
    initial = initialize()
    score = demo.score_answer(text, reference)
    report = demo.show_metrics(initial, score=score, display=False)
    assert report["accuracy"] == accuracy
    assert report["score"] == score
    assert len(initial.search.model.seen) == 7


@pytest.fixture
def batch_inputs(demonstrations_path, monkeypatch):
    from dlm_project.tasks.gsm8k import benchmark_dataset

    manifest = {"dataset": {"revision": "a" * 40}, "examples": [
        {"example_id": f"test:{i}", "row_id": i, "question": f"Held-out question {i}?",
         "reference_solution": "REFERENCE_ONLY\n#### 2"} for i in range(3)]}
    requests = []
    monkeypatch.setattr(benchmark_dataset, "get_manifest",
                        lambda path, **kw: requests.append((path, kw)) or manifest)
    model, tokenizer, prompts = FakeModel(), FakeTokenizer(), []
    decode_token = tokenizer.decode
    outputs = iter(["#### 2", "Answer: 2", "#### 3"])
    tokenizer.decode = lambda ids, **kw: next(outputs) if kw.get("skip_special_tokens") else decode_token(ids, **kw)
    wrapper = SimpleNamespace(model=model, tokenizer=tokenizer, mask_id=MASK,
                              encode_prompt=lambda text: prompts.append(text) or PROMPT.clone())
    return wrapper, prompts, requests


def test_optional_batch_reuses_model_pinned_first_rows_and_saves_metrics(batch_inputs, demonstrations_path, tmp_path):
    wrapper, prompts, requests = batch_inputs
    directory, summary = demo.run_demo_batch(
        wrapper, demonstrations_path, tmp_path / "gsm8k_test.json", tmp_path / "results",
        num_samples=2, gen_length=4, seed=7,
        mcts=MCTSConfig(init_tokens=2, num_candidates=3, top_k_tokens=2, top_k_actions=2))
    records = [json.loads(line) for line in (directory / "predictions.jsonl").read_text().splitlines()]
    assert [row["example_id"] for row in records] == ["test:0", "test:1"]
    assert [row["seed"] for row in records] == [7, 12]
    assert requests[0][1]["revision"] == "a" * 40
    assert len(wrapper.model.seen) == summary["total_nfe"] == 18
    assert summary["status"] == "completed" and summary["processed"] == summary["requested"] == 2
    assert summary["correct"] == 1 and summary["accuracy"] == .5
    assert summary["mean_nfe"] == 9 and summary["mean_wall_seconds"] == summary["total_wall_seconds"] / 2
    assert records[1]["metrics"]["score"]["parse_success"] is False
    assert all("REFERENCE_ONLY" not in prompt for prompt in prompts)
    assert summary["mean_tree"]["expanded_parents"] == 3
    assert summary["max_evaluated_depth"] == 2
    assert sum(row["nfe"] for row in summary["stages"]) == 18
    assert sum(row["wall_seconds"] for row in summary["stages"]) == pytest.approx(summary["total_wall_seconds"])
    assert json.loads((directory / "summary.json").read_text()) == summary
    assert json.loads((directory / "config.json").read_text())["dataset"]["revision"] == "a" * 40


@pytest.mark.parametrize("error,status", [(KeyboardInterrupt, "interrupted"), (RuntimeError, "failed")])
def test_batch_keeps_completed_records_on_interruption_or_failure(
        batch_inputs, demonstrations_path, tmp_path, monkeypatch, error, status):
    wrapper, _, _ = batch_inputs
    original = demo.run_completion
    calls = []

    def complete(*args, **kwargs):
        calls.append(None)
        if len(calls) == 2:
            raise error("Stopped during second example")
        return original(*args, **kwargs)

    monkeypatch.setattr(demo, "run_completion", complete)
    with pytest.raises(error):
        demo.run_demo_batch(wrapper, demonstrations_path, tmp_path / "gsm8k_test.json", tmp_path / "results",
                            num_samples=2, gen_length=4,
                            mcts=MCTSConfig(init_tokens=2, top_k_tokens=2, top_k_actions=2))
    summary_path = next((tmp_path / "results").rglob("summary.json"))
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == status and summary["processed"] == 1
    assert summary["requested"] == 2 and summary["accuracy"] == 1
    assert summary["total_nfe"] == 9
    assert summary["error"].startswith(error.__name__)
    assert len((summary_path.parent / "predictions.jsonl").read_text().splitlines()) == 1
