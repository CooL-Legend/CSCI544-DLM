"""Wrapper, CLI driver, JSONL artifacts and notebook presentation on CPU."""

import copy
import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from dlm_project.models.llada import LLaDA
from dlm_project.experiments.paths import experiment_name
from dlm_project.tasks.sudoku.inspect_mcts import inspect_puzzle
from dlm_project.decoding.mcts_decoding import MCTSConfig, MCTSSearch, resolve_decoding
from dlm_project.experiments.config import load_config, validate_config
from dlm_project.tasks.sudoku.runner import run_experiment
from tests.decoding.helpers import FakeModel, FakeTokenizer, MASK, PROMPT

from dlm_project.paths import PROJECT_ROOT as ROOT


def wrapper():
    result = LLaDA.__new__(LLaDA)
    result.device, result.mask_id, result.family = "cpu", MASK, "llada"
    result.model, result.tokenizer = FakeModel(), FakeTokenizer()
    result.identity = {"test_model": True}
    return result


def tiny_config(tmp_path):
    config = load_config(ROOT / "configs/sudoku/llada_mcts.yaml")
    config["experiment_root"] = str(tmp_path)
    config["model"].update(device="cpu", dtype="float32")
    config["generation"].update(gen_length=4, steps=4, block_length=4)
    config["decoding"]["mcts"].update(init_tokens=2, num_candidates=3, top_k_tokens=2, top_k_actions=2, max_model_calls=None)
    return config


def source(notebook, cell_id):
    return "".join(next(c for c in notebook["cells"] if c["id"] == cell_id)["source"])


@pytest.mark.parametrize("method,search_calls,completion_calls", [
    ("baseline", 0, 4), ("confidence", 0, 4), ("mcts_entropy", 7, 2),
])
def test_prediction_modes_forward_counts_text_interface_and_trace(method, search_calls, completion_calls):
    instance = wrapper()
    sink = []
    prediction = instance.predict_with_stats(
        "prompt", steps=4, gen_length=4, block_length=4,
        temperature=1.0 if method=="mcts_entropy" else 0.0,
        decoding={"method": method, "mcts": {"init_tokens":2,"num_candidates":3,"top_k_tokens":2,"top_k_actions":2}}, on_event=sink.append,
    )
    stats = prediction["stats"]
    assert stats["search_model_calls"] == search_calls
    assert stats["completion_model_calls"] == completion_calls
    assert stats["total_model_calls"] == len(instance.model.seen) == search_calls + completion_calls
    assert stats["method"] == method
    assert stats["latency_seconds"] >= stats["search_latency_seconds"] + stats["completion_latency_seconds"]
    assert stats["peak_gpu_memory_bytes"] is None
    assert stats["model_device"] == "cpu"
    assert stats["search_state_device"] == ("cpu" if method == "mcts_entropy" else None)
    assert prediction["trace"] == sink
    assert sink[-1]["stage"] == "prediction"
    assert prediction["text"] == instance.predict("prompt", steps=4, gen_length=4, block_length=4,
                                                 temperature=1.0 if method=="mcts_entropy" else 0.0,
                                                 decoding={"method": method,"mcts":{"init_tokens":2,"num_candidates":3,"top_k_tokens":2,"top_k_actions":2}})
    json.dumps(prediction, allow_nan=False)
    if method == "mcts_entropy":
        selected = next(e["selected_state"] for e in sink if e["stage"] == "termination")
        completed_positions = [e["action"]["output_position"] for e in sink if e["stage"] == "rollout_step" and e.get("phase") == "completion"]
        assert not set(completed_positions) & {a["output_position"] for a in selected["committed"]}
        first_completion_input = instance.model.seen[search_calls]
        for action in selected["committed"]:
            assert first_completion_input[0, 2 + action["output_position"]] == action["token_id"]


def test_baseline_default_delegates_to_original_sampler_with_unchanged_parameters(monkeypatch):
    from dlm_project.models import llada as Llada
    seen = []

    def original(model, prompt, **generation):
        seen.append(generation)
        return torch.cat([prompt, torch.tensor([[1, 2]])], dim=1)

    monkeypatch.setattr(Llada, "generate", original)
    instance = wrapper()
    assert instance.predict("prompt") == " onetwo"
    assert seen == [dict(steps=128, gen_length=128, block_length=32, temperature=0.,
                         remasking="low_confidence", mask_id=MASK)]


@pytest.mark.parametrize("phase,expected_calls,finished", [("one_step", 3, False), ("search", 7, True)])
def test_inspection_jsonl_contains_actual_events_and_no_completion(tmp_path, phase, expected_calls, finished):
    instance = wrapper()
    trace_path = tmp_path / "inspection" / f"{phase}.jsonl"
    result = inspect_puzzle(instance, "0" * 81, tiny_config(tmp_path), phase, trace_path)
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert events[0]["stage"] == "metadata"
    assert events[-1]["stage"] == "inspection"
    assert events[-1]["completion_model_calls"] == 0
    assert events[-1]["model_device"] == events[-1]["search_state_device"] == "cpu"
    assert events[-1]["finished"] is finished
    assert result.stats["search_model_calls"] == len(instance.model.seen) == expected_calls
    assert all(e["stage"] != "completion" for e in events)
    assert "logits" not in trace_path.read_text()
    assert "reference_solution" not in trace_path.read_text()


def test_runner_saves_resolved_config_search_stats_and_matching_trace(tmp_path):
    config = tiny_config(tmp_path)
    original = copy.deepcopy(config)
    instance = wrapper()
    run_dir, summary = run_experiment(config, [{"puzzle": "0" * 81, "solution": "1" * 81}], instance)
    assert config == original
    saved = json.loads((run_dir / "config.json").read_text())
    assert saved["decoding"] == resolve_decoding(config["decoding"], config["generation"])
    record = json.loads((run_dir / "predictions.jsonl").read_text())
    stats = record["decoding_stats"]
    assert len(record["raw_output"]) > 0  # Unmodified, unparseable output remains visible.
    assert record["parse_success"] is False
    assert record["decoding_method"] == "mcts_entropy"
    assert stats["selected_depth"] == 2 and stats["candidate_count"] == 3
    assert stats["mcts_config"] == saved["decoding"]["mcts"]
    assert stats["search_model_calls"] == summary["search_model_calls"] == 7
    assert stats["completion_model_calls"] == summary["completion_model_calls"] == 2
    assert summary["latency_seconds"] == stats["latency_seconds"]
    events = [json.loads(line) for line in (run_dir / record["trace_file"]).read_text().splitlines()]
    assert events[-1]["stats"] == stats
    assert events[-1]["text"] == record["raw_output"]
    assert summary["status"] == "completed"
    assert (run_dir / record["plot_file"]).is_file()


def test_interrupted_search_flushes_prior_trace_events(tmp_path):
    instance = wrapper()
    class Interrupted(FakeModel):
        def __call__(self, ids):
            if len(self.seen) == 2:
                raise RuntimeError("interrupted child")
            return super().__call__(ids)
    instance.model = Interrupted()
    with pytest.raises(RuntimeError, match="interrupted child"):
        run_experiment(tiny_config(tmp_path), [{"puzzle": "0" * 81, "solution": "1" * 81}], instance)
    trace_path = next(tmp_path.rglob("puzzle_000.jsonl"))
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert any(e["stage"] == "forward" for e in events)
    assert sum(e["stage"] == "backpropagation" for e in events) == 1
    assert events[-1]["stage"] == "forward"
    metrics = json.loads(next(tmp_path.rglob("metrics.json")).read_text())
    assert metrics["status"] == "failed" and metrics["processed"] == 0


def test_config_validation_and_method_sensitive_experiment_names(tmp_path):
    config = tiny_config(tmp_path)
    validate_config(config)
    baseline = experiment_name(config["generation"])
    assert experiment_name(config["generation"], {"method": "baseline"}) == baseline
    mcts = experiment_name(config["generation"], config["decoding"])
    assert "mcts_entropy" in mcts and "lc2-c3-k1_2-k2_2-budgetnone" in mcts
    for key in config["decoding"]["mcts"]:
        changed = copy.deepcopy(config["decoding"])
        if key == "stop_mode":
            continue
        if key == "max_model_calls":
            changed["mcts"][key] = 100
        else:
            changed["mcts"][key] *= 2
        assert experiment_name(config["generation"], changed) != mcts
    for method in ("confidence", "mcts_entropy"):
        for key, value in [("block_length", 2), ("steps", 2), ("temperature", .1 if method == "confidence" else 0), ("remasking", "random")]:
            invalid = copy.deepcopy(config)
            invalid["decoding"]["method"] = method
            invalid["generation"][key] = value
            with pytest.raises(ValueError, match="requires"):
                validate_config(invalid)
    for settings in [{"method": "invalid"}, {"seed": -1}, {"seed": True}, {"mcts": {"typo": 1}},
                     {"mcts": {"init_tokens": 5}}, {"unexpected": 1}]:
        with pytest.raises(ValueError):
            resolve_decoding({"method": "mcts_entropy", **settings}, config["generation"])


def test_notebook_inspection_uses_subprocess_and_viewer_reads_saved_traces(tmp_path, monkeypatch):
    notebook = json.loads((ROOT / "notebooks/sudoku/LLaDA_Sudoku_Colab.ipynb").read_text())
    frames = []
    import IPython.display
    monkeypatch.setattr(IPython.display, "display", frames.append)
    scope = {"Path": Path}
    exec(source(notebook, "mcts-trace-viewer"), scope)
    events = MCTSSearch(FakeModel(), PROMPT, MASK, 4, MCTSConfig(init_tokens=2, top_k_tokens=2, top_k_actions=2)).run().trace
    trace_path = tmp_path / "events.jsonl"
    trace_path.write_text("".join(json.dumps(e) + "\n" for e in events))
    scope["search_events"] = scope["read_trace"](trace_path)
    exec(source(notebook, "mcts-cycle-selector"), scope)
    assert scope["CYCLE_TO_VIEW"] == 1
    assert all(isinstance(frame, pd.DataFrame) for frame in frames)
    assert any("entropy_after" in frame for frame in frames)
    assert any("reward_sum_after" in frame for frame in frames)
    scope["show_mcts_cycle"](events, 2)
    assert any("ucb" in frame for frame in frames)
    for cell_id, phase in [("mcts-one-step", "one_step"), ("mcts-search", "search")]:
        cell = source(notebook, cell_id)
        assert "subprocess.run" in cell and f'"{phase}"' in cell and '"--trace-path"' in cell
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "from_pretrained(" not in code
    assert "LLaDA(" not in code


def test_notebook_runs_cuda_checks_in_subprocess_and_verifies_hf_device(tmp_path):
    notebook = json.loads((ROOT / "notebooks/sudoku/LLaDA_Sudoku_Colab.ipynb").read_text())
    calls = []
    scope = {
        "REPO_DIR": tmp_path,
        "sys": type("Python", (), {"executable": "python"}),
        "subprocess": type("Process", (), {"run": lambda *args, **kwargs: calls.append((args, kwargs))}),
    }
    exec(source(notebook, "cuda-search-tests"), scope)
    command = calls[0][0][0]
    assert command[:3] == ["python", "-u", "-c"]
    assert 'torch.cuda.is_available()' in command[3]
    assert 'raise RuntimeError' in command[3]  # Cannot report skipped tests as GPU validation.
    assert 'tests/decoding/test_mcts_cuda.py' in command[3]
    assert calls[0][1] == {"cwd": tmp_path, "check": True}
    for cell_id in ("mcts-one-step", "mcts-search", "mcts-gpu-verification"):
        assert '["model_device"]).type == "cuda"' in source(notebook, cell_id)
