"""Load one model and save one puzzle's MCTS trace, without final completion."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from dlm_project.decoding.mcts_decoding import MCTSConfig, MCTSSearch, model_device, resolve_decoding, state_record
from dlm_project.experiments.config import load_config, validate_config
from dlm_project.paths import PROJECT_ROOT as ROOT
from dlm_project.tasks.sudoku.dataset import format_prompt
from dlm_project.decoding.medal_prompting import sudoku_prompt
from dlm_project.tasks.sudoku.runner import _use_medal_decomposition


def inspect_puzzle(wrapper, puzzle, config, phase, trace_path):
    """Thin driver; accepts only a puzzle, never its reference solution.

    Each invocation starts a new search. `one_step` runs one complete cycle;
    `search` runs initialization to termination. Neither completes the output.
    """
    if phase not in {"one_step", "search"}:
        raise ValueError("phase must be one_step or search.")
    validate_config(config)
    options = resolve_decoding(config.get("decoding"), config["generation"])
    if options["method"] != "mcts_entropy":
        raise ValueError("Inspection requires decoding.method: mcts_entropy.")
    torch.manual_seed(options["seed"])
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(wrapper.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    with path.open("w") as handle:
        def emit(event):
            handle.write(json.dumps(event, allow_nan=False) + "\n")
            handle.flush()

        emit({"stage": "metadata", "phase": phase, "puzzle": puzzle,
              "generation": config["generation"], "decoding": options,
              "model": getattr(wrapper, "identity", None)})
        effective_prompt, decomposition = format_prompt(puzzle), None
        if _use_medal_decomposition({**config, "decoding": options}):
            effective_prompt, decomposition = sudoku_prompt(effective_prompt, 9, puzzle)
        emit({"stage": "prompt", "effective_prompt": effective_prompt, "decomposition": decomposition})
        prompt = wrapper.encode_prompt(effective_prompt)
        search = MCTSSearch(wrapper.model, prompt, wrapper.mask_id,
                            config["generation"]["gen_length"], MCTSConfig(**options["mcts"]),
                            tokenizer=wrapper.tokenizer, on_event=emit, seed=options["seed"])
        if phase == "one_step":
            search.step()
            result = search.result()
        else:
            result = search.run()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        emit({"stage": "inspection", "phase": phase, **result.stats,
              "model_device": str(model_device(wrapper.model)),
              "search_state_device": str(search.root.state.device),
              "completion_model_calls": 0, "latency_seconds": perf_counter() - started,
              "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
              if device.type == "cuda" else None,
              "selected_state": state_record(result.selected.state, search.prompt_len,
                                              search.mask_id, search.tokenizer)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("one_step", "search"), default="one_step")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sudoku/llada_mcts.yaml")
    parser.add_argument("--trace-path", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    from dlm_project.models.llada import LLaDA
    from dlm_project.tasks.sudoku.dataset import SudokuDataset

    dataset = SudokuDataset(n=1, cache_dir=config["data"].get("cache_dir"))
    wrapper = LLaDA(**config["model"])
    result = inspect_puzzle(wrapper, dataset[0]["puzzle"], config, args.phase, args.trace_path)
    print(json.dumps(result.stats, indent=2), flush=True)
    print(f"Saved trace: {args.trace_path}", flush=True)


if __name__ == "__main__":
    main()
