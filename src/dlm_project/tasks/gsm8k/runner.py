"""Inspect one GSM8K prompt and run the existing LLaDA wrapper in a subprocess."""

import argparse
import copy
from datetime import timezone
import json
from pathlib import Path
import subprocess

from dlm_project.paths import PROJECT_ROOT as ROOT
from dlm_project.experiments.paths import DEFAULT_TIMEZONE, create_run_directory, experiment_name
from dlm_project.decoding.mcts_decoding import resolve_decoding
from dlm_project.experiments.config import load_config, validate_config as validate_generation_config
from dlm_project.experiments.artifacts import save_json
from dlm_project.tasks.gsm8k.dataset import load_example, prompt_preview
from dlm_project.tasks.gsm8k.evaluation import score_answer
from dlm_project.decoding.medal_prompting import get_gsm8k_demonstrations, gsm8k_prompt
from dlm_project.experiments.manifests import digest


def validate_config(config):
    """This walkthrough intentionally runs one question at a time."""
    validate_generation_config(config)
    data = config["data"]
    if data["num_samples"] != 1:
        raise ValueError("The prompt walkthrough requires data.num_samples: 1.")
    if config["model"].get("family", "llada") != "llada":
        raise ValueError("This walkthrough uses model.family: llada.")
    if data.get("dataset_id") != "openai/gsm8k" or data.get("dataset_config") != "main":
        raise ValueError("Use the openai/gsm8k dataset's main configuration.")
    if data.get("split") not in {"train", "test"}:
        raise ValueError("data.split must be train or test.")
    index = data.get("question_index")
    if type(index) is not int or index < 0:
        raise ValueError("data.question_index must be a nonnegative integer.")
    prompt = config.get("prompt", {})
    if not isinstance(prompt.get("instructions"), str) or not prompt["instructions"].strip():
        raise ValueError("prompt.instructions must be nonempty text.")
    custom = prompt.get("custom_question")
    if custom is not None and (not isinstance(custom, str) or not custom.strip()):
        raise ValueError("prompt.custom_question must be null or nonempty text.")


def run_prompt_experiment(config, example, model):
    """Save the input, chat-token view, raw output and separate evaluation artifacts.

    Only user_prompt enters predict_with_stats(); the reference is used after
    generation. The existing wrapper owns decoding and all GPU operations.
    """
    validate_config(config)
    config = copy.deepcopy(config)
    config["decoding"] = resolve_decoding(config.get("decoding"), config["generation"])
    run_dir, started = create_run_directory(config, dataset_name="GSM8K")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    config["provenance"] = {"git_commit": commit, "model": getattr(model, "identity", None)}
    config["run"] = {"directory": str(run_dir), "dataset": "GSM8K",
                     "experiment_name": experiment_name(config["generation"], config["decoding"]),
                     "timezone": config.get("experiment_timezone", DEFAULT_TIMEZONE),
                     "started_at": started.isoformat(), "started_at_utc": started.astimezone(timezone.utc).isoformat()}
    save_json(run_dir / "config.json", config)
    save_json(run_dir / "metrics.json", {"status": "running", "processed": 0})
    preview = prompt_preview(config, example)
    if config["decoding"]["method"] == "mcts_entropy":
        from huggingface_hub import HfApi
        revision = config["data"].get("revision") or HfApi().dataset_info("openai/gsm8k").sha
        demonstrations = get_gsm8k_demonstrations(run_dir.parent / "gsm8k_decomposition_train.json",
                            revision=revision, cache_dir=config["data"].get("cache_dir"))
        base_prompt = preview["user_prompt"]
        effective, metadata = gsm8k_prompt(base_prompt, example["question"], demonstrations)
        preview.update(user_prompt=effective, base_prompt=base_prompt, decomposition=metadata,
                       messages=[{"role": "user", "content": effective}])
        save_json(run_dir / "decomposition_manifest.json", demonstrations)
    # Reuse the wrapper's exact formatting/tokenization method for inspection.
    # This is not a model forward and does not load a second model.
    ids = model.encode_prompt(preview["user_prompt"]).detach().cpu()[0].tolist()
    preview.update(
        rendered_chat_prompt=model.tokenizer.decode(ids, skip_special_tokens=False),
        input_token_ids=ids, input_tokens=model.tokenizer.convert_ids_to_tokens(ids),
        prompt_token_count=len(ids), output_mask_count=config["generation"]["gen_length"],
        mask_token_id=model.mask_id,
    )
    save_json(run_dir / "prompt.json", preview)
    print("User prompt:\n" + preview["user_prompt"], flush=True)
    print(f"Prompt tokens: {len(ids)} | output masks: {preview['output_mask_count']}", flush=True)
    with (run_dir / "trace.jsonl").open("w") as trace_handle:
        def save_event(event):
            trace_handle.write(json.dumps(event, allow_nan=False) + "\n")
            trace_handle.flush()

        prediction = model.predict_with_stats(preview["user_prompt"], **config["generation"],
                                              decoding=config["decoding"], on_event=save_event)
    # Preserve output even if later answer parsing/evaluation is interrupted.
    (run_dir / "response.txt").write_text(prediction["text"], encoding="utf-8")
    score = score_answer(prediction["text"], example["reference_solution"])
    record = {"index": 0, **example, "user_prompt": preview["user_prompt"],
              "raw_output": prediction["text"], **score,
              "base_prompt_hash": digest(preview.get("base_prompt", preview["user_prompt"])),
              "effective_prompt_hash": digest(preview["user_prompt"]), "decomposition": preview.get("decomposition"), "decoding_stats": prediction["stats"],
              "prompt_file": "prompt.json", "trace_file": "trace.jsonl"}
    (run_dir / "predictions.jsonl").write_text(json.dumps(record, allow_nan=False) + "\n")
    metrics = {"status": "completed", "processed": 1, **score, "decoding_stats": prediction["stats"]}
    save_json(run_dir / "metrics.json", metrics)
    print("\nLLaDA response:\n" + prediction["text"], flush=True)
    print(json.dumps(score, indent=2), flush=True)
    print(f"Saved artifacts: {run_dir}", flush=True)
    return run_dir, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/gsm8k/llada.yaml")
    parser.add_argument("--prompt-only", action="store_true", help="Preview user text without loading a model")
    parser.add_argument("--preview-path", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    if config.get("decoding", {}).get("method") == "mcts_entropy" and not config["data"].get("revision"):
        from huggingface_hub import HfApi
        config["data"]["revision"] = HfApi().dataset_info("openai/gsm8k").sha
    example = load_example(config)
    if args.prompt_only:
        preview = prompt_preview(config, example)
        if config.get("decoding", {}).get("method") == "mcts_entropy":
            root = Path(config.get("experiment_root", "results"))
            demonstrations = get_gsm8k_demonstrations(root / "manifests/gsm8k_decomposition_train.json",
                revision=config["data"]["revision"], cache_dir=config["data"].get("cache_dir"))
            base_prompt = preview["user_prompt"]
            effective, metadata = gsm8k_prompt(base_prompt, example["question"], demonstrations)
            preview.update(user_prompt=effective, base_prompt=base_prompt, decomposition=metadata,
                           messages=[{"role":"user", "content":effective}])
        print(preview["user_prompt"], flush=True)
        if args.preview_path:
            args.preview_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(args.preview_path, preview)
        return
    from dlm_project.models.llada import LLaDA

    model = LLaDA(**config["model"])
    run_dir, metrics = run_prompt_experiment(config, example, model)
    if args.run_manifest:
        args.run_manifest.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.run_manifest, {"run_dir": str(run_dir), "status": metrics["status"]})


if __name__ == "__main__":
    main()
