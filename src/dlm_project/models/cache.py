"""Download a pinned model to persistent storage and optionally stage it locally."""

import argparse
import json
from pathlib import Path
import shutil

from dlm_project.models.profiles import PROFILES, get_profile

MODEL_ID = get_profile()["repo_id"]
MODEL_REVISION = get_profile()["revision"]


def check_identity(directory, family):
    profile = get_profile(family)
    config_path = directory / "config.json"
    if config_path.is_file():
        if json.loads(config_path.read_text()).get("model_type") != family:
            raise ValueError(f"{directory} contains a different model family; choose a separate directory.")
    source = directory / ".model_source.json"
    if source.is_file():
        actual = json.loads(source.read_text())
        if any(actual.get(key) != profile[key] for key in ("repo_id", "revision")):
            raise ValueError(f"{directory} contains a different model/revision; choose a separate directory.")


def validate_snapshot(directory, family="llada"):
    directory = Path(directory)
    profile = get_profile(family)
    required = ["config.json", "tokenizer.json", "tokenizer_config.json",
                *profile["code_files"], "model.safetensors.index.json"]
    if family == "illada":
        required.append("chat_template.jinja")
    for name in required:
        if not (directory / name).is_file() or (directory / name).stat().st_size == 0:
            raise FileNotFoundError(f"Incomplete model snapshot: {directory / name}")
    check_identity(directory, family)
    tokenizer_config = json.loads((directory / "tokenizer_config.json").read_text())
    if not tokenizer_config.get("chat_template") and not (directory / "chat_template.jinja").is_file():
        raise FileNotFoundError("The model snapshot is missing its chat template.")
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    if not index.get("weight_map"):
        raise ValueError("The model weight index is empty.")
    for name in set(index["weight_map"].values()):
        shard = directory / name
        if shard.resolve().parent != directory.resolve():
            raise ValueError("Model shards must be files inside the snapshot directory.")
        if not shard.is_file() or shard.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty weight shard: {shard}")


def stage_model(source, destination, family="llada"):
    source, destination = Path(source), Path(destination)
    validate_snapshot(source, family)
    if source.resolve() == destination.resolve():
        return source
    destination.mkdir(parents=True, exist_ok=True)
    check_identity(destination, family)
    # Hugging Face download metadata stays on Drive; only model files go to the runtime.
    files = [path for path in source.iterdir() if path.is_file()
             and (not path.name.startswith(".") or path.name == ".model_source.json")]
    pending = [path for path in files if not (destination / path.name).is_file()
               or (destination / path.name).stat().st_size != path.stat().st_size
               or (destination / path.name).stat().st_mtime_ns != path.stat().st_mtime_ns]
    needed = sum(path.stat().st_size for path in pending)
    if shutil.disk_usage(destination).free < needed:
        raise OSError("Not enough runtime disk space. Use --drive-dir without --runtime-dir to load directly from Drive.")
    for path in pending:
        print(f"Copying {path.name} to runtime disk", flush=True)
        target = destination / path.name
        temporary = target.with_suffix(target.suffix + ".partial")
        shutil.copy2(path, temporary)
        temporary.replace(target)
    validate_snapshot(destination, family)
    return destination


def cache_model(drive_dir, runtime_dir=None, family="llada"):
    from huggingface_hub import snapshot_download

    profile = get_profile(family)
    drive_dir = Path(drive_dir)
    check_identity(drive_dir, family)
    snapshot_download(
        repo_id=profile["repo_id"],
        revision=profile["revision"],
        local_dir=str(drive_dir),
        allow_patterns=["*.json", "*.py", "*.jinja", "*.safetensors"],
        max_workers=2,
    )
    validate_snapshot(drive_dir, family)
    (drive_dir / ".model_source.json").write_text(json.dumps(
        {key: profile[key] for key in ("repo_id", "revision")}, indent=2,
    ) + "\n")
    return stage_model(drive_dir, runtime_dir, family) if runtime_dir else drive_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--model", choices=PROFILES, default="llada")
    args = parser.parse_args()
    model_path = cache_model(args.drive_dir, args.runtime_dir, args.model)
    print(f"Model ready: {model_path}", flush=True)


if __name__ == "__main__":
    main()
