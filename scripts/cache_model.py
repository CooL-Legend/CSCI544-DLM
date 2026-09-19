"""Download a fixed LLaDA snapshot to persistent storage and optionally stage it locally."""

import argparse
import json
from pathlib import Path
import shutil

MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
MODEL_REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"


def validate_snapshot(directory):
    directory = Path(directory)
    required = ["config.json", "tokenizer.json", "tokenizer_config.json",
                "configuration_llada.py", "modeling_llada.py", "model.safetensors.index.json"]
    for name in required:
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Incomplete model snapshot: {directory / name}")
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    for name in set(index["weight_map"].values()):
        shard = directory / name
        if not shard.is_file() or shard.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty weight shard: {shard}")


def stage_model(source, destination):
    source, destination = Path(source), Path(destination)
    validate_snapshot(source)
    if source.resolve() == destination.resolve():
        return source
    destination.mkdir(parents=True, exist_ok=True)
    # Hugging Face download metadata stays on Drive; only model files go to the runtime.
    files = [path for path in source.iterdir() if path.is_file() and not path.name.startswith(".")]
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
    validate_snapshot(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=str(args.drive_dir),
        allow_patterns=["*.json", "*.py", "*.safetensors"],
        max_workers=2,
    )
    validate_snapshot(args.drive_dir)
    (args.drive_dir / ".model_source.json").write_text(json.dumps({"repo_id": MODEL_ID, "revision": MODEL_REVISION}, indent=2) + "\n")
    model_path = stage_model(args.drive_dir, args.runtime_dir) if args.runtime_dir else args.drive_dir
    print(f"Model ready: {model_path}", flush=True)


if __name__ == "__main__":
    main()
