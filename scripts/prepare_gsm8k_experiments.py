"""Verify shared samples offline, or reconstruct them from the pinned source."""

import argparse
from pathlib import Path

from dlm_project.experiments.artifacts import save_json
from dlm_project.experiments.manifests import digest
from dlm_project.paths import PROJECT_ROOT
from dlm_project.tasks.gsm8k.experiment_protocol import (
    E2_EVALUATION_COUNT, E2_MANIFEST_PATH, STAGE_COUNTS,
    build_manifest, get_e2_manifest, get_extended_manifest, get_manifest, load_shared_protocol, validate_manifest,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Alternate copy; contents must match the canonical manifest.")
    parser.add_argument("--rebuild", action="store_true", help="Download only the pinned revision and verify the expected hash.")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--test-count", type=int, default=None, help="Optional test split count (e.g. 500 for extended test manifest).")
    parser.add_argument("--eval-count", type=int, default=None, help="Optional evaluation split count (e.g. 70 for E2 manifest).")
    parser.add_argument("--e2", action="store_true", help="Build or verify the E2 manifest with 70 evaluation samples.")
    args = parser.parse_args()
    common = load_shared_protocol()
    if args.e2 and args.eval_count is None:
        args.eval_count = E2_EVALUATION_COUNT
    destination = args.output or (E2_MANIFEST_PATH if args.e2 else PROJECT_ROOT / common["data"]["manifest_path"])
    target_counts = dict(STAGE_COUNTS)
    if args.test_count is not None:
        target_counts["test"] = int(args.test_count)
    if args.eval_count is not None:
        target_counts["evaluation"] = int(args.eval_count)
    if args.rebuild:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as parquet

        rows = {}
        for split in ("train", "test"):
            source = hf_hub_download("openai/gsm8k", f"main/{split}-00000-of-00001.parquet",
                                     repo_type="dataset", revision=common["data"]["revision"],
                                     cache_dir=args.cache_dir)
            rows[split] = parquet.read_table(source).to_pylist()
        manifest = build_manifest(
            rows["train"], rows["test"], revision=common["data"]["revision"],
            test_count=args.test_count, eval_count=args.eval_count
        )
        validate_manifest(manifest, common, expected_stage_counts=target_counts)
        if destination.exists():
            if args.eval_count is not None:
                get_e2_manifest(destination, common=common, eval_count=args.eval_count)
            elif args.test_count is None:
                get_manifest(destination, common=common)  # Never overwrite a mismatched artifact.
            else:
                get_extended_manifest(destination, common=common, test_count=args.test_count)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            save_json(destination, manifest)
    if args.eval_count is not None:
        manifest = get_e2_manifest(destination, common=common, eval_count=args.eval_count)
    elif args.test_count is None:
        manifest = get_manifest(destination, common=common)
    else:
        manifest = get_extended_manifest(destination, common=common, test_count=args.test_count)
    print(f"Verified {destination}\nRevision: {manifest['dataset']['revision']}\n"
          f"Samples: {target_counts}\nManifest SHA256: {digest(manifest)}")


if __name__ == "__main__":
    main()
