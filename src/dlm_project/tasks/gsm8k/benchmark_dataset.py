"""One frozen, ordered GSM8K test split shared by all decoder benchmarks."""

import json
import os
from pathlib import Path
import re
import tempfile

from filelock import SoftFileLock

from dlm_project.experiments.manifests import digest
from dlm_project.tasks.gsm8k.evaluation import extract_answer


TEST_SIZE = 1319
PROFILE_SIZES = {"smoke": 20, "initial": 100, "full": TEST_SIZE}


def validate_manifest(manifest, revision=None):
    source = manifest.get("dataset", {})
    if (manifest.get("version") != 1 or manifest.get("kind") != "gsm8k_test_benchmark"
            or source.get("repo_id") != "openai/gsm8k" or source.get("subset") != "main"
            or source.get("split") != "test"
            or not re.fullmatch(r"[0-9a-f]{40}", str(source.get("revision", "")))):
        raise ValueError("Manifest must identify a pinned openai/gsm8k main test split.")
    if revision is not None and source["revision"] != revision:
        raise ValueError("Requested dataset revision differs from the frozen manifest.")
    examples = manifest.get("examples", [])
    if len(examples) != TEST_SIZE:
        raise ValueError(f"The full GSM8K test manifest must contain {TEST_SIZE} examples.")
    for index, example in enumerate(examples):
        if (type(example.get("row_id")) is not int or example["row_id"] != index
                or example.get("example_id") != f"test:{index}"):
            raise ValueError("Manifest rows must preserve GSM8K test order and source IDs.")
        content = {key: example.get(key) for key in ("question", "reference_solution")}
        if any(not isinstance(value, str) or not value.strip() for value in content.values()):
            raise ValueError("Manifest questions and references must be nonempty text.")
        if example.get("content_hash") != digest(content):
            raise ValueError("Manifest content hash mismatch.")
        if extract_answer(content["reference_solution"]) is None:
            raise ValueError("Malformed GSM8K reference answer in manifest.")
    return manifest


def get_manifest(path, *, cache_dir=None, revision=None):
    """Reuse offline or freeze all test rows at an immutable Hub revision.

    Serialize publication with an exclusive-create lock, then rename a complete
    temporary file. This also works on mounted Google Drive, which does not
    support hard links. Concurrent writers reuse the first published snapshot.
    """
    path = Path(path).expanduser().resolve()
    if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
        raise ValueError("data.revision must be an immutable 40-character commit SHA.")
    if path.exists():
        return validate_manifest(json.loads(path.read_text(encoding="utf-8")), revision)
    from datasets import load_dataset

    if revision is None:
        from huggingface_hub import HfApi
        revision_to_load = HfApi().dataset_info("openai/gsm8k").sha
    else:
        revision_to_load = revision
    if not re.fullmatch(r"[0-9a-f]{40}", str(revision_to_load)):
        raise ValueError("Could not resolve an immutable GSM8K dataset revision.")
    dataset = load_dataset("openai/gsm8k", "main", split="test", revision=revision_to_load,
                           cache_dir=cache_dir)
    examples = []
    for index, row in enumerate(dataset):
        content = {"question": row["question"], "reference_solution": row["answer"]}
        examples.append({"row_id": index, "example_id": f"test:{index}",
                         **content, "content_hash": digest(content)})
    manifest = {"version": 1, "kind": "gsm8k_test_benchmark",
                "dataset": {"repo_id": "openai/gsm8k", "subset": "main", "split": "test",
                            "revision": revision_to_load,
                            "fingerprint": getattr(dataset, "_fingerprint", None)},
                "examples": examples}
    validate_manifest(manifest, revision)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(manifest, handle, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        # SoftFileLock uses exclusive file creation rather than flock or hard
        # links, so publication works on Colab's Drive filesystem as well.
        with SoftFileLock(str(path) + ".lock", timeout=60):
            if path.exists():
                manifest = json.loads(path.read_text(encoding="utf-8"))
            else:
                os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return validate_manifest(manifest, revision)


def select_examples(manifest, profile, num_samples=None):
    if profile not in PROFILE_SIZES:
        raise ValueError("evaluation.profile must be smoke, initial, or full.")
    count = PROFILE_SIZES[profile] if num_samples is None else num_samples
    if type(count) is not int or not 1 <= count <= len(manifest["examples"]):
        raise ValueError(f"num_samples must be between 1 and {len(manifest['examples'])}.")
    return manifest["examples"][:count]
