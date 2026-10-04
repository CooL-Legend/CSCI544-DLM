"""Frozen, self-contained task samples shared by all sweep conditions.

References live in these artifacts for offline scoring, but prompt construction
receives only questions (GSM8K) or the released user message (Sudoku).
"""

import hashlib
import json
from pathlib import Path
import random
import re

from dlm_project.experiments.artifacts import save_json

POOL_SIZES = {"gsm8k": {"development": 20, "evaluation": 100},
              "sudoku": {"development": 10, "evaluation": 30}}


def digest(value):
    """Stable SHA256 for text or JSON-compatible content."""
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _content(task, row):
    keys = ("question", "reference_solution") if task == "gsm8k" else ("puzzle", "solution", "prompt")
    return {key: row[key] for key in keys}


def _input_key(task, row):
    return row["question"].strip() if task == "gsm8k" else row["puzzle"]


def _pick(task, rows, count, rng, seen):
    order = list(range(len(rows)))
    rng.shuffle(order)
    chosen = []
    for index in order:
        row = rows[index]
        key = digest(_input_key(task, row))
        if key not in seen:
            seen.add(key)
            chosen.append(row)
            if len(chosen) == count:
                return chosen
    raise ValueError(f"Not enough distinct {task} examples for a pool of {count}.")


def create_manifest(task, sample_seed=1, cache_dir=None, revision=None):
    """Resolve a dataset snapshot once, then sample disjoint ordered pools."""
    from datasets import load_dataset

    rng, seen = random.Random(sample_seed), set()
    if task == "gsm8k":
        if revision is None:
            from huggingface_hub import HfApi
            revision = HfApi().dataset_info("openai/gsm8k").sha
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("GSM8K revision must be an immutable 40-character commit SHA.")
        sources, fingerprints = {}, {}
        for partition, split in (("development", "train"), ("evaluation", "test")):
            dataset = load_dataset("openai/gsm8k", "main", split=split,
                                   revision=revision, cache_dir=cache_dir)
            fingerprints[split] = getattr(dataset, "_fingerprint", None)
            rows = [{"row_id": i, "source_split": split, "question": row["question"],
                     "reference_solution": row["answer"]} for i, row in enumerate(dataset)]
            sources[partition] = _pick(task, rows, POOL_SIZES[task][partition], rng, seen)
        source = {"repo_id": "openai/gsm8k", "subset": "main", "revision": revision,
                  "fingerprints": fingerprints}
    elif task == "sudoku":
        from dlm_project.tasks.sudoku.dataset import BGPOSudokuDataset, BGPO_PROVENANCE
        if revision is not None and revision != BGPO_PROVENANCE["revision"]:
            raise ValueError("Sudoku revision must match the pinned BGPO 4x4 release.")
        dataset = BGPOSudokuDataset(n=500, cache_dir=cache_dir)
        rows = [{"row_id": row["source_index"], "source_split": "test",
                 "puzzle": row["puzzle"], "solution": row["solution"], "prompt": row["prompt"]}
                for row in dataset]
        chosen = _pick(task, rows, sum(POOL_SIZES[task].values()), rng, seen)
        sources = {"development": chosen[:10], "evaluation": chosen[10:]}
        source = dict(BGPO_PROVENANCE)
    else:
        raise ValueError(f"Unknown sweep task: {task}")
    examples = []
    for partition, rows in sources.items():
        for row in rows:
            examples.append({**row, "example_id": f"{row['source_split']}:{row['row_id']}",
                             "partition": partition, "position": len(examples),
                             "content_hash": digest(_content(task, row))})
    manifest = {"version": 1, "task": task, "sample_seed": sample_seed,
                "dataset": source, "examples": examples}
    validate_manifest(manifest, task, sample_seed)
    return manifest


def validate_manifest(manifest, task, sample_seed=None):
    """Reject changed content, duplicate inputs, overlapping pools or wrong sources."""
    if manifest.get("version") != 1 or manifest.get("task") != task:
        raise ValueError("Manifest version/task mismatch.")
    if sample_seed is not None and manifest.get("sample_seed") != sample_seed:
        raise ValueError("Manifest sample seed differs from the configuration.")
    source = manifest["dataset"]
    if task == "gsm8k":
        if (source.get("repo_id") != "openai/gsm8k" or source.get("subset") != "main"
                or not re.fullmatch(r"[0-9a-f]{40}", str(source.get("revision", "")))):
            raise ValueError("Manifest requires a pinned openai/gsm8k main snapshot.")
    else:
        from dlm_project.tasks.sudoku.dataset import BGPO_PROVENANCE
        if source != BGPO_PROVENANCE:
            raise ValueError("Manifest requires the pinned BGPO 4x4 dataset.")
    counts = dict.fromkeys(POOL_SIZES[task], 0)
    ids, hashes = set(), set()
    for position, row in enumerate(manifest["examples"]):
        partition = row["partition"]
        if partition not in counts or row["position"] != position:
            raise ValueError("Invalid manifest partition or ordering.")
        expected_split = ("train" if partition == "development" else "test") if task == "gsm8k" else "test"
        if row["source_split"] != expected_split or row["example_id"] != f"{expected_split}:{row['row_id']}":
            raise ValueError("Invalid source split or example ID.")
        content = _content(task, row)
        if any(not isinstance(value, str) or not value.strip() for value in content.values()):
            raise ValueError("Manifest content must be nonempty text.")
        if row["content_hash"] != digest(content):
            raise ValueError("Manifest content hash mismatch.")
        input_hash = digest(_input_key(task, row))
        if row["example_id"] in ids or input_hash in hashes:
            raise ValueError("Duplicate inputs or overlapping manifest partitions.")
        ids.add(row["example_id"])
        hashes.add(input_hash)
        if task == "gsm8k":
            from dlm_project.tasks.gsm8k.evaluation import extract_answer
            if extract_answer(row["reference_solution"]) is None:
                raise ValueError("Malformed GSM8K reference answer.")
        else:
            from dlm_project.tasks.sudoku.evaluation import is_valid_sudoku, preserves_clues
            if (not re.fullmatch(r"[0-4]{16}", row["puzzle"])
                    or not is_valid_sudoku(row["solution"], 4)
                    or not preserves_clues(row["puzzle"], row["solution"], 4)):
                raise ValueError("Malformed 4x4 puzzle or reference.")
        counts[partition] += 1
    if counts != POOL_SIZES[task]:
        raise ValueError(f"Manifest requires pools of {POOL_SIZES[task]}; got {counts}.")


def get_manifest(path, task, *, sample_seed=1, cache_dir=None, revision=None):
    """Reuse an existing manifest offline, or create one without overwriting it."""
    path = Path(path).expanduser().resolve()
    if path.exists():
        manifest = json.loads(path.read_text())
        validate_manifest(manifest, task, sample_seed)
        if revision is not None and manifest["dataset"]["revision"] != revision:
            raise ValueError("Requested revision differs from the frozen manifest.")
    else:
        manifest = create_manifest(task, sample_seed, cache_dir, revision)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_json(path, manifest)
    return manifest


def select_examples(manifest, profile, num_samples):
    partition = "development" if profile == "smoke" else "evaluation"
    pool = [row for row in manifest["examples"] if row["partition"] == partition]
    if type(num_samples) is not int or not 1 <= num_samples <= len(pool):
        raise ValueError(f"Request 1–{len(pool)} examples from the {partition} pool.")
    return pool[:num_samples]
