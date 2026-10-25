"""Legacy E0 v1-v4 pools; new experiments use experiment_protocol's fixed inputs."""

import json
from pathlib import Path
import random
import re

from filelock import SoftFileLock

from dlm_project.experiments.artifacts import save_json
from dlm_project.experiments.manifests import digest
from dlm_project.tasks.gsm8k.evaluation import extract_answer


STAGE_COUNTS = {"dev": 5, "evaluation": 20, "test": 25}
TRAIN_COUNTS = {"dev": 5, "evaluation": 20}


def question_key(question):
    return digest(" ".join(question.split()).casefold())


def build_manifest(train, test, *, revision, sample_seed=1):
    """Shuffle once; exclude any train question also present in the test split."""
    rng = random.Random(sample_seed)
    test_keys = {question_key(row["question"]) for row in test}
    seen, examples = set(), []
    train_order, test_order = list(range(len(train))), list(range(len(test)))
    rng.shuffle(train_order)
    rng.shuffle(test_order)
    picked = []
    for index in train_order:
        key = question_key(train[index]["question"])
        if key not in seen and key not in test_keys:
            seen.add(key)
            picked.append(index)
        if len(picked) == sum(TRAIN_COUNTS.values()):
            break
    if len(picked) != sum(TRAIN_COUNTS.values()):
        raise ValueError("E0 needs 25 distinct training questions disjoint from the test split.")
    pools = {"dev": ("train", picked[:5]), "evaluation": ("train", picked[5:]),
             "test": ("test", test_order)}
    for stage, (split, order) in pools.items():
        source = train if split == "train" else test
        for index in order:
            row = source[index]
            key = question_key(row["question"])
            if split == "test":
                if key in seen:
                    continue
                seen.add(key)
            content = {"question": row["question"], "reference_solution": row["answer"]}
            examples.append({"example_id": f"{split}:{index}", "row_id": index,
                             "source_split": split, "stage": stage,
                             "content_hash": digest(content), **content})
    manifest = {"version": 1, "kind": "gsm8k_e0", "sample_seed": sample_seed,
                "dataset": {"repo_id": "openai/gsm8k", "subset": "main", "revision": revision},
                "source_counts": {"train": len(train), "test": len(test)},
                "examples": examples}
    return validate_manifest(manifest, revision=revision, sample_seed=sample_seed)


def validate_manifest(manifest, *, revision=None, sample_seed=1):
    source = manifest.get("dataset", {})
    if (manifest.get("version") != 1 or manifest.get("kind") != "gsm8k_e0"
            or source.get("repo_id") != "openai/gsm8k" or source.get("subset") != "main"
            or not re.fullmatch(r"[0-9a-f]{40}", str(source.get("revision", "")))):
        raise ValueError("E0 requires a pinned openai/gsm8k main manifest.")
    if revision is not None and source["revision"] != revision:
        raise ValueError("Dataset revision differs from the frozen E0 manifest.")
    if manifest.get("sample_seed") != sample_seed:
        raise ValueError("Sample seed differs from the frozen E0 manifest.")
    ids, questions, counts = set(), set(), dict.fromkeys(STAGE_COUNTS, 0)
    for row in manifest["examples"]:
        stage = row["stage"]
        split = "test" if stage == "test" else "train"
        content = {key: row[key] for key in ("question", "reference_solution")}
        if (stage not in counts or row["source_split"] != split
                or type(row["row_id"]) is not int or row["row_id"] < 0
                or row["example_id"] != f"{split}:{row['row_id']}"
                or any(not isinstance(value, str) or not value.strip() for value in content.values())
                or row["content_hash"] != digest(content)
                or extract_answer(row["reference_solution"]) is None):
            raise ValueError("Invalid E0 example identity, content, or reference.")
        key = question_key(row["question"])
        if row["example_id"] in ids or key in questions:
            raise ValueError("Duplicate questions or overlapping E0 stages.")
        ids.add(row["example_id"])
        questions.add(key)
        counts[stage] += 1
    if any(counts[stage] != count for stage, count in TRAIN_COUNTS.items()) or not counts["test"]:
        raise ValueError("E0 requires 5 development, 20 evaluation, and a nonempty test pool.")
    return manifest


def get_manifest(path, *, revision=None, sample_seed=1, cache_dir=None):
    """Publish once under a Drive-compatible lock; subsequent reads work offline."""
    if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
        raise ValueError("Dataset revision must be an immutable 40-character commit SHA.")
    if type(sample_seed) is not int or sample_seed < 0:
        raise ValueError("Sample seed must be a nonnegative integer.")
    path = Path(path).expanduser().resolve()
    if path.exists():
        return validate_manifest(json.loads(path.read_text()), revision=revision, sample_seed=sample_seed)
    from datasets import load_dataset
    from huggingface_hub import HfApi

    resolved = revision or HfApi().dataset_info("openai/gsm8k").sha
    sources = {split: load_dataset("openai/gsm8k", "main", split=split,
                                   revision=resolved, cache_dir=cache_dir)
               for split in ("train", "test")}
    manifest = build_manifest(sources["train"], sources["test"],
                              revision=resolved, sample_seed=sample_seed)
    manifest["dataset"]["fingerprints"] = {
        split: getattr(dataset, "_fingerprint", None) for split, dataset in sources.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with SoftFileLock(str(path) + ".lock", timeout=60):
        if path.exists():
            manifest = json.loads(path.read_text())
        else:
            save_json(path, manifest)
    return validate_manifest(manifest, revision=revision, sample_seed=sample_seed)


def select_examples(manifest, stage, num_samples=None):
    if stage not in STAGE_COUNTS:
        raise ValueError("E0 stage must be dev, evaluation, or test.")
    count = STAGE_COUNTS[stage] if num_samples is None else num_samples
    pool = [row for row in manifest["examples"] if row["stage"] == stage]
    if type(count) is not int or not 1 <= count <= len(pool):
        raise ValueError(f"Request 1–{len(pool)} questions from the frozen {stage} pool.")
    return pool[:count]
