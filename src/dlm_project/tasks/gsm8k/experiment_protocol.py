"""Frozen GSM8K inputs and settings shared by E0, E1 and E2.

Inference only reads the checked-in manifest. Dataset access is confined to
the explicit preparation command; references never enter prompt construction.
"""

import copy
import json
from pathlib import Path
import random
import re

import yaml

from dlm_project.experiments.manifests import digest
from dlm_project.models.profiles import get_profile
from dlm_project.paths import PROJECT_ROOT
from dlm_project.tasks.gsm8k.evaluation import extract_answer


COMMON_PATH = PROJECT_ROOT / "configs/gsm8k/experiments_common.yaml"
PROTOCOL_VERSION = "gsm8k_shared_v2"
STAGE_COUNTS = {"dev": 5, "evaluation": 20, "test": 200}
EXTENDED_TEST_COUNT = 500
EXTENDED_STAGE_COUNTS = {"dev": 5, "evaluation": 20, "test": EXTENDED_TEST_COUNT}
EXTENDED_MANIFEST_PATH = PROJECT_ROOT / "configs/gsm8k/experiment_samples_500.json"
E2_EVALUATION_COUNT = 70
E2_STAGE_COUNTS = {"dev": 5, "evaluation": E2_EVALUATION_COUNT, "test": 200}
E2_MANIFEST_PATH = PROJECT_ROOT / "configs/gsm8k/experiment_samples_e2.json"
# Lc is identical for every experiment and condition; only the budget varies.
FIXED_DEPTH = 10


def load_shared_protocol(path=None):
    """Return a fresh copy; alternate paths must contain the canonical protocol."""
    canonical = yaml.safe_load(COMMON_PATH.read_text())
    common = yaml.safe_load(Path(path).read_text()) if path else canonical
    if common != canonical:
        raise ValueError("Common experimental settings differ from the frozen shared protocol.")
    profile = get_profile()
    if (common["protocol_version"] != PROTOCOL_VERSION
            or common["model"] != {"family": "llada", "repo_id": profile["repo_id"],
                                    "revision": profile["revision"], "dtype": "bfloat16"}
            or common["data"]["stage_counts"] != STAGE_COUNTS
            or common["data"]["sample_seed"] != 1
            or common["generation"] != dict(gen_length=256, steps=256, block_length=256)
            or common["decoding"] != {"seed": 1, "completion_method": "llada", "completion_temperature": 0.0,
                                      "mcts": {"top_k_tokens": 3, "top_k_actions": 5, "num_candidates": 3}}
            or common["search"]["simulation"] != "immediate_child"
            or [(p["label"], p["depth"], p["budget"]) for p in common["search"]["conditions"]]
               not in (
                   [("A", FIXED_DEPTH, 256), ("B", FIXED_DEPTH, 512)],
                   [("A", FIXED_DEPTH, 256), ("B", FIXED_DEPTH, 512), ("C", FIXED_DEPTH, 1024)],
               )
            or common["prompt"]["medal_decomposition"] is not False
            or common["prompt"]["template"].count("{GSM8K_question}") != 1):
        raise ValueError("Invalid frozen common experimental protocol.")
    return copy.deepcopy(common)


def runtime_defaults(common):
    """Project shared protocol metadata onto the existing runner config shape."""
    return {
        "model": {"family": common["model"]["family"], "model_path": common["model"]["repo_id"],
                  "dtype": common["model"]["dtype"], "device": "cuda", "local_files_only": False},
        "data": {"revision": common["data"]["revision"], "sample_seed": common["data"]["sample_seed"],
                 "num_samples": None, "cache_dir": None},
        "prompt": copy.deepcopy(common["prompt"]),
        "generation": copy.deepcopy(common["generation"]),
        "decoding": copy.deepcopy(common["decoding"]),
    }


def merge_settings(defaults, supplied):
    result = copy.deepcopy(defaults)
    for key, value in supplied.items():
        result[key] = (merge_settings(result[key], value)
                       if isinstance(value, dict) and isinstance(result.get(key), dict)
                       else copy.deepcopy(value))
    return result


def resolve_common_reference(source):
    """Paths in experiment configs are repository-relative, never cwd-relative."""
    reference = Path(source.get("common_config", COMMON_PATH))
    common = load_shared_protocol(reference if reference.is_absolute() else PROJECT_ROOT / reference)
    return common, merge_settings(runtime_defaults(common), source)


def build_prompt(question, settings=None):
    template = (settings if settings is not None else load_shared_protocol()["prompt"])["template"]
    return template.strip().replace("{GSM8K_question}", question.strip())


def example_seed(row_id, seed=1):
    return (seed + 5 * row_id) % 2**63


def question_key(question):
    return digest(" ".join(question.split()).casefold())


def build_manifest(train, test, *, revision, sample_seed=1, test_count=None, eval_count=None):
    """Keep the historical E0 RNG/shuffle order, freezing 5/20/200 (or extended test/eval counts)."""
    target_counts = dict(STAGE_COUNTS)
    if test_count is not None:
        target_counts["test"] = int(test_count)
    if eval_count is not None:
        target_counts["evaluation"] = int(eval_count)
    needed_train = target_counts["dev"] + target_counts["evaluation"]
    rng = random.Random(sample_seed)
    train_order, test_order = list(range(len(train))), list(range(len(test)))
    rng.shuffle(train_order)
    rng.shuffle(test_order)
    test_keys = {question_key(row["question"]) for row in test}
    seen, picked = set(), []
    for index in train_order:
        key = question_key(train[index]["question"])
        if key not in seen and key not in test_keys:
            seen.add(key)
            picked.append(index)
        if len(picked) == needed_train:
            break
    if len(picked) != needed_train:
        raise ValueError(f"Need {needed_train} distinct training questions disjoint from test.")
    examples = []
    dev_count = target_counts["dev"]
    pools = {"dev": ("train", picked[:dev_count]), "evaluation": ("train", picked[dev_count:]),
             "test": ("test", test_order)}
    for stage, (split, order) in pools.items():
        source = train if split == "train" else test
        count = 0
        for index in order:
            row = source[index]
            if split == "test":
                key = question_key(row["question"])
                if key in seen:
                    continue
                seen.add(key)
            content = {"question": row["question"], "reference_solution": row["answer"]}
            examples.append({"example_id": f"{split}:{index}", "row_id": index,
                             "source_split": split, "stage": stage,
                             "content_hash": digest(content), **content})
            count += 1
            if count == target_counts[stage]:
                break
    manifest = {"version": 2, "kind": "gsm8k_shared", "sample_seed": sample_seed,
                "dataset": {"repo_id": "openai/gsm8k", "subset": "main", "revision": revision},
                "source_counts": {"train": len(train), "test": len(test)}, "examples": examples}
    return validate_manifest(manifest, expected_stage_counts=target_counts)


def validate_manifest(manifest, common=None, *, expected_stage_counts=None):
    """Validate structure and, for official runs, the full ordered content hash."""
    target_counts = expected_stage_counts or STAGE_COUNTS
    try:
        data = manifest["dataset"]
        if (manifest["version"] != 2 or manifest["kind"] != "gsm8k_shared"
                or manifest["sample_seed"] != 1 or data["repo_id"] != "openai/gsm8k"
                or data["subset"] != "main" or not re.fullmatch(r"[0-9a-f]{40}", data["revision"])):
            raise ValueError("Invalid shared GSM8K manifest identity.")
        counts, ids, questions = dict.fromkeys(target_counts, 0), set(), set()
        stages = []
        for row in manifest["examples"]:
            stage = row["stage"]
            split = "test" if stage == "test" else "train"
            content = {key: row[key] for key in ("question", "reference_solution")}
            if (stage not in counts or row["source_split"] != split
                    or type(row["row_id"]) is not int
                    or not 0 <= row["row_id"] < manifest["source_counts"][split]
                    or row["example_id"] != f"{split}:{row['row_id']}"
                    or any(not isinstance(v, str) or not v.strip() for v in content.values())
                    or row["content_hash"] != digest(content)
                    or extract_answer(row["reference_solution"]) is None):
                raise ValueError("Invalid shared example identity, content or answer.")
            key = question_key(row["question"])
            if row["example_id"] in ids or key in questions:
                raise ValueError("Duplicate questions or overlapping shared stages.")
            ids.add(row["example_id"])
            questions.add(key)
            counts[stage] += 1
            stages.append(stage)
        if counts != target_counts or stages != [s for s, count in target_counts.items() for _ in range(count)]:
            raise ValueError("Shared manifest requires ordered dev/evaluation/test pools of 5/20/200.")
        if common is not None:
            if data["revision"] != common["data"]["revision"]:
                raise ValueError("Dataset revision differs from the shared protocol.")
            if target_counts == STAGE_COUNTS:
                if digest(manifest) != common["data"]["manifest_hash"]:
                    raise ValueError("Shared manifest hash mismatch: content or sample order changed.")
            else:
                canonical = get_manifest(common=common)
                can_dev = [e for e in canonical["examples"] if e["stage"] == "dev"]
                can_eval = [e for e in canonical["examples"] if e["stage"] == "evaluation"]
                can_test = [e for e in canonical["examples"] if e["stage"] == "test"]
                man_dev = [e for e in manifest["examples"] if e["stage"] == "dev"]
                man_eval = [e for e in manifest["examples"] if e["stage"] == "evaluation"]
                man_test = [e for e in manifest["examples"] if e["stage"] == "test"]
                if man_dev[:len(can_dev)] != can_dev:
                    raise ValueError("Manifest dev prefix does not match canonical dev.")
                if man_eval[:len(can_eval)] != can_eval:
                    raise ValueError("Manifest evaluation prefix does not match canonical evaluation.")
                if man_test[:min(len(can_test), len(man_test))] != can_test[:min(len(can_test), len(man_test))]:
                    raise ValueError("Manifest test prefix does not match canonical test.")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Malformed shared GSM8K manifest.") from error
    return manifest


def get_manifest(path=None, *, common=None):
    common = common or load_shared_protocol()
    path = Path(path) if path is not None else PROJECT_ROOT / common["data"]["manifest_path"]
    if not path.is_file():
        raise ValueError("Shared manifest missing. Run scripts/prepare_gsm8k_experiments.py --rebuild.")
    return validate_manifest(json.loads(path.read_text()), common)


def get_extended_manifest(path=None, *, common=None, test_count=EXTENDED_TEST_COUNT):
    common = common or load_shared_protocol()
    path = Path(path) if path is not None else EXTENDED_MANIFEST_PATH
    if not path.is_file():
        raise ValueError(
            f"Extended manifest missing at {path}. "
            "Run scripts/prepare_gsm8k_experiments.py --rebuild --test-count 500 "
            "--output configs/gsm8k/experiment_samples_500.json."
        )
    target_counts = dict(STAGE_COUNTS)
    target_counts["test"] = int(test_count)
    return validate_manifest(json.loads(path.read_text()), common, expected_stage_counts=target_counts)


def get_e2_manifest(path=None, *, common=None, eval_count=E2_EVALUATION_COUNT):
    common = common or load_shared_protocol()
    path = Path(path) if path is not None else E2_MANIFEST_PATH
    if not path.is_file():
        raise ValueError(
            f"E2 manifest missing at {path}. "
            "Run scripts/prepare_gsm8k_experiments.py --rebuild --e2 "
            f"--output {E2_MANIFEST_PATH}."
        )
    target_counts = dict(STAGE_COUNTS)
    target_counts["evaluation"] = int(eval_count)
    return validate_manifest(json.loads(path.read_text()), common, expected_stage_counts=target_counts)


def select_examples(manifest, stage, *, expected_count=None):
    if stage not in STAGE_COUNTS:
        raise ValueError("Stage must be dev, evaluation, or test.")
    rows = [row for row in manifest["examples"] if row["stage"] == stage]
    target = expected_count if expected_count is not None else STAGE_COUNTS[stage]
    if len(rows) != target:
        raise ValueError("Wrong shared stage size.")
    return rows


def load_samples(stage, *, manifest_path=None):
    """Experiment-independent, offline public entry point for fixed samples."""
    return select_examples(get_manifest(manifest_path), stage)
