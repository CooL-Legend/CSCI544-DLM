"""Answer-isolated inline decomposition prompts for MEDAL task runners."""

import json
import os
from pathlib import Path
import re
import tempfile
from filelock import SoftFileLock

from dlm_project.experiments.manifests import digest
from dlm_project.tasks.gsm8k.evaluation import extract_answer

PROMPT_VERSION = "medal_inline_v1"


def get_gsm8k_demonstrations(path, *, revision, cache_dir=None):
    """Freeze training rows 0/1 at the evaluation manifest's immutable revision."""
    path = Path(path)
    if not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
        raise ValueError("Decomposition demonstrations require a pinned dataset revision.")

    def validate(value):
        examples = value.get("examples", [])
        if (value.get("version") != PROMPT_VERSION or value.get("dataset") !=
                {"repo_id": "openai/gsm8k", "subset": "main", "split": "train", "revision": revision}
                or [e.get("row_id") for e in examples] != [0, 1]):
            raise ValueError("Decomposition manifest must contain pinned training rows 0 and 1.")
        for example in examples:
            content = {k: example.get(k) for k in ("question", "reference_solution")}
            if (any(not isinstance(v, str) or not v.strip() for v in content.values())
                    or example.get("content_hash") != digest(content)
                    or extract_answer(content["reference_solution"]) is None):
                raise ValueError("Invalid decomposition demonstration content hash or answer.")
        return value

    if path.exists():
        return validate(json.loads(path.read_text()))
    from datasets import load_dataset
    dataset = load_dataset("openai/gsm8k", "main", split="train[:2]", revision=revision, cache_dir=cache_dir)
    if len(dataset) != 2:
        raise ValueError("Expected exactly two training demonstrations.")
    examples = []
    for index, row in enumerate(dataset):
        content = {"question": row["question"], "reference_solution": row["answer"]}
        examples.append({"row_id": index, **content, "content_hash": digest(content)})
    value = validate({"version": PROMPT_VERSION, "dataset": {"repo_id": "openai/gsm8k", "subset": "main",
                     "split": "train", "revision": revision}, "examples": examples})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name+".", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, allow_nan=False)
        with SoftFileLock(str(path)+".lock", timeout=60):
            if path.exists():
                return validate(json.loads(path.read_text()))
            os.replace(temporary, path)
        return value
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def _normalize(text):
    return re.sub(r"\s+", " ", text).strip().casefold()

# GSM8K prompt template for MEDAL task runners, using two training demonstrations.
# Prompt Decomposition: 
'''
    1. Identify the quantities and the quantity being asked for.
    2. Calculate using the relevant quantities: <reasoning>
    3. State the resulting quantity.
'''
def gsm8k_prompt(base_prompt, question, demonstrations):
    examples = demonstrations["examples"]
    if _normalize(question) in {_normalize(e["question"]) for e in examples}:
        raise ValueError("Evaluation question overlaps a decomposition demonstration.")
    rendered = []
    for example in examples:
        answer = extract_answer(example["reference_solution"])
        reasoning = example["reference_solution"].rsplit("####", 1)[0].strip()
        reasoning = re.sub(r"<<([^>]+)>>", r"\1", reasoning)
        rendered.append(f"Problem: {example['question']}\n"
                        "1. Identify the quantities and the quantity being asked for.\n"
                        f"2. Calculate using the relevant quantities: {reasoning}\n"
                        f"3. State the resulting quantity.\n#### {answer}")
    prompt = ("Decompose the current problem into exactly three short subtasks and solve them in order. "
              "Show your own intermediate calculations, then end with exactly #### followed by a plain number.\n\n"
              + "\n\n".join(rendered) + "\n\nCurrent problem:\n" + base_prompt)
    metadata = {"version": PROMPT_VERSION, "template": "gsm8k_train_two_shot",
                "num_subtasks": 3, "demonstration_manifest_hash": digest(demonstrations),
                "demonstrations_dataset": demonstrations["dataset"], "demonstration_rows": [0, 1],
                "base_prompt_hash": digest(base_prompt), "effective_prompt_hash": digest(prompt),
                "planning_model_calls": 0}
    return prompt, metadata


def sudoku_prompt(base_prompt, board_size=9, puzzle=None):
    if board_size not in (4, 9):
        raise ValueError("Sudoku demonstrations require board size 4 or 9.")
    box = 2 if board_size == 4 else 3
    demos = []
    for variant in (0, 1):
        solution = ''.join(str((r*box+r//box+c+variant) % board_size+1)
                           for r in range(board_size) for c in range(board_size))
        given = ''.join('0' if (i+variant) % 3 == 0 else token for i, token in enumerate(solution))
        if puzzle == given:
            raise ValueError("Evaluation puzzle overlaps a synthetic decomposition demonstration.")
        final = f"<answer>{solution}</answer>" if board_size == 4 else solution
        demos.append(f"Synthetic {board_size}x{board_size} puzzle (0 is blank): {given}\n"
                     "1. Inspect the givens and locate empty cells.\n"
                     "2. Apply row, column, and box constraints to fill the empty cells.\n"
                     f"3. Check all constraints and preserve the givens.\n{final}")
    prompt = ("Use the following synthetic examples to decompose the current Sudoku into three short subtasks.\n\n"
              + '\n\n'.join(demos) + "\n\nCurrent puzzle:\n" + base_prompt
              + "\n\nFor this method, show three short reasoning steps before the final answer, overriding any "
                "instruction above to omit explanations. Keep the original final-answer format exactly.")
    return prompt, {"version": PROMPT_VERSION, "template": f"sudoku_{board_size}_synthetic_two_shot",
                    "num_subtasks": 3, "demonstration_manifest_hash": digest(demos),
                    "base_prompt_hash": digest(base_prompt), "effective_prompt_hash": digest(prompt),
                    "planning_model_calls": 0, "demonstration_source": "synthetic; no held-out solutions"}
