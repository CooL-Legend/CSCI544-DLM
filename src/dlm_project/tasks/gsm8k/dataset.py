"""One GSM8K example and answer-free user-prompt construction."""


def load_example(config):
    """Load one question/reference pair, or a custom question without a reference."""
    custom = config["prompt"].get("custom_question")
    if custom is not None:
        return {"question": custom.strip(), "reference_solution": None, "source": {"kind": "custom"}}
    from datasets import load_dataset

    data = config["data"]
    index = data["question_index"]
    split = f"{data['split']}[{index}:{index + 1}]"
    dataset = load_dataset(data["dataset_id"], data["dataset_config"], split=split,
                           cache_dir=data.get("cache_dir"))
    if len(dataset) != 1:
        raise ValueError(f"Expected one GSM8K example at {split}; got {len(dataset)}.")
    row = dataset[0]
    return {"question": row["question"], "reference_solution": row["answer"],
            "source": {"kind": "gsm8k", "dataset_id": data["dataset_id"],
                       "dataset_config": data["dataset_config"], "split": data["split"],
                       "question_index": index, "fingerprint": getattr(dataset, "_fingerprint", None)}}


def build_prompt(question, instructions):
    """Build only the user message. The wrapper adds the HF chat template."""
    return f"{instructions.strip()}\n\nQuestion:\n{question.strip()}"


def prompt_preview(config, example):
    """Preview without model weights, tokenization, or the reference answer."""
    prompt = build_prompt(example["question"], config["prompt"]["instructions"])
    return {"source": example["source"], "question": example["question"],
            "user_prompt": prompt, "messages": [{"role": "user", "content": prompt}]}
