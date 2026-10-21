"""Resolve LLaDA termination tokens and score only their completed output prefix."""


def resolve_termination_tokens(tokenizer, model, mask_id):
    """Resolve both EOS and EOT, without copying upstream's numeric constants."""
    vocabulary = tokenizer.get_vocab()
    eot = vocabulary.get("<|eot_id|>")
    eos = next((vocabulary[name] for name in ("<|endoftext|>", "<|end_of_text|>")
                if name in vocabulary), getattr(tokenizer, "eos_token_id", None))
    declared = []
    for source in (tokenizer, getattr(model, "config", None), getattr(model, "generation_config", None)):
        value = getattr(source, "eos_token_id", None)
        declared.extend(value if isinstance(value, (list, tuple)) else [] if value is None else [value])
    if type(eos) is not int or type(eot) is not int or eos == eot:
        raise ValueError("Cannot resolve distinct EOS and <|eot_id|> tokens from the pinned LLaDA tokenizer.")
    ids = [eos, eot, *declared]
    known = set(vocabulary.values())
    size = getattr(getattr(model, "config", None), "vocab_size", max(known) + 1)
    if any(type(token) is not int or token not in known or not 0 <= token < size
           or token == mask_id for token in ids):
        raise ValueError("Invalid termination token ID in tokenizer/model configuration.")
    ids = sorted(set(ids))
    return {"eos_token_id": eos, "eot_token_id": eot, "token_ids": ids,
            "tokens": {str(token): tokenizer.convert_ids_to_tokens(token) for token in ids},
            "search_policy": "exclude_eos_and_eot", "completion_policy": "allow",
            "scoring_policy": "prefix_before_first_terminator", "early_stopping": False}


def decode_output(tokenizer, output_ids, termination_ids):
    """Trim only the generated suffix, after decoding has filled every position."""
    ids = [int(token) for token in output_ids]
    first = next((i for i, token in enumerate(ids) if token in termination_ids), None)
    prefix = ids if first is None else ids[:first]
    return {"full_output_ids": ids, "scoring_output_ids": prefix,
            "raw_output": tokenizer.decode(ids, skip_special_tokens=False),
            "scoring_text": tokenizer.decode(prefix, skip_special_tokens=True),
            "first_terminator_position": first,
            "first_terminator_id": None if first is None else ids[first],
            "tokens_after_terminator": 0 if first is None else len(ids) - first - 1}
