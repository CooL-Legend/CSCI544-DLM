"""Shared configuration loading and model/generation validation."""

from pathlib import Path
from zoneinfo import ZoneInfo
import yaml

from dlm_project.models.profiles import get_profile
from dlm_project.decoding.mcts_decoding import resolve_decoding
from dlm_project.experiments.paths import DEFAULT_TIMEZONE


def load_config(path):
    with Path(path).open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The configuration must be a YAML mapping.")
    if "common_config" in config:
        from dlm_project.tasks.gsm8k.experiment_protocol import resolve_common_reference
        _, config = resolve_common_reference(config)
    return config


def validate_config(config):
    get_profile(config["model"].get("family", "llada"))
    ZoneInfo(config.get("experiment_timezone", DEFAULT_TIMEZONE))
    n = config["data"]["num_samples"]
    maximum = 500 if config["data"].get("source") == "bgpo_4x4" else 1_000_000
    if maximum == 500 and (type(n) is not int or not 1 <= n <= maximum):
        raise ValueError("data.num_samples must be between 1 and 500 for bgpo_4x4.")
    if type(n) is not int or not 1 <= n <= maximum:
        raise ValueError("data.num_samples must be between 1 and 1,000,000.")
    generation = config["generation"]
    for key in ("steps", "gen_length", "block_length"):
        if type(generation[key]) is not int or generation[key] < 1:
            raise ValueError(f"generation.{key} must be a positive integer.")
    steps, length, block = (generation[k] for k in ("steps", "gen_length", "block_length"))
    if length % block or steps > length or steps % (length // block):
        raise ValueError("gen_length must divide into blocks, and steps must be <= gen_length and divisible by the number of blocks.")
    if generation["temperature"] < 0:
        raise ValueError("temperature must be nonnegative.")
    if generation["remasking"] not in {"low_confidence", "random"}:
        raise ValueError("remasking must be low_confidence or random.")
    if config["model"]["dtype"] not in {"bfloat16", "float16", "float32"}:
        raise ValueError("dtype must be bfloat16, float16, or float32.")
    resolve_decoding(config.get("decoding"), generation)
