"""Consistent experiment names and collision-safe local-time run directories."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dlm_project.models.profiles import get_profile
from dlm_project.decoding.mcts_decoding import resolve_decoding

from dlm_project.paths import PROJECT_ROOT as ROOT
DEFAULT_TIMEZONE = "America/Los_Angeles"


def experiment_name(generation, decoding=None):
    remasking = generation["remasking"].replace("_confidence", "conf")
    name = (f"gen{generation['gen_length']}-steps{generation['steps']}-"
            f"blk{generation['block_length']}-temp{generation['temperature']:g}-{remasking}")
    options = resolve_decoding(decoding)
    if options["method"] == "baseline":
        return name  # Preserve existing experiment directories.
    m = options["mcts"]
    suffix = (f"-{options['method']}-gamma{m['margin_gamma']:g}"
              f"-eps{m['epsilon']:g}-seed{options['seed']}")
    if options["method"] == "mcts_entropy":
        suffix += (f"-lc{m['init_tokens']}-c{m['num_candidates']}"
                   f"-k1_{m['top_k_tokens']}-k2_{m['top_k_actions']}"
                   f"-budget{m['max_model_calls'] if m['max_model_calls'] is not None else 'none'}-ucb{m['exploration_c']:g}-medal-v2")
        if m["stop_mode"] != "candidates":
            suffix += f"-stop-{m['stop_mode']}"
    return name + suffix


def experiment_parent(config, *, dataset_name=None):
    if dataset_name is None:
        dataset_name = "Sudoku4x4" if config.get("data", {}).get("source") == "bgpo_4x4" else "Sudoku"
    if "experiment_root" in config:
        root = Path(config["experiment_root"]).expanduser()
        label = get_profile(config["model"].get("family", "llada"))["label"]
        parent = root / label / dataset_name / experiment_name(config["generation"], config.get("decoding"))
    else:
        # Backward compatibility for existing callers supplying an exact parent.
        parent = Path(config["output_dir"]).expanduser()
    return (parent if parent.is_absolute() else ROOT / parent).resolve()


def create_run_directory(config, now=None, *, dataset_name=None):
    zone = ZoneInfo(config.get("experiment_timezone", DEFAULT_TIMEZONE))
    started = datetime.now(zone) if now is None else now.astimezone(zone)
    parent = experiment_parent(config, dataset_name=dataset_name)
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = started.strftime("%d:%m:%y:%H:%M:%S")
    suffix = 1
    while True:
        directory = parent / (timestamp if suffix == 1 else f"{timestamp}-{suffix:02d}")
        try:
            directory.mkdir()
            return directory, started
        except FileExistsError:
            suffix += 1
