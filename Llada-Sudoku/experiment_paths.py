"""Consistent experiment names and collision-safe local-time run directories."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from model_profiles import get_profile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIMEZONE = "America/Los_Angeles"


def experiment_name(generation):
    remasking = generation["remasking"].replace("_confidence", "conf")
    return (f"gen{generation['gen_length']}-steps{generation['steps']}-"
            f"blk{generation['block_length']}-temp{generation['temperature']:g}-{remasking}")


def experiment_parent(config):
    if "experiment_root" in config:
        root = Path(config["experiment_root"]).expanduser()
        label = get_profile(config["model"].get("family", "llada"))["label"]
        parent = root / label / "Sudoku" / experiment_name(config["generation"])
    else:
        # Backward compatibility for existing callers supplying an exact parent.
        parent = Path(config["output_dir"]).expanduser()
    return (parent if parent.is_absolute() else ROOT / parent).resolve()


def create_run_directory(config, now=None):
    zone = ZoneInfo(config.get("experiment_timezone", DEFAULT_TIMEZONE))
    started = datetime.now(zone) if now is None else now.astimezone(zone)
    parent = experiment_parent(config)
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
