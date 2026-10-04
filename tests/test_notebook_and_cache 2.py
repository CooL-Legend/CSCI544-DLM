import json
from pathlib import Path

import nbformat
import pytest
import yaml

from cache_model import stage_model, validate_snapshot

ROOT = Path(__file__).resolve().parents[1]


def make_snapshot(directory):
    directory.mkdir()
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "configuration_llada.py", "modeling_llada.py"]:
        (directory / name).write_text("{}")
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "model-1.safetensors"}}))
    (directory / "model-1.safetensors").write_bytes(b"fake test weights")


def test_incomplete_snapshot_is_rejected(tmp_path):
    source = tmp_path / "source"
    make_snapshot(source)
    (source / "model-1.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="weight shard"):
        validate_snapshot(source)


def test_staging_skips_complete_copy_and_repairs_truncation(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "runtime"
    make_snapshot(source)
    (source / ".cache").mkdir()
    (source / ".cache/token").write_text("metadata")
    stage_model(source, destination)
    target = destination / "model-1.safetensors"
    before = target.stat().st_mtime_ns
    stage_model(source, destination)
    assert target.stat().st_mtime_ns == before
    assert not (destination / ".cache").exists()
    target.write_bytes(b"partial")
    stage_model(source, destination)
    assert target.read_bytes() == b"fake test weights"


def test_notebook_is_valid_and_contains_no_saved_outputs():
    notebook = nbformat.read(ROOT / "notebooks/LLaDA_Sudoku_Colab.ipynb", as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            assert cell.outputs == []
            assert cell.execution_count is None
            compile(cell.source, f"cell-{index}", "exec")


def test_yaml_files_parse():
    for path in [ROOT / "environment.yaml", ROOT / "configs/sudoku.yaml", ROOT / ".github/workflows/checks.yaml"]:
        assert isinstance(yaml.safe_load(path.read_text()), dict)
