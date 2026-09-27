import json
from pathlib import Path

import nbformat
import pytest
import yaml

from dlm_project.models.cache import cache_model, stage_model, validate_snapshot
from dlm_project.models.profiles import get_profile

from dlm_project.paths import PROJECT_ROOT as ROOT


def make_snapshot(directory, family="llada"):
    directory.mkdir()
    for name in ["tokenizer.json", *get_profile(family)["code_files"]]:
        (directory / name).write_text("{}")
    (directory / "config.json").write_text(json.dumps({"model_type": family}))
    (directory / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ messages }}"} if family == "llada" else {}))
    if family == "illada":
        (directory / "chat_template.jinja").write_text("{{ messages }}")
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"weight": "model-1.safetensors"}}))
    (directory / "model-1.safetensors").write_bytes(b"fake test weights")


@pytest.mark.parametrize("family", ["llada", "illada"])
def test_incomplete_snapshot_is_rejected(tmp_path, family):
    source = tmp_path / "source"
    make_snapshot(source, family)
    (source / "model-1.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="weight shard"):
        validate_snapshot(source, family)


@pytest.mark.parametrize("family", ["llada", "illada"])
def test_staging_skips_complete_copy_and_repairs_truncation(tmp_path, family):
    source, destination = tmp_path / "source", tmp_path / "runtime"
    make_snapshot(source, family)
    (source / ".cache").mkdir()
    (source / ".cache/token").write_text("metadata")
    stage_model(source, destination, family)
    target = destination / "model-1.safetensors"
    before = target.stat().st_mtime_ns
    stage_model(source, destination, family)
    assert target.stat().st_mtime_ns == before
    assert not (destination / ".cache").exists()
    target.write_bytes(b"partial")
    stage_model(source, destination, family)
    assert target.read_bytes() == b"fake test weights"


@pytest.mark.parametrize("name", ["LLaDA", "iLLaDA"])
def test_notebook_is_valid_and_contains_no_saved_outputs(name):
    notebook = nbformat.read(ROOT / f"notebooks/sudoku/{name}_Sudoku_Colab.ipynb", as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            assert cell.outputs == []
            assert cell.execution_count is None
            compile(cell.source, f"cell-{index}", "exec")
    source = "\n".join(cell.source for cell in notebook.cells)
    assert 'BRANCH = "adi"' in source
    assert f"blob/adi/notebooks/sudoku/{name}_Sudoku_Colab.ipynb" in source
    assert "--run-manifest" in source
    assert 'Image(filename=str(plot_path))' in source
    assert "Compare 128" not in source


def test_yaml_files_parse():
    for path in [ROOT / "environment.yaml", ROOT / "configs/sudoku/llada.yaml", ROOT / "configs/sudoku/llada_mcts.yaml", ROOT / "configs/sudoku/illada.yaml", ROOT / ".github/workflows/checks.yaml"]:
        assert isinstance(yaml.safe_load(path.read_text()), dict)


def test_illada_requires_its_own_files_and_template(tmp_path):
    source = tmp_path / "illada"
    make_snapshot(source, "illada")
    validate_snapshot(source, "illada")
    (source / "chat_template.jinja").unlink()
    with pytest.raises(FileNotFoundError, match="chat_template"):
        validate_snapshot(source, "illada")


def test_wrong_family_cannot_overwrite_a_cached_model(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    make_snapshot(source, "illada")
    make_snapshot(destination, "llada")
    before = (destination / "config.json").read_bytes()
    with pytest.raises(ValueError, match="different model family"):
        stage_model(source, destination, "illada")
    assert (destination / "config.json").read_bytes() == before


@pytest.mark.parametrize("family", ["llada", "illada"])
def test_download_selects_pinned_snapshot_and_reuses_staged_weights(tmp_path, monkeypatch, family):
    import huggingface_hub

    source, destination = tmp_path / "drive", tmp_path / "runtime"
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        if not source.exists():
            make_snapshot(source, family)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    cache_model(source, destination, family)
    before = (destination / "model-1.safetensors").stat().st_mtime_ns
    cache_model(source, destination, family)
    assert (destination / "model-1.safetensors").stat().st_mtime_ns == before
    assert len(calls) == 2
    assert calls[0]["repo_id"] == get_profile(family)["repo_id"]
    assert calls[0]["revision"] == get_profile(family)["revision"]
    assert "*.jinja" in calls[0]["allow_patterns"]
    source_info = json.loads((destination / ".model_source.json").read_text())
    assert source_info["revision"] == get_profile(family)["revision"]
