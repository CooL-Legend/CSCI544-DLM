"""Unit tests for the unified MEDAL and AB-MEDAL batch processing verification."""

from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

from dlm_project.tasks.gsm8k import ab_development as dev
from scripts.verify_batch_processing import verify_method_batching, run_verification, format_summary_table
from tests.decoding.test_ab_batch import BatchModel


class Tokenizer:
    eos_token_id = pad_token_id = 4

    def get_vocab(self):
        return {"[MASK]": 0, "one": 1, "two": 2, "three": 3, "<|endoftext|>": 4, "<|eot_id|>": 5}

    def convert_ids_to_tokens(self, i):
        return next(k for k, v in self.get_vocab().items() if v == i)

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(self.convert_ids_to_tokens(i) for i in ids if not skip_special_tokens or i not in (0, 4, 5))


def mock_wrapper(model=None):
    ctx = dev.context()
    return SimpleNamespace(
        model=model or BatchModel(),
        tokenizer=Tokenizer(),
        mask_id=0,
        identity={k: ctx["common"]["model"][k] for k in ("family", "repo_id", "revision")},
        encode_prompt=lambda text: torch.tensor([[3] * (2 + len(text) % 4)]),
    )


@pytest.mark.parametrize("method", ["medal", "ab_medal"])
def test_verify_single_method_batching_synthetic(tmp_path, method):
    w = mock_wrapper()
    ctx = dev.context(stage="dev", condition="A", indices=(0, 1), method=method)
    prepared = dev.prepare(w, ctx)

    results = verify_method_batching(
        w,
        method=method,
        prepared=prepared,
        base_ctx=ctx,
        output_dir=tmp_path,
        batch_sizes=(2,),
    )

    assert results["all_passed"]
    assert results["method"] == method
    assert "2" in results["batches"]
    batch_res = results["batches"]["2"]
    assert batch_res["padding_probe_passed"]
    assert batch_res["parity_passed"]
    assert (tmp_path / method / "verification_results.json").is_file()


def test_run_verification_end_to_end_synthetic(tmp_path):
    w = mock_wrapper()
    all_results, success = run_verification(
        w,
        methods=("medal", "ab_medal"),
        batch_sizes=(2,),
        num_samples=2,
        condition="A",
        output_dir=tmp_path,
    )

    assert success
    assert "medal" in all_results
    assert "ab_medal" in all_results
    assert all_results["medal"]["batches"]["2"]["parity_passed"]
    assert all_results["ab_medal"]["batches"]["2"]["parity_passed"]

    summary_file = tmp_path / "summary.json"
    summary_md = tmp_path / "summary.md"
    assert summary_file.is_file()
    assert summary_md.is_file()

    table = format_summary_table(all_results)
    assert "MEDAL" in table
    assert "AB_MEDAL" in table
    assert "PASS" in table


def test_verify_batch_processing_notebook_validity():
    import nbformat
    nb_path = Path("notebooks/gsm8k/Verify_Batch_Processing.ipynb")
    assert nb_path.is_file(), "Notebook file must exist."
    nb = nbformat.read(nb_path, as_version=4)
    nbformat.validate(nb)

    # Compile every code cell to verify Python syntax
    code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    assert len(code_cells) >= 8
    for idx, cell in enumerate(code_cells):
        compile(cell.source, f"cell_{idx}", "exec")


def test_verify_batch_processing_cli_help():
    import subprocess
    import sys
    res = subprocess.run([sys.executable, "scripts/verify_batch_processing.py", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "--methods" in res.stdout
    assert "--batch-sizes" in res.stdout
    assert "--fast" in res.stdout

