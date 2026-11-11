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

    # Verify configuration cell specifies batch of 4 and Condition A (B=256)
    config_cell = nb.cells[10].source
    assert "BATCH_SIZES = [4]" in config_cell
    assert "NUM_QUESTIONS = 4" in config_cell
    assert 'CONDITION = "A"' in config_cell


@pytest.mark.parametrize("method", ["medal", "ab_medal"])
def test_verify_batch_of_4_condition_a_synthetic(tmp_path, method):
    w = mock_wrapper()
    ctx = dev.context(stage="dev", condition="A", indices=(0, 1, 2, 3), method=method)
    prepared = dev.prepare(w, ctx)

    results = verify_method_batching(
        w,
        method=method,
        prepared=prepared,
        base_ctx=ctx,
        output_dir=tmp_path,
        batch_sizes=(4,),
    )

    assert results["all_passed"]
    assert results["method"] == method
    assert "4" in results["batches"]
    batch_res = results["batches"]["4"]
    assert batch_res["padding_probe_passed"]
    assert batch_res["parity_passed"]
    assert (tmp_path / method / "verification_results.json").is_file()


def test_verify_batch_processing_cli_help():
    import subprocess
    import sys
    res = subprocess.run([sys.executable, "scripts/verify_batch_processing.py", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "--methods" in res.stdout
    assert "--batch-sizes" in res.stdout
    assert "--fast" in res.stdout


def test_install_varlen_flash_attention_eliminates_padding_bias():
    from dlm_project.models.llada import install_varlen_flash_attention
    from dlm_project.decoding.ab_batch import pack_states

    sdpa_masks_seen = []

    class MockRotary(torch.nn.Module):
        def forward(self, q, k):
            pos = torch.arange(q.shape[-2], device=q.device, dtype=q.dtype).view(1, 1, -1, 1)
            return q + 0.01 * pos, k + 0.01 * pos

    class MockLLaDABlock(torch.nn.Module):
        def __init__(self, d_model=16, n_heads=2):
            super().__init__()
            self.config = SimpleNamespace(
                d_model=d_model,
                n_heads=n_heads,
                effective_n_kv_heads=n_heads,
                rope=True,
                attention_dropout=0.0,
            )
            self.q_norm = None
            self.k_norm = None
            self.q_proj = torch.nn.Linear(d_model, d_model, bias=False)
            self.k_proj = torch.nn.Linear(d_model, d_model, bias=False)
            self.v_proj = torch.nn.Linear(d_model, d_model, bias=False)
            self.attn_out = torch.nn.Linear(d_model, d_model, bias=False)
            self.rotary_emb = MockRotary()

        def _scaled_dot_product_attention(self, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
            sdpa_masks_seen.append(attn_mask)
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal
            )

        def attention(self, q, k, v, attention_bias=None, layer_past=None, use_cache=False):
            B, T, C = q.shape
            nh = self.config.n_heads
            hs = C // nh
            q = q.view(B, T, nh, hs).transpose(1, 2)
            k = k.view(B, T, nh, hs).transpose(1, 2)
            v = v.view(B, T, nh, hs).transpose(1, 2)
            q, k = self.rotary_emb(q, k)
            att = self._scaled_dot_product_attention(q, k, v, attn_mask=attention_bias, is_causal=False)
            att = att.transpose(1, 2).contiguous().view(B, T, C)
            return self.attn_out(att), None

        def forward(self, x, attention_bias=None):
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            h, _ = self.attention(q, k, v, attention_bias=attention_bias)
            return x + h

    class MockInnerLLaDA(torch.nn.Module):
        def __init__(self, vocab_size=32, d_model=16):
            super().__init__()
            self.wte = torch.nn.Embedding(vocab_size, d_model)
            self.blocks = torch.nn.ModuleList([MockLLaDABlock(d_model, 2), MockLLaDABlock(d_model, 2)])
            self.ff_out = torch.nn.Linear(d_model, vocab_size, bias=False)
            self.transformer = SimpleNamespace(wte=self.wte, blocks=self.blocks, ff_out=self.ff_out)

        def forward(
            self,
            input_ids,
            input_embeddings=None,
            attention_mask=None,
            attention_bias=None,
            past_key_values=None,
            use_cache=False,
            last_logits_only=False,
            output_hidden_states=None,
        ):
            if attention_mask is not None and 0.0 in attention_mask:
                # Replicate modeling_llada.py's 4D attention_bias construction
                mask_f = (attention_mask == 0.0).to(dtype=torch.float32) * torch.finfo(torch.float32).min
                attention_bias = mask_f[:, None, None, :]
            x = self.wte(input_ids)
            for block in self.blocks:
                x = block(x, attention_bias=attention_bias)
            logits = self.ff_out(x)
            from collections import namedtuple
            Out = namedtuple("LLaDAOutput", ["logits", "attn_key_values", "hidden_states"])
            return Out(logits=logits, attn_key_values=None, hidden_states=None)

    class MockLLaDAModelLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = MockInnerLLaDA()

        def forward(self, input_ids, attention_mask=None, **kwargs):
            return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    torch.manual_seed(42)
    model = MockLLaDAModelLM().eval()
    assert install_varlen_flash_attention(model) is True

    # Four variable-length sequences matching the 4-question batch structure
    states = [
        torch.randint(1, 30, (1, 23)),
        torch.randint(1, 30, (1, 15)),
        torch.randint(1, 30, (1, 19)),
        torch.randint(1, 30, (1, 11)),
    ]
    ids, attention, lengths = pack_states(states, pad_token_id=0, device=torch.device("cpu"))

    sdpa_masks_seen.clear()
    batched_out = model(ids, attention_mask=attention)
    # Verify every single SDPA call received attn_mask=None (enabling FlashAttention-2)
    assert len(sdpa_masks_seen) == len(states) * 2
    assert all(m is None for m in sdpa_masks_seen)

    for i, (state, length) in enumerate(zip(states, lengths)):
        unpadded_out = model(state)
        padded_serial_out = model(ids[i : i + 1], attention_mask=attention[i : i + 1])

        diff_ps = (unpadded_out.logits[0] - padded_serial_out.logits[0, :length]).abs().max().item()
        diff_batch = (unpadded_out.logits[0] - batched_out.logits[i, :length]).abs().max().item()
        assert diff_ps == 0.0, f"Row {i} padded_serial diff was {diff_ps}"
        assert diff_batch < 1e-6, f"Row {i} batched diff was {diff_batch}"


