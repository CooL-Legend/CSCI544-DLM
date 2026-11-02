"""Shared fixed-length inference for original LLaDA, BGPO, and iLLaDA."""

import json
from pathlib import Path
from time import perf_counter
import types

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from dlm_project.decoding.baseline import generate
from dlm_project.models.profiles import PROFILES, get_profile
from dlm_project.decoding.mcts_decoding import MCTSConfig, MCTSSearch, complete_confidence, model_device, resolve_decoding


def install_varlen_flash_attention(model):
    """Patch LLaDAModel to use varlen token packing and per-sequence unmasked FlashAttention-2.

    HuggingFace's modeling_llada.py converts any 2D attention_mask containing 0s into a
    dense 4D float attention_bias tensor (B, 1, T, T). Passing that 4D float bias into
    F.scaled_dot_product_attention disables PyTorch FlashAttention-2 and (when T % 8 != 0
    or cached bias slices are non-contiguous) falls back to unfused Math SDPA, causing
    both >2x slowdowns and large bfloat16 logit drift.

    Because every layer in LLaDAModel outside _scaled_dot_product_attention is strictly
    token-wise and our batches are right-padded:
      1. For B == 1 with right-padding (e.g. padded_serial), we slice to the valid prefix
         length L_0 and call the unmasked forward pass (attn_mask=None), guaranteeing
         0.000000 bit-for-bit parity with unpadded serial execution.
      2. For B > 1 with right-padding, we pack all valid tokens [input_ids[i, :L_i]] into
         a single contiguous sequence (1, sum(L_i)) for all token-wise linear/MLP/LM-head
         layers (sharing 100% of weight reads with zero wasted pad FLOPs), and inside each
         block's attention() split Q, K, V by lengths to apply RoPE and unmasked
         FlashAttention-2 (attn_mask=None) per sequence at its exact unpadded length L_i.
    """
    llada_inner = getattr(model, "model", None)
    transformer = getattr(llada_inner, "transformer", None)
    if llada_inner is None or transformer is None or getattr(llada_inner, "_varlen_patched", False):
        return False

    orig_inner_forward = llada_inner.forward

    def _patch_block(block):
        orig_attn = block.attention

        def varlen_attention(self, q, k, v, attention_bias=None, layer_past=None, use_cache=False):
            v_lengths = getattr(llada_inner, "_varlen_lengths", None)
            if v_lengths is None:
                return orig_attn(q, k, v, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache)
            B, T, C = q.size()
            dtype = k.dtype
            if getattr(self, "q_norm", None) is not None and getattr(self, "k_norm", None) is not None:
                q = self.q_norm(q).to(dtype=dtype)
                k = self.k_norm(k).to(dtype=dtype)
            nh = self.config.n_heads
            n_kv = self.config.effective_n_kv_heads
            hs = C // nh
            q = q.view(B, T, nh, hs).transpose(1, 2)
            k = k.view(B, T, n_kv, hs).transpose(1, 2)
            v = v.view(B, T, n_kv, hs).transpose(1, 2)

            q_segs = q.split(v_lengths, dim=2)
            k_segs = k.split(v_lengths, dim=2)
            v_segs = v.split(v_lengths, dim=2)
            att_segs = []
            dropout_p = 0.0 if not self.training else self.config.attention_dropout
            for qi, ki, vi in zip(q_segs, k_segs, v_segs):
                if self.config.rope:
                    qi, ki = self.rotary_emb(qi, ki)
                atti = self._scaled_dot_product_attention(
                    qi, ki, vi, attn_mask=None, dropout_p=dropout_p, is_causal=False
                )
                att_segs.append(atti)
            att = torch.cat(att_segs, dim=2)
            att = att.transpose(1, 2).contiguous().view(B, T, C)
            return self.attn_out(att), None

        block.attention = types.MethodType(varlen_attention, block)

    for module in llada_inner.modules():
        if hasattr(module, "attention") and hasattr(module, "_scaled_dot_product_attention"):
            _patch_block(module)

    def varlen_inner_forward(
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
        if (
            attention_mask is not None
            and input_ids is not None
            and input_embeddings is None
            and attention_bias is None
            and past_key_values is None
            and not use_cache
            and not last_logits_only
        ):
            B, T = input_ids.shape
            row_lengths = [int(attention_mask[i].sum().item()) for i in range(B)]
            is_right_padded = all(
                0 < L <= T
                and bool((attention_mask[i, :L] == 1).all())
                and bool((attention_mask[i, L:] == 0).all())
                for i, L in enumerate(row_lengths)
            )
            if is_right_padded:
                if B == 1:
                    L0 = row_lengths[0]
                    if L0 == T:
                        return orig_inner_forward(
                            input_ids=input_ids,
                            input_embeddings=None,
                            attention_mask=None,
                            attention_bias=None,
                            past_key_values=None,
                            use_cache=False,
                            last_logits_only=False,
                            output_hidden_states=output_hidden_states,
                        )
                    out = orig_inner_forward(
                        input_ids=input_ids[:, :L0],
                        input_embeddings=None,
                        attention_mask=None,
                        attention_bias=None,
                        past_key_values=None,
                        use_cache=False,
                        last_logits_only=False,
                        output_hidden_states=output_hidden_states,
                    )
                    padded_logits = out.logits.new_zeros((1, T, out.logits.shape[-1]))
                    padded_logits[:, :L0, :] = out.logits
                    return out._replace(logits=padded_logits)

                if getattr(self, "_varlen_exact_gemm", False):
                    seg_outs = [
                        orig_inner_forward(
                            input_ids=input_ids[i : i + 1, :L],
                            input_embeddings=None,
                            attention_mask=None,
                            attention_bias=None,
                            past_key_values=None,
                            use_cache=False,
                            last_logits_only=False,
                            output_hidden_states=output_hidden_states,
                        )
                        for i, L in enumerate(row_lengths)
                    ]
                    padded_logits = seg_outs[0].logits.new_zeros((B, T, seg_outs[0].logits.shape[-1]))
                    for i, (L, seg_out) in enumerate(zip(row_lengths, seg_outs)):
                        padded_logits[i, :L, :] = seg_out.logits[0]
                    return seg_outs[0]._replace(logits=padded_logits)

                packed_ids = torch.cat([input_ids[i, :L] for i, L in enumerate(row_lengths)], dim=0).unsqueeze(0)
                self._varlen_lengths = row_lengths
                try:
                    out = orig_inner_forward(
                        input_ids=packed_ids,
                        input_embeddings=None,
                        attention_mask=None,
                        attention_bias=None,
                        past_key_values=None,
                        use_cache=False,
                        last_logits_only=False,
                        output_hidden_states=output_hidden_states,
                    )
                finally:
                    self._varlen_lengths = None
                padded_logits = out.logits.new_zeros((B, T, out.logits.shape[-1]))
                for i, (L, seg) in enumerate(zip(row_lengths, out.logits[0].split(row_lengths, dim=0))):
                    padded_logits[i, :L, :] = seg
                return out._replace(logits=padded_logits)

        return orig_inner_forward(
            input_ids=input_ids,
            input_embeddings=input_embeddings,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            past_key_values=past_key_values,
            use_cache=use_cache,
            last_logits_only=last_logits_only,
            output_hidden_states=output_hidden_states,
        )

    llada_inner.forward = types.MethodType(varlen_inner_forward, llada_inner)
    llada_inner._varlen_patched = True
    llada_inner._varlen_exact_gemm = False
    return True


class _CountForwards:
    """Count actual baseline forwards without changing the existing sampler."""

    def __init__(self, model):
        self.model, self.calls = model, 0

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.model(*args, **kwargs)


class LLaDA:
    def __init__(self, device="cuda", model_path=None,
                 dtype="bfloat16", local_files_only=False, family="llada"):
        profile = get_profile(family)
        model_path = model_path or profile["repo_id"]
        if model_path in {item["repo_id"] for item in PROFILES.values()} and model_path != profile["repo_id"]:
            raise ValueError("model.family does not match the selected Hub checkpoint.")
        self.family = family
        self.mask_id = profile["mask_id"]
        self.device = device
        if torch.device(device).type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Select a GPU runtime in Colab.")
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]

        local_path = Path(model_path)
        source = None
        if local_path.is_dir():
            source_file = local_path / ".model_source.json"
            if source_file.is_file():
                source = json.loads(source_file.read_text())
                if source.get("repo_id") != profile["repo_id"] or source.get("revision") != profile["revision"]:
                    raise ValueError("The cached snapshot does not match the selected model family/revision.")
            config_file = local_path / "config.json"
            if config_file.is_file():
                model_type = json.loads(config_file.read_text()).get("model_type")
                if model_type in ("llada", "illada") and model_type != profile["model_type"]:
                    raise ValueError("model.family does not match the local checkpoint.")
        # The registered Hub checkpoints are pinned; arbitrary local checkpoints
        # retain their own identity rather than being mislabeled as the official revision.
        hub_options = {"revision": profile["revision"]} if model_path == profile["repo_id"] else {}
        self.identity = {
            "family": family, "model_path": str(model_path), "mask_id": self.mask_id,
            "repo_id": source["repo_id"] if source else (model_path if not local_path.is_dir() else None),
            "revision": source["revision"] if source else hub_options.get("revision"),
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
            **hub_options,
        )
        if not self.tokenizer.chat_template and (local_path / "chat_template.jinja").is_file():
            self.tokenizer.chat_template = (local_path / "chat_template.jinja").read_text()
        if family == "illada":
            if not self.tokenizer.chat_template:
                raise ValueError("iLLaDA requires its checkpoint's chat_template.jinja.")
            if self.tokenizer.convert_tokens_to_ids("<[MASK]>") != self.mask_id:
                raise ValueError("The iLLaDA tokenizer must map <[MASK]> to token 5.")

        model_options = {}
        if family == "llada_bgpo":
            # The released checkpoint calls flash_attn_func even when the
            # optional package is absent. Select its native PyTorch SDPA branch.
            model_config = AutoConfig.from_pretrained(
                model_path, trust_remote_code=True,
                local_files_only=local_files_only, **hub_options,
            )
            model_config.flash_attention = False
            model_config.flash_attn_varlen = False
            model_options["config"] = model_config
            self.identity["attention_overrides"] = {
                "flash_attention": False, "flash_attn_varlen": False,
            }

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
            device_map={"": device},
            **model_options,
            **hub_options,
        ).eval()
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        install_varlen_flash_attention(self.model)

    def encode_prompt(self, prompt):
        """Format one chat prompt and return input IDs [1, P] on the model device."""
        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self.device)

    def predict(self, prompt, steps=128, gen_length=128, block_length=32,
                temperature=0.0, remasking="low_confidence", decoding=None):
        """Compatibility interface: return text, with the original baseline default."""
        return self.predict_with_stats(
            prompt, steps, gen_length, block_length, temperature, remasking,
            decoding=decoding,
        )["text"]

    @torch.inference_mode()
    def predict_with_stats(self, prompt, steps=128, gen_length=128, block_length=32,
                           temperature=0.0, remasking="low_confidence", decoding=None,
                           on_event=None):
        """Return {text, stats, trace}; optionally stream each trace event to a sink.

        MCTS initializes a partial state and passes that exact state and its
        independent candidate evaluations into adjusted-score completion. No model is
        reloaded here. Timings exclude model loading and synchronize CUDA.
        """
        generation = dict(steps=steps, gen_length=gen_length, block_length=block_length,
                          temperature=temperature, remasking=remasking)
        options = resolve_decoding(decoding, generation)
        config = MCTSConfig(**options["mcts"])
        method = options["method"]
        # Bare predict() preserves the caller's RNG behavior. Experiment configs
        # explicitly supply a seed, recorded with the saved prediction.
        if decoding is not None:
            torch.manual_seed(options["seed"])
        device = torch.device(self.device)
        is_cuda = device.type == "cuda"

        def sync():
            if is_cuda:
                torch.cuda.synchronize(device)

        sync()
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        started = perf_counter()
        input_ids = self.encode_prompt(prompt)
        prompt_len = input_ids.shape[1]
        trace = []

        def emit(event):
            trace.append(event)
            if on_event is not None:
                on_event(event)

        stats = {"method": method, "seed": options["seed"] if decoding is not None else None,
                 "mcts_config": options["mcts"],
                 "model_device": str(model_device(self.model)),
                 "search_state_device": "cpu" if method == "mcts_entropy" else None,
                 "search_model_calls": 0, "completion_model_calls": 0,
                 "search_latency_seconds": 0.0, "selected_depth": 0,
                 "candidate_count": 0, "fallback_used": False, "stop_reason": None}
        if method == "baseline":
            counted_model = _CountForwards(self.model)
            sync()
            completion_started = perf_counter()
            output = generate(counted_model, input_ids, **generation, mask_id=self.mask_id)
            stats["completion_model_calls"] = counted_model.calls
        elif method == "mcts_entropy":
            from dlm_project.decoding.mcts_confidence import decode
            result = decode(self.model, input_ids, self.mask_id, steps=steps, gen_length=gen_length,
                            block_length=block_length, temperature=temperature, seed=options["seed"],
                            mcts=config, tokenizer=self.tokenizer, on_event=emit)
            output = result.token_ids
            stats.update(result.details)
            completion_started = None
        else:
            state = torch.full((1, prompt_len + gen_length), self.mask_id, dtype=torch.long)
            state[:, :prompt_len] = input_ids.cpu()
            sync()
            completion_started = perf_counter()
            completion = complete_confidence(self.model, state, prompt_len, self.mask_id, config,
                                             tokenizer=self.tokenizer, on_event=emit)
            output = completion.token_ids
            stats["completion_model_calls"] = completion.model_calls
        sync()
        if completion_started is not None:
            stats["completion_latency_seconds"] = perf_counter() - completion_started
        text = self.tokenizer.batch_decode(output[:, prompt_len:], skip_special_tokens=True)[0]
        stats["latency_seconds"] = perf_counter() - started
        stats["total_model_calls"] = stats["search_model_calls"] + stats["completion_model_calls"]
        stats["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(device) if is_cuda else None
        emit({"stage": "prediction", "stats": dict(stats), "text": text})
        return {"text": text, "stats": stats, "trace": trace}
