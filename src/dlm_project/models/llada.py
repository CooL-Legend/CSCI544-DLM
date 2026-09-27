"""Shared fixed-length inference for LLaDA and iLLaDA checkpoints."""

import json
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModel, AutoTokenizer
from dlm_project.decoding.baseline import generate
from dlm_project.models.profiles import PROFILES, get_profile
from dlm_project.decoding.mcts_decoding import MCTSConfig, MCTSSearch, complete_confidence, model_device, resolve_decoding


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
                if model_type in ("llada", "illada") and model_type != family:
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

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
            device_map={"": device},
            **hub_options,
        ).eval()

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
        cached distribution statistics into greedy completion. No model is
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
        else:
            analysis = None
            if method == "mcts_entropy":
                sync()
                search_started = perf_counter()
                search = MCTSSearch(self.model, input_ids, self.mask_id, gen_length, config,
                                    tokenizer=self.tokenizer, on_event=emit)
                result = search.run()
                sync()
                stats.update(result.stats)
                stats["search_latency_seconds"] = perf_counter() - search_started
                state, analysis = result.selected.state, result.selected.analysis
            else:
                state = torch.full((1, prompt_len + gen_length), self.mask_id, dtype=torch.long)
                state[:, :prompt_len] = input_ids.cpu()
            sync()
            completion_started = perf_counter()
            completion = complete_confidence(
                self.model, state, prompt_len, self.mask_id, config,
                initial_analysis=analysis, tokenizer=self.tokenizer, on_event=emit,
            )
            output = completion.token_ids
            stats["completion_model_calls"] = completion.model_calls
        sync()
        stats["completion_latency_seconds"] = perf_counter() - completion_started
        text = self.tokenizer.batch_decode(output[:, prompt_len:], skip_special_tokens=True)[0]
        stats["latency_seconds"] = perf_counter() - started
        stats["total_model_calls"] = stats["search_model_calls"] + stats["completion_model_calls"]
        stats["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(device) if is_cuda else None
        emit({"stage": "prediction", "stats": dict(stats), "text": text})
        return {"text": text, "stats": stats, "trace": trace}
