"""Shared fixed-length inference for LLaDA and iLLaDA checkpoints."""

import json
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer
from generate import generate
from model_profiles import PROFILES, get_profile

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

    @torch.no_grad()
    def predict(self, prompt, steps=128, gen_length=128, block_length=32,
                temperature=0.0, remasking="low_confidence"):
        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        input_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(self.device)

        output = generate(
            self.model,
            input_ids,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            remasking=remasking,
            mask_id=self.mask_id,
        )

        return self.tokenizer.batch_decode(
            output[:, input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0]
