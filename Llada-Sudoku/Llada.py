# llada.py
import torch
from transformers import AutoModel, AutoTokenizer
from generate import generate

class LLaDA:
    def __init__(self, device="cuda", model_path="GSAI-ML/LLaDA-8B-Instruct",
                 dtype="bfloat16", local_files_only=False):
        self.device = device
        if torch.device(device).type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Select a GPU runtime in Colab.")
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
            device_map={"": device},
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
        )

        return self.tokenizer.batch_decode(
            output[:, input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0]
