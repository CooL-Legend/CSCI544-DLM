"""Controlled CPU model and tokenizer used by decoder and workflow tests."""

from types import SimpleNamespace
import torch


MASK = 0
PROMPT = torch.tensor([[3, MASK]])  # A prompt mask must also remain untouched.


class FakeModel:
    device = torch.device("cpu")

    def __init__(self, policy=None):
        self.policy = policy or (lambda output: [0.0, 0.7, 0.2, 0.1])
        self.seen = []

    def __call__(self, ids, **kwargs):
        assert not torch.is_grad_enabled()
        assert ids.shape[0] == 1  # Siblings are evaluated sequentially.
        self.seen.append(ids.cpu().clone())
        p = torch.tensor(self.policy(ids[0, PROMPT.shape[1]:]), dtype=torch.float32)
        if p.ndim == 1:
            logits = p.log().expand(1, ids.shape[1], -1).clone()
        else:
            logits = torch.cat([p[:1].expand(PROMPT.shape[1], -1), p]).log()[None]
        return SimpleNamespace(logits=logits)


class FakeTokenizer:
    def decode(self, ids, **kwargs):
        return "".join({0: "[MASK]", 1: " one", 2: "two", 3: "3"}[i] for i in ids)

    def batch_decode(self, ids, **kwargs):
        return [self.decode(row.tolist(), **kwargs) for row in ids]

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=PROMPT.clone())
