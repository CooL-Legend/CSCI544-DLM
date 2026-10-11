"""Small result contract shared by the four non-search decoders."""

from dataclasses import dataclass, field

import torch


@dataclass
class DecodeResult:
    """Completed IDs [1, prompt+output], actual model calls, and compact events."""

    token_ids: torch.Tensor
    generation_nfes: int
    scoring_nfes: int = 0
    details: dict = field(default_factory=dict)
    trace: list[dict] = field(default_factory=list)

    @property
    def total_nfes(self):
        return self.generation_nfes + self.scoring_nfes


class CountingModel:
    """Count forwards while retaining the wrapped model's device and interface."""

    def __init__(self, model):
        self.model = model
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.model(*args, **kwargs)
