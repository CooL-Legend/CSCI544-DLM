"""Run on a CUDA runtime (Colab); skip locally when CUDA is unavailable.

These tiny controlled models validate actual device transfers and CUDA math
without loading LLaDA. The notebook's later HF run validates the 8B model.
"""

from types import SimpleNamespace

import pytest
import torch

from dlm_project.decoding.mcts_decoding import MCTSConfig, MCTSSearch, complete_medal

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime required")


class ControlledModel(torch.nn.Module):
    def __init__(self, device, root_probabilities, child_probabilities):
        super().__init__()
        self.register_buffer("root_probabilities", torch.tensor(root_probabilities, device=device))
        self.register_buffer("child_probabilities", torch.tensor(child_probabilities, device=device))
        self.forward_devices = []

    @property
    def device(self):
        return self.root_probabilities.device

    def forward(self, ids):
        assert ids.device == self.device
        assert not torch.is_grad_enabled()
        self.forward_devices.append(str(ids.device))
        is_root = (ids[0, 2:] == 0).all()
        probabilities = torch.where(is_root, self.root_probabilities, self.child_probabilities)
        return SimpleNamespace(logits=probabilities.log().expand(1, ids.shape[1], -1))


@pytest.mark.parametrize("root_probabilities,child_probabilities", [
    ([0., .7, .2, .1], [0., .7, .2, .1]),
    ([0., .99, .005, .005], [0., 1/3, 1/3, 1/3]),  # Negative first-edge rewards.
    ([0., 1., 0., 0.], [0., 1., 0., 0.]),  # Zero entropy stays finite on CUDA.
])
def test_cuda_search_and_completion_match_cpu(root_probabilities, child_probabilities):
    # A mask in the prompt is fixed; only output masks may be committed.
    prompt = torch.tensor([[3, 0]], device="cuda")
    config = MCTSConfig(init_tokens=2, top_k_tokens=2, top_k_actions=2)
    gpu_model = ControlledModel("cuda", root_probabilities, child_probabilities)
    gpu = MCTSSearch(gpu_model, prompt, mask_id=0, gen_length=4, config=config)
    first_cycle = gpu.step()
    assert gpu.model_calls == len(gpu_model.forward_devices) == 3
    initialization = next(e for e in first_cycle if e["stage"] == "initialization")
    assert torch.device(initialization["model_device"]).type == "cuda"
    assert initialization["search_state_device"] == "cpu"
    gpu_result = gpu.run()

    cpu_model = ControlledModel("cpu", root_probabilities, child_probabilities)
    cpu = MCTSSearch(cpu_model, prompt.cpu(), mask_id=0, gen_length=4, config=config)
    cpu_result = cpu.run()
    assert gpu_result.stats["candidate_count"] == cpu_result.stats["candidate_count"]
    assert gpu_result.stats["selected_depth"] == cpu_result.stats["selected_depth"]
    assert torch.equal(gpu_result.selected.state, cpu_result.selected.state)
    assert gpu_result.stats["selected_score"] == pytest.approx(cpu_result.stats["selected_score"], abs=2e-5)
    gpu_evaluations = [e for e in gpu_result.trace if e["stage"] == "evaluation"]
    cpu_evaluations = [e for e in cpu_result.trace if e["stage"] == "evaluation"]
    assert [e["reward"] for e in gpu_evaluations] == pytest.approx(
        [e["reward"] for e in cpu_evaluations], abs=2e-5, rel=2e-5)
    assert all(node.state.device.type == "cpu" and node.analysis.entropies.device.type == "cpu"
               for node in gpu.nodes)

    completion = complete_medal(gpu_model, gpu_result.selected.state, 2, 0, config, seed=1)
    cpu_completion = complete_medal(cpu_model, cpu_result.selected.state, 2, 0, config, seed=1)
    # CPU and CUDA generators need not draw identical categorical samples.
    again = complete_medal(gpu_model, gpu_result.selected.state, 2, 0, config, seed=1)
    assert torch.equal(completion.token_ids, again.token_ids)
    assert (cpu_completion.token_ids[:, 2:] != 0).all()
    assert torch.equal(completion.token_ids[:, :2], prompt.cpu())
    committed = gpu_result.selected.state[:, 2:] != 0
    assert torch.equal(completion.token_ids[:, 2:][committed],
                       gpu_result.selected.state[:, 2:][committed])
    assert completion.model_calls == 2  # Fresh distribution per remaining mask.
    assert len(gpu_model.forward_devices) == gpu.model_calls + completion.model_calls + again.model_calls
    assert all(torch.device(device).type == "cuda" for device in gpu_model.forward_devices)
