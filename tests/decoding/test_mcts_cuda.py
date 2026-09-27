"""Run on a CUDA runtime (Colab); skip locally when CUDA is unavailable.

These tiny controlled models validate actual device transfers and CUDA math
without loading LLaDA. The notebook's later HF run validates the 8B model.
"""

from types import SimpleNamespace

import pytest
import torch

from dlm_project.decoding.mcts_decoding import MCTSConfig, MCTSSearch, complete_confidence

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
    config = MCTSConfig()
    gpu_model = ControlledModel("cuda", root_probabilities, child_probabilities)
    gpu = MCTSSearch(gpu_model, prompt, mask_id=0, gen_length=4, config=config)
    first_cycle = gpu.step()
    assert gpu.model_calls == len(gpu_model.forward_devices) == 3
    assert torch.device(first_cycle[0]["model_device"]).type == "cuda"
    assert first_cycle[0]["search_state_device"] == "cpu"
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

    completion = complete_confidence(gpu_model, gpu_result.selected.state, 2, 0, config,
                                     initial_analysis=gpu_result.selected.analysis)
    cpu_completion = complete_confidence(cpu_model, cpu_result.selected.state, 2, 0, config,
                                         initial_analysis=cpu_result.selected.analysis)
    assert torch.equal(completion.token_ids, cpu_completion.token_ids)
    assert torch.equal(completion.token_ids[:, :2], prompt.cpu())
    committed = gpu_result.selected.state[:, 2:] != 0
    assert torch.equal(completion.token_ids[:, 2:][committed],
                       gpu_result.selected.state[:, 2:][committed])
    assert completion.model_calls == 1  # One cached commitment, then one CUDA forward.
    assert len(gpu_model.forward_devices) == gpu.model_calls + completion.model_calls
    assert all(torch.device(device).type == "cuda" for device in gpu_model.forward_devices)
