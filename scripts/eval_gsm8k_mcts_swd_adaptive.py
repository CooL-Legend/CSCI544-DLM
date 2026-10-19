"""LLaDA GSM8K with KL-adaptive MCTS controls."""

from dlm_project.paths import PROJECT_ROOT
from dlm_project.tasks.gsm8k.benchmark import main


if __name__ == "__main__":
    main("mcts_swd_adaptive", default_config=PROJECT_ROOT / "configs/gsm8k/mcts_swd_adaptive.yaml",
         expected_family="llada")
