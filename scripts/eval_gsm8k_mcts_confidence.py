"""LLaDA GSM8K accuracy with MCTS initialization and confidence completion."""

from dlm_project.paths import PROJECT_ROOT
from dlm_project.tasks.gsm8k.benchmark import main


if __name__ == "__main__":
    main("mcts_confidence", default_config=PROJECT_ROOT / "configs/gsm8k/mcts_confidence.yaml",
         expected_family="llada")
