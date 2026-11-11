# Experiment 0: Search-Free Baselines

Experiment 0 evaluates pure search-free LLaDA baselines on GSM8K using the shared protocol (`gsm8k_shared_v2`):
- **LLaDA Confidence** (`confidence_t0`, 256 NFEs, $T=0$) — Primary baseline.
- **LLaDA Entropy** (`entropy_t0`, 256 NFEs, $T=0$) — Optional baseline (via `--include-entropy` or `--methods confidence entropy`).

All MCTS search conditions (MEDAL vs. AB-MEDAL across budgets 256, 512, 1024) are unified and evaluated in Experiment 1 / Experiment 3 via `scripts/run_e1_e3.py`.

See the [shared experiment protocol guide](experiments.md) for the frozen samples, exact two-shot prompt, and metric definitions. The [historical implementation guide](experiment_0/README.md) records legacy v1–v4 behavior.
