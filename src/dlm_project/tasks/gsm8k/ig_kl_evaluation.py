"""Run the sean-1105 one-step IG-KL method on the frozen 70-question pool.

Sample selection follows E2's 70-question pool. Search and completion use the E1
runner with independent per-question IG-KL history in serial or cooperative
batches, without FIG lookahead or simulation caps.
"""

from dlm_project.experiments.manifests import digest
from dlm_project.tasks.gsm8k.experiment_1 import (
    CHECKPOINT_BUDGETS, MAX_SEARCH_DEPTH, MIN_CANDIDATES,
    context as e1_context, load_model, run_e1_stage,
)
from dlm_project.tasks.gsm8k.experiment_protocol import (
    E2_EVALUATION_COUNT, get_e2_manifest, select_examples,
)


def context(*, stage="evaluation", methods=("medal", "ab_medal"),
            budgets=CHECKPOINT_BUDGETS, max_depth=MAX_SEARCH_DEPTH,
            num_candidates=MIN_CANDIDATES, sample_slice=None, num_samples=None,
            reward_mode="ig_kl", kl_weight=0.05, kl_top_k=32):
    """Keep the old evaluation settings while selecting a fixed 70-pool slice."""
    if stage not in ("evaluation", "eval"):
        raise ValueError("The 70-question IG-KL notebook requires stage='evaluation'.")
    if sample_slice is not None and num_samples is not None:
        raise ValueError("Choose sample_slice or num_samples, not both.")
    count = E2_EVALUATION_COUNT
    if sample_slice is not None:
        if (not isinstance(sample_slice, (tuple, list)) or len(sample_slice) != 2
                or any(type(value) is not int for value in sample_slice)):
            raise ValueError("sample_slice must contain two integer bounds.")
        start, end = sample_slice
        if not 0 <= start < end <= count:
            raise ValueError(f"sample_slice must satisfy 0 <= start < end <= {count}.")
    else:
        if num_samples is not None and (type(num_samples) is not int
                                       or not 1 <= num_samples <= count):
            raise ValueError(f"num_samples must be an integer from 1 to {count}.")
        start, end = 0, count if num_samples is None else num_samples

    ctx = e1_context(
        stage="evaluation", methods=methods, budgets=budgets,
        max_depth=max_depth, num_candidates=num_candidates,
        reward_mode=reward_mode, kl_weight=kl_weight, kl_top_k=kl_top_k,
    )
    manifest = get_e2_manifest(common=ctx["common"])
    examples = select_examples(manifest, "evaluation", expected_count=count)
    ctx.update(
        examples=examples[start:end], manifest=manifest["dataset"],
        manifest_hash=digest(manifest), evaluation_pool="e2_frozen_70",
        sample_slice=[start, end],
    )
    return ctx
