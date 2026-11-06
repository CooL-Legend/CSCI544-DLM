"""Names shared by the E1 runner and its offline E3 reports."""

VARIANTS = {
    "medal": {"base_method": "medal", "reward_mode": "ig", "label": "MEDAL"},
    "ab_medal": {"base_method": "ab_medal", "reward_mode": "ig", "label": "AB-MEDAL"},
    "medal_adalook": {"base_method": "medal", "reward_mode": "adalook_adaptive", "label": "MEDAL + AdaLook"},
    "ab_medal_adalook": {"base_method": "ab_medal", "reward_mode": "adalook_adaptive", "label": "AB-MEDAL + AdaLook"},
}
METHODS = tuple(VARIANTS)
ADALOOK_DEFAULTS = {
    "adalook_alpha": 0.1,
    "adalook_confidence_threshold": 0.8,
    "adalook_max_depth": 2,
    "adalook_variance_threshold": 0.01,
    "adalook_trigger_threshold": 0.2,
    "adalook_min_masks": 2,
}


def variant(method):
    if method not in VARIANTS:
        raise ValueError(f"Unknown E1 method {method!r}; expected one of {METHODS}.")
    return dict(VARIANTS[method])


def is_adaptive(method):
    """Whether the outer search uses Bayesian GEN/CONT (independent of reward)."""
    return variant(method)["base_method"] == "ab_medal"
