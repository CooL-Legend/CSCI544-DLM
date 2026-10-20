"""Shared E0 template rendering, including historical offline-report support."""

from dlm_project.tasks.gsm8k.dataset import build_prompt as build_legacy_prompt


def build_e0_prompt(question, settings):
    """Insert the question literally and preserve the supplied examples."""
    if "template" in settings:
        return settings["template"].strip().replace("{GSM8K_question}", question.strip())
    # Saved v1 runs used instructions followed by a Question: block.
    # Offline reports must reconstruct their original prompt, not today's one.
    return build_legacy_prompt(question, settings["instructions"])
