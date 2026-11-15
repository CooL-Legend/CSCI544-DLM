"""Local model lookup for offline CARC jobs."""

from pathlib import Path


def resolve_local_llada(model_dir, *, model_path=None, hub_cache=None):
    """Find an existing LLaDA snapshot without downloads or cache reconstruction.

    A MODEL_PATH override is authoritative. MODEL_DIR accepts the historical
    flat download or common Hugging Face cache layouts; automatic hub lookup
    uses only the project's pinned revision, never an arbitrary latest snapshot.
    """
    from dlm_project.models.profiles import get_profile

    if model_path is not None:
        explicit = Path(model_path).expanduser().resolve()
        if (explicit / "config.json").is_file():
            return explicit
        raise FileNotFoundError(
            f"MODEL_PATH has no config.json: {explicit}. "
            "Point MODEL_PATH at the local snapshot, or unset it to use MODEL_DIR."
        )

    profile = get_profile("llada")
    name = profile["repo_id"].split("/")[-1]
    cache_name = "models--" + profile["repo_id"].replace("/", "--")
    root = Path(model_dir).expanduser().resolve()
    candidates = [root, root / name, root / "models" / name,
                  root / "snapshots" / profile["revision"]]
    hubs = [root, root / "hub", root / "huggingface" / "hub"]
    if hub_cache is not None:
        hubs.append(Path(hub_cache).expanduser().resolve())
    candidates.extend(hub / cache_name / "snapshots" / profile["revision"] for hub in hubs)
    for candidate in candidates:
        if (candidate / "config.json").is_file():
            return candidate
    searched = "\n".join(f"  {path}" for path in dict.fromkeys(candidates))
    raise FileNotFoundError(
        "No local LLaDA snapshot found. Set MODEL_DIR to the prepared model folder "
        "or MODEL_PATH to its snapshot directory. No download was attempted.\n"
        f"Searched for config.json in:\n{searched}"
    )
