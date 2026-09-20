"""Checkpoint identities shared by inference, downloading, and notebooks."""

PROFILES = {
    "llada": {
        "label": "LLaDA",
        "repo_id": "GSAI-ML/LLaDA-8B-Instruct",
        "revision": "08b83a6feb34df1a6011b80c3c00c7563e963b07",
        "mask_id": 126336,
        "code_files": ("configuration_llada.py", "modeling_llada.py"),
    },
    "illada": {
        "label": "iLLaDA",
        "repo_id": "GSAI-ML/iLLaDA-8B-Instruct",
        "revision": "5769f049224b2cff61106ebce0429ede5deeb434",
        "mask_id": 5,
        "code_files": ("configuration_illada.py", "modeling_illada.py"),
    },
}


def get_profile(family="llada"):
    if family not in PROFILES:
        raise ValueError("model.family must be llada or illada.")
    return dict(PROFILES[family])
