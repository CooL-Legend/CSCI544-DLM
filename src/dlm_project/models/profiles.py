"""Checkpoint identities shared by inference, downloading, and notebooks."""

PROFILES = {
    "llada": {
        "model_type": "llada",
        "label": "LLaDA",
        "repo_id": "GSAI-ML/LLaDA-8B-Instruct",
        "revision": "08b83a6feb34df1a6011b80c3c00c7563e963b07",
        "mask_id": 126336,
        "code_files": ("configuration_llada.py", "modeling_llada.py"),
    },
    "illada": {
        "model_type": "illada",
        "label": "iLLaDA",
        "repo_id": "GSAI-ML/iLLaDA-8B-Instruct",
        "revision": "5769f049224b2cff61106ebce0429ede5deeb434",
        "mask_id": 5,
        "code_files": ("configuration_illada.py", "modeling_illada.py"),
    },
    "llada_bgpo": {
        "label": "LLaDA-BGPO",
        "model_type": "llada",
        "repo_id": "THU-KEG/LLaDA-8B-BGPO-sudoku",
        "revision": "101ce156b05e4f9fef9688569b4ddc3241f81356",
        "mask_id": 126336,
        "code_files": ("configuration_llada.py", "modeling_llada.py"),
    },
}


def get_profile(family="llada"):
    if family not in PROFILES:
        raise ValueError(f"model.family must be one of: {', '.join(PROFILES)}.")
    return dict(PROFILES[family])
