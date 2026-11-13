"""Keep generated SRIG evaluation notebook controls aligned with CARC runners."""


def configure_srig_eval(notebook):
    if any("PROFILE_RESOURCES = True" in "".join(cell["source"]) for cell in notebook["cells"]):
        return notebook
    for cell in notebook["cells"]:
        was_list = isinstance(cell["source"], list)
        text = "".join(cell["source"]) if was_list else cell["source"]
        if cell["cell_type"] == "code":
            if "REPO_URL =" in text:
                text += '''
if not IN_COLAB:
    candidates = ([Path(os.environ['REPO_DIR'])] if 'REPO_DIR' in os.environ else []) + [Path.cwd(), *Path.cwd().parents]
    REPO_DIR = next((p for p in candidates if (p / 'src/dlm_project').is_dir()), None)
    if REPO_DIR is None:
        raise RuntimeError('Set REPO_DIR to an existing repository checkout.')
    os.chdir(REPO_DIR)
    for entry in (str(REPO_DIR), str(REPO_DIR / 'src')):
        if entry not in sys.path:
            sys.path.insert(0, entry)
'''
            if "Dependencies installed successfully." in text:
                text += '''
if not IN_COLAB:
    import importlib.util
    missing = [name for name in ('torch', 'transformers', 'accelerate', 'yaml', 'matplotlib', 'pandas', 'pytest')
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(f'Prepare the CARC environment before launching this kernel. Missing: {missing}')
'''
            if "OUTPUT_DIR = Path('results/e2_srig_eval')" in text:
                text += '''
if not IN_COLAB:
    PERSIST_ROOT = Path(os.environ.get('PERSIST_ROOT', REPO_DIR / '.cache/carc')).resolve()
    OUTPUT_DIR = Path(os.environ.get('ARTIFACT_ROOT', PERSIST_ROOT / 'results/e2_srig_eval')).resolve()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('HF_HOME', str(PERSIST_ROOT / 'cache/huggingface'))
'''
            if "BATCH_SIZE = 2" in text:
                text = text.replace("# On NVIDIA L4 (22.4 GiB VRAM), BATCH_SIZE = 2 is recommended for stability;\n# BATCH_SIZE = 4 is suited for A100 (40/80 GB) or High-RAM runtimes.\nBATCH_SIZE = 2",
                                    "# Baseline from the observed L40 run; measure peaks on each target GPU.\nBATCH_SIZE = int(os.environ.get('BATCH_SIZE', '4'))\nPROFILE_RESOURCES = True")
                text = text.replace("print('Using model location:', MODEL_PATH)",
                                    "MODEL_PATH = os.environ.get('MODEL_PATH', MODEL_PATH)\nprint('Using model location:', MODEL_PATH)")
            if "batch_size=BATCH_SIZE," in text:
                text = text.replace("    batch_size=BATCH_SIZE,", "    batch_size=BATCH_SIZE,\n    profile_resources=PROFILE_RESOURCES,")
            # Resolve link targets so a relative non-Colab output directory works too.
            text = text.replace(".symlink_to(q_dir, target_is_directory=True)", ".symlink_to(q_dir.resolve(), target_is_directory=True)")
        else:
            text = text.replace("batch size 2", "batch size 4 with resource profiling")
        cell["source"] = text.splitlines(keepends=True) if was_list else text
    return notebook
