"""Generator script to build Experiment 0 (Confidence and Entropy) Test Colab notebooks.

Constructs:
    1. notebooks/gsm8k/E0_Experiment_Confidence_Test.ipynb (200 Primary + 300 Extended = 500 Test Questions)
    2. notebooks/gsm8k/E0_Experiment_Entropy_Test.ipynb    (200 Primary + 300 Extended = 500 Test Questions)
"""

from pathlib import Path
import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks/gsm8k"

E0_NOTEBOOK_SPECS = {
    "confidence": {
        "mode": "confidence",
        "companion_mode": "entropy",
        "upper": "Confidence",
        "companion_upper": "Entropy",
        "condition_id": "confidence_t0",
        "selection_rule": "token_probability",
        "full_name": "LLaDA Confidence Decoding (Token-Probability Position Selection at T=0)",
        "filename": "E0_Experiment_Confidence_Test.ipynb",
        "output_subdir": "results/e0_confidence_test",
        "companion_output_subdir": "results/e0_entropy_test",
        "equation_md": (
            "At each of the 256 unmasking steps ($T=0$, 0 search NFEs, 256 completion NFEs), "
            "for current masked state $s_t$ with remaining masked positions $\\mathcal{M}(s_t)$, "
            "each masked position $j \\in \\mathcal{M}(s_t)$ proposes its highest-probability non-mask token "
            "$\\hat{v}_j = \\arg\\max_{v \\neq [\\text{MASK}]} p_\\theta(v \\mid j, s_t)$ and commits the single position "
            "$j^*$ maximizing token confidence:\n"
            "$$\n"
            "j^* = \\arg\\max_{j \\in \\mathcal{M}(s_t)} \\max_{v \\neq [\\text{MASK}]} p_\\theta(v \\mid j, s_t), "
            "\\qquad s_{t+1} = s_t \\oplus (j^*, \\hat{v}_{j^*})\n"
            "$$\n\n"
        ),
    },
    "entropy": {
        "mode": "entropy",
        "companion_mode": "confidence",
        "upper": "Entropy",
        "companion_upper": "Confidence",
        "condition_id": "entropy_t0",
        "selection_rule": "lowest_entropy",
        "full_name": "LLaDA Entropy Decoding (Lowest-Predictive-Entropy Position Selection at T=0)",
        "filename": "E0_Experiment_Entropy_Test.ipynb",
        "output_subdir": "results/e0_entropy_test",
        "companion_output_subdir": "results/e0_confidence_test",
        "equation_md": (
            "At each of the 256 unmasking steps ($T=0$, 0 search NFEs, 256 completion NFEs), "
            "for current masked state $s_t$ with remaining masked positions $\\mathcal{M}(s_t)$, "
            "each masked position $j \\in \\mathcal{M}(s_t)$ is scored by its predictive Shannon entropy "
            "$h_j(s_t)$ over non-mask vocabulary tokens, and the position $j^*$ with the **lowest predictive entropy** "
            "commits its greedy token $\\hat{v}_{j^*} = \\arg\\max_{v \\neq [\\text{MASK}]} p_\\theta(v \\mid j^*, s_t)$:\n"
            "$$\n"
            "h_j(s_t) = -\\sum_{v \\neq [\\text{MASK}]} p_\\theta(v \\mid j, s_t) \\log p_\\theta(v \\mid j, s_t), "
            "\\qquad j^* = \\arg\\min_{j \\in \\mathcal{M}(s_t)} h_j(s_t)\n"
            "$$\n\n"
        ),
    },
}


def _create_e0_test_notebook(spec: dict) -> Path:
    mode = spec["mode"]
    upper = spec["upper"]
    companion_mode = spec["companion_mode"]
    companion_upper = spec["companion_upper"]
    nb = nbf.v4.new_notebook()
    cells = []

    # Badge & Header
    cells.append(nbf.v4.new_markdown_cell(
        f'<a href="https://colab.research.google.com/github/CooL-Legend/CSCI544-DLM/blob/adi-e2/notebooks/gsm8k/{spec["filename"]}" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>'
    ))
    cells.append(nbf.v4.new_markdown_cell(
        f"# Experiment 0: Search-Free {spec['full_name']} on GSM8K — Test Split (200 Primary + 300 Extended Questions)\n\n"
        f"This notebook runs **Experiment 0 (E0)** on the predetermined GSM8K `test` split with `LLaDA-8B-Instruct` using search-free **{upper} Decoding (`{spec['condition_id']}`, 256 NFEs, $T=0$)** under the shared two-shot decomposition protocol (`gsm8k_shared_v2`):\n"
        + spec["equation_md"]
        + "### Test Dataset Split Organization (Aligned with E1 & E2 Test Runs):\n"
        "1. **Primary 200 Test Samples** (`sample_slice=(0, 200)`): The exact 200 predetermined test questions from `configs/gsm8k/experiment_samples.json` (`sample_seed=1`).\n"
        "2. **Extended 300 Test Samples** (`sample_slice=(200, 500)`): 300 additional held-out test questions from `configs/gsm8k/experiment_samples_500.json` under the identical `sample_seed=1` ordering.\n"
        "3. **Combined 500 Test Samples**: Full 500-question test pool evaluation, Wilson 95% confidence intervals, and paired comparison.\n\n"
        "### Protocol & Execution Features:\n"
        "- **Cooperative Varlen FlashAttention-2 Batching**: Packs active states across questions in each batch (`BATCH_SIZE = 2` on L4, `4` or `8` on A100) with 100% bit-for-bit numerical parity with serial decoding.\n"
        "- **Incremental Google Drive Persistence & Resumption**: Saves per-question artifacts (`metrics.json`, `completion.json`, `trace.jsonl`, `complete.json`) and summary CSVs after every batch so interrupted Colab sessions resume automatically with 0 redundant forward passes.\n"
    ))

    # Section 1: GPU Hardware
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Verify GPU Hardware\n"
        "Verify CUDA GPU supporting bfloat16 (recommended: L4 or A100 with $\\ge 22$ GiB VRAM)."
    ))
    cells.append(nbf.v4.new_code_cell(
        "import torch\n\n"
        "if not torch.cuda.is_available():\n"
        "    raise RuntimeError('Select a CUDA GPU runtime in Colab (L4 or A100), then reconnect.')\n"
        "gpu = torch.cuda.get_device_properties(0)\n"
        "vram_gb = gpu.total_memory / (1024 ** 3)\n"
        "print(f'GPU: {gpu.name} | Total VRAM: {vram_gb:.2f} GiB')\n"
        "if vram_gb < 20.0:\n"
        "    raise RuntimeError('This notebook uses unquantized BF16 weights. Select an L4 or A100 GPU runtime.')\n"
        "if not torch.cuda.is_bf16_supported():\n"
        "    raise RuntimeError('This GPU does not support bfloat16 inference.')\n"
        "print('✓ GPU hardware verification passed.')\n"
    ))

    # Section 2: Repo Setup
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Repository Setup\n"
        "Clone or update the GitHub repository on branch `adi-e2`."
    ))
    cells.append(nbf.v4.new_code_cell(
        "from pathlib import Path\n"
        "import os\n"
        "import subprocess\n"
        "import sys\n\n"
        "IN_COLAB = 'google.colab' in sys.modules\n"
        "REPO_URL = 'https://github.com/CooL-Legend/CSCI544-DLM.git'\n"
        "BRANCH = 'adi-e2'\n"
        "REPO_DIR = Path('/content/CSCI544-DLM') if IN_COLAB else Path.cwd().resolve()\n\n"
        "if IN_COLAB:\n"
        "    if not (REPO_DIR / '.git').exists():\n"
        "        token = os.environ.get('GITHUB_TOKEN', '')\n"
        "        if not token:\n"
        "            try:\n"
        "                from google.colab import userdata\n"
        "                token = userdata.get('GITHUB_TOKEN')\n"
        "            except Exception:\n"
        "                pass\n"
        "        clone_url = REPO_URL.replace('https://', f'https://{token}@') if token else REPO_URL\n"
        "        subprocess.run(['git', 'clone', '-b', BRANCH, clone_url, str(REPO_DIR)], check=True)\n"
        "    else:\n"
        "        subprocess.run(['git', '-C', str(REPO_DIR), 'fetch', 'origin', BRANCH], check=True)\n"
        "        subprocess.run(['git', '-C', str(REPO_DIR), 'checkout', BRANCH], check=True)\n"
        "        subprocess.run(['git', '-C', str(REPO_DIR), 'pull', 'origin', BRANCH], check=True)\n"
        "    os.chdir(REPO_DIR)\n"
        "    if str(REPO_DIR / 'src') not in sys.path:\n"
        "        sys.path.insert(0, str(REPO_DIR / 'src'))\n\n"
        "print(f'Working directory: {Path.cwd()}')\n"
    ))

    # Section 3: Dependencies
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Install Dependencies\n"
        "Install pinned dependencies and editable package."
    ))
    cells.append(nbf.v4.new_code_cell(
        "import site\n"
        "import subprocess\n"
        "import sys\n\n"
        "if IN_COLAB:\n"
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(REPO_DIR / 'requirements/llada-colab.txt')], check=True)\n"
        "    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(REPO_DIR)], check=True)\n"
        "    import importlib\n"
        "    importlib.invalidate_caches()\n"
        "    user_site = site.getusersitepackages()\n"
        "    if user_site not in sys.path:\n"
        "        sys.path.append(user_site)\n"
        "print('Dependencies installed successfully.')\n"
    ))

    # Section 4: Unit tests
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Run Algorithmic Pre-Flight Unit Tests\n"
        f"Verify E0 {upper} test-split alignment with E1/E2 (200 primary and 500 extended manifests), batched decoding parity, and checkpoint resumption before loading weights."
    ))
    cells.append(nbf.v4.new_code_cell(
        "import subprocess\n"
        "import sys\n\n"
        "res = subprocess.run(\n"
        "    [sys.executable, '-m', 'pytest', 'tests/tasks/test_e0_test_stage.py', '-v'],\n"
        "    capture_output=True, text=True\n"
        ")\n"
        "print(res.stdout)\n"
        "if res.returncode != 0:\n"
        "    print(res.stderr)\n"
        "    raise RuntimeError('Pre-flight E0 test stage unit tests failed!')\n"
        "print('All E0 algorithmic unit tests passed.')\n"
    ))

    # Section 5: Mount Drive
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Mount Google Drive & Configure Cache\n"
        "Mount Google Drive for persistent artifact and model caching."
    ))
    cells.append(nbf.v4.new_code_cell(
        "from pathlib import Path\n"
        "import os\n\n"
        "DRIVE_ROOT = Path('/content/drive/MyDrive/CSCI544-DLM')\n"
        f"OUTPUT_DIR = Path({spec['output_subdir']!r})\n\n"
        "if IN_COLAB:\n"
        "    from google.colab import drive\n"
        "    drive.mount('/content/drive')\n"
        f"    OUTPUT_DIR = DRIVE_ROOT / {spec['output_subdir']!r}\n"
        "    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
        "    cache_dir = DRIVE_ROOT / 'cache/huggingface'\n"
        "    cache_dir.mkdir(parents=True, exist_ok=True)\n"
        "    os.environ['HF_HOME'] = str(cache_dir)\n"
        "    os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_dir)\n\n"
        "print(f'Output directory: {OUTPUT_DIR.resolve()}')\n"
    ))

    # Section 6: Controls & Model Loading
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Experiment Controls & Staged Model Loading\n"
        f"Configure search-free E0 {upper} decoding (`{spec['condition_id']}`, 256 steps at $T=0$), cooperative batch size 2, and load `LLaDA-8B-Instruct` from staged NVMe storage."
    ))
    cells.append(nbf.v4.new_code_cell(
        "import shutil\n"
        "import subprocess\n"
        "import sys\n"
        "from pathlib import Path\n"
        "from dlm_project.models.profiles import get_profile\n"
        "from dlm_project.tasks.gsm8k.experiment_0 import context, load_model\n\n"
        "STAGE = 'test'\n"
        f"METHODS = ({mode!r},)\n"
        "# On NVIDIA L4 (22.4 GiB VRAM), BATCH_SIZE = 2 is recommended for stability;\n"
        "# BATCH_SIZE = 4 or 8 is suited for A100 (40/80 GB) or High-RAM runtimes.\n"
        "BATCH_SIZE = 2\n"
        "RESUME = True\n\n"
        "# Load base context on the 500-question test split to initialize model identity\n"
        "base_ctx = context(stage=STAGE, methods=METHODS, sample_slice=(0, 500))\n\n"
        "profile = get_profile('llada')\n"
        "model_name = profile['repo_id'].split('/')[-1]\n"
        "drive_model_dir = DRIVE_ROOT / 'models' / model_name\n"
        "runtime_model_dir = Path('/content/models') / model_name\n\n"
        "# Fast local NVMe cache in Colab runtime if available on Drive (or cache to Drive first)\n"
        "if IN_COLAB:\n"
        "    if not (drive_model_dir / 'config.json').is_file() and not (runtime_model_dir / 'config.json').is_file():\n"
        "        subprocess.run([\n"
        "            sys.executable, '-u', str(REPO_DIR / 'scripts/cache_model.py'),\n"
        "            '--model', 'llada', '--drive-dir', str(drive_model_dir),\n"
        "            '--runtime-dir', str(runtime_model_dir),\n"
        "        ], check=True)\n"
        "    elif drive_model_dir.is_dir() and not (runtime_model_dir / 'config.json').is_file():\n"
        "        print(f'Copying model from Google Drive to local fast runtime storage: {runtime_model_dir}...')\n"
        "        shutil.copytree(drive_model_dir, runtime_model_dir)\n\n"
        "MODEL_PATH = str(runtime_model_dir if runtime_model_dir.is_dir() else (drive_model_dir if drive_model_dir.is_dir() else profile['repo_id']))\n"
        "print('Using model location:', MODEL_PATH)\n"
        "print('Loading LLaDA-8B-Instruct wrapper on GPU in bfloat16...')\n"
        "wrapper = load_model(base_ctx, model_path=MODEL_PATH, device='cuda', local_files_only=Path(MODEL_PATH).is_dir())\n"
        "print(f'✓ Model loaded: {wrapper.identity[\"family\"]} ({wrapper.identity[\"repo_id\"]} @ {wrapper.identity[\"revision\"]})')\n"
        f"print(f'✓ Stage: {{STAGE}} | Method: {{METHODS[0]}} ({spec['condition_id']}) | Total Test Pool: {{len(base_ctx[\"examples\"])}} questions | Batch Size: {{BATCH_SIZE}}')\n"
    ))

    # Section 7: Run Primary 200 Test Samples
    cells.append(nbf.v4.new_markdown_cell(
        "## 7. Execute Part 1: Primary Test Set (Original 200 Test Questions)\n"
        "Evaluates the exact 200 predetermined test questions (`sample_slice=(0, 200)`) from `configs/gsm8k/experiment_samples.json`."
    ))
    cells.append(nbf.v4.new_code_cell(
        "from time import perf_counter\n"
        "from dlm_project.tasks.gsm8k.experiment_0 import run_e0_stage\n\n"
        "ctx_200 = context(stage=STAGE, methods=METHODS, sample_slice=(0, 200))\n"
        "ctx_200['setup'] = dict(base_ctx.get('setup', {}))\n"
        f"print('Starting Part 1 (Primary 200 Test Questions) with E0 {upper} decoding...')\n"
        "t0 = perf_counter()\n"
        "q_rows_200, agg_rows_200 = run_e0_stage(\n"
        "    wrapper,\n"
        "    ctx_200,\n"
        "    OUTPUT_DIR / 'primary_200',\n"
        "    methods=METHODS,\n"
        "    resume=RESUME,\n"
        "    batch_size=BATCH_SIZE,\n"
        ")\n"
        "print(f'Part 1 Complete in {(perf_counter() - t0) / 60:.2f} minutes ({len(q_rows_200)} questions).')\n"
    ))

    # Section 8: Run Extended 300 Test Samples
    cells.append(nbf.v4.new_markdown_cell(
        "## 8. Execute Part 2: Extended Test Set (Additional 300 Test Questions)\n"
        "Evaluates the additional 300 predetermined test questions (`sample_slice=(200, 500)`) from `configs/gsm8k/experiment_samples_500.json`."
    ))
    cells.append(nbf.v4.new_code_cell(
        "ctx_300 = context(stage=STAGE, methods=METHODS, sample_slice=(200, 500))\n"
        "ctx_300['setup'] = dict(base_ctx.get('setup', {}))\n"
        f"print('Starting Part 2 (Extended 300 Test Questions) with E0 {upper} decoding...')\n"
        "t0 = perf_counter()\n"
        "q_rows_300, agg_rows_300 = run_e0_stage(\n"
        "    wrapper,\n"
        "    ctx_300,\n"
        "    OUTPUT_DIR / 'extended_300',\n"
        "    methods=METHODS,\n"
        "    resume=RESUME,\n"
        "    batch_size=BATCH_SIZE,\n"
        ")\n"
        "print(f'Part 2 Complete in {(perf_counter() - t0) / 60:.2f} minutes ({len(q_rows_300)} questions).')\n"
    ))

    # Section 9: Aggregate & Paired Results
    cells.append(nbf.v4.new_markdown_cell(
        "## 9. Results & Summary Metrics (Primary 200, Extended 300, and Combined 500 Test Questions)\n"
        "Display aggregate accuracy, Wilson 95% confidence intervals, parse rate, NFE counts, and wall-clock latency across Primary 200, Extended 300, and Combined 500 test questions."
    ))
    cells.append(nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "from IPython.display import display\n"
        "from dlm_project.experiments.sweep_report import write_csv\n"
        "from dlm_project.tasks.gsm8k.experiment_0 import (\n"
        "    compute_e0_aggregate_rows,\n"
        "    compute_paired_comparison,\n"
        ")\n\n"
        "summaries_dir = OUTPUT_DIR / 'summaries'\n"
        "summaries_dir.mkdir(parents=True, exist_ok=True)\n\n"
        "agg_cols = [\n"
        "    'method', 'condition', 'sample_count', 'correct_count', 'accuracy_percent',\n"
        "    'ci_95_low', 'ci_95_high', 'parse_rate', 'mean_search_nfe',\n"
        "    'mean_completion_nfe', 'mean_total_nfe', 'mean_completion_wall_seconds',\n"
        "]\n\n"
        "combined_q_rows = q_rows_200 + q_rows_300\n"
        "combined_agg_rows = compute_e0_aggregate_rows(combined_q_rows, STAGE, METHODS)\n\n"
        "for title, rows, agg, slug in [\n"
        "    ('Part 1: Primary 200 Test Questions', q_rows_200, agg_rows_200, 'primary_200'),\n"
        "    ('Part 2: Extended 300 Test Questions', q_rows_300, agg_rows_300, 'extended_300'),\n"
        "    ('Combined Pool: All 500 Test Questions', combined_q_rows, combined_agg_rows, 'combined_500'),\n"
        "]:\n"
        "    if rows:\n"
        f"        write_csv(summaries_dir / f'e0_{mode}_{{slug}}_per_question.csv', rows)\n"
        "    if agg:\n"
        f"        write_csv(summaries_dir / f'e0_{mode}_{{slug}}_aggregate.csv', agg)\n"
        "        df_agg = pd.DataFrame(agg)\n"
        "        print(f'=== {title} Aggregate Metrics ===')\n"
        "        display(df_agg[[c for c in agg_cols if c in df_agg.columns]])\n\n"
        "if combined_q_rows:\n"
        "    write_csv(summaries_dir / 'e0_test_per_question.csv', combined_q_rows)\n"
        "if combined_agg_rows:\n"
        "    write_csv(summaries_dir / 'e0_test_aggregate.csv', combined_agg_rows)\n\n"
        "# Optional cross-baseline paired comparison if companion E0 test notebook results exist on Drive\n"
        f"companion_pq = (DRIVE_ROOT if IN_COLAB else Path('.')) / {spec['companion_output_subdir']!r} / 'summaries/e0_test_per_question.csv'\n"
        "if companion_pq.is_file() and combined_q_rows:\n"
        "    comp_rows = pd.read_csv(companion_pq).to_dict('records')\n"
        "    paired = compute_paired_comparison(combined_q_rows + comp_rows, baseline_method='confidence', comparison_method='entropy')\n"
        "    if paired:\n"
        f"        print('=== Paired Comparison (E0 Confidence vs. E0 Entropy on Common Test Questions) ===')\n"
        "        display(pd.DataFrame(paired))\n"
        "        write_csv(summaries_dir / 'e0_test_paired_comparison.csv', paired)\n\n"
        "if combined_q_rows:\n"
        "    df_pq = pd.DataFrame(combined_q_rows)\n"
        "    pq_cols = [\n"
        "        'example_id', 'method', 'condition', 'exact_match', 'parse_success',\n"
        "        'predicted_answer', 'reference_answer', 'completion_nfes', 'total_nfes',\n"
        "        'first_terminator_position', 'completion_wall_seconds',\n"
        "    ]\n"
        "    print('\\n=== Sample Per-Question Test Outcomes (First 20 Questions) ===')\n"
        "    display(df_pq[[c for c in pq_cols if c in df_pq.columns]].head(20))\n"
    ))

    # Section 10: Interactive Question & Unmasking Trace Inspector
    cells.append(nbf.v4.new_markdown_cell(
        "## 10. Interactive Test Question & Unmasking Trace Inspector\n"
        "Inspect the decoded reasoning steps, first EOS/EOT termination boundary, extracted numerical answer, and unmasking commit trace for any evaluated test question."
    ))
    cells.append(nbf.v4.new_code_cell(
        "import json\n"
        "from IPython.display import Code, display\n\n"
        "INSPECT_SPLIT = 'primary_200'  # 'primary_200' or 'extended_300'\n"
        "INSPECT_QUESTION_IDX = 0\n\n"
        "split_ctx = ctx_200 if INSPECT_SPLIT == 'primary_200' else ctx_300\n"
        "sample_ex = split_ctx['examples'][INSPECT_QUESTION_IDX]\n"
        "sample_ex_id = sample_ex['example_id']\n"
        f"q_dir = OUTPUT_DIR / INSPECT_SPLIT / STAGE / {mode!r} / sample_ex_id\n"
        "metrics_json = q_dir / 'metrics.json'\n"
        "trace_jsonl = q_dir / 'trace.jsonl'\n\n"
        "if metrics_json.is_file():\n"
        "    met = json.loads(metrics_json.read_text())\n"
        f"    print(f'=== E0 {upper} Test Outcome ({{sample_ex_id}}) ===')\n"
        "    print('Question:\\n', met['question'])\n"
        "    print(f'Exact Match: {met[\"exact_match\"]} | Predicted: {met[\"predicted_answer\"]} | Reference: {met[\"reference_answer\"]}')\n"
        "    print(f'NFEs (search/completion/total): {met[\"search_nfes\"]}/{met[\"completion_nfes\"]}/{met[\"total_nfes\"]} | First EOS/EOT Pos: {met[\"first_terminator_position\"]}')\n"
        "    print('\\nScored Prefix (before first EOS/EOT):')\n"
        "    display(Code(met['scoring_text'], language='text'))\n\n"
        "if trace_jsonl.is_file():\n"
        "    commits = [json.loads(line) for line in trace_jsonl.read_text().splitlines() if line.strip() and '\"stage\": \"commit\"' in line]\n"
        "    print(f'\\nUnmasking Commit Trace ({len(commits)} steps recorded; showing first 3 commits):')\n"
        "    for event in commits[:3]:\n"
        "        print(json.dumps(event, indent=2))\n"
    ))

    # Section 11: Drive Flush
    cells.append(nbf.v4.new_markdown_cell(
        "## 11. Synchronize All Artifacts to Google Drive\n"
        "Flush and sync all in-flight buffers to Google Drive."
    ))
    cells.append(nbf.v4.new_code_cell(
        "if IN_COLAB:\n"
        "    from google.colab import drive\n"
        "    drive.flush_and_unmount()\n"
        "    print('Google Drive flushed successfully.')\n"
    ))

    nb.cells = cells
    out_path = NOTEBOOKS_DIR / spec["filename"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        nbf.write(nb, f)
    print(f"Generated {out_path} ({len(cells)} cells)")
    return out_path


def create_e0_confidence_test_notebook() -> Path:
    return _create_e0_test_notebook(E0_NOTEBOOK_SPECS["confidence"])


def create_e0_entropy_test_notebook() -> Path:
    return _create_e0_test_notebook(E0_NOTEBOOK_SPECS["entropy"])


def main() -> None:
    create_e0_confidence_test_notebook()
    create_e0_entropy_test_notebook()


if __name__ == "__main__":
    main()
