"""Generator script to build Experiment 2 Dev and Eval Colab-ready notebooks.

Constructs:
    1. notebooks/gsm8k/Experiment_2_Dev.ipynb   (5 questions)
    2. notebooks/gsm8k/Experiment_2_Eval.ipynb  (Original 20 questions + additional 50 questions cell)
"""

import json
from pathlib import Path
import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks/gsm8k"


def create_e2_dev_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Badge & Header
    cells.append(nbf.v4.new_markdown_cell(
        '<a href="https://colab.research.google.com/github/CooL-Legend/CSCI544-DLM/blob/adi/notebooks/gsm8k/Experiment_2_Dev.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>'
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "# Experiment 2: Two-Step Lookahead MEDAL vs. AB-MEDAL on GSM8K — Dev (5 Questions)\n\n"
        "This notebook evaluates **Experiment 2 (E2)** comparing **MEDAL** and **AB-MEDAL** on GSM8K using a **Two-Step Lookahead Fractional Information Gain (FIG)** reward function with `LLaDA-8B-Instruct`.\n\n"
        "### Mathematical Formulation:\n"
        "Rather than computing immediate 1-step information gain, for each candidate child state $s_i$, search projects the highest-confidence action $a^*_i \\in A(s_i)$ to form grandchild $g_i = s_i \\oplus a^*_i$, computing:\n"
        "$$\n"
        "R_{\\mathrm{FIG}}(s, s_i) = \\frac{H(s) - H(g_i)}{\\max(H(s), \\epsilon)}\n"
        "$$\n\n"
        "- **MEDAL (Two-Step Ahead)**: Expands all $K_2$ child actions at once; for each child $s_i$, evaluates $s_i$ and its greedy grandchild $g_i$, caching both states and computing $R_{\\mathrm{FIG}}(s, s_i)$. Backpropagates mean backed-up rewards.\n"
        "- **AB-MEDAL (Two-Step Ahead)**: When selecting `GEN`, expands the top remaining action from $s \\to s_i$, then greedily takes the top action from $s_i \\to g_i$, evaluates both forwards, updates Student-$t$ Gaussian posteriors with $R_{\\mathrm{FIG}}$, and backpropagates.\n"
        "- **State Evaluation Caching**: Grandchild analyses are cached in `cached_analyses` so that subsequent tree expansions incur **0 redundant NFEs**.\n"
    ))

    # Section 1: Verify GPU
    cells.append(nbf.v4.new_markdown_cell("## 1. Verify GPU Hardware\nVerify that a CUDA GPU supporting bfloat16 (recommended: L4 or A100 with $\\ge 22$ GiB VRAM) is active."))
    cells.append(nbf.v4.new_code_cell(
        "import torch\n\n"
        "if not torch.cuda.is_available():\n"
        "    raise RuntimeError('Select a CUDA GPU runtime in Colab (L4 or A100), then reconnect.')\n"
        "gpu = torch.cuda.get_device_properties(0)\n"
        "vram_gb = gpu.total_memory / (1024 ** 3)\n"
        "print(f'GPU: {gpu.name} | Compute Capability: {gpu.major}.{gpu.minor} | Total VRAM: {vram_gb:.2f} GiB')\n"
        "if vram_gb < 20.0:\n"
        "    print('Warning: VRAM is below 22 GiB. If OOM occurs, use --batch-size 1.')\n"
        "else:\n"
        "    print('VRAM check passed.')\n"
    ))

    # Section 2: Repository Setup
    cells.append(nbf.v4.new_markdown_cell("## 2. Repository Setup\nClone or update the GitHub repository on branch `adi`."))
    cells.append(nbf.v4.new_code_cell(
        "from pathlib import Path\n"
        "import getpass\n"
        "import os\n"
        "import subprocess\n"
        "import sys\n\n"
        "IN_COLAB = 'google.colab' in sys.modules\n"
        "REPO_URL = 'https://github.com/CooL-Legend/CSCI544-DLM.git'\n"
        "BRANCH = 'adi'\n"
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
    cells.append(nbf.v4.new_markdown_cell("## 3. Install Dependencies\nInstall pinned dependencies and install editable package."))
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
    cells.append(nbf.v4.new_markdown_cell("## 4. Run Algorithmic Pre-Flight Unit Tests\nRun unit tests to verify 2-step lookahead FIG reward computation, state caching, and batched parity."))
    cells.append(nbf.v4.new_code_cell(
        "import subprocess\n"
        "import sys\n\n"
        "res = subprocess.run(\n"
        "    [sys.executable, '-m', 'pytest', 'tests/tasks/test_experiment_2.py', '-v'],\n"
        "    capture_output=True, text=True\n"
        ")\n"
        "print(res.stdout)\n"
        "if res.returncode != 0:\n"
        "    print(res.stderr)\n"
        "    raise RuntimeError('Pre-flight E2 unit tests failed!')\n"
        "print('All E2 algorithmic unit tests passed.')\n"
    ))

    # Section 5: Drive Mount
    cells.append(nbf.v4.new_markdown_cell("## 5. Mount Google Drive & Configure Cache\nMount Google Drive for persistent artifact and model caching."))
    cells.append(nbf.v4.new_code_cell(
        "from pathlib import Path\n"
        "import os\n\n"
        "DRIVE_ROOT = Path('/content/drive/MyDrive/CSCI544-DLM')\n"
        "OUTPUT_DIR = Path('results/e2_dev')\n\n"
        "if IN_COLAB:\n"
        "    from google.colab import drive\n"
        "    drive.mount('/content/drive')\n"
        "    OUTPUT_DIR = DRIVE_ROOT / 'results/e2_dev'\n"
        "    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
        "    cache_dir = DRIVE_ROOT / 'cache/huggingface'\n"
        "    cache_dir.mkdir(parents=True, exist_ok=True)\n"
        "    os.environ['HF_HOME'] = str(cache_dir)\n"
        "    os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_dir)\n\n"
        "print(f'Output directory: {OUTPUT_DIR.resolve()}')\n"
    ))

    # Section 6: Experiment Controls
    cells.append(nbf.v4.new_markdown_cell("## 6. Experiment Controls & Configuration\nConfigure 5-question frozen `dev` set, matched search budgets [256, 512], and cooperative batch execution."))
    cells.append(nbf.v4.new_code_cell(
        "from dlm_project.tasks.gsm8k.experiment_2 import (\n"
        "    CHECKPOINT_BUDGETS, MAX_SEARCH_DEPTH, MIN_CANDIDATES, context\n"
        ")\n\n"
        "STAGE = 'dev'\n"
        "METHODS = ('medal', 'ab_medal')\n"
        "BUDGETS = (256, 512)\n"
        "BATCH_SIZE = 2\n"
        "RESUME = True\n\n"
        "ctx = context(\n"
        "    stage=STAGE,\n"
        "    methods=METHODS,\n"
        "    budgets=BUDGETS,\n"
        "    max_depth=MAX_SEARCH_DEPTH,\n"
        "    num_candidates=MIN_CANDIDATES,\n"
        ")\n"
        "print(f'Stage: {ctx[\"stage\"]} | Questions: {len(ctx[\"examples\"])} | Budgets: {BUDGETS} | Methods: {METHODS}')\n"
    ))

    # Section 7: Load Model
    cells.append(nbf.v4.new_markdown_cell("## 7. Cache & Load LLaDA-8B-Instruct\nLoad `LLaDA-8B-Instruct` model and tokenizer on CUDA in bfloat16, and prepare tokenized evaluation prompts."))
    cells.append(nbf.v4.new_code_cell(
        "from dlm_project.tasks.gsm8k.experiment_2 import load_model, prepare\n\n"
        "print('Loading model and tokenizer on GPU in bfloat16...')\n"
        "wrapper = load_model(ctx, device='cuda')\n"
        "prepared = prepare(wrapper, ctx)\n"
        "print(f'Prepared {len(prepared)} tokenized evaluation prompts.')\n"
    ))

    # Section 8: Execute Full E2 Dev
    cells.append(nbf.v4.new_markdown_cell("## 8. Execute Experiment 2 (Dev (5 Questions))\nExecute MEDAL and AB-MEDAL with the 2-step lookahead FIG reward across budgets 256 and 512."))
    cells.append(nbf.v4.new_code_cell(
        "from time import perf_counter\n"
        "from dlm_project.tasks.gsm8k.experiment_2 import run_e2_stage\n\n"
        "print(f'Starting E2 execution on {len(ctx[\"examples\"])} questions for {METHODS} at budgets {BUDGETS}...')\n"
        "t_start = perf_counter()\n"
        "q_rows, agg_rows = run_e2_stage(\n"
        "    wrapper,\n"
        "    ctx,\n"
        "    OUTPUT_DIR,\n"
        "    methods=METHODS,\n"
        "    budgets=BUDGETS,\n"
        "    max_depth=MAX_SEARCH_DEPTH,\n"
        "    num_candidates=MIN_CANDIDATES,\n"
        "    resume=RESUME,\n"
        "    batch_size=BATCH_SIZE,\n"
        ")\n"
        "total_elapsed = perf_counter() - t_start\n"
        "print(f'\\nE2 Dev execution complete in {total_elapsed / 60:.2f} minutes ({len(q_rows)} checkpoint outcomes).')\n"
    ))

    # Section 9: Results & Paired Comparison
    cells.append(nbf.v4.new_markdown_cell("## 9. Experiment 2 Results & Paired Comparison\nDisplay aggregate accuracy and paired comparison (McNemar exact test) at matched budgets."))
    cells.append(nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "from IPython.display import display\n"
        "from scripts.run_e2 import compute_paired_comparison\n\n"
        "agg_file = OUTPUT_DIR / 'summaries' / f'e2_{STAGE}_aggregate.csv'\n"
        "if agg_file.is_file():\n"
        "    df_agg = pd.read_csv(agg_file)\n"
        "    print('=== E2 Dev Aggregate Metrics ===')\n"
        "    display(df_agg[['method', 'search_budget', 'sample_count', 'accuracy_percent', 'ci_95_low', 'ci_95_high', 'mean_total_nfe', 'mean_selected_depth']])\n\n"
        "paired = compute_paired_comparison(q_rows, budgets=BUDGETS)\n"
        "if paired:\n"
        "    df_paired = pd.DataFrame(paired)\n"
        "    print('=== Paired Comparison (AB-MEDAL vs. MEDAL at Matched Search Budget) ===')\n"
        "    display(df_paired[['search_budget', 'paired_questions', 'medal_accuracy', 'ab_medal_accuracy', 'accuracy_delta_percent', 'ab_medal_only_correct', 'medal_only_correct', 'mcnemar_exact_p']])\n"
    ))

    # Section 10: Search Tree Inspector
    cells.append(nbf.v4.new_markdown_cell("## 10. Interactive Search Tree & Decision Trace Inspector\nExamine candidate shortlists, 2-step FIG rewards, and lookahead state entropy drops."))
    cells.append(nbf.v4.new_code_cell(
        "import json\n\n"
        "sample_ex_id = prepared[0]['example']['example_id']\n"
        "chk_dir = OUTPUT_DIR / STAGE / 'ab_medal' / sample_ex_id / 'checkpoints/256'\n"
        "if (chk_dir / 'metrics.json').is_file():\n"
        "    met = json.loads((chk_dir / 'metrics.json').read_text())\n"
        "    print(f'Sample {sample_ex_id} @ Budget 256:')\n"
        "    print(f'  Selected Node: {met[\"selected_node_id\"]} | Depth: {met[\"selected_depth\"]} | Q: {met[\"selected_q\"]:.4f}')\n"
        "    print(f'  Exact Match: {met[\"exact_match\"]} | Pred: {met[\"predicted_answer\"]} | Ref: {met[\"reference_answer\"]}')\n"
    ))

    # Section 11: Drive Sync
    cells.append(nbf.v4.new_markdown_cell("## 11. Synchronize All Artifacts to Google Drive\nExplicitly flush and sync all in-flight buffers to Google Drive."))
    cells.append(nbf.v4.new_code_cell(
        "if IN_COLAB:\n"
        "    from google.colab import drive\n"
        "    drive.flush_and_unmount()\n"
        "    print('Google Drive flushed successfully.')\n"
    ))

    nb.cells = cells
    out_path = NOTEBOOKS_DIR / "Experiment_2_Dev.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        nbf.write(nb, f)
    print(f"Generated {out_path} ({len(cells)} cells)")


def create_e2_eval_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Badge & Header
    cells.append(nbf.v4.new_markdown_cell(
        '<a href="https://colab.research.google.com/github/CooL-Legend/CSCI544-DLM/blob/adi/notebooks/gsm8k/Experiment_2_Eval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>'
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "# Experiment 2: Two-Step Lookahead MEDAL vs. AB-MEDAL on GSM8K — Evaluation (20 Canonical + 50 Extended Questions)\n\n"
        "This notebook runs **Experiment 2 (E2)** on the GSM8K validation/evaluation pool with `LLaDA-8B-Instruct` using the **Two-Step Lookahead FIG Reward**:\n"
        "$$\n"
        "R_{\\mathrm{FIG}}(s, s_i) = \\frac{H(s) - H(g_i)}{\\max(H(s), \\epsilon)}\n"
        "$$\n\n"
        "### Dataset Split Organization:\n"
        "1. **Canonical 20 Samples** (`sample_slice=(0, 20)`): The exact 20 evaluation questions from Experiment 1, preserving historical prefix ordering.\n"
        "2. **Additional 50 Samples** (`sample_slice=(20, 70)`): 50 new validation questions sampled under the deterministic `sample_seed=1` order.\n"
        "3. **Combined 70 Samples**: Full validation pool evaluation and paired statistical tests.\n"
    ))

    # Section 1: GPU Hardware
    cells.append(nbf.v4.new_markdown_cell("## 1. Verify GPU Hardware\nVerify CUDA GPU supporting bfloat16 (recommended: L4 or A100 with $\\ge 22$ GiB VRAM)."))
    cells.append(nbf.v4.new_code_cell(
        "import torch\n\n"
        "if not torch.cuda.is_available():\n"
        "    raise RuntimeError('Select a CUDA GPU runtime in Colab (L4 or A100), then reconnect.')\n"
        "gpu = torch.cuda.get_device_properties(0)\n"
        "vram_gb = gpu.total_memory / (1024 ** 3)\n"
        "print(f'GPU: {gpu.name} | Total VRAM: {vram_gb:.2f} GiB')\n"
    ))

    # Section 2: Repo Setup
    cells.append(nbf.v4.new_markdown_cell("## 2. Repository Setup\nClone or update the GitHub repository on branch `adi`."))
    cells.append(nbf.v4.new_code_cell(
        "from pathlib import Path\n"
        "import os\n"
        "import subprocess\n"
        "import sys\n\n"
        "IN_COLAB = 'google.colab' in sys.modules\n"
        "REPO_URL = 'https://github.com/CooL-Legend/CSCI544-DLM.git'\n"
        "BRANCH = 'adi'\n"
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
    cells.append(nbf.v4.new_markdown_cell("## 3. Install Dependencies\nInstall pinned dependencies and editable package."))
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
    cells.append(nbf.v4.new_markdown_cell("## 4. Run Algorithmic Pre-Flight Unit Tests\nVerify 2-step lookahead FIG reward computation, state caching, and batched parity before loading weights."))
    cells.append(nbf.v4.new_code_cell(
        "import subprocess\n"
        "import sys\n\n"
        "res = subprocess.run(\n"
        "    [sys.executable, '-m', 'pytest', 'tests/tasks/test_experiment_2.py', '-v'],\n"
        "    capture_output=True, text=True\n"
        ")\n"
        "print(res.stdout)\n"
        "if res.returncode != 0:\n"
        "    print(res.stderr)\n"
        "    raise RuntimeError('Pre-flight E2 unit tests failed!')\n"
        "print('All E2 algorithmic unit tests passed.')\n"
    ))

    # Section 5: Mount Drive
    cells.append(nbf.v4.new_markdown_cell("## 5. Mount Google Drive & Configure Cache\nMount Google Drive for persistent artifact and model caching."))
    cells.append(nbf.v4.new_code_cell(
        "from pathlib import Path\n"
        "import os\n\n"
        "DRIVE_ROOT = Path('/content/drive/MyDrive/CSCI544-DLM')\n"
        "OUTPUT_DIR = Path('results/e2_eval')\n\n"
        "if IN_COLAB:\n"
        "    from google.colab import drive\n"
        "    drive.mount('/content/drive')\n"
        "    OUTPUT_DIR = DRIVE_ROOT / 'results/e2_eval'\n"
        "    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
        "    cache_dir = DRIVE_ROOT / 'cache/huggingface'\n"
        "    cache_dir.mkdir(parents=True, exist_ok=True)\n"
        "    os.environ['HF_HOME'] = str(cache_dir)\n"
        "    os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_dir)\n\n"
        "print(f'Output directory: {OUTPUT_DIR.resolve()}')\n"
    ))

    # Section 6: Controls
    cells.append(nbf.v4.new_markdown_cell("## 6. Experiment Controls & Shared Model Loading\nConfigure budgets [256, 512], depth 10, batch size 4, and load `LLaDA-8B-Instruct`."))
    cells.append(nbf.v4.new_code_cell(
        "from dlm_project.tasks.gsm8k.experiment_2 import (\n"
        "    CHECKPOINT_BUDGETS, MAX_SEARCH_DEPTH, MIN_CANDIDATES, context, load_model\n"
        ")\n\n"
        "METHODS = ('medal', 'ab_medal')\n"
        "BUDGETS = (256, 512)\n"
        "BATCH_SIZE = 4\n"
        "RESUME = True\n\n"
        "# Load base context to initialize model identity\n"
        "base_ctx = context(stage='evaluation', methods=METHODS, budgets=BUDGETS)\n"
        "print('Loading LLaDA-8B-Instruct wrapper on GPU in bfloat16...')\n"
        "wrapper = load_model(base_ctx, device='cuda')\n"
        "print('Model successfully loaded.')\n"
    ))

    # Section 7: Run Canonical 20 Samples
    cells.append(nbf.v4.new_markdown_cell(
        "## 7. Execute Part 1: Canonical Evaluation Set (Original 20 Questions)\n"
        "Evaluates the exact 20 evaluation questions from Experiment 1 (`sample_slice=(0, 20)`)."
    ))
    cells.append(nbf.v4.new_code_cell(
        "from time import perf_counter\n"
        "from dlm_project.tasks.gsm8k.experiment_2 import run_e2_stage\n\n"
        "ctx_20 = context(stage='evaluation', methods=METHODS, budgets=BUDGETS, sample_slice=(0, 20))\n"
        "print(f'Starting Part 1 (Canonical 20 Questions)...')\n"
        "t0 = perf_counter()\n"
        "q_rows_20, agg_rows_20 = run_e2_stage(\n"
        "    wrapper,\n"
        "    ctx_20,\n"
        "    OUTPUT_DIR / 'canonical_20',\n"
        "    methods=METHODS,\n"
        "    budgets=BUDGETS,\n"
        "    max_depth=MAX_SEARCH_DEPTH,\n"
        "    num_candidates=MIN_CANDIDATES,\n"
        "    resume=RESUME,\n"
        "    batch_size=BATCH_SIZE,\n"
        ")\n"
        "print(f'Part 1 Complete in {(perf_counter() - t0) / 60:.2f} minutes.')\n"
    ))

    # Section 8: Run Additional 50 Samples
    cells.append(nbf.v4.new_markdown_cell(
        "## 8. Execute Part 2: Extended Validation Set (Additional 50 Questions)\n"
        "Evaluates the additional 50 validation questions (`sample_slice=(20, 70)`)."
    ))
    cells.append(nbf.v4.new_code_cell(
        "ctx_50 = context(stage='evaluation', methods=METHODS, budgets=BUDGETS, sample_slice=(20, 70))\n"
        "print(f'Starting Part 2 (Additional 50 Questions)...')\n"
        "t0 = perf_counter()\n"
        "q_rows_50, agg_rows_50 = run_e2_stage(\n"
        "    wrapper,\n"
        "    ctx_50,\n"
        "    OUTPUT_DIR / 'extended_50',\n"
        "    methods=METHODS,\n"
        "    budgets=BUDGETS,\n"
        "    max_depth=MAX_SEARCH_DEPTH,\n"
        "    num_candidates=MIN_CANDIDATES,\n"
        "    resume=RESUME,\n"
        "    batch_size=BATCH_SIZE,\n"
        ")\n"
        "print(f'Part 2 Complete in {(perf_counter() - t0) / 60:.2f} minutes.')\n"
    ))

    # Section 9: Aggregate & Paired Results
    cells.append(nbf.v4.new_markdown_cell("## 9. Results & Paired Comparisons (Canonical 20, Extended 50, and Combined 70)"))
    cells.append(nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "from IPython.display import display\n"
        "from scripts.run_e2 import compute_paired_comparison\n\n"
        "print('=== Part 1: Canonical 20 Questions Paired Comparison ===')\n"
        "paired_20 = compute_paired_comparison(q_rows_20, budgets=BUDGETS)\n"
        "if paired_20:\n"
        "    display(pd.DataFrame(paired_20)[['search_budget', 'paired_questions', 'medal_accuracy', 'ab_medal_accuracy', 'accuracy_delta_percent', 'ab_medal_only_correct', 'medal_only_correct', 'mcnemar_exact_p']])\n\n"
        "print('=== Part 2: Additional 50 Questions Paired Comparison ===')\n"
        "paired_50 = compute_paired_comparison(q_rows_50, budgets=BUDGETS)\n"
        "if paired_50:\n"
        "    display(pd.DataFrame(paired_50)[['search_budget', 'paired_questions', 'medal_accuracy', 'ab_medal_accuracy', 'accuracy_delta_percent', 'ab_medal_only_correct', 'medal_only_correct', 'mcnemar_exact_p']])\n\n"
        "print('=== Combined Pool (All 70 Questions) Paired Comparison ===')\n"
        "combined_q_rows = q_rows_20 + q_rows_50\n"
        "paired_combined = compute_paired_comparison(combined_q_rows, budgets=BUDGETS)\n"
        "if paired_combined:\n"
        "    display(pd.DataFrame(paired_combined)[['search_budget', 'paired_questions', 'medal_accuracy', 'ab_medal_accuracy', 'accuracy_delta_percent', 'ab_medal_only_correct', 'medal_only_correct', 'mcnemar_exact_p']])\n"
    ))

    # Section 10: Drive Flush
    cells.append(nbf.v4.new_markdown_cell("## 10. Synchronize All Artifacts to Google Drive\nFlush and sync all in-flight buffers to Google Drive."))
    cells.append(nbf.v4.new_code_cell(
        "if IN_COLAB:\n"
        "    from google.colab import drive\n"
        "    drive.flush_and_unmount()\n"
        "    print('Google Drive flushed successfully.')\n"
    ))

    nb.cells = cells
    out_path = NOTEBOOKS_DIR / "Experiment_2_Eval.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        nbf.write(nb, f)
    print(f"Generated {out_path} ({len(cells)} cells)")


def main():
    create_e2_dev_notebook()
    create_e2_eval_notebook()


if __name__ == "__main__":
    main()

