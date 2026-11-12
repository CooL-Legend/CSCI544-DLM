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
        '<a href="https://colab.research.google.com/github/CooL-Legend/CSCI544-DLM/blob/adi-e2/notebooks/gsm8k/Experiment_2_Dev.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>'
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
    cells.append(nbf.v4.new_markdown_cell("## 2. Repository Setup\nClone or update the GitHub repository on branch `adi-e2`."))
    cells.append(nbf.v4.new_code_cell(
        "from pathlib import Path\n"
        "import getpass\n"
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
        "# On NVIDIA L4 (22.4 GiB VRAM), BATCH_SIZE = 2 is recommended for stability;\n"
        "# BATCH_SIZE = 4 is suited for A100 (40/80 GB) or High-RAM runtimes.\n"
        "BATCH_SIZE = 2\n"
        "RESUME = True\n\n"
        "ctx = context(\n"
        "    stage=STAGE,\n"
        "    methods=METHODS,\n"
        "    budgets=BUDGETS,\n"
        "    max_depth=MAX_SEARCH_DEPTH,\n"
        "    num_candidates=MIN_CANDIDATES,\n"
        ")\n"
        "print(f'Stage: {ctx[\"stage\"]} | Questions: {len(ctx[\"examples\"])} | Budgets: {BUDGETS} | Methods: {METHODS} | Batch Size: {BATCH_SIZE}')\n"
    ))

    # Section 7: Load Model
    cells.append(nbf.v4.new_markdown_cell("## 7. Cache & Load LLaDA-8B-Instruct\nStage model to fast local NVMe storage and load `LLaDA-8B-Instruct` model and tokenizer on CUDA in bfloat16."))
    cells.append(nbf.v4.new_code_cell(
        "import shutil\n"
        "from pathlib import Path\n"
        "from dlm_project.models.profiles import get_profile\n"
        "from dlm_project.tasks.gsm8k.experiment_2 import load_model, prepare\n\n"
        "profile = get_profile('llada')\n"
        "model_name = profile['repo_id'].split('/')[-1]\n"
        "drive_model_dir = DRIVE_ROOT / 'models' / model_name\n"
        "runtime_model_dir = Path('/content/models') / model_name\n\n"
        "# Fast local NVMe cache in Colab runtime if available on Drive\n"
        "if IN_COLAB and drive_model_dir.is_dir() and not (runtime_model_dir / 'config.json').is_file():\n"
        "    print(f'Copying model from Google Drive to local fast runtime storage: {runtime_model_dir}...')\n"
        "    shutil.copytree(drive_model_dir, runtime_model_dir)\n\n"
        "MODEL_PATH = str(runtime_model_dir if runtime_model_dir.is_dir() else (drive_model_dir if drive_model_dir.is_dir() else profile['repo_id']))\n"
        "print('Using model location:', MODEL_PATH)\n"
        "print('Loading model and tokenizer on GPU in bfloat16...')\n"
        "wrapper = load_model(ctx, model_path=MODEL_PATH, device='cuda', local_files_only=Path(MODEL_PATH).is_dir())\n"
        "prepared = prepare(wrapper, ctx)\n"
        "print(f'✓ Model loaded: {wrapper.identity[\"family\"]} ({wrapper.identity[\"repo_id\"]} @ {wrapper.identity[\"revision\"]})')\n"
        "print(f'✓ Prepared {len(prepared)} tokenized evaluation prompts.')\n"
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

    # Section 11: E3 Offline Metric Extraction & Publication Figures
    cells.append(nbf.v4.new_markdown_cell(
        "## 11. E3 Offline Metric Extraction & Publication Figures\n"
        "Run offline tree allocation, branching dynamics, and structural analysis on saved E2 checkpoints (zero GPU compute), and generate 6 publication figures."
    ))
    cells.append(nbf.v4.new_code_cell(
        "from dlm_project.tasks.gsm8k.e3_analysis import analyze_e3_stage\n"
        "from IPython.display import Image, display, Markdown\n\n"
        "print('Running E3 Offline Tree Allocation & Topology Analysis on E2 traces...')\n"
        "e3_q, e3_agg = analyze_e3_stage(\n"
        "    results_dir=OUTPUT_DIR / STAGE,\n"
        "    output_dir=OUTPUT_DIR / 'summaries',\n"
        "    stage=STAGE,\n"
        "    methods=tuple(METHODS),\n"
        "    budgets=tuple(BUDGETS),\n"
        ")\n"
        "print(f'✓ E3 Analysis Completed successfully: {len(e3_q)} question outcomes evaluated.')\n\n"
        "figures_dir = OUTPUT_DIR / 'summaries' / 'e3_figures'\n"
        "fig_titles = {\n"
        "    'fig1_nfe_by_depth': 'Figure 1: NFE Allocation by Tree Depth — MEDAL vs AB-MEDAL across budgets',\n"
        "    'fig2_branching_factors': 'Figure 2: Branching Factor Distribution — Children per expanded parent',\n"
        "    'fig3_candidate_density': 'Figure 3: Candidate Density vs Depth — Minimum-C availability and selection depth',\n"
        "    'fig4_root_subtrees': 'Figure 4: Root-Subtree Compute Concentration — Search effort across root branches',\n"
        "    'fig5_reward_redundancy': 'Figure 5: Reward and Search Redundancy — Negative IG rates and duplicate states',\n"
        "    'fig6_accuracy_vs_budget': 'Figure 6: Accuracy vs Search Budget — Performance scaling curve',\n"
        "}\n\n"
        "for fig_key, title in fig_titles.items():\n"
        "    png_path = figures_dir / f'{fig_key}.png'\n"
        "    if png_path.is_file():\n"
        "        display(Markdown(f'### {title}'))\n"
        "        display(Image(filename=str(png_path), width=750))\n"
    ))

    # Section 12: Drive Sync
    cells.append(nbf.v4.new_markdown_cell("## 12. Synchronize All Artifacts to Google Drive\nExplicitly flush and sync all in-flight buffers to Google Drive."))
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
        '<a href="https://colab.research.google.com/github/CooL-Legend/CSCI544-DLM/blob/adi-e2/notebooks/gsm8k/Experiment_2_Eval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>'
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
    cells.append(nbf.v4.new_markdown_cell("## 2. Repository Setup\nClone or update the GitHub repository on branch `adi-e2`."))
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

    # Section 6: Controls & Model Loading
    cells.append(nbf.v4.new_markdown_cell("## 6. Experiment Controls & Staged Model Loading\nConfigure budgets [256, 512], depth 10, batch size 2, and load `LLaDA-8B-Instruct` from staged NVMe storage."))
    cells.append(nbf.v4.new_code_cell(
        "import shutil\n"
        "from pathlib import Path\n"
        "from dlm_project.models.profiles import get_profile\n"
        "from dlm_project.tasks.gsm8k.experiment_2 import (\n"
        "    CHECKPOINT_BUDGETS, MAX_SEARCH_DEPTH, MIN_CANDIDATES, context, load_model\n"
        ")\n\n"
        "STAGE = 'evaluation'\n"
        "METHODS = ('medal', 'ab_medal')\n"
        "BUDGETS = (256, 512)\n"
        "# On NVIDIA L4 (22.4 GiB VRAM), BATCH_SIZE = 2 is recommended for stability;\n"
        "# BATCH_SIZE = 4 is suited for A100 (40/80 GB) or High-RAM runtimes.\n"
        "BATCH_SIZE = 2\n"
        "RESUME = True\n\n"
        "# Load base context to initialize model identity\n"
        "base_ctx = context(stage='evaluation', methods=METHODS, budgets=BUDGETS)\n\n"
        "profile = get_profile('llada')\n"
        "model_name = profile['repo_id'].split('/')[-1]\n"
        "drive_model_dir = DRIVE_ROOT / 'models' / model_name\n"
        "runtime_model_dir = Path('/content/models') / model_name\n\n"
        "# Fast local NVMe cache in Colab runtime if available on Drive\n"
        "if IN_COLAB and drive_model_dir.is_dir() and not (runtime_model_dir / 'config.json').is_file():\n"
        "    print(f'Copying model from Google Drive to local fast runtime storage: {runtime_model_dir}...')\n"
        "    shutil.copytree(drive_model_dir, runtime_model_dir)\n\n"
        "MODEL_PATH = str(runtime_model_dir if runtime_model_dir.is_dir() else (drive_model_dir if drive_model_dir.is_dir() else profile['repo_id']))\n"
        "print('Using model location:', MODEL_PATH)\n"
        "print('Loading LLaDA-8B-Instruct wrapper on GPU in bfloat16...')\n"
        "wrapper = load_model(base_ctx, model_path=MODEL_PATH, device='cuda', local_files_only=Path(MODEL_PATH).is_dir())\n"
        "print(f'✓ Model loaded: {wrapper.identity[\"family\"]} ({wrapper.identity[\"repo_id\"]} @ {wrapper.identity[\"revision\"]})')\n"
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
        "from scripts.run_e2 import compute_paired_comparison\n"
        "from dlm_project.experiments.sweep_report import write_csv\n\n"
        "summaries_dir = OUTPUT_DIR / 'summaries'\n"
        "summaries_dir.mkdir(parents=True, exist_ok=True)\n\n"
        "print('=== Part 1: Canonical 20 Questions Paired Comparison ===')\n"
        "paired_20 = compute_paired_comparison(q_rows_20, budgets=BUDGETS)\n"
        "if paired_20:\n"
        "    df_p20 = pd.DataFrame(paired_20)[['search_budget', 'paired_questions', 'medal_accuracy', 'ab_medal_accuracy', 'accuracy_delta_percent', 'ab_medal_only_correct', 'medal_only_correct', 'mcnemar_exact_p']]\n"
        "    display(df_p20)\n"
        "    write_csv(summaries_dir / 'e2_canonical_20_paired_comparison.csv', paired_20)\n\n"
        "print('=== Part 2: Additional 50 Questions Paired Comparison ===')\n"
        "paired_50 = compute_paired_comparison(q_rows_50, budgets=BUDGETS)\n"
        "if paired_50:\n"
        "    df_p50 = pd.DataFrame(paired_50)[['search_budget', 'paired_questions', 'medal_accuracy', 'ab_medal_accuracy', 'accuracy_delta_percent', 'ab_medal_only_correct', 'medal_only_correct', 'mcnemar_exact_p']]\n"
        "    display(df_p50)\n"
        "    write_csv(summaries_dir / 'e2_extended_50_paired_comparison.csv', paired_50)\n\n"
        "print('=== Combined Pool (All 70 Questions) Paired Comparison ===')\n"
        "combined_q_rows = q_rows_20 + q_rows_50\n"
        "paired_combined = compute_paired_comparison(combined_q_rows, budgets=BUDGETS)\n"
        "if paired_combined:\n"
        "    df_pcomb = pd.DataFrame(paired_combined)[['search_budget', 'paired_questions', 'medal_accuracy', 'ab_medal_accuracy', 'accuracy_delta_percent', 'ab_medal_only_correct', 'medal_only_correct', 'mcnemar_exact_p']]\n"
        "    display(df_pcomb)\n"
        "    write_csv(summaries_dir / 'e2_combined_70_paired_comparison.csv', paired_combined)\n"
        "    write_csv(summaries_dir / 'e2_evaluation_paired_comparison.csv', paired_combined)\n"
    ))

    # Section 10: Interactive Search Tree & Decision Trace Inspector
    cells.append(nbf.v4.new_markdown_cell(
        "## 10. Interactive Search Tree & Decision Trace Inspector\n"
        "Inspect the search tree topology, candidate nodes, and decision trace for any evaluated question."
    ))
    cells.append(nbf.v4.new_code_cell(
        "from dlm_project.tasks.gsm8k.tree_viz import print_ascii_tree, summarize_decision_trace\n"
        "import json\n\n"
        "# Select a sample question to inspect from canonical_20\n"
        "sample_ex_id = ctx_20['examples'][0]['example_id']\n"
        "chk_dir = OUTPUT_DIR / 'canonical_20' / 'evaluation' / 'ab_medal' / sample_ex_id / 'checkpoints/256'\n"
        "tree_json = chk_dir / 'tree.json'\n"
        "metrics_json = chk_dir / 'metrics.json'\n\n"
        "if tree_json.is_file():\n"
        "    print(f'=== AB-MEDAL Search Tree at B=256 for {sample_ex_id} ===')\n"
        "    print(print_ascii_tree(tree_json, max_depth=5, max_children_per_node=3))\n\n"
        "if metrics_json.is_file():\n"
        "    met = json.loads(metrics_json.read_text())\n"
        "    print(f'=== Checkpoint Metrics ({sample_ex_id} @ B=256) ===')\n"
        "    print(f'  Selected Node: {met[\"selected_node_id\"]} | Depth: {met[\"selected_depth\"]} | Q: {met[\"selected_q\"]:.4f}')\n"
        "    print(f'  Exact Match: {met[\"exact_match\"]} | Pred: {met[\"predicted_answer\"]} | Ref: {met[\"reference_answer\"]}')\n"
    ))

    # Section 11: E3 Offline Metric Extraction & Publication Figures
    cells.append(nbf.v4.new_markdown_cell(
        "## 11. E3 Offline Metric Extraction & Publication Figures\n"
        "Run offline tree allocation, branching dynamics, and structural analysis on saved evaluation checkpoints (zero GPU compute), and generate 6 publication figures."
    ))
    cells.append(nbf.v4.new_code_cell(
        "import shutil\n"
        "from dlm_project.tasks.gsm8k.e3_analysis import analyze_e3_stage\n"
        "from IPython.display import Image, display, Markdown\n\n"
        "# Build combined trace directory linking canonical_20 and extended_50\n"
        "combined_results_dir = OUTPUT_DIR / 'combined_70' / 'evaluation'\n"
        "combined_results_dir.mkdir(parents=True, exist_ok=True)\n"
        "for part_dir in [OUTPUT_DIR / 'canonical_20' / 'evaluation', OUTPUT_DIR / 'extended_50' / 'evaluation']:\n"
        "    if part_dir.is_dir():\n"
        "        for m in METHODS:\n"
        "            m_src = part_dir / m\n"
        "            if m_src.is_dir():\n"
        "                m_dst = combined_results_dir / m\n"
        "                m_dst.mkdir(parents=True, exist_ok=True)\n"
        "                for q_dir in m_src.iterdir():\n"
        "                    if q_dir.is_dir() and not (m_dst / q_dir.name).exists():\n"
        "                        try:\n"
        "                            (m_dst / q_dir.name).symlink_to(q_dir, target_is_directory=True)\n"
        "                        except OSError:\n"
        "                            shutil.copytree(q_dir, m_dst / q_dir.name)\n\n"
        "# Select results directory: combined_70 if populated, else canonical_20\n"
        "has_combined = any((combined_results_dir / m).is_dir() and any((combined_results_dir / m).iterdir()) for m in METHODS)\n"
        "eval_results_dir = combined_results_dir if has_combined else (OUTPUT_DIR / 'canonical_20' / 'evaluation')\n"
        "e3_output_dir = OUTPUT_DIR / ('combined_70' if has_combined else 'canonical_20') / 'summaries'\n"
        "e3_output_dir.mkdir(parents=True, exist_ok=True)\n\n"
        "print(f'Running E3 Offline Tree Analysis on {eval_results_dir}...')\n"
        "e3_q, e3_agg = analyze_e3_stage(\n"
        "    results_dir=eval_results_dir,\n"
        "    output_dir=e3_output_dir,\n"
        "    stage='evaluation',\n"
        "    methods=tuple(METHODS),\n"
        "    budgets=tuple(BUDGETS),\n"
        ")\n"
        "print(f'✓ E3 Analysis Completed successfully: {len(e3_q)} question outcomes evaluated.')\n\n"
        "figures_dir = e3_output_dir / 'e3_figures'\n"
        "fig_titles = {\n"
        "    'fig1_nfe_by_depth': 'Figure 1: NFE Allocation by Tree Depth — MEDAL vs AB-MEDAL across budgets',\n"
        "    'fig2_branching_factors': 'Figure 2: Branching Factor Distribution — Children per expanded parent',\n"
        "    'fig3_candidate_density': 'Figure 3: Candidate Density vs Depth — Minimum-C availability and selection depth',\n"
        "    'fig4_root_subtrees': 'Figure 4: Root-Subtree Compute Concentration — Search effort across root branches',\n"
        "    'fig5_reward_redundancy': 'Figure 5: Reward and Search Redundancy — Negative IG rates and duplicate states',\n"
        "    'fig6_accuracy_vs_budget': 'Figure 6: Accuracy vs Search Budget — Performance scaling curve',\n"
        "}\n\n"
        "for fig_key, title in fig_titles.items():\n"
        "    png_path = figures_dir / f'{fig_key}.png'\n"
        "    if png_path.is_file():\n"
        "        display(Markdown(f'### {title}'))\n"
        "        display(Image(filename=str(png_path), width=750))\n"
    ))

    # Section 12: Drive Flush
    cells.append(nbf.v4.new_markdown_cell("## 12. Synchronize All Artifacts to Google Drive\nFlush and sync all in-flight buffers to Google Drive."))
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


def create_e2_srig_dev_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Badge & Header
    cells.append(nbf.v4.new_markdown_cell(
        '<a href="https://colab.research.google.com/github/CooL-Legend/CSCI544-DLM/blob/adi-e2/notebooks/gsm8k/E2_Experiment_SRIG_Dev.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>'
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "# Experiment 2: Two-Step Lookahead MEDAL vs. AB-MEDAL on GSM8K with Sum of Relative Information Gains (SRIG) — Dev (5 Questions)\n\n"
        "This notebook evaluates **Experiment 2 (E2)** comparing **MEDAL** and **AB-MEDAL** on GSM8K using the **Two-Step Sum of Relative Information Gains (SRIG / $\\text{Future}R_{ig}$)** reward function with `LLaDA-8B-Instruct`.\n\n"
        "### Mathematical Formulation:\n"
        "For each candidate child state $s_i$, search projects the highest-confidence action $a^*_i \\in A(s_i)$ to form virtual grandchild $g_i^* = s_i \\oplus a^*_i$, computing the two-step sum of relative information gains:\n"
        "$$\n"
        "\\text{Future}R_{ig}(s, a_i) = R_{ig}(s, s_i) + \\lambda \\, R_{ig}(s_i, g_i^*) = \\frac{H(s) - H(s_i)}{\\max(H(s), \\epsilon)} + \\lambda \\frac{H(s_i) - H(g_i^*)}{\\max(H(s_i), \\epsilon)}\n"
        "$$\n\n"
        "### Key Algorithmic & Protocol Rules:\n"
        "- **Search Budget $B \\in \\{256, 512\\}$**: Measured in actual sequence model forward passes (per-sequence NFEs). Caching saves model forwards without consuming budget.\n"
        "- **Simulation Cap Safeguard ($4B$)**: Capped at $4B$ completed simulations per question to prevent nonterminating or runaway tree traversal loops over cached states. The NFE utilization metric $\\frac{\\text{actual NFEs}}{B}$ is monitored to verify whether the cap binds.\n"
        "- **Virtual Grandchildren ($g_i^*$)**: Evaluated and cached in `cached_analyses` to eliminate redundant forwards, but **never attached to the MCTS tree** and never assigned GEN/CONT posteriors.\n"
        "- **AB-MEDAL Second-Step Selection**: On `GEN`, the first step creates child $s_i$. The second step performs a forced greedy GEN ($a_i^* = A(s_i)[0]$) to construct $g_i^*$ without mutating $s_i$'s GEN/CONT posteriors or visit counts.\n"
        "- **Terminal vs. Actionless Children**: Terminal children ($0$ masks remaining) have $H(s_i) = 0$ and $R_{ig}(s_i, g_i^*) = 0$. Actionless non-terminal children fallback to 1-step $R_{ig}(s, s_i)$ without fabricating $H(g) = 0$.\n"
        "- **Exact Checkpoint Boundaries**: Snapshots at $B=256$ use non-mutating hypothetical boundary backups on isolated copies so that the snapshot exactly matches an independent $B=256$ run while the live tree continues seamlessly to $B=512$.\n"
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
    cells.append(nbf.v4.new_markdown_cell("## 2. Repository Setup\nClone or update the GitHub repository on branch `adi-e2`."))
    cells.append(nbf.v4.new_code_cell(
        "from pathlib import Path\n"
        "import getpass\n"
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
    cells.append(nbf.v4.new_markdown_cell("## 4. Run Algorithmic Pre-Flight Unit Tests\nRun unit tests to verify 2-step lookahead SRIG reward computation, exact checkpoint boundaries, and state caching."))
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
        "OUTPUT_DIR = Path('results/e2_srig_dev')\n\n"
        "if IN_COLAB:\n"
        "    from google.colab import drive\n"
        "    drive.mount('/content/drive')\n"
        "    OUTPUT_DIR = DRIVE_ROOT / 'results/e2_srig_dev'\n"
        "    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
        "    cache_dir = DRIVE_ROOT / 'cache/huggingface'\n"
        "    cache_dir.mkdir(parents=True, exist_ok=True)\n"
        "    os.environ['HF_HOME'] = str(cache_dir)\n"
        "    os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_dir)\n\n"
        "print(f'Output directory: {OUTPUT_DIR.resolve()}')\n"
    ))

    # Section 6: Experiment Controls
    cells.append(nbf.v4.new_markdown_cell("## 6. Experiment Controls & Configuration\nConfigure 5-question frozen `dev` set, matched search budgets [256, 512], SRIG reward mode with $\\lambda=1.0$, and $4B$ simulation cap."))
    cells.append(nbf.v4.new_code_cell(
        "from dlm_project.tasks.gsm8k.experiment_2 import (\n"
        "    CHECKPOINT_BUDGETS, MAX_SEARCH_DEPTH, MIN_CANDIDATES, context\n"
        ")\n\n"
        "STAGE = 'dev'\n"
        "METHODS = ('medal', 'ab_medal')\n"
        "BUDGETS = (256, 512)\n"
        "REWARD_MODE = 'srig'\n"
        "REWARD_LAMBDA = 1.0\n"
        "SIM_CAP_MULTIPLIER = 4\n"
        "# On NVIDIA L4 (22.4 GiB VRAM), BATCH_SIZE = 2 is recommended for stability;\n"
        "# BATCH_SIZE = 4 is suited for A100 (40/80 GB) or High-RAM runtimes.\n"
        "BATCH_SIZE = 2\n"
        "RESUME = True\n\n"
        "ctx = context(\n"
        "    stage=STAGE,\n"
        "    methods=METHODS,\n"
        "    budgets=BUDGETS,\n"
        "    max_depth=MAX_SEARCH_DEPTH,\n"
        "    num_candidates=MIN_CANDIDATES,\n"
        "    reward_mode=REWARD_MODE,\n"
        "    reward_lambda=REWARD_LAMBDA,\n"
        "    sim_cap_multiplier=SIM_CAP_MULTIPLIER,\n"
        ")\n"
        "print(f'Stage: {ctx[\"stage\"]} | Questions: {len(ctx[\"examples\"])} | Budgets: {BUDGETS} | Methods: {METHODS} | Reward: {REWARD_MODE} (λ={REWARD_LAMBDA}) | Sim Cap: {SIM_CAP_MULTIPLIER}B | Batch Size: {BATCH_SIZE}')\n"
    ))

    # Section 7: Load Model
    cells.append(nbf.v4.new_markdown_cell("## 7. Cache & Load LLaDA-8B-Instruct\nStage model to fast local NVMe storage and load `LLaDA-8B-Instruct` model and tokenizer on CUDA in bfloat16."))
    cells.append(nbf.v4.new_code_cell(
        "import shutil\n"
        "from pathlib import Path\n"
        "from dlm_project.models.profiles import get_profile\n"
        "from dlm_project.tasks.gsm8k.experiment_2 import load_model, prepare\n\n"
        "profile = get_profile('llada')\n"
        "model_name = profile['repo_id'].split('/')[-1]\n"
        "drive_model_dir = DRIVE_ROOT / 'models' / model_name\n"
        "runtime_model_dir = Path('/content/models') / model_name\n\n"
        "# Fast local NVMe cache in Colab runtime if available on Drive\n"
        "if IN_COLAB and drive_model_dir.is_dir() and not (runtime_model_dir / 'config.json').is_file():\n"
        "    print(f'Copying model from Google Drive to local fast runtime storage: {runtime_model_dir}...')\n"
        "    shutil.copytree(drive_model_dir, runtime_model_dir)\n\n"
        "MODEL_PATH = str(runtime_model_dir if runtime_model_dir.is_dir() else (drive_model_dir if drive_model_dir.is_dir() else profile['repo_id']))\n"
        "print('Using model location:', MODEL_PATH)\n"
        "print('Loading model and tokenizer on GPU in bfloat16...')\n"
        "wrapper = load_model(ctx, model_path=MODEL_PATH, device='cuda', local_files_only=Path(MODEL_PATH).is_dir())\n"
        "prepared = prepare(wrapper, ctx)\n"
        "print(f'✓ Model loaded: {wrapper.identity[\"family\"]} ({wrapper.identity[\"repo_id\"]} @ {wrapper.identity[\"revision\"]})')\n"
        "print(f'✓ Prepared {len(prepared)} tokenized evaluation prompts.')\n"
    ))

    # Section 8: Execute Full E2 Dev
    cells.append(nbf.v4.new_markdown_cell("## 8. Execute Experiment 2 (Dev (5 Questions))\nExecute MEDAL and AB-MEDAL with the 2-step lookahead SRIG reward across budgets 256 and 512."))
    cells.append(nbf.v4.new_code_cell(
        "from time import perf_counter\n"
        "from dlm_project.tasks.gsm8k.experiment_2 import run_e2_stage\n\n"
        "print(f'Starting E2 SRIG execution on {len(ctx[\"examples\"])} questions for {METHODS} at budgets {BUDGETS}...')\n"
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
    cells.append(nbf.v4.new_markdown_cell("## 9. Experiment 2 Results, NFE Utilization & Paired Comparison\nDisplay aggregate accuracy, NFE utilization, simulation cap binding rate, and McNemar exact test at matched budgets."))
    cells.append(nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "from IPython.display import display\n"
        "from scripts.run_e2 import compute_paired_comparison\n\n"
        "agg_file = OUTPUT_DIR / 'summaries' / f'e2_{STAGE}_aggregate.csv'\n"
        "if agg_file.is_file():\n"
        "    df_agg = pd.read_csv(agg_file)\n"
        "    print('=== E2 Dev Aggregate Metrics (SRIG Reward) ===')\n"
        "    cols = ['method', 'search_budget', 'sample_count', 'accuracy_percent', 'mean_nfe_utilization', 'mean_search_nfe', 'mean_simulations', 'sim_cap_bind_rate', 'mean_total_nfe', 'mean_selected_depth']\n"
        "    available_cols = [c for c in cols if c in df_agg.columns]\n"
        "    display(df_agg[available_cols])\n\n"
        "paired = compute_paired_comparison(q_rows, budgets=BUDGETS)\n"
        "if paired:\n"
        "    df_paired = pd.DataFrame(paired)\n"
        "    print('=== Paired Comparison (AB-MEDAL vs. MEDAL at Matched Search Budget) ===')\n"
        "    display(df_paired[['search_budget', 'paired_questions', 'medal_accuracy', 'ab_medal_accuracy', 'accuracy_delta_percent', 'ab_medal_only_correct', 'medal_only_correct', 'mcnemar_exact_p']])\n"
    ))

    # Section 10: Search Tree Inspector
    cells.append(nbf.v4.new_markdown_cell("## 10. Interactive Search Tree & Decision Trace Inspector\nExamine candidate shortlists, 2-step SRIG reward breakdowns, and lookahead state entropy drops."))
    cells.append(nbf.v4.new_code_cell(
        "import json\n\n"
        "sample_ex_id = prepared[0]['example']['example_id']\n"
        "chk_dir = OUTPUT_DIR / STAGE / 'ab_medal' / sample_ex_id / 'checkpoints/256'\n"
        "if (chk_dir / 'metrics.json').is_file():\n"
        "    met = json.loads((chk_dir / 'metrics.json').read_text())\n"
        "    print(f'Sample {sample_ex_id} @ Budget 256:')\n"
        "    print(f'  Selected Node: {met[\"selected_node_id\"]} | Depth: {met[\"selected_depth\"]} | Q: {met[\"selected_q\"]:.4f}')\n"
        "    print(f'  Exact Match: {met[\"exact_match\"]} | Pred: {met[\"predicted_answer\"]} | Ref: {met[\"reference_answer\"]}')\n"
        "    print(f'  NFE Utilization: {met.get(\"nfe_utilization\", 1.0)*100:.1f}% | Sims: {met.get(\"simulations_completed\", \"N/A\")}/{met.get(\"simulation_cap\", \"N/A\")}')\n"
    ))

    # Section 11: E3 Offline Metric Extraction & Publication Figures
    cells.append(nbf.v4.new_markdown_cell(
        "## 11. E3 Offline Metric Extraction & Publication Figures\n"
        "Run offline tree allocation, branching dynamics, and structural analysis on saved E2 checkpoints (zero GPU compute), and generate 6 publication figures."
    ))
    cells.append(nbf.v4.new_code_cell(
        "from dlm_project.tasks.gsm8k.e3_analysis import analyze_e3_stage\n"
        "from IPython.display import Image, display, Markdown\n\n"
        "print('Running E3 Offline Tree Allocation & Topology Analysis on E2 traces...')\n"
        "e3_q, e3_agg = analyze_e3_stage(\n"
        "    results_dir=OUTPUT_DIR / STAGE,\n"
        "    output_dir=OUTPUT_DIR / 'summaries',\n"
        "    stage=STAGE,\n"
        "    methods=tuple(METHODS),\n"
        "    budgets=tuple(BUDGETS),\n"
        ")\n"
        "print(f'✓ E3 Analysis Completed successfully: {len(e3_q)} question outcomes evaluated.')\n\n"
        "figures_dir = OUTPUT_DIR / 'summaries' / 'e3_figures'\n"
        "fig_titles = {\n"
        "    'fig1_nfe_by_depth': 'Figure 1: NFE Allocation by Tree Depth — MEDAL vs AB-MEDAL across budgets',\n"
        "    'fig2_branching_factors': 'Figure 2: Branching Factor Distribution — Children per expanded parent',\n"
        "    'fig3_candidate_density': 'Figure 3: Candidate Density vs Depth — Minimum-C availability and selection depth',\n"
        "    'fig4_root_subtrees': 'Figure 4: Root-Subtree Compute Concentration — Search effort across root branches',\n"
        "    'fig5_reward_redundancy': 'Figure 5: Reward and Search Redundancy — Negative IG rates and duplicate states',\n"
        "    'fig6_accuracy_vs_budget': 'Figure 6: Accuracy vs Search Budget — Performance scaling curve',\n"
        "}\n\n"
        "for fig_key, title in fig_titles.items():\n"
        "    png_path = figures_dir / f'{fig_key}.png'\n"
        "    if png_path.is_file():\n"
        "        display(Markdown(f'### {title}'))\n"
        "        display(Image(filename=str(png_path), width=750))\n"
    ))

    # Section 12: Drive Sync
    cells.append(nbf.v4.new_markdown_cell("## 12. Synchronize All Artifacts to Google Drive\nExplicitly flush and sync all in-flight buffers to Google Drive."))
    cells.append(nbf.v4.new_code_cell(
        "if IN_COLAB:\n"
        "    from google.colab import drive\n"
        "    drive.flush_and_unmount()\n"
        "    print('Google Drive flushed successfully.')\n"
    ))

    nb.cells = cells
    out_path = NOTEBOOKS_DIR / "E2_Experiment_SRIG_Dev.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        nbf.write(nb, f)
    print(f"Generated {out_path} ({len(cells)} cells)")


def create_e2_srig_eval_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Badge & Header
    cells.append(nbf.v4.new_markdown_cell(
        '<a href="https://colab.research.google.com/github/CooL-Legend/CSCI544-DLM/blob/adi-e2/notebooks/gsm8k/E2_Experiment_SRIG_Eval.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>'
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "# Experiment 2: Two-Step Lookahead MEDAL vs. AB-MEDAL on GSM8K with Sum of Relative Information Gains (SRIG) — Evaluation (20 Canonical + 50 Extended Questions)\n\n"
        "This notebook runs **Experiment 2 (E2)** on the GSM8K validation/evaluation pool with `LLaDA-8B-Instruct` using the **Two-Step Sum of Relative Information Gains (SRIG / $\\text{Future}R_{ig}$)** reward function:\n"
        "$$\n"
        "\\text{Future}R_{ig}(s, a_i) = R_{ig}(s, s_i) + \\lambda \\, R_{ig}(s_i, g_i^*) = \\frac{H(s) - H(s_i)}{\\max(H(s), \\epsilon)} + \\lambda \\frac{H(s_i) - H(g_i^*)}{\\max(H(s_i), \\epsilon)}\n"
        "$$\n\n"
        "### Dataset Split Organization:\n"
        "1. **Canonical 20 Samples** (`sample_slice=(0, 20)`): The exact 20 evaluation questions from Experiment 1, preserving historical prefix ordering.\n"
        "2. **Additional 50 Samples** (`sample_slice=(20, 70)`): 50 new validation questions sampled under the deterministic `sample_seed=1` order.\n"
        "3. **Combined 70 Samples**: Full validation pool evaluation and paired statistical tests.\n\n"
        "### Guardrails & Protocols:\n"
        "- Capped at $4B$ simulations per question as an exploration runaway safeguard; actual NFE utilization is logged.\n"
        "- Virtual grandchildren are evaluated and cached in `cached_analyses` but never attached to the tree or assigned GEN/CONT posteriors.\n"
        "- Non-mutating hypothetical boundary snapshots ensure standalone $B=256$ equivalence without altering live tree execution toward $B=512$.\n"
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
    cells.append(nbf.v4.new_markdown_cell("## 2. Repository Setup\nClone or update the GitHub repository on branch `adi-e2`."))
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
    cells.append(nbf.v4.new_markdown_cell("## 4. Run Algorithmic Pre-Flight Unit Tests\nVerify 2-step lookahead SRIG reward computation, state caching, and batched parity before loading weights."))
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
        "OUTPUT_DIR = Path('results/e2_srig_eval')\n\n"
        "if IN_COLAB:\n"
        "    from google.colab import drive\n"
        "    drive.mount('/content/drive')\n"
        "    OUTPUT_DIR = DRIVE_ROOT / 'results/e2_srig_eval'\n"
        "    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
        "    cache_dir = DRIVE_ROOT / 'cache/huggingface'\n"
        "    cache_dir.mkdir(parents=True, exist_ok=True)\n"
        "    os.environ['HF_HOME'] = str(cache_dir)\n"
        "    os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_dir)\n\n"
        "print(f'Output directory: {OUTPUT_DIR.resolve()}')\n"
    ))

    # Section 6: Controls & Model Loading
    cells.append(nbf.v4.new_markdown_cell("## 6. Experiment Controls & Staged Model Loading\nConfigure budgets [256, 512], depth 10, batch size 2, SRIG reward mode, and load `LLaDA-8B-Instruct` from staged NVMe storage."))
    cells.append(nbf.v4.new_code_cell(
        "import shutil\n"
        "from pathlib import Path\n"
        "from dlm_project.models.profiles import get_profile\n"
        "from dlm_project.tasks.gsm8k.experiment_2 import (\n"
        "    CHECKPOINT_BUDGETS, MAX_SEARCH_DEPTH, MIN_CANDIDATES, context, load_model\n"
        ")\n\n"
        "STAGE = 'evaluation'\n"
        "METHODS = ('medal', 'ab_medal')\n"
        "BUDGETS = (256, 512)\n"
        "REWARD_MODE = 'srig'\n"
        "REWARD_LAMBDA = 1.0\n"
        "SIM_CAP_MULTIPLIER = 4\n"
        "# On NVIDIA L4 (22.4 GiB VRAM), BATCH_SIZE = 2 is recommended for stability;\n"
        "# BATCH_SIZE = 4 is suited for A100 (40/80 GB) or High-RAM runtimes.\n"
        "BATCH_SIZE = 2\n"
        "RESUME = True\n\n"
        "# Load base context to initialize model identity\n"
        "base_ctx = context(\n"
        "    stage='evaluation',\n"
        "    methods=METHODS,\n"
        "    budgets=BUDGETS,\n"
        "    reward_mode=REWARD_MODE,\n"
        "    reward_lambda=REWARD_LAMBDA,\n"
        "    sim_cap_multiplier=SIM_CAP_MULTIPLIER,\n"
        ")\n\n"
        "profile = get_profile('llada')\n"
        "model_name = profile['repo_id'].split('/')[-1]\n"
        "drive_model_dir = DRIVE_ROOT / 'models' / model_name\n"
        "runtime_model_dir = Path('/content/models') / model_name\n\n"
        "# Fast local NVMe cache in Colab runtime if available on Drive\n"
        "if IN_COLAB and drive_model_dir.is_dir() and not (runtime_model_dir / 'config.json').is_file():\n"
        "    print(f'Copying model from Google Drive to local fast runtime storage: {runtime_model_dir}...')\n"
        "    shutil.copytree(drive_model_dir, runtime_model_dir)\n\n"
        "MODEL_PATH = str(runtime_model_dir if runtime_model_dir.is_dir() else (drive_model_dir if drive_model_dir.is_dir() else profile['repo_id']))\n"
        "print('Using model location:', MODEL_PATH)\n"
        "print('Loading LLaDA-8B-Instruct wrapper on GPU in bfloat16...')\n"
        "wrapper = load_model(base_ctx, model_path=MODEL_PATH, device='cuda', local_files_only=Path(MODEL_PATH).is_dir())\n"
        "print(f'✓ Model loaded: {wrapper.identity[\"family\"]} ({wrapper.identity[\"repo_id\"]} @ {wrapper.identity[\"revision\"]})')\n"
    ))

    # Section 7: Run Canonical 20 Samples
    cells.append(nbf.v4.new_markdown_cell(
        "## 7. Execute Part 1: Canonical Evaluation Set (Original 20 Questions)\n"
        "Evaluates the exact 20 evaluation questions from Experiment 1 (`sample_slice=(0, 20)`)."
    ))
    cells.append(nbf.v4.new_code_cell(
        "from time import perf_counter\n"
        "from dlm_project.tasks.gsm8k.experiment_2 import run_e2_stage\n\n"
        "ctx_20 = context(\n"
        "    stage='evaluation',\n"
        "    methods=METHODS,\n"
        "    budgets=BUDGETS,\n"
        "    sample_slice=(0, 20),\n"
        "    reward_mode=REWARD_MODE,\n"
        "    reward_lambda=REWARD_LAMBDA,\n"
        "    sim_cap_multiplier=SIM_CAP_MULTIPLIER,\n"
        ")\n"
        "print(f'Starting Part 1 (Canonical 20 Questions) with SRIG reward...')\n"
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
        "ctx_50 = context(\n"
        "    stage='evaluation',\n"
        "    methods=METHODS,\n"
        "    budgets=BUDGETS,\n"
        "    sample_slice=(20, 70),\n"
        "    reward_mode=REWARD_MODE,\n"
        "    reward_lambda=REWARD_LAMBDA,\n"
        "    sim_cap_multiplier=SIM_CAP_MULTIPLIER,\n"
        ")\n"
        "print(f'Starting Part 2 (Additional 50 Questions) with SRIG reward...')\n"
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
        "from scripts.run_e2 import compute_paired_comparison\n"
        "from dlm_project.experiments.sweep_report import write_csv\n\n"
        "summaries_dir = OUTPUT_DIR / 'summaries'\n"
        "summaries_dir.mkdir(parents=True, exist_ok=True)\n\n"
        "print('=== Part 1: Canonical 20 Questions Paired Comparison ===')\n"
        "paired_20 = compute_paired_comparison(q_rows_20, budgets=BUDGETS)\n"
        "if paired_20:\n"
        "    df_p20 = pd.DataFrame(paired_20)[['search_budget', 'paired_questions', 'medal_accuracy', 'ab_medal_accuracy', 'accuracy_delta_percent', 'ab_medal_only_correct', 'medal_only_correct', 'mcnemar_exact_p']]\n"
        "    display(df_p20)\n"
        "    write_csv(summaries_dir / 'e2_srig_canonical_20_paired_comparison.csv', paired_20)\n\n"
        "print('=== Part 2: Additional 50 Questions Paired Comparison ===')\n"
        "paired_50 = compute_paired_comparison(q_rows_50, budgets=BUDGETS)\n"
        "if paired_50:\n"
        "    df_p50 = pd.DataFrame(paired_50)[['search_budget', 'paired_questions', 'medal_accuracy', 'ab_medal_accuracy', 'accuracy_delta_percent', 'ab_medal_only_correct', 'medal_only_correct', 'mcnemar_exact_p']]\n"
        "    display(df_p50)\n"
        "    write_csv(summaries_dir / 'e2_srig_extended_50_paired_comparison.csv', paired_50)\n\n"
        "print('=== Combined Pool (All 70 Questions) Paired Comparison ===')\n"
        "combined_q_rows = q_rows_20 + q_rows_50\n"
        "paired_combined = compute_paired_comparison(combined_q_rows, budgets=BUDGETS)\n"
        "if paired_combined:\n"
        "    df_pcomb = pd.DataFrame(paired_combined)[['search_budget', 'paired_questions', 'medal_accuracy', 'ab_medal_accuracy', 'accuracy_delta_percent', 'ab_medal_only_correct', 'medal_only_correct', 'mcnemar_exact_p']]\n"
        "    display(df_pcomb)\n"
        "    write_csv(summaries_dir / 'e2_srig_combined_70_paired_comparison.csv', paired_combined)\n"
        "    write_csv(summaries_dir / 'e2_srig_evaluation_paired_comparison.csv', paired_combined)\n"
    ))

    # Section 10: Interactive Search Tree & Decision Trace Inspector
    cells.append(nbf.v4.new_markdown_cell(
        "## 10. Interactive Search Tree & Decision Trace Inspector\n"
        "Inspect the search tree topology, candidate nodes, and decision trace for any evaluated question."
    ))
    cells.append(nbf.v4.new_code_cell(
        "from dlm_project.tasks.gsm8k.tree_viz import print_ascii_tree, summarize_decision_trace\n"
        "import json\n\n"
        "# Select a sample question to inspect from canonical_20\n"
        "sample_ex_id = ctx_20['examples'][0]['example_id']\n"
        "chk_dir = OUTPUT_DIR / 'canonical_20' / 'evaluation' / 'ab_medal' / sample_ex_id / 'checkpoints/256'\n"
        "tree_json = chk_dir / 'tree.json'\n"
        "metrics_json = chk_dir / 'metrics.json'\n\n"
        "if tree_json.is_file():\n"
        "    print(f'=== AB-MEDAL Search Tree at B=256 for {sample_ex_id} ===')\n"
        "    print(print_ascii_tree(tree_json, max_depth=5, max_children_per_node=3))\n\n"
        "if metrics_json.is_file():\n"
        "    met = json.loads(metrics_json.read_text())\n"
        "    print(f'=== Checkpoint Metrics ({sample_ex_id} @ B=256) ===')\n"
        "    print(f'  Selected Node: {met[\"selected_node_id\"]} | Depth: {met[\"selected_depth\"]} | Q: {met[\"selected_q\"]:.4f}')\n"
        "    print(f'  Exact Match: {met[\"exact_match\"]} | Pred: {met[\"predicted_answer\"]} | Ref: {met[\"reference_answer\"]}')\n"
        "    print(f'  NFE Utilization: {met.get(\"nfe_utilization\", 1.0)*100:.1f}% | Sims: {met.get(\"simulations_completed\", \"N/A\")}/{met.get(\"simulation_cap\", \"N/A\")}')\n"
    ))

    # Section 11: E3 Offline Metric Extraction & Publication Figures
    cells.append(nbf.v4.new_markdown_cell(
        "## 11. E3 Offline Metric Extraction & Publication Figures\n"
        "Run offline tree allocation, branching dynamics, and structural analysis on saved evaluation checkpoints (zero GPU compute), and generate 6 publication figures."
    ))
    cells.append(nbf.v4.new_code_cell(
        "import shutil\n"
        "from dlm_project.tasks.gsm8k.e3_analysis import analyze_e3_stage\n"
        "from IPython.display import Image, display, Markdown\n\n"
        "# Build combined trace directory linking canonical_20 and extended_50\n"
        "combined_results_dir = OUTPUT_DIR / 'combined_70' / 'evaluation'\n"
        "combined_results_dir.mkdir(parents=True, exist_ok=True)\n"
        "for part_dir in [OUTPUT_DIR / 'canonical_20' / 'evaluation', OUTPUT_DIR / 'extended_50' / 'evaluation']:\n"
        "    if part_dir.is_dir():\n"
        "        for m in METHODS:\n"
        "            m_src = part_dir / m\n"
        "            if m_src.is_dir():\n"
        "                m_dst = combined_results_dir / m\n"
        "                m_dst.mkdir(parents=True, exist_ok=True)\n"
        "                for q_dir in m_src.iterdir():\n"
        "                    if q_dir.is_dir() and not (m_dst / q_dir.name).exists():\n"
        "                        try:\n"
        "                            (m_dst / q_dir.name).symlink_to(q_dir, target_is_directory=True)\n"
        "                        except OSError:\n"
        "                            shutil.copytree(q_dir, m_dst / q_dir.name)\n\n"
        "# Select results directory: combined_70 if populated, else canonical_20\n"
        "has_combined = any((combined_results_dir / m).is_dir() and any((combined_results_dir / m).iterdir()) for m in METHODS)\n"
        "eval_results_dir = combined_results_dir if has_combined else (OUTPUT_DIR / 'canonical_20' / 'evaluation')\n"
        "e3_output_dir = OUTPUT_DIR / ('combined_70' if has_combined else 'canonical_20') / 'summaries'\n"
        "e3_output_dir.mkdir(parents=True, exist_ok=True)\n\n"
        "print(f'Running E3 Offline Tree Analysis on {eval_results_dir}...')\n"
        "e3_q, e3_agg = analyze_e3_stage(\n"
        "    results_dir=eval_results_dir,\n"
        "    output_dir=e3_output_dir,\n"
        "    stage='evaluation',\n"
        "    methods=tuple(METHODS),\n"
        "    budgets=tuple(BUDGETS),\n"
        ")\n"
        "print(f'✓ E3 Analysis Completed successfully: {len(e3_q)} question outcomes evaluated.')\n\n"
        "figures_dir = e3_output_dir / 'e3_figures'\n"
        "fig_titles = {\n"
        "    'fig1_nfe_by_depth': 'Figure 1: NFE Allocation by Tree Depth — MEDAL vs AB-MEDAL across budgets',\n"
        "    'fig2_branching_factors': 'Figure 2: Branching Factor Distribution — Children per expanded parent',\n"
        "    'fig3_candidate_density': 'Figure 3: Candidate Density vs Depth — Minimum-C availability and selection depth',\n"
        "    'fig4_root_subtrees': 'Figure 4: Root-Subtree Compute Concentration — Search effort across root branches',\n"
        "    'fig5_reward_redundancy': 'Figure 5: Reward and Search Redundancy — Negative IG rates and duplicate states',\n"
        "    'fig6_accuracy_vs_budget': 'Figure 6: Accuracy vs Search Budget — Performance scaling curve',\n"
        "}\n\n"
        "for fig_key, title in fig_titles.items():\n"
        "    png_path = figures_dir / f'{fig_key}.png'\n"
        "    if png_path.is_file():\n"
        "        display(Markdown(f'### {title}'))\n"
        "        display(Image(filename=str(png_path), width=750))\n"
    ))

    # Section 12: Drive Flush
    cells.append(nbf.v4.new_markdown_cell("## 12. Synchronize All Artifacts to Google Drive\nFlush and sync all in-flight buffers to Google Drive."))
    cells.append(nbf.v4.new_code_cell(
        "if IN_COLAB:\n"
        "    from google.colab import drive\n"
        "    drive.flush_and_unmount()\n"
        "    print('Google Drive flushed successfully.')\n"
    ))

    nb.cells = cells
    out_path = NOTEBOOKS_DIR / "E2_Experiment_SRIG_Eval.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        nbf.write(nb, f)
    print(f"Generated {out_path} ({len(cells)} cells)")


def main():
    create_e2_dev_notebook()
    create_e2_eval_notebook()
    create_e2_srig_dev_notebook()
    create_e2_srig_eval_notebook()


if __name__ == "__main__":
    main()

