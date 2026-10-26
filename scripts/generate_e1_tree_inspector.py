"""Script to generate notebooks/gsm8k/E1_Tree_Inspector.ipynb."""

from pathlib import Path
import nbformat as nbf


def generate_notebook(output_path: Path):
    nb = nbf.v4.new_notebook()

    # Cell 1: Title and Documentation with Colab badge
    c1_md = """<a href="https://colab.research.google.com/github/CooL-Legend/CSCI544-DLM/blob/adi/notebooks/gsm8k/E1_Tree_Inspector.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a>

# Experiment 1 & 3: Tree Structure & Candidate Inspector
### Offline AB-MCTS Tree Visualization & Candidate Analysis for MEDAL and AB-MEDAL on GSM8K

This notebook inspects saved **Experiment 1 (E1)** and **Experiment 3 (E3)** search trees and candidate evaluations stored on **Google Drive** or locally.
It requires **zero GPU, model weights, or inference passes**—it runs 100% offline on standard CPU Colab instances.

---
### Key Features:
1. **Google Drive Integration**: Automatically mounts Drive and locates your saved E1 checkpoints.
2. **Early Score Comparison**: Compares the scores and final candidates selected by MEDAL and AB-MEDAL across all evaluated questions, accompanied by aggregate accuracy and efficiency metrics.
3. **Distinct Candidate Analysis**: Automatically identifies candidate depth $d^*$ (the deepest depth holding $\\ge C$ distinct candidate states, default $C=3$) and reports the candidate count, full depth distribution, and candidate state properties.
4. **AB-MCTS Tree Visualization**: Generates full comparative tree structures (Immediate IG and Backed-up Mean Q) matching [AB-MCTS Figure 7](https://arxiv.org/html/2503.04412v5#S4.F7), inlining PNG previews and saving vector PDFs.
"""
    nb.cells.append(nbf.v4.new_markdown_cell(c1_md))

    # Cell 2: Setup Repo and Environment
    c2_code = """from pathlib import Path
import getpass
import os
import subprocess
import sys
import tempfile

REPOSITORY = "CooL-Legend/CSCI544-DLM"
BRANCH = "adi"
REPO_URL = f"https://github.com/{REPOSITORY}.git"
IN_COLAB = Path("/content").is_dir()
REPO_DIR = Path("/content/CSCI544-DLM") if IN_COLAB else next(
    path for path in (Path.cwd(), *Path.cwd().parents)
    if (path / "src/dlm_project").is_dir()
)

def run_git(args, token=None):
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    with tempfile.TemporaryDirectory() as tmp:
        if token:
            askpass = Path(tmp) / "askpass.py"
            askpass.write_text(
                "#!/usr/bin/env python3\\nimport os, sys\\n"
                "print('x-access-token' if 'username' in sys.argv[1].lower() else os.environ['COLAB_GH_TOKEN'])\\n"
            )
            askpass.chmod(0o700)
            env.update(GIT_ASKPASS=str(askpass), COLAB_GH_TOKEN=token)
        return subprocess.run(["git", "-c", "credential.helper=", *args], env=env, capture_output=True, text=True)

def remote_git(args):
    result = run_git(args)
    if result.returncode:
        try:
            from google.colab import userdata
            token = userdata.get("GITHUB_TOKEN")
        except Exception:
            token = None
        if not token:
            token = getpass.getpass("GitHub token with read access to private repo CooL-Legend/CSCI544-DLM: ")
        try:
            result = run_git(args, token)
        finally:
            del token
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "GitHub clone/update failed.")

def ensure_checkout():
    def local_git(*args):
        return subprocess.check_output(["git", "-C", str(REPO_DIR), *args], text=True).strip()

    if REPO_DIR.exists():
        if not (REPO_DIR / ".git").is_dir():
            raise RuntimeError(f"{REPO_DIR} exists but is not a Git checkout.")
        fetch_specs = local_git("config", "--get-all", "remote.origin.fetch").splitlines()
        branch_ref = f"refs/heads/{BRANCH}:refs/remotes/origin/{BRANCH}"
        if not any(spec.lstrip("+") in (branch_ref, "refs/heads/*:refs/remotes/origin/*") for spec in fetch_specs):
            local_git("remote", "set-branches", "--add", "origin", BRANCH)
        remote_git(["-C", str(REPO_DIR), "fetch", "origin", branch_ref])
        exists = subprocess.run(["git", "-C", str(REPO_DIR), "show-ref", "--verify", "--quiet", f"refs/heads/{BRANCH}"]).returncode == 0
        if exists:
            local_git("switch", BRANCH)
        else:
            local_git("switch", "--create", BRANCH, "--track", f"origin/{BRANCH}")
        local_git("merge", "--ff-only", f"origin/{BRANCH}")
    else:
        remote_git(["clone", "--branch", BRANCH, "--single-branch", REPO_URL, str(REPO_DIR)])
    print(f"Code ready: {REPO_DIR} | branch: {BRANCH} | commit: {local_git('rev-parse', '--short', 'HEAD')}")

if IN_COLAB:
    ensure_checkout()
else:
    print(f"Using local repository checkout: {REPO_DIR}")

os.chdir(REPO_DIR)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# Install dependencies and editable package
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_DIR)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "matplotlib", "pandas", "seaborn"], check=True)

print(f"✓ Dependencies installed and codebase ready at: {REPO_DIR}")
"""
    nb.cells.append(nbf.v4.new_code_cell(c2_code))

    # Cell 3: Mount Google Drive & Data Path Configuration
    c3_code = """from pathlib import Path
import os

# 1. Mount Google Drive if running in Google Colab
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")

# 2. Experiment Settings
STAGE = "dev"       # 'dev', 'evaluation', or 'test'
BUDGET = 512        # Checkpoint budget: 256, 512, or 1024
MIN_C = 3           # Minimum distinct candidates threshold C (paper rule)

# 3. Auto-detect results directory matching Dev & Eval Colab paths
# You can also explicitly override DATA_DIR by setting it to a specific Path:
# DATA_DIR = Path("/content/drive/MyDrive/CSCI544-DLM/results/e1_e3")
search_roots = [
    Path("/content/drive/MyDrive/CSCI544-DLM/LLaDA/GSM8K/experiment_1/results/e1_e3"),
    Path("/content/drive/MyDrive/CSCI544-DLM/results/e1_e3"),
    REPO_DIR / "results/e1_e3",
    Path("/content/CSCI544-DLM/results/e1_e3"),
]

DATA_DIR = next(
    (p for p in search_roots if p is not None and (p / STAGE).is_dir()),
    search_roots[0]
)

RESULTS_STAGE_DIR = (DATA_DIR / STAGE) if (DATA_DIR / STAGE).is_dir() else DATA_DIR

print(f"Experiment Stage:    {STAGE}")
print(f"Search Budget:       {BUDGET}")
print(f"Minimum Candidates: C = {MIN_C}")
print(f"Data Directory:      {DATA_DIR}")
print(f"Resolved Stage Dir:  {RESULTS_STAGE_DIR} (exists: {RESULTS_STAGE_DIR.exists()})")
"""
    nb.cells.append(nbf.v4.new_code_cell(c3_code))

    # Cell 4: Auxiliary & Helper Functions
    c4_code = """import json
from pathlib import Path
import pandas as pd
from typing import Dict, Any, List

from dlm_project.tasks.gsm8k.candidate_history import (
    extract_distinct_candidates,
    extract_candidate_progression,
    load_question_comparison,
)
from dlm_project.tasks.gsm8k.e3_tree_plots import plot_e3_tree_comparison, COLOR_METRICS
from IPython.display import Image, Markdown, display

def discover_stage_questions(stage_dir: Path, budget: int) -> List[Dict[str, Any]]:
    \"\"\"Scan results directory and list all evaluated questions for the given budget.\"\"\"
    questions = {}
    for method in ("medal", "ab_medal"):
        m_dir = stage_dir / method
        if not m_dir.is_dir():
            continue
        for q_dir in sorted(m_dir.iterdir()):
            if not q_dir.is_dir():
                continue
            ex_id = q_dir.name
            chk_dir = q_dir / "checkpoints" / str(budget)
            met_file = chk_dir / "metrics.json"
            if met_file.is_file():
                if ex_id not in questions:
                    questions[ex_id] = {
                        "example_id": ex_id,
                        "methods": [],
                        "question": "",
                        "reference_answer": "",
                    }
                questions[ex_id]["methods"].append(method)
                try:
                    met = json.loads(met_file.read_text())
                    questions[ex_id]["question"] = met.get("question", "")
                    questions[ex_id]["reference_answer"] = met.get("reference_answer", "")
                except Exception:
                    pass
    return list(questions.values())

print("✓ Auxiliary and inspection helper functions loaded successfully.")
"""
    nb.cells.append(nbf.v4.new_code_cell(c4_code))

    # Cell 5: Early Section: Per-Question & Aggregate Score Comparison
    c5_code = """print("================================================================================")
print(f"EARLY SECTION: PER-QUESTION & AGGREGATE SCORE COMPARISON (Stage={STAGE}, Budget={BUDGET})")
print("================================================================================")

# 1. Load Per-Question Comparison
q_rows, agg_summary = load_question_comparison(RESULTS_STAGE_DIR, stage=STAGE, budget=BUDGET)

if not q_rows:
    print(f"No completed question checkpoints found in {RESULTS_STAGE_DIR} for budget {BUDGET}.")
    print("Verify that DATA_DIR contains evaluated results.")
else:
    # Build clean Per-Question DataFrame
    table_rows = []
    for r in q_rows:
        m_correct = r.get("medal_exact_match")
        ab_correct = r.get("ab_medal_exact_match")
        m_sym = "✓" if m_correct else ("✗" if m_correct is False else "-")
        ab_sym = "✓" if ab_correct else ("✗" if ab_correct is False else "-")
        
        table_rows.append({
            "Example ID": r["example_id"],
            "Ref Answer": r.get("reference_answer"),
            "MEDAL Pred": f"{m_sym} {r.get('medal_predicted') or '-'}",
            "MEDAL Score": r.get("medal_score"),
            "MEDAL Depth (d*)": r.get("medal_selected_depth"),
            "MEDAL NFEs": r.get("medal_total_nfes"),
            "AB-MEDAL Pred": f"{ab_sym} {r.get('ab_medal_predicted') or '-'}",
            "AB-MEDAL Score": r.get("ab_medal_score"),
            "AB-MEDAL Depth (d*)": r.get("ab_medal_selected_depth"),
            "AB-MEDAL NFEs": r.get("ab_medal_total_nfes"),
            "Outcome": r.get("outcome"),
        })
    
    df_questions = pd.DataFrame(table_rows)
    display(Markdown(f"### Per-Question Selected Candidate Comparison (Budget B={BUDGET})"))
    display(df_questions)
    
    # 2. Display Aggregate Analysis
    display(Markdown(f"### Aggregate Performance & Efficiency Summary"))
    
    agg_table = []
    for m in ("medal", "ab_medal"):
        if m in agg_summary.get("methods", {}):
            s = agg_summary["methods"][m]
            agg_table.append({
                "Method": m.upper(),
                "Sample Count": s["sample_count"],
                "Correct": s["correct_count"],
                "Accuracy (%)": f"{s['accuracy_percent']:.1f}%",
                "95% Wilson CI": f"[{s['ci_95_low']*100:.1f}%, {s['ci_95_high']*100:.1f}%]",
                "Mean Selected Depth (d*)": f"{s['mean_selected_depth']:.2f}",
                "Mean Selected Q": f"{s['mean_selected_q']:.4f}",
                "Mean Total NFEs": f"{s['mean_total_nfes']:.1f}",
            })
    
    if agg_table:
        display(pd.DataFrame(agg_table))
    
    h2h = agg_summary.get("head_to_head", {})
    display(Markdown(f"**Head-to-Head Breakdown (N={h2h.get('compared_count', 0)})**: "
                     f"AB-MEDAL Wins: **{h2h.get('ab_medal_wins', 0)}** | "
                     f"MEDAL Wins: **{h2h.get('medal_wins', 0)}** | "
                     f"Both Correct Ties: **{h2h.get('both_correct_ties', 0)}** | "
                     f"Both Incorrect Ties: **{h2h.get('both_incorrect_ties', 0)}**"))
"""
    nb.cells.append(nbf.v4.new_code_cell(c5_code))

    # Cell 6: Question Selector
    c6_code = """# Discover all available evaluated questions
available_questions = discover_stage_questions(RESULTS_STAGE_DIR, BUDGET)

print(f"Available evaluated questions in {STAGE} ({len(available_questions)} total):")
for idx, q in enumerate(available_questions):
    print(f"  [{idx}] {q['example_id']:<15} | Methods: {q['methods']} | {q['question'][:55]}...")

# SELECT QUESTION: Choose by index, example_id, or search string
QUESTION_INDEX = 0         # Change this to 0, 1, 2, 3... to pick by index
EXAMPLE_ID_OVERRIDE = None # Or set to a specific ID e.g. 'train:6499'
SEARCH_TEXT = None         # Or set to a search word e.g. 'bookstore'

selected_q = None
if EXAMPLE_ID_OVERRIDE:
    selected_q = next((q for q in available_questions if q["example_id"] == EXAMPLE_ID_OVERRIDE), None)
elif SEARCH_TEXT:
    selected_q = next((q for q in available_questions if SEARCH_TEXT.lower() in q["question"].lower()), None)
elif available_questions and 0 <= QUESTION_INDEX < len(available_questions):
    selected_q = available_questions[QUESTION_INDEX]

if not selected_q:
    print(f"Warning: Selected question not found. Defaulting to available questions list.")
    selected_ex_id = available_questions[0]["example_id"] if available_questions else "example_not_found"
else:
    selected_ex_id = selected_q["example_id"]
    display(Markdown(f"### Selected Question: `{selected_ex_id}`"))
    print("Question:\\n", selected_q.get("question", ""))
    print("\\nReference Answer:", selected_q.get("reference_answer", ""))

SELECTED_EXAMPLE_ID = selected_ex_id
"""
    nb.cells.append(nbf.v4.new_code_cell(c6_code))

    # Cell 7: Distinct Candidate Analysis at Depth with Candidates >= C
    c7_code = """display(Markdown(f"### Distinct Candidates Analysis at Qualifying Depth ($d^*$) with Candidates $\\\\ge {MIN_C}$"))

candidate_reports = {}

for method in ("medal", "ab_medal"):
    chk_dir = RESULTS_STAGE_DIR / method / SELECTED_EXAMPLE_ID / "checkpoints" / str(BUDGET)
    tree_path = chk_dir / "tree.json"
    sel_path = chk_dir / "selection.json"
    
    if not tree_path.is_file():
        print(f"[Not Found] {method.upper()} tree file missing: {tree_path}")
        continue
        
    tree_data = json.loads(tree_path.read_text())
    sel_data = json.loads(sel_path.read_text()) if sel_path.is_file() else {}
    
    stats = extract_distinct_candidates(tree_data, min_c=MIN_C, selection_record=sel_data)
    candidate_reports[method] = stats
    
    d_star = stats["candidate_depth"]
    cand_count = stats["candidate_count"]
    dist = stats["distinct_candidates_by_depth"]
    qualifying = stats["qualifying_depths"]
    
    print(f"\\n{'=' * 30} {method.upper()} CANDIDATE SUMMARY {'=' * 30}")
    print(f"  Max Depth Reached in Tree:           {stats['max_depth_reached']}")
    print(f"  Candidate Selection Depth (d*):      {d_star}")
    print(f"  Number of Distinct Candidates at d*: {cand_count} (>= {MIN_C}: {cand_count >= MIN_C})")
    print(f"  Qualifying Depths (count >= {MIN_C}):    {qualifying if qualifying else 'None (fallback used)'}")
    print(f"  Distinct Candidates by Depth:        {dist}")
    
    # Format candidates at depth d* into a DataFrame
    cand_rows = []
    for c in stats["candidates"]:
        action = c.get("action")
        action_str = f"pos {action['output_position']} -> tok {action['token_id']}" if action else "root"
        cand_rows.append({
            "Node ID": c["node_id"],
            "Visits": c["visits"],
            "Mean Q": f"{c['mean_reward']:.4f}",
            "Immediate IG": f"{c['immediate_reward']:.4f}" if c.get("immediate_reward") is not None else "-",
            "Eval Order": c["evaluation_order"],
            "Action": action_str,
            "Shortlist": "★ Yes" if c["is_shortlist"] else "No",
            "Selected": "✓ Chosen" if c["is_selected"] else "-",
        })
    
    df_cands = pd.DataFrame(cand_rows)
    display(Markdown(f"**{method.upper()} Candidates at Depth $d^* = {d_star}$ (Ranked by Mean Q)**:"))
    display(df_cands.head(15))
"""
    nb.cells.append(nbf.v4.new_code_cell(c7_code))

    # Cell 8: AB-MCTS Tree Structure Visualization
    c8_code = """display(Markdown(f"### Full Search Tree Structure Visualization: `{SELECTED_EXAMPLE_ID}` (B={BUDGET})"))
print("Rendering comparative AB-MCTS trees (MEDAL vs AB-MEDAL side-by-side)...")

tree_output_dir = (
    RESULTS_STAGE_DIR.parent / "summaries" / "e3_figures" / "trees"
    / RESULTS_STAGE_DIR.name / SELECTED_EXAMPLE_ID
)

exported_figures = {}
for metric_key, metric_title in (
    ("immediate_reward", "Immediate Information Gain (IG)"),
    ("mean_reward", "Backed-up Mean Q-Value"),
):
    try:
        paths = plot_e3_tree_comparison(
            results_dir=RESULTS_STAGE_DIR,
            output_dir=tree_output_dir,
            example_id=SELECTED_EXAMPLE_ID,
            budget=BUDGET,
            color_by=metric_key,
        )
        exported_figures[metric_key] = paths
        
        display(Markdown(f"#### {metric_title} — MEDAL vs AB-MEDAL"))
        display(Image(filename=str(paths["png"]), width=1400))
        print(f"  ✓ PNG Preview: {paths['png']}")
        print(f"  ✓ Vector PDF:  {paths['pdf']} (zoom to read all evaluation order numbers)")
    except Exception as e:
        print(f"Could not render {metric_title} tree: {e}")

display(Markdown(f"✓ Tree inspection complete. All figure files saved in: `{tree_output_dir}`"))
"""
    nb.cells.append(nbf.v4.new_code_cell(c8_code))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, output_path)
    print(f"Successfully generated {output_path} ({len(nb.cells)} cells).")


if __name__ == "__main__":
    generate_notebook(Path("notebooks/gsm8k/E1_Tree_Inspector.ipynb"))
