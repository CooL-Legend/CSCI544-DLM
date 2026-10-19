# Milestone 1: paired LLaDA decoder sweeps

Run [`scripts/run_gsm8k_sweep.py`](../scripts/run_gsm8k_sweep.py) and
[`scripts/run_sudoku_sweep.py`](../scripts/run_sudoku_sweep.py). Each process loads
the **same original LLaDA-8B-Instruct** checkpoint once, then evaluates examples
and conditions sequentially. Sudoku uses **BGPO's released 4×4 dataset**, not its
fine-tuned weights. The existing BGPO model comparison and single-example
notebooks retain their behavior.

These experiments use MEDAL-style immediate parent-to-child information gain and inline decomposition, followed by standard LLaDA confidence completion from the selected partial state. Finite caps are budget-limited ablations; author-code parity is unverified. Historical report-only artifacts remain supported. See the [walkthrough](mcts_walkthrough.md) for the assumptions and one-forward-per-child cost accounting.

## Conditions and frozen data

| Setting | Default |
|---|---|
| Conditions | `baseline`, `confidence`, `mcts_128`, `mcts_256`, `mcts_512` |
| Model revision | `08b83a6feb34df1a6011b80c3c00c7563e963b07` |
| Generation | BF16 on CUDA, batch 1, temperature 1 for MCTS and 0 for standalone baselines; length = steps = block length = 256 |
| Search | Lc=20, K1=3, K2=5, gamma=10, exploration=0.01, epsilon=1e-8 |
| Stopping | `budget`: all tree-analysis, simulation and candidate-evaluation forwards count against each cap; target-depth nodes stay closed |
| Smoke profile | First 3 frozen development examples, all five conditions |
| Milestone profile | First 50 frozen GSM8K evaluation questions / all 30 Sudoku evaluation puzzles |
| Sampling | Seed 1; sample once, then reuse the manifest |

GSM8K manifests contain 20 development questions from `main/train` and an ordered
pool of 100 evaluation questions from `main/test`. The first manifest creation
resolves the dataset's current commit SHA and loads **both splits at that SHA**.
Set `data.revision` to an immutable SHA to choose it explicitly. A saved manifest
can be reused offline without looking up a new revision.

Sudoku manifests contain 10 development and 30 disjoint evaluation puzzles from
the existing 500-puzzle release, pinned at BGPO commit
`2d3e819a65b42ff5bd9cce958fb4412d57c1609d`. Selection shuffles original row IDs once,
skips duplicate inputs, and retains order, partition, source revision and content
hashes. Manifests include task content so later sweeps do not need another dataset
download. Their references enter scoring and inspection only.

Generation seed = `(decoding.seed + manifest position) % 2**63`. Positions are
global and zero-based: GSM8K evaluation starts at 20; Sudoku evaluation starts at
10. All conditions for an example use the same prompt, seed and fully masked
initial output. Each MCTS condition constructs a fresh tree.

## Launch from a configured CUDA environment

```bash
python scripts/run_gsm8k_sweep.py --profile smoke
python scripts/run_sudoku_sweep.py --profile smoke

# Reuse the same manifests for the milestone evaluation partition.
python scripts/run_gsm8k_sweep.py --profile milestone
python scripts/run_sudoku_sweep.py --profile milestone

# Optional override: use a prefix of the frozen pool and explicit caps.
python scripts/run_gsm8k_sweep.py --profile milestone --num-samples 10 \
  --budgets 128 256 512 --manifest results/manifests/gsm8k-seed1.json

# Works offline without loading weights or downloading data.
python scripts/run_gsm8k_sweep.py --report-only /path/to/exact/run/directory
python scripts/run_sudoku_sweep.py --report-only /path/to/exact/run/directory
```

Default YAML files are [`configs/gsm8k/llada_sweep.yaml`](../configs/gsm8k/llada_sweep.yaml)
and [`configs/sudoku/llada_sweep.yaml`](../configs/sudoku/llada_sweep.yaml).
Both commands accept `--config`, `--profile`, `--num-samples`, `--budgets`,
`--manifest`, `--output-dir` and `--report-only`. `--output-dir` is a **parent**;
each run creates a new timestamped child. The default parents are
`results/LLaDA/{GSM8K,Sudoku4x4}/sweeps/`. Reruns create new folders, not resumed
trees. Reuse manifests when comparing settings.

Edit YAML for search/generation settings. `C` remains `num_candidates: 3` for
compatibility but does not stop budget-mode sweeps. Existing decoders still
default to `stop_mode: candidates`; their legacy experiment names stay the same.
Budget-mode names append `-stop-budget`.

## Colab subprocess smoke verification

Use the existing LLaDA notebook's checkout, Drive mount and original-model cache
steps. The checkout must contain the new scripts. Install the original LLaDA
dependency profile even if the previous session used the BGPO comparison:

```python
from pathlib import Path
import subprocess, sys

REPO_DIR = Path('/content/CSCI544-DLM')  # Your existing checkout.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r',
                str(REPO_DIR / 'requirements/llada-colab.txt'), '-r',
                str(REPO_DIR / 'requirements/dev.txt')], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '-e',
                str(REPO_DIR)], check=True)

subprocess.run([sys.executable, '-u', '-c', '''
import torch, pytest
assert torch.cuda.is_available(), 'Select a CUDA GPU runtime'
assert torch.cuda.is_bf16_supported(), 'This sweep defaults to BF16'
print(torch.cuda.get_device_name(0))
raise SystemExit(pytest.main(['-q', 'tests/decoding/test_mcts_cuda.py']))
'''], cwd=REPO_DIR, check=True)
```

Keep the model out of the notebook kernel. Reuse the original LLaDA snapshot
staged by `cache_model.py`; its `.model_source.json` must retain the pinned model
identity. Do not point these sweeps at BGPO weights.

```python
import yaml

MODEL_PATH = Path('/content/models/LLaDA-8B-Instruct')
SWEEP_ROOT = Path('/content/drive/MyDrive/CSCI544-DLM/milestone1')
SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
assert (MODEL_PATH / '.model_source.json').is_file()

for task in ('gsm8k', 'sudoku'):
    config = yaml.safe_load((REPO_DIR / f'configs/{task}/llada_sweep.yaml').read_text())
    config['model'].update(model_path=str(MODEL_PATH), local_files_only=True)
    config['experiment_root'] = str(SWEEP_ROOT)
    config['data']['cache_dir'] = '/content/drive/MyDrive/CSCI544-DLM/cache/datasets'
    (SWEEP_ROOT / f'{task}.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
```

Launch three actual HF examples per task, preserving live progress and the exact
returned directory. Each task process loads one model and releases it on exit:

```python
def launch_sweep(task, profile='smoke'):
    command = [sys.executable, '-u', str(REPO_DIR / f'scripts/run_{task}_sweep.py'),
               '--config', str(SWEEP_ROOT / f'{task}.yaml'), '--profile', profile,
               '--manifest', str(SWEEP_ROOT / 'manifests' / f'{task}-seed1.json')]
    run_dir = None
    with subprocess.Popen(command, cwd=REPO_DIR, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end='')
            if line.startswith('Sweep artifacts: '):
                run_dir = Path(line.removeprefix('Sweep artifacts: ').strip())
        if process.wait() != 0:
            raise RuntimeError(f'{task} failed; inspect partial artifacts at {run_dir}')
    assert run_dir is not None
    return run_dir

smoke_runs = {task: launch_sweep(task) for task in ('gsm8k', 'sudoku')}
```

Inspect saved artifacts without constructing another model:

```python
import json
import pandas as pd
from IPython.display import display, Image, Markdown

for task, run_dir in smoke_runs.items():
    summary = json.loads((run_dir / 'summary.json').read_text())
    rows = [json.loads(line) for line in (run_dir / 'predictions.jsonl').read_text().splitlines()]
    assert summary['status'] == 'completed' and summary['common_n'] == 3
    assert len(rows) == 15 and summary['setup']['model_load_count'] == 1
    assert all(row['model_device'].startswith('cuda') for row in rows)
    assert all(row['peak_gpu_memory_bytes'] > 0 for row in rows)
    assert all(row['total_nfes'] == row['search_nfes'] + row['completion_nfes'] for row in rows)
    assert all(row['search_nfes'] <= row['requested_search_budget']
               for row in rows if row['method'] == 'mcts_entropy')
    display(pd.read_csv(run_dir / 'summary.csv'))
    display(Image(filename=str(run_dir / 'plots/accuracy_vs_nfes.png')))
    display(Image(filename=str(run_dir / 'plots/search_diagnostics.png')))
    first_search = next(row for row in rows if row['method'] == 'mcts_entropy')
    print(task, first_search['example_id'], first_search['raw_output'])
    events = [json.loads(line) for line in (run_dir / first_search['trace_file']).read_text().splitlines()]
    CYCLE_TO_VIEW = 1
    for stage in ('selection', 'expansion', 'evaluation', 'backpropagation'):
        display(pd.json_normalize([event for event in events
                                   if event.get('cycle') == CYCLE_TO_VIEW and event['stage'] == stage]))
    if task == 'sudoku':
        display(Image(filename=str(sorted((run_dir / 'examples').glob('*.png'))[0])))
```

After both smoke runs pass, call `launch_sweep(task, profile='milestone')`. Avoid
tuning on evaluation examples. A completed smoke run establishes functioning
inference/accounting/artifacts, not model accuracy or equivalence.

## Cost and interpretation

One NFE is one call to the DLM at batch size one. Search includes one root-analysis forward and one forward per evaluated immediate child. Candidate ranking reuses cached mean tree rewards and adds no forwards. Fresh final completion costs `256 - selected_depth`. Failed child evaluations never update visits. If no target-depth candidate exists, fallback selects a deepest evaluated state, or root.

Caps 128/256/512 allow at most 127/255/511 immediate-child evaluations after root analysis. These are explicit budget-limited runs; they do not guarantee the candidate quota. Read saved phase counts and selected depths for each example.

Use smoke latency to estimate milestone runtime on your allocated GPU. Model
loading and warm-up time are reported separately; plotting/scoring are outside
decoding latency. Trace capture during decoding is included with the same policy
across conditions. `sweep.warmup: false` disables the setup forward.

The primary comparison uses exact accuracy and **actual total NFEs**, with Wilson
95% intervals and sample counts. Parsing failures count as wrong. A failed
generation produces no prediction row and marks the run incomplete. Main plots
use the common set completed under all conditions; `available_n` and
`observed_conditions` expose individually completed results excluded by pairing.
Realized branching is evaluated children divided by distinct evaluated parents;
the summary shows its mean across examples and leaves non-search values empty.

Saved artifacts:

| File/folder | Contents |
|---|---|
| `config.json`, `manifest.json`, `run_state.json` | Resolved settings, frozen content/provenance, setup costs and execution state |
| `predictions.jsonl`, `results.csv` | Raw outputs, prompt hashes/seeds, task scores, NFEs, synchronized latency, memory and search diagnostics |
| `summary.json`, `summary.csv`, `report.md` | Paired quality/compute summaries, Wilson intervals, status, definitions and plot links |
| `plots/` | Accuracy–NFE, accuracy–latency, NFE breakdown, correctness heatmap and search diagnostics; PNG and PDF |
| `plot_data.json` | Exact paired numbers used to draw the comparison figures |
| `gsm8k_examples.csv` | GSM8K question, response, parsed/reference answers and per-example metrics |
| `examples/` | Sudoku puzzle/reference/all-condition figures, including invalid/incomplete outputs |
| `traces/` | Every search cycle and completion event, flushed incrementally; no vocabulary logits |
| `inspection/` | First three selected examples' MCTS selected-path entropy and action reward/context gain plots, plus plot data |

`--report-only` regenerates derived files from completed records and compact
traces. It does not contact the Hub. An incomplete final JSONL line left by a hard
interrupt is ignored with a visible recovery notice; earlier malformed records
are rejected. Raw records are never overwritten during reporting.

## Code and verification

- `experiments/manifests.py`: frozen split/row sampling and validation.
- `experiments/sweep.py`: sequential execution, task adapters, warm-up, metrics and incremental saving.
- `experiments/sweep_report.py`: offline aggregation, Wilson intervals, CSV/Markdown and incomplete-run handling.
- `experiments/sweep_plots.py`: saved-metric figures and trace inspection.
- `decoding/mcts_decoding.py`: all search logic, including both stopping modes and immediate-child entropy rewards.

Run `python -m pytest -q` in the pinned LLaDA dependency environment. New tests
exercise both stop modes, sibling caps, deduplication, immediate-child phase accounting,
frozen manifests, scoring failures, paired samples, offline report regeneration,
one model load, incremental saves and figure data. Controlled CPU tests and
controlled CUDA tests are separate from **actual HF GPU smoke validation**, which
requires the two three-example runs above. The local implementation was developed
without a CUDA runtime; no real-model benchmark scores are supplied as test data.
