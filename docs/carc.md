# E1/E2 on USC CARC

## Group 37: submit E1 directly with Slurm

On branch `adi-e2-carc`, `scripts/run_carc_e1_test.sbatch` runs the same E1
experiment as `notebooks/gsm8k/Experiment_1_Test.ipynb` without starting Jupyter.
It reuses the prepared environment, model and data. The job does not install
packages or download assets, and it does not require a benchmark report.

Submit from your `adi-e2-carc` repository root on CARC (adjust the checkout
directory if you use a different copy):

```bash
cd /home1/xchen664/CSCI544-DLM
mkdir -p results/slurm
REPO_DIR="$PWD" sbatch scripts/run_carc_e1_test.sbatch
```

The two optional arguments are **number of test questions** (default `2`) and
**batch size** (default `4`). To test a full batch of four questions:

```bash
sbatch scripts/run_carc_e1_test.sbatch 4 4
```

For a first smoke run on `debug`, override the GPU type and time as well as
the partition. Discovery documents A40 GPUs on `debug` and a one-hour limit;
L40S GPUs are on `gpu`. Use a separate output directory for this test:

```bash
cd /home1/xchen664/CSCI544-DLM
mkdir -p results/slurm
REPO_DIR="$PWD" ARTIFACT_ROOT="$PWD/results/e1_debug_smoke" sbatch \
  --partition=debug \
  --gpus-per-task=a40:1 \
  --time=01:00:00 \
  scripts/run_carc_e1_test.sbatch 2 4
```

This runs two questions across all four method/budget conditions. Use `4 4`
to test a full batch of four. See the [CARC job limits](https://www.carc.usc.edu/user-guides/hpc-systems/using-our-hpc-systems/running-jobs).

The script requests account `xiangren_1987`, partition `gpu`, one `l40s`,
four CPU cores, **64 GB host RAM** and **two hours**. Each L40S has **48 GB GPU
memory**, separate from host RAM. These are initial resource settings, not
measured memory or runtime guarantees. See the
[CARC GPU specifications](https://www.carc.usc.edu/user-guides/hpc-systems/discovery/resource-overview-discovery)
and [GPU job options](https://www.carc.usc.edu/user-guides/advanced-hpc-programming/gpu-programming).

Prepared paths are reused as follows:

| Resource | Default |
| --- | --- |
| Python environment | `/project2/xiangren_1987/group_37/envs/dlm` |
| Model directory | `/project2/xiangren_1987/group_37/model-cache` |
| Dataset cache | `/project2/xiangren_1987/group_37/datasets` |
| Experiment results | `REPO_DIR/results/e1_e3_test_slurm` |
| Runtime Hugging Face metadata/custom code | `REPO_DIR/.cache/carc/huggingface` |
| Slurm logs | `results/slurm/e1-<jobid>.out` and `.err` |

The existing Group 37 model directory has `config.json`, tokenizer files and
six weight shards directly in it, so it is loaded as a flat local model.
A Hugging Face cache with the pinned LLaDA snapshot is also supported.
`MODEL_DIR` overrides that location;
`MODEL_PATH` selects an explicit local snapshot. `HF_DATASETS_CACHE` can
override the dataset-cache location. Set `ARTIFACT_ROOT` for an exact output
directory, for example:

```bash
ARTIFACT_ROOT="$PWD/results/e1_other_run" sbatch scripts/run_carc_e1_test.sbatch 4 4
```

The job runs the environment's Python directly, so `conda activate` is
unnecessary. Prepare any missing dependencies before submitting, using
`requirements/carc.txt` and `python -m pip install --no-deps -e .` with that
environment's Python. Reuse its compatible CUDA-enabled PyTorch.

E1 reads questions and reference answers from the checked-in
`configs/gsm8k/experiment_samples.json` and `experiment_samples_500.json`.
It does **not** load a separate Hugging Face dataset or validate a dataset
cache. The configured dataset cache remains available to other experiments.
Model loading is offline and uses existing local files without copying weights.

The runner compares **MEDAL** and **AB-MEDAL** at budgets **256 and 512**, with
`Lc = 10`, minimum candidates `C = 3`, the original normalized IG reward,
standard LLaDA confidence completion, automatic compatible-run resumption,
resource profiling and offline E3 analysis. It saves per-question outputs,
checkpoints, traces, aggregate scores and figures under the results directory.
Both `/results/` and `.cache/` are ignored by Git.

For larger test sets, pass `200` or `500`. The wall times below are examples;
adjust them after measuring a smoke run or pilot on the allocated GPU:

```bash
sbatch --time=24:00:00 scripts/run_carc_e1_test.sbatch 200 4
sbatch --time=24:00:00 scripts/run_carc_e1_test.sbatch 500 4
```

Run these as successive stages when extending the same output directory, not
concurrently. A compatible completed prefix can be reused. Changing batch size
or experimental settings requires a separate output directory.

Slurm prints a job ID at submission. Monitor it with:

```bash
squeue -u "$USER"
tail -f results/slurm/e1-JOB_ID.out
```

Replace `JOB_ID` with the submitted job ID; errors are in the matching
`e1-JOB_ID.err`. Create `results/slurm` **before** submission because Slurm
opens log files before the script starts. Submit from the repository root, or
set `REPO_DIR` explicitly and provide suitable `sbatch --output` and
`--error` paths. Slurm executes a spooled copy of the script, so its own file
location cannot identify the repository.

## Group 37: two-step SWD on the same 500 test questions

`scripts/run_carc_e2_swd_test.sbatch` runs the **two-step SWD reward from
`notebooks/gsm8k/E2_Experiment_SWD_Eval.ipynb`** on the same frozen 500-question
GSM8K test pool and ordering used by the E1 run. The original notebook's
20 + 50 evaluation split remains unchanged. The CARC launcher selects
`--stage test --num-samples 500` for the larger comparison.

The notebook's reward sums grandchild confidence weighted by stability along
the parent → child → virtual-grandchild trajectory, with **beta = 1.0**.
This entry runs that E2 implementation, including its one-step stability
fallback and compact distribution cache. It does not select the separate
`IG − lambda * KL` reward modification from `sean-1114`.

The requested production defaults are:

| Setting | Value |
| --- | --- |
| Slurm account / partition | `xiangren_1987` / `gpu` |
| GPU / wall time | one `a40` / `24:00:00` |
| CPU cores / host RAM | `4` / `64G` |
| Questions / batch size | `500` / `8` |
| Methods / search checkpoints | MEDAL + AB-MEDAL / `256`, `512` |
| Search depth / minimum candidates | `10` / `3` |
| Reward | `swd`, beta `1.0`, lambda `1.0` |
| Simulation cap | `4 * B` per question |
| Results | `REPO_DIR/results/e2_swd_test_500` |
| Logs | `results/slurm/e2-swd-<jobid>.out` and `.err` |

Lambda is passed for consistency with the notebook; beta controls the SWD
stability penalty. The simulation cap can stop a search before its NFE budget
is exhausted; E2 reports the actual search NFE and utilization.

Submit all 500 questions from the CARC checkout:

```bash
cd /home1/xchen664/CSCI544-DLM
mkdir -p results/slurm
ARTIFACT_ROOT="$PWD/results/e2_swd_test_500" \
  sbatch scripts/run_carc_e2_swd_test.sbatch 500 8
```

Omitting both positional arguments also selects `500 8`. The arguments are
question count and batch size; `NUM_SAMPLES` and `BATCH_SIZE` can provide
defaults when the corresponding positional argument is omitted.

For a two-question smoke run in its own output directory:

```bash
cd /home1/xchen664/CSCI544-DLM
mkdir -p results/slurm
ARTIFACT_ROOT="$PWD/results/e2_swd_debug_smoke" sbatch \
  --partition=debug --gpus-per-task=a40:1 --time=01:00:00 \
  scripts/run_carc_e2_swd_test.sbatch 2 2
```

The launcher reuses the existing environment
`/project2/xiangren_1987/group_37/envs/dlm` and model
`/project2/xiangren_1987/group_37/model-cache`. It keeps the configured dataset
cache at `/project2/xiangren_1987/group_37/datasets`, while reading all test
questions and reference answers directly from the repository manifests.
It performs no package installation, model download, dataset download or
Google Drive operation. Model weights are read in place, and writable runtime
metadata goes to `REPO_DIR/.cache/carc` as in the E1 launcher. The same
`REPO_DIR`, `ENV_DIR`/`DLM_ENV`, `MODEL_DIR`, `MODEL_PATH`, `HF_DATASETS_CACHE`,
`RESULTS_DIR`, `ARTIFACT_ROOT` and `PERSIST_ROOT` overrides are supported.

The job runs GPU search and standard LLaDA confidence completion, saves
checkpoints and resource profiles, and then generates paired comparisons and
offline E3 diagnostics. Its outputs include:

| Output under `results/e2_swd_test_500/` | Contents |
| --- | --- |
| `summaries/e2_test_aggregate.csv` | Accuracy, parse rate, NFE, timing and search statistics by method/budget |
| `summaries/e2_test_paired_comparison.csv` | Same-question MEDAL vs. AB-MEDAL comparison at each budget |
| `summaries/e2_test_per_question.csv` | Per-question predictions, references and scores |
| `summaries/e3_test_aggregate.csv` | Offline search-tree metrics |
| `summaries/e3_figures/` | Six diagnostic figures, PNG and PDF |
| `test/<method>/<example_id>/` | Per-question traces, completion markers and checkpoints |
| `run_manifest.json` | Configuration, sample identities and run completion status |
| `resource_summary.json`, `resource_attempts/` | Current and historical attempt resource measurements |

The shared E3 renderer retains historical labels such as `ig_mean` and
"Negative IG"; for this run they describe the stored **SWD immediate reward**,
not raw information gain. Its depth/root-subtree counts describe attached
search-tree nodes rather than exact forward allocation, because E2 also
evaluates virtual grandchildren and reuses cached states. Use E2 search NFE,
utilization and resource profiles to compare compute usage.

A complete run has four aggregate rows with `sample_count=500`, two paired
rows with `paired_questions=500`, and `run_manifest.json` status `completed`.
There are 2,000 question/method/budget outcomes. Slurm should also show
`COMPLETED` with exit code `0:0` when reporting and E3 analysis have finished.

If the job times out, resubmit the **same command, output directory, batch size
and source/configuration**. Compatible completed question/method results are
reused; interrupted searches are recomputed. A completed E1 directory such as
`results/e1_test_500` is not an SWD resume directory. Keep it for comparison.
Changing SWD settings or batch size requires a separate output directory.

The default batch size 8 is the requested setting; the two-question smoke run
does not establish full-batch GPU memory or runtime. Inspect the resource
measurements for the new SWD run rather than treating E1 measurements as SWD
measurements. This direct launcher does not depend on the separate benchmark
gate described below.

## Group 37: two-step CMER on the same 500 test questions

`scripts/run_carc_e2_cmer_test.sbatch` adapts the
[CMER Test notebook on `adi-e2`](https://github.com/CooL-Legend/CSCI544-DLM/blob/08efc305b35311c3643eb77f1a8b6b0ac8e86c7c/notebooks/gsm8k/E2_Experiment_CMER_Test.ipynb).
It combines the notebook's primary 200 and extended 300 questions into one
500-question run, preserving their order and the same frozen test pool as E1
and SWD. Batch size is increased from the notebook's 2 to the requested 8.

The search uses **CMER: common-mask entropy reduction**, with the notebook's
two-step lookahead, one-step fallback, and standard LLaDA confidence completion.
Both MEDAL and AB-MEDAL run at checkpoints 256 and 512, with maximum depth 10,
minimum 3 candidates and simulation cap `4 * B`. Lambda `1.0` is passed for
notebook consistency but does not enter the CMER reward formula.

The launcher defaults to **500 questions, batch 8, one A40, 48 hours**, account
`xiangren_1987`, partition `gpu`, 4 CPUs and 64 GB host RAM. It reuses:

| Setting | Path |
| --- | --- |
| Environment | `/project2/xiangren_1987/group_37/envs/dlm` |
| Model | `/project2/xiangren_1987/group_37/model-cache` |
| Dataset cache | `/project2/xiangren_1987/group_37/datasets` |
| Results | `REPO_DIR/results/e2_cmer_test_500` |
| Slurm logs | `results/slurm/e2-cmer-<jobid>.out` and `.err` |

Questions and answers are read from repository manifests. No environment
installation or model/dataset download takes place. The same path overrides
and writable runtime metadata directory as the SWD launcher are supported.

Submit the full experiment from the repository root:

```bash
cd /home1/xchen664/CSCI544-DLM
mkdir -p results/slurm
sbatch scripts/run_carc_e2_cmer_test.sbatch 500 8
```

For a separate two-question smoke run:

```bash
ARTIFACT_ROOT="$PWD/results/e2_cmer_debug_smoke" sbatch \
  --partition=debug --gpus-per-task=a40:1 --time=01:00:00 \
  --job-name=e2-cmer-smoke scripts/run_carc_e2_cmer_test.sbatch 2 2
```

Results use the same E2/E3 layout as SWD, in the separate CMER directory:

- `summaries/e2_test_aggregate.csv`: accuracy, NFE and timing by method/budget.
- `summaries/e2_test_paired_comparison.csv`: paired MEDAL/AB-MEDAL comparisons.
- `summaries/e2_test_per_question.csv`: predictions and scores for each outcome.
- `test/<method>/<example_id>/checkpoints/<budget>/tree.json`: checkpoint trees
  with node states, parent/child links, actions, rewards, visits and Q values.
- Per-question `nodes.jsonl`, `trace.jsonl`, checkpoint completions, offline E3
  diagnostics and resource profiles are also retained.

A complete run produces four aggregate rows with `sample_count=500`, two paired
rows with `paired_questions=500`, and 2,000 checkpoint trees. Check that
`run_manifest.json` reports `completed` and Slurm reports `COMPLETED`, exit `0:0`.
The shared E3 labels such as "Negative IG" refer to CMER immediate rewards for
this run; virtual lookahead states are not all attached search-tree nodes.

Resume is enabled. After a timeout, resubmit with the **same output directory,
batch size and source/configuration**. Completed compatible question/method
results are reused; interrupted searches are recomputed. Use a separate
directory for smoke runs or changed settings. A two-question smoke run verifies
the pipeline, not full batch-8 GPU memory or runtime.

## Existing notebook and benchmark workflow

The sections below retain the original Jupyter and measured-pilot/production
workflow. They are separate from the direct notebook-equivalent Slurm entry
above; its smoke and test runs do not depend on the production benchmark gate.

Use GPU-backed JupyterLab for smoke tests and inspection, and the existing Python
runners under Slurm for the long experiment. Start with **four questions per
batch** based on the observed L40 SRIG run. The reported approximately 16 GB is
an observation, not a verified peak or a host-RAM requirement.

CARC supports [JupyterLab through OnDemand](https://www.carc.usc.edu/user-guides/data-science/data-science-ondemand).
Connect through USC Secure networking or the USC VPN. Jupyter still needs a GPU
allocation and the prepared environment's kernel. Run inference on compute nodes.

## Prepare once

Keep the complete repository, environment, pinned model and results in your
assigned `/project2/<project>/<user>/` storage. Scratch can be purged. The job
does not install packages, clone repositories or download model weights.

Example shell variables (replace paths with your allocation):

```bash
export REPO_DIR=/project2/PROJECT/USER/CSCI544-DLM
export ENV_DIR=/project2/PROJECT/USER/envs/dlm
export PERSIST_ROOT=/project2/PROJECT/USER/dlm
export MODEL_PATH="$PERSIST_ROOT/models/LLaDA-8B-Instruct"
export ARTIFACT_ROOT="$PERSIST_ROOT/e1-carc"
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_DIR"
```

Use Python 3.11 and a CUDA-enabled PyTorch build compatible with the compute-node
driver. Inspect CARC's available modules with `module spider python` and
`module spider cuda`; check `nvidia-smi` inside a GPU allocation. Use the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/)
for the CUDA wheel rather than copying a CPU or macOS environment.

After creating the environment and installing CUDA-enabled PyTorch:

```bash
"$ENV_DIR/bin/python" -m pip install -r requirements/carc.txt
"$ENV_DIR/bin/python" -m pip install --no-deps -e .
"$ENV_DIR/bin/python" -m pip check
"$ENV_DIR/bin/python" -m pip freeze > "$PERSIST_ROOT/environment.lock.txt"
"$ENV_DIR/bin/python" -m ipykernel install --user --name dlm-carc --display-name "DLM CARC"
"$ENV_DIR/bin/python" scripts/cache_model.py --model llada --drive-dir "$MODEL_PATH"
```

The historical `--drive-dir` argument accepts any local directory; Google Drive
is not involved. This command downloads the pinned revision, validates all
weight shards and writes `.model_source.json` for provenance. Inference reads the
checked-in GSM8K manifests and needs no dataset download.

## Jupyter smoke check

Select the `DLM CARC` kernel in a GPU-backed OnDemand session. Open
`notebooks/gsm8k/Experiment_1_Test.ipynb`, use `NUM_SAMPLES = 4`, `BATCH_SIZE = 4`
and `PROFILE_RESOURCES = True`. Set `REPO_DIR`, `MODEL_PATH`, `PERSIST_ROOT` and a
separate `ARTIFACT_ROOT` before starting the kernel. Four samples exercise the
whole configured batch; the notebook retains a two-sample quick-check default.

The SRIG evaluation notebook supports the same paths and profiling and retains
its separate canonical-20 and extended-50 directories. Its two calls have
separate resource summaries. CLI summaries additionally include post-run E3 and
reporting time; notebook stage summaries cover `run_e*_stage` unless explicitly
finalized by the notebook. Do not sum model-load costs across notebook stages
that share one loaded model.

## Full pipeline pilot and production

The Slurm template defaults to one **L40S**, four CPU cores, **64 GB host RAM**,
and 24 hours. L40S is CARC's documented GPU type; an L40 observation is not a
measurement of L40S throughput. Check `noderes -c -g`, `noderes -f -g` and
`myaccount` for your allocation. Override `--gpus-per-task` for another type and
run its pilot separately. [CARC GPU resource syntax](https://www.carc.usc.edu/user-guides/advanced-hpc-programming/gpu-programming)

Create `logs` **before** submission; Slurm opens logs before the script starts.
Run from the repository root:

```bash
mkdir -p logs
export RUN_MODE=smoke
sbatch --account=YOUR_ACCOUNT --time=02:00:00 slurm/e1_test.sbatch

export RUN_MODE=benchmark
sbatch --account=YOUR_ACCOUNT slurm/e1_test.sbatch
```

The benchmark uses isolated processes and empty result directories for the
same frozen 20 evaluation questions at both 256/512 budgets and both methods:

1. Measure full E1 execution at batch sizes 1 and 4, including completion and
   artifact writes. Compare trees, selected states and completion tokens.
2. For each size, tokenize the 500-question manifest and stress the longest
   actual prompt-plus-generation length at widths `batch` and `2 * batch`.
   This synthetic forward stress check covers worst-case completion width; its
   projected runtime is not used for scheduling.
3. Try batch 8 only when batch 4 passes strict parity, the length stress check,
   and at least 20% measured memory headroom. Choose 8 only with the same safety
   checks and at least 10% faster complete E1 evaluation. Otherwise retain 4.
4. If strict batch parity fails, the report retains the differences and chooses
   serial execution when its memory/stress checks pass. It never relabels
   differing trees as a parity success. A failing serial pilot blocks selection.

`pilot/benchmark_report.json` contains the selected size and estimated time:

```text
model_load + 1.25 * (500 / 20) * E1_evaluation_seconds + reporting_allowance
```

The reporting allowance scales measured reporting plus E3 time by the same
sample ratio and headroom. Queue waiting is excluded. Discovery documents a
[48-hour GPU job limit](https://www.carc.usc.edu/user-guides/hpc-systems/using-our-hpc-systems/running-jobs).
If the estimate exceeds the limit, use successive compatible resumed jobs.

After inspecting the successful report, request appropriate wall time:

```bash
export RUN_MODE=production
export BENCHMARK_REPORT="$ARTIFACT_ROOT/pilot/benchmark_report.json"
sbatch --account=YOUR_ACCOUNT --time=24:00:00 slurm/e1_test.sbatch
```

Production validates GPU model, PyTorch/CUDA, repository commit, source
fingerprint (including uncommitted code/configuration), and model revision.
It runs 500 questions and validates 2,000 distinct outcomes, four 500-question
aggregate rows, complete checkpoint artifacts and all six E3 PNG/PDF figures.
Changing source files after the pilot requires a fresh pilot directory.

The underlying CLIs remain usable directly:

```bash
python scripts/run_e1_test.py --num-samples 500 --batch-size 4 --profile-resources --model-path "$MODEL_PATH" --local-files-only --output-dir "$ARTIFACT_ROOT/manual-e1"
python scripts/run_e2.py --stage evaluation --eval-split canonical_20 --reward-mode srig --batch-size 4 --profile-resources --model-path "$MODEL_PATH" --local-files-only --output-dir "$ARTIFACT_ROOT/srig"
```

## Resource measurements

`--profile-resources` is optional on all three E1/E2 CLIs and
`profile_resources=True` is optional on `run_e1_stage` and `run_e2_stage`.
Algorithm outputs and return shapes are unchanged. Existing checkpoint timing
columns now contain measured snapshot latency rather than budget-ratio estimates.

| Output | Meaning |
| --- | --- |
| `resource_batches.jsonl` | Append-only phase records, identified by attempt ID and batch ID; one record per actual search, completion or artifact phase |
| `resource_summary.json` | Latest attempt: environment, model baseline, phase peaks/counts, lengths, elapsed time, fresh/resumed counts and throughput |
| `resource_attempts/<id>.json` | Preserved summaries from earlier launches, including failed attempts |

- GPU values are **GiB**. Allocated memory covers live PyTorch tensors; reserved
  memory covers the allocator. Peaks are reset at search/completion boundaries
  and read before another measured GPU phase can reset them. Cache cleanup does
  not erase the high-water counters.
- Device-wide memory is sampled every second and includes runtime overhead and
  other processes. It can miss short spikes; PyTorch peak counters complement it.
  Headroom uses the larger of sampled device usage and phase peak reservation.
- Host RSS is separate. Linux sampling reads process RSS; process-lifetime
  high-water RSS is also recorded. On platforms without `/proc`, RSS sampling
  falls back to the process-lifetime high-water value.
- Physical model invocations are counted once at dispatch, including failed
  attempts; sequence evaluations count active rows. Completion cache hits add no
  calls. Per-budget nominal algorithm NFEs remain in the ordinary result tables.
- Shared batch latency is not summed over questions. Artifact timers subtract
  nested completion time. Overall evaluation time includes tokenization,
  resumption checks, all methods, both budgets, bookkeeping and artifact I/O.
- Questions/hour counts a question only after every requested condition is
  complete. It is null for any attempt mixing resumed work or ending in failure.
  Fresh, resumed and mixed question counts are always reported separately.

After the Slurm job exits and accounting appears, attach step-level MaxRSS:

```bash
python scripts/record_slurm_resources.py --job-id JOB_ID --output-dir "$ARTIFACT_ROOT/production"
```

Use measured RSS plus headroom to revise the initial 64 GB host request. A 16 GB
GPU reading cannot supply that number. See
[PyTorch CUDA memory semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management).

## Resume and validation

Resume validates model, protocol, source fingerprint, batch size, decoding,
prompt tokens, sample content and seeds before overwriting the manifest. It
permits an identical frozen-prefix extension, including 200 to 500. Each
question/method needs its completion marker, valid trace and all requested tree,
selection, completion and metrics files. Incomplete sets are recomputed.

Legacy output directories lack these identity checks and must be preserved in
place; start a fresh directory for this implementation. An interrupted in-flight
search is recomputed; completed question/method artifacts are reused. A current
batch may be smaller when some questions were already finished.

Run the CPU checks before allocating the GPU:

```bash
PYTHONPATH=src:. python -m pytest tests/tasks/test_resource_profiling.py tests/workflows/test_carc_workflow.py tests/tasks/test_experiment_1.py tests/tasks/test_experiment_2.py tests/decoding/test_checkpoint_equivalence.py tests/tasks/test_e3_analysis.py
```

These test four unequal-length SRIG questions for both methods, profiling
equivalence, eight distinct completion states, cache hits, partial batches,
resume rejection/recovery, peak preservation, and measured checkpoint timing.
Only the target-GPU pilot establishes real-model parity, memory and throughput.
