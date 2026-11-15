"""Opt-in, attempt-local measurements for the E1/E2 execution paths.

Sequence NFEs are counted at the model boundary, independently of checkpoint
accounting. Shared batch latency is never summed over question result rows.
"""

from collections import defaultdict
from contextvars import ContextVar
from functools import wraps
import inspect
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
from time import perf_counter
from uuid import uuid4

import torch

from dlm_project.decoding.mcts_decoding import model_device
from dlm_project.experiments.artifacts import save_json

GIB = 1024 ** 3
_CURRENT = ContextVar("experiment_resource_profile", default=None)


def synchronize(model):
    device = model_device(model)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def watch_checkpoints(tree, model, started):
    """Timestamp actual snapshots, including E2's hypothetical boundary copies."""
    for name in ("_snapshot_checkpoint", "_snapshot_boundary_checkpoint"):
        original = getattr(tree, name, None)
        if original is None:
            continue

        def timed(*args, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            synchronize(model)
            result["search_elapsed_seconds"] = perf_counter() - started
            return result

        setattr(tree, name, timed)


def _rss_gib():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024 / GIB
    except OSError:
        pass
    return _lifetime_rss_gib()


def _lifetime_rss_gib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value * (1 if sys.platform == "darwin" else 1024) / GIB


def _revision():
    try:
        from dlm_project.paths import PROJECT_ROOT
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def source_fingerprint():
    """Include uncommitted inference/config changes in provenance and resume checks."""
    from dlm_project.paths import PROJECT_ROOT
    checksum = hashlib.sha256()
    for directory in ("src", "scripts", "configs"):
        for path in sorted((PROJECT_ROOT / directory).rglob("*")):
            if path.is_file() and path.suffix in (".py", ".yaml", ".json"):
                checksum.update(str(path.relative_to(PROJECT_ROOT)).encode())
                checksum.update(path.read_bytes())
    return checksum.hexdigest()


class _ForwardProbe:
    def __init__(self, model, record):
        self._model, self._record = model, record

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __call__(self, *args, **kwargs):
        ids = args[0] if args else kwargs["input_ids"]
        count = int(ids.shape[0])
        self._record["model_invocations"] += 1
        self._record["sequence_evaluations"] += count
        self._record["max_active_sequences"] = max(self._record["max_active_sequences"], count)
        return self._model(*args, **kwargs)


class ResourceProfiler:
    def __init__(self, wrapper, ctx, output_dir, *, methods, budgets, batch_size, resume):
        self.model = wrapper.model
        self.device = model_device(self.model)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.attempt_id = uuid4().hex
        self.started = perf_counter()
        self.records, self.stack = [], []
        self.batch = {}
        self.computed, self.resumed = set(), set()
        self.methods, self.budgets = tuple(methods), tuple(budgets)
        self.sample_ids = [ex["example_id"] for ex in ctx["examples"]]
        self.prompt_lengths = []
        self.gen_length = ctx["common"]["generation"]["gen_length"]
        self.metadata = {
            "model": wrapper.identity, "protocol_version": ctx["protocol_version"],
            "repository_commit": _revision(), "torch_version": torch.__version__,
            "source_fingerprint": source_fingerprint(),
            "cuda_version": torch.version.cuda, "device": str(self.device),
            "configured_question_batch_size": batch_size, "resume_requested": resume,
            "stage": ctx["stage"], "reward_mode": ctx.get("reward_mode"),
            "reward_lambda": ctx.get("reward_lambda"), "reward_beta": ctx.get("reward_beta"),
            "sample_ids": self.sample_ids,
            "methods": list(methods), "budgets": list(budgets),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "model_load_seconds": ctx.get("setup", {}).get("model_load_seconds",
                                                         getattr(wrapper, "model_load_seconds", None)),
        }
        self.gpu_total = None
        self.baseline = self.memory()
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            self.gpu_total = props.total_memory / GIB
            self.metadata.update(gpu_name=props.name, gpu_total_gib=self.gpu_total)
        self.sampled_device_peak = None
        self.sampled_rss_peak = 0.0
        self._stop = threading.Event()
        self.sample()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def memory(self):
        if self.device.type != "cuda":
            return {"allocated_gib": None, "reserved_gib": None}
        return {"allocated_gib": torch.cuda.memory_allocated(self.device) / GIB,
                "reserved_gib": torch.cuda.memory_reserved(self.device) / GIB}

    def sample(self):
        self.sampled_rss_peak = max(self.sampled_rss_peak, _rss_gib())
        if self.device.type == "cuda":
            try:
                free, total = torch.cuda.mem_get_info(self.device)
                used = (total - free) / GIB
                self.sampled_device_peak = max(self.sampled_device_peak or 0, used)
            except RuntimeError:
                pass  # GPU counters remain authoritative if device sampling is unavailable.

    def _sample_loop(self):
        while not self._stop.wait(1.0):
            self.sample()

    def write_record(self, record):
        record.update(attempt_id=self.attempt_id)
        self.records.append(record)
        with (self.output_dir / "resource_batches.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def finish(self, status, error=None):
        self._stop.set()
        self._thread.join(timeout=2)
        self.sample()
        elapsed = perf_counter() - self.started
        phases = {}
        for name in sorted({r["phase"] for r in self.records}):
            rows = [r for r in self.records if r["phase"] == name]
            calls = sum(r["model_invocations"] for r in rows)
            seqs = sum(r["sequence_evaluations"] for r in rows)
            seconds = sum(r["wall_seconds"] for r in rows)
            phases[name] = {
                "wall_seconds": seconds, "model_invocations": calls,
                "sequence_evaluations": seqs,
                "mean_active_sequences": seqs / calls if calls else None,
                "max_active_sequences": max(r["max_active_sequences"] for r in rows),
                "model_calls_per_second": calls / seconds if seconds else None,
                "sequence_evaluations_per_second": seqs / seconds if seconds else None,
            }
            for key in ("peak_allocated_gib", "peak_reserved_gib"):
                values = [r[key] for r in rows if r.get(key) is not None]
                phases[name][key] = max(values) if values else None
            for key in ("completion_requests", "unique_completion_requests", "cached_completion_requests"):
                phases[name][key] = sum(r.get(key, 0) for r in rows)
        fresh = [ex for ex in self.sample_ids if all((ex, m) in self.computed for m in self.methods)]
        reused = [ex for ex in self.sample_ids if all((ex, m) in self.resumed for m in self.methods)]
        completed = [ex for ex in self.sample_ids
                     if all((ex, m) in self.computed | self.resumed for m in self.methods)]
        measured_peaks = [p["peak_reserved_gib"] for p in phases.values() if p["peak_reserved_gib"] is not None]
        if self.sampled_device_peak is not None:
            measured_peaks.append(self.sampled_device_peak)
        used_peak = max(measured_peaks) if measured_peaks else None
        summary = {
            "schema_version": 1, "attempt_id": self.attempt_id, "status": status, "error": error,
            **self.metadata, "model_loaded_baseline": self.baseline, "phases": phases,
            "evaluation_wall_seconds": elapsed,
            "total_elapsed_seconds": elapsed + (self.metadata["model_load_seconds"] or 0),
            "fully_fresh_questions": len(fresh), "fully_resumed_questions": len(reused),
            "completed_questions": len(completed), "mixed_questions": len(completed) - len(fresh) - len(reused),
            "resumed_question_methods": len(self.resumed),
            "fresh_questions_per_hour": len(fresh) * 3600 / elapsed
                if status == "completed" and not self.resumed and elapsed else None,
            "sampled_device_peak_gib": self.sampled_device_peak,
            "device_memory_scope": "whole_device_including_other_processes; sampled_every_1s",
            "sampled_process_peak_rss_gib": self.sampled_rss_peak,
            "process_lifetime_peak_rss_gib": _lifetime_rss_gib(),
            "gpu_headroom_fraction": (self.gpu_total - used_peak) / self.gpu_total
                if self.gpu_total and used_peak is not None else None,
            "prompt_token_lengths": self.prompt_lengths,
            "max_prompt_plus_generation_tokens": max(self.prompt_lengths, default=0) + self.gen_length,
            "slurm_max_rss": None,
        }
        save_json(self.output_dir / "resource_summary.json", summary)
        # Keep prior launch measurements when the same experiment resumes.
        attempt_dir = self.output_dir / "resource_attempts"
        attempt_dir.mkdir(exist_ok=True)
        save_json(attempt_dir / f"{self.attempt_id}.json", summary)


def resource_phase(name):
    """Measure a phase without changing the model's inputs or outputs."""
    def decorate(function):
        @wraps(function)
        def measured(wrapper, *args, **kwargs):
            profiler = _CURRENT.get()
            if profiler is None:
                return function(wrapper, *args, **kwargs)
            gpu_phase = name in ("search", "completion")
            model = wrapper.model if gpu_phase else profiler.model
            if gpu_phase:
                synchronize(model)
                if profiler.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(profiler.device)
            record = {"phase": name, **profiler.batch, "model_invocations": 0,
                      "sequence_evaluations": 0, "max_active_sequences": 0}
            frame = {"children_seconds": 0.0}
            profiler.stack.append(frame)
            started = perf_counter()
            if gpu_phase:
                wrapper.model = _ForwardProbe(model, record)
            try:
                result = function(wrapper, *args, **kwargs)
                if name == "completion":
                    values = list(result.values()) if isinstance(result, dict) else [result]
                    record["completion_requests"] = len(values)
                    record["cached_completion_requests"] = sum(bool(v[-1]) for v in values)
                    record["unique_completion_requests"] = len(values) - record["cached_completion_requests"]
                record["status"] = "completed"
                return result
            except BaseException as error:
                record.update(status="failed", error=type(error).__name__)
                raise
            finally:
                if gpu_phase:
                    wrapper.model = model
                if gpu_phase:
                    synchronize(model)
                elapsed = perf_counter() - started
                record["wall_seconds"] = max(0.0, elapsed - frame["children_seconds"])
                record["inclusive_wall_seconds"] = elapsed
                record["peak_allocated_gib"] = None
                record["peak_reserved_gib"] = None
                if gpu_phase and profiler.device.type == "cuda":
                    record["peak_allocated_gib"] = torch.cuda.max_memory_allocated(profiler.device) / GIB
                    record["peak_reserved_gib"] = torch.cuda.max_memory_reserved(profiler.device) / GIB
                profiler.stack.pop()
                if profiler.stack:
                    profiler.stack[-1]["children_seconds"] += elapsed
                profiler.sample()
                profiler.write_record(record)
        return measured
    return decorate


def profiled_stage(function):
    @wraps(function)
    def measured(*args, **kwargs):
        bound = inspect.signature(function).bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        if not values["profile_resources"]:
            return function(*args, **kwargs)
        profiler = ResourceProfiler(values["wrapper"], values["ctx"], values["output_dir"],
                                    **{k: values[k] for k in ("methods", "budgets", "batch_size", "resume")})
        token = _CURRENT.set(profiler)
        status, error = "failed", None
        try:
            result = function(*args, **kwargs)
            status = "completed"
            return result
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _CURRENT.reset(token)
            profiler.finish(status, error)
    return measured


def record_prepared(prepared):
    profiler = _CURRENT.get()
    if profiler:
        profiler.prompt_lengths = [int(p["ids"].shape[1]) for p in prepared]


def record_batch(pending, cached, method):
    profiler = _CURRENT.get()
    if profiler:
        profiler.batch = {"batch_id": uuid4().hex, "method": method,
                          "example_ids": [p["example"]["example_id"] for p in pending]}
        profiler.resumed.update((ex, method) for ex in cached)


def record_completed(rows):
    profiler = _CURRENT.get()
    if profiler:
        grouped = defaultdict(set)
        for row in rows:
            grouped[(row["example_id"], row["method"])].add(row["search_budget"])
        profiler.computed.update(key for key, budgets in grouped.items() if set(profiler.budgets) <= budgets)


def finish_reporting(output_dir, *, reporting_seconds, offline_seconds, total_seconds):
    path = Path(output_dir) / "resource_summary.json"
    if not path.exists():
        return
    summary = json.loads(path.read_text())
    summary.update(reporting_wall_seconds=reporting_seconds, offline_analysis_wall_seconds=offline_seconds,
                   total_elapsed_seconds=total_seconds)
    save_json(path, summary)
    save_json(Path(output_dir) / "resource_attempts" / f"{summary['attempt_id']}.json", summary)
