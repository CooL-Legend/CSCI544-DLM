"""Validate one pinned GSM8K example and save the complete Adaptive MEDAL audit.

Example (CUDA / Colab, target-reaching and uncapped):
    python scripts/run_ab_medal_single.py --example-index 0 --seed 1 \
        --Lc 20 --C 3 --K1 3 --K2 5 --output-dir results/ab_medal/single

Finite synthetic trees test the mechanism on CPU. They do not establish that a
real DLM reaches the target; metrics.json records that gate separately.
"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from time import perf_counter

import torch

from dlm_project.decoding.ab_medal import ABMEDALConfig, ABMEDALSearch, tree_record, tree_statistics
from dlm_project.decoding import mcts_decoding as medal
from dlm_project.decoding.medal_demo import (
    InitializationResult, _sync, load_demo_model, prepare_demo_prompt, run_completion, show_metrics,
)
from dlm_project.experiments.artifacts import save_json
from dlm_project.experiments.paths import create_run_directory
from dlm_project.tasks.gsm8k.benchmark_dataset import get_manifest
from dlm_project.tasks.gsm8k.evaluation import score_answer


class ForwardCounter:
    """Count invocations and sequence NFEs where the underlying model is called."""

    def __init__(self, model):
        self.model = model
        self.calls = self.sequence_nfes = 0

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, ids, **kwargs):
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("These experiments require batch size 1.")
        self.calls += 1
        self.sequence_nfes += ids.shape[0]
        return self.model(ids, **kwargs)


def plot_tree(tree, path, *, max_nodes=120, max_depth=20):
    """Render cached nodes; virtual GEN/CONT decisions add no inference.

    The JSON always contains the full tree. Plot limits only affect the image.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if max_nodes < 1 or max_depth < 0:
        raise ValueError("Tree plot limits require max_nodes >= 1 and max_depth >= 0.")
    nodes = {n["node_id"]: n for n in tree["nodes"]}
    visible = []

    def visit(node_id):
        node = nodes[node_id]
        if len(visible) >= max_nodes or node["depth"] > max_depth:
            return
        visible.append(node_id)
        for child_id in node["children"]:
            visit(child_id)

    visit(0)
    shown = set(visible)
    positions, next_leaf = {}, [0]

    def layout(node_id):
        children = [c for c in nodes[node_id]["children"] if c in shown]
        xs = [layout(c) for c in children]
        if not xs:
            x = next_leaf[0]
            next_leaf[0] += 1
        else:
            x = sum(xs) / len(xs)
        positions[node_id] = x
        return x

    layout(0)
    depth = max(nodes[i]["depth"] for i in visible)
    width = min(36, max(12, next_leaf[0] * 3.2))
    figure = Figure(figsize=(width, max(4, (depth + 1) * 1.45)), constrained_layout=True)
    axis = figure.subplots()
    axis.set_axis_off()

    def text(x, y, label, color, size=7):
        axis.text(x, y, label, ha="center", va="center", fontsize=size,
                  bbox={"boxstyle": "round,pad=0.3", "fc": color, "ec": "#667085", "lw": .7}, zorder=3)

    def edge(x1, y1, x2, y2, label=None):
        axis.plot([x1, x2], [y1, y2], color="#98a2b3", linewidth=.7, zorder=1)
        if label:
            axis.text((x1 + x2) / 2, (y1 + y2) / 2, label, fontsize=6, ha="center", va="center",
                      bbox={"facecolor": "white", "edgecolor": "none", "pad": 1}, zorder=2)

    for node_id in visible:
        node = nodes[node_id]
        x, y = positions[node_id], -node["depth"] * 3
        models = node.get("posteriors", {})
        location = f"\nm={models['node']['m']:+.3g}" if models else ""
        label = f"N{node_id}  d={node['depth']}  visits={node['visits']}{location}"
        if node["candidate"]:
            label += "\nTERMINAL CANDIDATE"
        text(x, y, label, "#d1fadf" if node["candidate"] else "#eff4ff")
        if not node["candidate"]:
            for name, offset in (("GEN", -.23), ("CONT", .23)):
                count = sum(d["choice"] == name for d in node.get("decisions", []))
                model = models.get(name)
                details = f"\nm={model['m']:+.3g}, k={model['kappa']:g}" if model else ""
                text(x + offset, y - .9, f"{name} ({count}){details}", "#fff4cc" if name == "GEN" else "#f2e8ff", 6)
                edge(x, y - .2, x + offset, y - .7)
        for child_id in node["children"]:
            if child_id not in shown:
                continue
            child = nodes[child_id]
            action = child["action"]
            reward = child["immediate_reward"]
            label = f"p{action['output_position']}=t{action['token_id']}\nr={reward:+.3g}" if reward is not None else "unevaluated"
            edge(x + .23, y - 1.1, positions[child_id], -child["depth"] * 3 + .3, label)
        omitted_roots = [c for c in node["children"] if c not in shown]
        if omitted_roots:
            pending, omitted = list(omitted_roots), 0
            while pending:
                item = nodes[pending.pop()]
                omitted += 1
                pending.extend(item["children"])
            text(x, y - 1.65, f"… {omitted} nodes omitted\n(plot limit)", "#fff1f3", 6)
    axis.set_xlim(-.7, max(positions.values()) + .7)
    axis.set_ylim(-depth * 3 - 2.3, 1)
    axis.set_title(f"Adaptive MEDAL: {len(visible)}/{len(nodes)} states shown; "
                   "GEN generates one action, CONT chooses an existing child\n"
                   "m = posterior location; green = distinct terminal candidate; full tree in tree.json", fontsize=10)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(figure).print_png(str(path))
    return {"shown_nodes": len(visible), "omitted_nodes": len(nodes) - len(visible),
            "max_nodes": max_nodes, "max_depth": max_depth}


def run_attempt(wrapper, prompt_ids, config, directory, *, method="ab_medal", seed=1,
                gen_length=256, temperature=1.0, reference=None, prompt_seconds=0.0,
                setup_seconds=0.0, metadata=None, print_history=True,
                plot_max_nodes=120, plot_max_depth=20):
    """Save diagnostics even for capped, exhausted or interrupted attempts.

    Used by the paired runner too, with the same already-loaded model and prompt.
    Only a search with the complete requested candidate quota is decoded.
    """
    if method not in ("medal", "ab_medal"):
        raise ValueError("Unknown search method.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory / "config.json", {
        "method": method, "seed": seed, "completion_seed": medal.derive_seed(seed, "final_completion"),
        "gen_length": gen_length, "temperature": temperature, "search": asdict(config),
        "model": getattr(wrapper, "identity", {}), "example": metadata or {}, "batch_size": 1,
    })
    counted = ForwardCounter(wrapper.model)
    search_class = ABMEDALSearch if method == "ab_medal" else medal.MCTSSearch
    error, interrupted, failed, completion, text, score = None, False, False, None, None, None
    decoding_seconds = 0.0
    with (directory / "trace.jsonl").open("w", encoding="utf-8") as trace:
        def emit(event):
            line = json.dumps(event, allow_nan=False)
            trace.write(line + "\n")
            trace.flush()
            if print_history:
                print(line, flush=True)

        search = search_class(counted, prompt_ids, wrapper.mask_id, gen_length, config,
                              tokenizer=wrapper.tokenizer, on_event=emit, seed=seed)
        _sync(counted)
        started = perf_counter()
        try:
            search.run()
        except medal.SearchIncompleteError as exc:
            error = str(exc)
        except (Exception, KeyboardInterrupt) as exc:
            error = f"{type(exc).__name__}: {exc}"
            interrupted = isinstance(exc, KeyboardInterrupt)
            failed = True
        finally:
            _sync(counted)
        initialization_seconds = perf_counter() - started
        initialization_calls, initialization_nfes = counted.calls, counted.sequence_nfes
        reached = len(search.candidates) >= config.num_candidates
        initial = None
        if search.root.analysis is not None:
            initial = InitializationResult(search, search.result(), initialization_seconds)
        if reached and error is None:
            decoding_started = perf_counter()
            try:
                completion = run_completion(initial, seed=seed, temperature=temperature, tokenizer=wrapper.tokenizer)
                for event in completion.trace:
                    emit({**event, "phase": "decoding"})
                text = wrapper.tokenizer.decode(completion.token_ids[0, search.prompt_len:].tolist(), skip_special_tokens=True)
                score = score_answer(text, reference) if reference is not None else score_answer(text, None)
                save_json(directory / "completion.json", {"text": text, "score": score,
                                                           "token_ids": completion.token_ids.tolist()})
            except (Exception, KeyboardInterrupt) as exc:
                error = f"{type(exc).__name__}: {exc}"
                interrupted = isinstance(exc, KeyboardInterrupt)
                failed = True
            finally:
                _sync(counted)
                decoding_seconds = completion.wall_seconds if completion is not None else perf_counter() - decoding_started
        decoding_calls = counted.calls - initialization_calls
        decoding_nfes = counted.sequence_nfes - initialization_nfes
    if initial is not None:
        report = show_metrics(initial, completion, prompt_seconds=prompt_seconds,
                              setup_seconds=setup_seconds, score=score, display=False)
        stats = search.result().stats
    else:
        report = {"stages": [], "initialization_wall_seconds": initialization_seconds,
                  "decoding_wall_seconds": 0.0, "total_wall_seconds": prompt_seconds + initialization_seconds,
                  "setup_wall_seconds": setup_seconds, "accuracy": None}
        stats = {"stop_reason": "error_before_root_analysis", "candidate_count": 0}
    status = "interrupted" if interrupted else "error" if failed else "complete" if reached else "incomplete"
    report.update(method=method, status=status, error=error, target_reached=reached,
                  decoding_performed=completion is not None, response=text, score=score,
                  initialization_model_calls=initialization_calls, initialization_nfe=initialization_nfes,
                  decoding_model_calls=decoding_calls, decoding_nfe=decoding_nfes,
                  decoding_wall_seconds=decoding_seconds,
                  total_wall_seconds=prompt_seconds + initialization_seconds + decoding_seconds,
                  total_model_calls=counted.calls, total_nfe=counted.sequence_nfes, model_evaluations=counted.calls,
                  search=stats, structure=tree_statistics(search), seed=seed,
                  metadata=metadata or {}, artifact_directory=str(directory))
    report["full_target_dlm_gate"] = bool(reached and config.init_tokens == 20 and config.num_candidates == 3
                                          and method == "ab_medal" and config.search_policy == "gaussian"
                                          and getattr(wrapper, "identity", {}).get("family") == "llada"
                                          and all(n.depth == 20 for n in search.candidates.values()))
    save_json(directory / "metrics.json", report)
    tree = tree_record(search)
    save_json(directory / "tree.json", tree)
    save_json(directory / "history.json", search.history if hasattr(search, "history") else search.trace)
    report["tree_plot"] = plot_tree(tree, directory / ("ab_medal_tree.png" if method == "ab_medal" else "medal_tree.png"),
                                   max_nodes=plot_max_nodes, max_depth=plot_max_depth)
    save_json(directory / "metrics.json", report)
    print(f"{method}: {status}; candidates={len(search.candidates)}/{config.num_candidates}; "
          f"initialization={initialization_nfes} NFE; decoding={decoding_nfes} NFE; artifacts={directory}", flush=True)
    return report


def add_common_arguments(parser):
    parser.add_argument("--model-path", help="Local/Drive checkpoint or Hub ID; default is pinned LLaDA.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--manifest-path", type=Path, default=Path(".cache/medal/gsm8k_test.json"))
    parser.add_argument("--demonstrations-path", type=Path, default=Path(".cache/medal/gsm8k_decomposition_train.json"))
    parser.add_argument("--dataset-revision", help="Immutable GSM8K revision; existing manifests are reused offline.")
    parser.add_argument("--cache-dir", help="Hugging Face datasets cache directory.")
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--Lc", type=int, default=20)
    parser.add_argument("--C", type=int, default=3)
    parser.add_argument("--K1", type=int, default=3)
    parser.add_argument("--K2", type=int, default=5)
    parser.add_argument("--prior-m", type=float, default=0.0)
    parser.add_argument("--prior-kappa", type=float, default=1.0)
    parser.add_argument("--prior-nu", type=float, default=1.0)
    parser.add_argument("--prior-tau2", type=float, default=.1)
    parser.add_argument("--search-policy", choices=("gaussian", "medal_equivalent"), default="gaussian")
    parser.add_argument("--plot-max-nodes", type=int, default=120)
    parser.add_argument("--plot-max-depth", type=int, default=20)


def search_config(args, max_nfe=None, *, adaptive=True):
    settings = dict(init_tokens=args.Lc, num_candidates=args.C, top_k_tokens=args.K1,
                    top_k_actions=args.K2, max_model_calls=max_nfe)
    if adaptive:
        settings.update(prior_m=args.prior_m, prior_kappa=args.prior_kappa, prior_nu=args.prior_nu,
                        prior_tau2=args.prior_tau2, search_policy=args.search_policy)
    config = ABMEDALConfig(**settings) if adaptive else medal.MCTSConfig(**settings)
    config.validate_length(args.gen_length)
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("temperature must be positive and finite.")
    if args.plot_max_nodes < 1 or args.plot_max_depth < 0:
        raise ValueError("Invalid tree visualization limits.")
    return config


def load_inputs(args):
    # Check the requested device before fetching a dataset or an 8B checkpoint.
    if torch.device(args.device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run the full-target DLM gate in a Colab GPU runtime.")
    started = perf_counter()
    manifest = get_manifest(args.manifest_path, revision=args.dataset_revision, cache_dir=args.cache_dir)
    manifest_seconds = perf_counter() - started
    started = perf_counter()
    wrapper = load_demo_model(model_path=args.model_path, device=args.device, dtype=args.dtype,
                              local_files_only=args.local_files_only)
    return wrapper, manifest, {"manifest_wall_seconds": manifest_seconds,
                               "model_loading_wall_seconds": perf_counter() - started,
                               "model_load_count": 1, "warmup_model_calls": 0, "warmup_wall_seconds": 0.0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--example-index", type=int, default=0, help="Zero-based pinned GSM8K test index.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-nfe", type=int, help="Optional initialization-only cap including the root forward.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/ab_medal/single"))
    args = parser.parse_args(argv)
    config = search_config(args, args.max_nfe)
    medal.derive_seed(args.seed, "validation")
    if not 0 <= args.example_index < 1319:
        parser.error("example-index must be between 0 and 1318.")
    wrapper, manifest, setup = load_inputs(args)
    from dlm_project.tasks.gsm8k.benchmark import example_seed
    example = manifest["examples"][args.example_index]
    directory, _ = create_run_directory({"output_dir": str(args.output_dir)})
    save_json(directory / "setup.json", {**setup, "dataset": manifest["dataset"],
                                          "cli": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})
    prompt, ids, prompt_meta = prepare_demo_prompt(wrapper, example["question"], args.demonstrations_path,
                                                  revision=manifest["dataset"]["revision"], cache_dir=args.cache_dir)
    save_json(directory / "prompt.json", {"text": prompt, "metadata": prompt_meta})
    report = run_attempt(wrapper, ids, config, directory,
                         seed=example_seed(args.seed, example["row_id"]), gen_length=args.gen_length,
                         temperature=args.temperature, reference=example["reference_solution"],
                         prompt_seconds=prompt_meta["prompt_wall_seconds"],
                         setup_seconds=prompt_meta["setup_wall_seconds"], metadata=example,
                         plot_max_nodes=args.plot_max_nodes, plot_max_depth=args.plot_max_depth)
    print(f"Full-target DLM gate: {'observed' if report['full_target_dlm_gate'] else 'NOT satisfied'}", flush=True)
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
