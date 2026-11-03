"""Offline, sequential comparisons; per-question durable outputs and strict resume.

Uses the current branch's frozen shared inputs, independently of formal E0/E1
condition restrictions. Partial subsets are diagnostics, not full-stage results.
"""

from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from dlm_project.decoding.mcts_decoding import MCTSConfig, complete_llada, derive_seed
from dlm_project.decoding import mcts_confidence
from dlm_project.experiments.manifests import digest
from dlm_project.paths import PROJECT_ROOT
from dlm_project.tasks.gsm8k.experiment_protocol import get_manifest, select_examples, build_prompt, example_seed
from dlm_project.tasks.gsm8k.e0_tokens import resolve_termination_tokens, decode_output
from dlm_project.tasks.gsm8k.evaluation import score_answer


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def implementation_hash():
    files = ['decoding/mcts_decoding.py', 'decoding/adalook_rollout.py',
             'decoding/mcts_confidence.py', 'decoding/common.py', 'decoding/baseline.py',
             'tasks/gsm8k/local_experiments.py', 'tasks/gsm8k/experiment_protocol.py',
             'tasks/gsm8k/e0_tokens.py', 'tasks/gsm8k/evaluation.py', 'models/llada.py']
    h = hashlib.sha256()
    for name in files:
        h.update(name.encode())
        h.update((PROJECT_ROOT / 'src/dlm_project' / name).read_bytes())
    return h.hexdigest()


def read_predictions(path):
    """Discard only an incomplete final append left by process/power failure."""
    if not path.exists():
        return []
    data = path.read_bytes()
    end = data.rfind(b'\n') + 1
    rows = [json.loads(line) for line in data[:end].splitlines()]
    if end != len(data):
        with path.open('r+b') as f:
            f.truncate(end)
    return rows


def report(directory, rows, conditions, status, error=None, *,
           title='LLaDA / MEDAL AdaLook comparison', extra_metrics=()):
    # A retried question contributes only its latest attempt.
    latest = {(r['condition'], r['example_id']): r for r in rows}
    summary = dict(status=status, error=error, conditions=[])
    for condition in conditions:
        selected = [r for r in latest.values() if r['condition'] == condition]
        completed = [r for r in selected if r['status'] == 'completed']
        item = dict(id=condition, attempted=len(selected), completed=len(completed),
                    accuracy=mean(r['correct'] for r in completed) if completed else None)
        for key in ('search_nfes', 'completion_nfes', 'total_nfes', 'decoding_seconds',
                    'adalook_extra_nfes', 'adalook_mean_depth', *extra_metrics):
            item['mean_' + key] = mean(r.get(key, 0) for r in completed) if completed else None
        summary['conditions'].append(item)
    save(directory / 'summary.json', summary)
    target = directory / 'report'
    target.mkdir(exist_ok=True)
    with (target / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary['conditions'][0]))
        writer.writeheader()
        writer.writerows(summary['conditions'])
    lines = ['# ' + title, '', f'Status: {status}', '',
             'Accuracy below uses completed questions only; incomplete runs are not final comparisons.', '',
             '| Condition | Completed | Accuracy | Mean total NFE | Mean seconds |', '|---|---:|---:|---:|---:|']
    for r in summary['conditions']:
        lines.append(f"| {r['id']} | {r['completed']} | {r['accuracy']} | {r['mean_total_nfes']} | {r['mean_decoding_seconds']} |")
    (target / 'report.md').write_text('\n'.join(lines) + '\n')
    if status == 'completed':
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        items = summary['conditions']
        for axis, key, label in zip(axes, ('accuracy', 'mean_total_nfes'), ('Accuracy', 'Total sequence NFE')):
            axis.bar([r['id'] for r in items], [r[key] or 0 for r in items])
            axis.set_ylabel(label)
            axis.tick_params(axis='x', labelrotation=20, labelsize=7)
        fig.tight_layout()
        fig.savefig(target / 'comparison.png', dpi=140)
        plt.close(fig)
    return summary


def run_experiments(config, *, resume=None, wrapper=None):
    config = json.loads(json.dumps(config))
    config['implementation_hash'] = implementation_hash()
    manifest = get_manifest(config['paths']['DATASET_MANIFEST'], common=config['common'])
    examples = select_examples(manifest, config['stage'])[:config['num_samples']]
    config['manifest_hash'] = digest(manifest)
    mode = config['search']['reward_mode']
    condition = f"medal_budget_b{config['search']['max_model_calls']}_{mode}_{digest(config['search'])[:10]}"
    conditions = (['confidence_t0'] if config['method'] in ('both', 'llada') else [])
    if config['method'] in ('both', 'mcts'):
        conditions.append(condition)
    config['conditions'] = conditions
    selection = [r['example_id'] for r in examples]
    if resume:
        directory = Path(resume).resolve()
        if json.loads((directory / 'config.json').read_text()) != config:
            raise ValueError('Resume configuration or implementation differs from this run.')
        if json.loads((directory / 'selection.json').read_text()) != selection:
            raise ValueError('Resume selection differs.')
    else:
        directory = (Path(config['paths']['RESULTS_DIR']) / 'LLaDA/GSM8K/llada_vs_mcts' /
                     config['method'] / config['stage'] / f"budget-lc{config['search']['init_tokens']}" /
                     datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
        directory.mkdir(parents=True, exist_ok=False)
        save(directory / 'config.json', config)
        save(directory / 'manifest.json', manifest)
        save(directory / 'selection.json', selection)
    (directory / 'traces').mkdir(exist_ok=True)
    prediction_path = directory / 'predictions.jsonl'
    rows = read_predictions(prediction_path)
    done = {(r['condition'], r['example_id']) for r in rows if r['status'] == 'completed'}
    attempts = Counter((r['condition'], r['example_id']) for r in rows)
    print(f'Results: {directory}', flush=True)
    status, error = 'completed', None
    try:
        if all((c, e['example_id']) in done for c in conditions for e in examples):
            report(directory, rows, conditions, 'completed')
            return directory
        if wrapper is None:
            from dlm_project.models.llada import LLaDA
            wrapper = LLaDA(model_path=config['common']['model']['repo_id'], device=config['device'],
                            dtype=config['common']['model']['dtype'], local_files_only=True)
        if any(wrapper.identity.get(k) != config['common']['model'][k] for k in ('family', 'repo_id', 'revision')):
            raise ValueError('Model identity differs from the pinned shared protocol.')
        termination = resolve_termination_tokens(wrapper.tokenizer, wrapper.model, wrapper.mask_id)
        generation = config['common']['generation']
        def sync():
            if torch.device(wrapper.model.device).type == 'cuda':
                torch.cuda.synchronize(wrapper.model.device)
        for current in conditions:
            for example in examples:
                key = current, example['example_id']
                if key in done:
                    continue
                attempts[key] += 1
                trace_path = directory / 'traces' / f"{current}_{example['example_id'].replace(':', '_')}_attempt{attempts[key]}.jsonl"
                while trace_path.exists():
                    attempts[key] += 1
                    trace_path = directory / 'traces' / f"{current}_{example['example_id'].replace(':', '_')}_attempt{attempts[key]}.jsonl"
                ids = wrapper.encode_prompt(build_prompt(example['question'], config['common']['prompt']))
                seed = example_seed(example['row_id'], config['seed'])
                row = dict(condition=current, example_id=example['example_id'], seed=seed,
                           status='running', trace_file=str(trace_path.relative_to(directory)),
                           prompt_ids_hash=digest(ids.cpu().tolist()), reward_mode=mode if current != 'confidence_t0' else None)
                started = perf_counter()
                search_calls = completion_calls = 0
                try:
                    with trace_path.open('w') as trace:
                        def emit(event):
                            nonlocal search_calls, completion_calls
                            if event.get('stage') == 'forward':
                                if event.get('phase') == 'completion':
                                    completion_calls += 1
                                else:
                                    search_calls += 1
                            trace.write(json.dumps(event, allow_nan=False) + '\n')
                            trace.flush()
                        sync()
                        started = perf_counter()
                        if current == 'confidence_t0':
                            state = torch.cat((ids.cpu(), torch.full((1, generation['gen_length']), wrapper.mask_id, dtype=torch.long)), dim=1)
                            result = complete_llada(wrapper.model, state, ids.shape[1], wrapper.mask_id,
                                MCTSConfig(**config['search']), seed=derive_seed(seed, 'final_completion'),
                                temperature=0., on_event=lambda e: emit({**e, 'phase': 'completion'}))
                            details = dict(search_model_calls=0, completion_model_calls=result.model_calls)
                        else:
                            result = mcts_confidence.decode(wrapper.model, ids, wrapper.mask_id,
                                **generation, temperature=0., seed=seed, mcts=config['search'],
                                completion_method='llada', search_excluded_token_ids=termination['token_ids'], on_event=emit)
                            details = result.details
                        sync()
                        decoded = decode_output(wrapper.tokenizer, result.token_ids[0, ids.shape[1]:].tolist(), termination['token_ids'])
                        scored = score_answer(decoded['scoring_text'], example['reference_solution'])
                        row.update(status='completed', correct=scored['exact_match'], **decoded, score=scored,
                                   details=details, adalook_extra_nfes=details.get('adalook_extra_nfes', 0),
                                   adalook_mean_depth=details.get('adalook_mean_depth', 0),
                                   search_nfes=details['search_model_calls'], completion_nfes=details['completion_model_calls'])
                except (Exception, KeyboardInterrupt) as exc:
                    row.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'error',
                               error=f'{type(exc).__name__}: {exc}', search_nfes=search_calls, completion_nfes=completion_calls)
                    raise
                finally:
                    row.update(total_nfes=row.get('search_nfes', 0) + row.get('completion_nfes', 0),
                               decoding_seconds=perf_counter() - started)
                    with prediction_path.open('a') as output:
                        output.write(json.dumps(row, allow_nan=False) + '\n')
                        output.flush()
                        os.fsync(output.fileno())
                    rows.append(row)
                    report(directory, rows, conditions, 'running')
                print(f"{current}: {example['example_id']} correct={row['correct']} NFE={row['total_nfes']}", flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        status = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'error'
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report(directory, rows, conditions, status, error)
    return directory
