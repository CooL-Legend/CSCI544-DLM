"""Paired AB-MEDAL reward experiments with durable attempts and strict resume.

All inference goes through ab_development (ABMEDALSearch), never the ordinary
MEDAL comparison decoder. Report summaries use the same core fields as MEDAL.
"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from uuid import uuid4

from dlm_project.paths import PROJECT_ROOT
from dlm_project.experiments.sweep_report import write_csv
from . import ab_development as dev
from .local_experiments import read_predictions, report, save


def contexts(*, stage='dev', condition='A', num_samples=5,
             reward_modes=('ig', 'adalook_adaptive'), adalook_options=None):
    modes = tuple(reward_modes)
    if not modes or len(set(modes)) != len(modes):
        raise ValueError('Choose at least one distinct reward mode.')
    options = dict(adalook_options or {})
    if any(not key.startswith('adalook_') for key in options):
        raise ValueError('adalook_options must contain only AdaLook settings.')
    return [dev.context(stage=stage, condition=condition, indices=tuple(range(num_samples)),
                        method='ab_medal', reward_options=dict(options, reward_mode=mode))
            for mode in modes]


def implementation_hash():
    h = hashlib.sha256()
    source = PROJECT_ROOT / 'src/dlm_project'
    # Include shared completion, model, audit, serialization and report helpers.
    for package in ('decoding', 'tasks/gsm8k', 'models', 'experiments'):
        for path in sorted((source / package).rglob('*.py')):
            h.update(str(path.relative_to(source)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()


def configuration(variants):
    if not variants:
        raise ValueError('At least one AB variant is required.')
    keys = ('implementation_version', 'method', 'common', 'common_protocol_hash',
            'prompt_template_hash', 'manifest_hash', 'examples', 'stage', 'condition',
            'search', 'reward_options', 'variant_hash')
    configs = []
    for ctx in variants:
        dev.validate_context(ctx)
        if ctx['method'] != 'ab_medal':
            raise ValueError('AB comparison requires method=ab_medal.')
        configs.append({key: ctx[key] for key in keys})
    first = configs[0]
    for ctx in configs[1:]:
        for key in ('common_protocol_hash', 'manifest_hash', 'examples', 'stage', 'condition'):
            if ctx[key] != first[key]:
                raise ValueError('AB variants must share frozen inputs and budget conditions.')
        fixed = lambda c: {k: v for k, v in c['search'].items()
                           if k != 'reward_mode' and not k.startswith('adalook_')}
        if fixed(ctx) != fixed(first):
            raise ValueError('Only reward/AdaLook settings may differ across variants.')
    names = ['ab_medal_' + c['search']['reward_mode'] for c in configs]
    if len(set(names)) != len(names):
        raise ValueError('Reward modes must be distinct in this comparison.')
    return dict(version='ab_comparison_v1', implementation_hash=implementation_hash(),
                conditions=names, variants=configs)


def write_report(directory, rows, config, status, error=None):
    extra = ('temporary_candidate_nfes', 'adalook_total_extra_nfes', 'adalook_trigger_rate',
             'evaluated_children', 'max_depth', 'selected_depth', 'branching_factor_mean', 'parsing_rate')
    summary = report(directory, rows, config['conditions'], status, error,
                     title='AB-MEDAL / AB-MEDAL + AdaLook comparison', extra_metrics=extra)
    latest = {(r['condition'], r['example_id']): r for r in rows}
    complete = {key: row for key, row in latest.items() if row['status'] == 'completed'}
    control = 'ab_medal_ig' if 'ab_medal_ig' in config['conditions'] else config['conditions'][0]
    paired = []
    for condition in config['conditions']:
        if condition == control:
            continue
        for example in config['variants'][0]['examples']:
            example_id = example['example_id']
            a, b = complete.get((control, example_id)), complete.get((condition, example_id))
            if a is None or b is None:
                continue
            if any(a[key] != b[key] for key in ('seed', 'prompt_ids_hash', 'common_protocol_hash')):
                raise ValueError('Completed pair has inconsistent inputs.')
            paired.append(dict(example_id=example_id, control=control, variant=condition,
                               control_correct=a['correct'], variant_correct=b['correct'],
                               wrong_to_right=not a['correct'] and b['correct'],
                               right_to_wrong=a['correct'] and not b['correct'],
                               delta_total_nfes=b['total_nfes'] - a['total_nfes'],
                               delta_decoding_seconds=b['decoding_seconds'] - a['decoding_seconds']))
    summary['requested_per_condition'] = len(config['variants'][0]['examples'])
    summary['paired_comparisons'] = []
    for condition in config['conditions']:
        if condition == control:
            continue
        pairs = [p for p in paired if p['variant'] == condition]
        summary['paired_comparisons'].append(dict(control=control, variant=condition,
            completed_pairs=len(pairs), wrong_to_right=sum(p['wrong_to_right'] for p in pairs),
            right_to_wrong=sum(p['right_to_wrong'] for p in pairs)))
    details = []
    for condition in config['conditions']:
        selected = [r for (c, _), r in complete.items() if c == condition]
        decisions, reasons = Counter(), Counter()
        for row in selected:
            decisions.update(row.get('decision_counts') or {})
            reasons.update(row.get('adalook_stop_reasons') or {})
        total = sum(decisions.values())
        details.append(dict(condition=condition, decision_counts=dict(decisions),
            gen_percentage=100 * decisions['GEN'] / total if total else None,
            cont_percentage=100 * decisions['CONT'] / total if total else None,
            adalook_stop_reasons=dict(reasons)))
    summary['adaptive_diagnostics'] = details
    save(directory / 'summary.json', summary)
    save(directory / 'report/paired_results.json', paired)
    write_csv(directory / 'report/paired_results.csv', paired)
    write_csv(directory / 'report/per_question.csv', list(latest.values()))
    lines = ['\n## Paired changes (completed pairs only)', '']
    if error:
        lines += [f'Run error: {error}', '']
    for pair in summary['paired_comparisons']:
        lines.append(f"- {pair['variant']} vs {control}: {pair['completed_pairs']} pairs; "
                     f"wrong → right: {pair['wrong_to_right']}; right → wrong: {pair['right_to_wrong']}.")
    lines += ['', '## Search and AdaLook diagnostics', '',
              '| Condition | Mean tree children | Mean selected depth | Mean temporary comparison NFE | Mean rollout NFE | Mean rollout trigger rate |',
              '|---|---:|---:|---:|---:|---:|']
    for item in summary['conditions']:
        lines.append('| ' + ' | '.join(str(item[k]) for k in (
            'id', 'mean_evaluated_children', 'mean_selected_depth', 'mean_temporary_candidate_nfes',
            'mean_adalook_extra_nfes', 'mean_adalook_trigger_rate')) + ' |')
    for item in details:
        lines.append(f"\n- {item['condition']}: GEN={item['gen_percentage']}%, CONT={item['cont_percentage']}%; "
                     f"AdaLook stops: {item['adalook_stop_reasons']}.")
    lines += ['', 'Trigger rate is the fraction of finalized groups reaching depth > 1. '
              'Temporary comparison NFEs are separate from rollout NFEs; both consume the search budget.',
              'IG and AdaLook use different reward scales with identical AB priors. '
              'Use adalook_one_step to isolate confidence scoring from adaptive lookahead.']
    with (directory / 'report/report.md').open('a') as f:
        f.write('\n'.join(lines) + '\n')
    return summary


def run_comparison(variants, *, output_root=None, resume=None, wrapper=None):
    config = configuration(variants)
    if resume is not None:
        directory = Path(resume)
        if json.loads((directory / 'config.json').read_text()) != config:
            raise ValueError('Resume configuration or implementation differs from this run.')
        if json.loads((directory / 'selection.json').read_text()) != [e['example_id'] for e in variants[0]['examples']]:
            raise ValueError('Resume question selection differs from this run.')
    else:
        if output_root is None:
            raise ValueError('Supply output_root for a new comparison.')
        label = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8]
        directory = Path(output_root) / variants[0]['stage'] / variants[0]['condition'] / label
        directory.mkdir(parents=True, exist_ok=False)
        save(directory / 'config.json', config)
        save(directory / 'selection.json', [e['example_id'] for e in variants[0]['examples']])
        save(directory / 'manifest.json', dev.get_manifest(common=variants[0]['common']))
    (directory / 'traces').mkdir(exist_ok=True)
    rows = read_predictions(directory / 'predictions.jsonl')
    completed = {(r['condition'], r['example_id']) for r in rows if r['status'] == 'completed'}
    expected = {(name, e['example_id']) for name in config['conditions'] for e in variants[0]['examples']}
    if not completed <= expected:
        raise ValueError('Saved predictions contain unexpected conditions or questions.')
    if completed == expected:
        write_report(directory, rows, config, 'completed')
        return directory
    print('AB comparison artifacts:', directory, flush=True)
    write_report(directory, rows, config, 'running')
    status, error = 'running', None
    try:
        if wrapper is None:
            raise ValueError('Supply the notebook-loaded model to run unfinished questions.')
        for condition, ctx in zip(config['conditions'], variants):
            if all((condition, e['example_id']) in completed for e in ctx['examples']):
                continue
            prepared = dev.prepare(wrapper, ctx)
            for item in prepared:
                example_id = item['example']['example_id']
                if (condition, example_id) in completed:
                    continue
                attempt_id = uuid4().hex[:12]
                safe_id = example_id.replace(':', '_').replace('/', '_')
                question_dir = directory / 'questions' / condition / safe_id / attempt_id
                question_dir.mkdir(parents=True)
                initialization = completion = None
                row = None
                try:
                    initialization = dev.initialize(wrapper, item, ctx, question_dir)
                    completion = dev.complete(wrapper, item, initialization, ctx, question_dir)
                    row = dev.finish_single(wrapper, item, ctx, initialization, completion, question_dir)
                except (Exception, KeyboardInterrupt) as failure:
                    partial = question_dir / 'metrics.json'
                    if partial.is_file():
                        row = json.loads(partial.read_text())
                    else:
                        row = dict(example_id=example_id, method='ab_medal',
                                   reward_mode=ctx['search']['reward_mode'],
                                   search_nfes=initialization.search.model_calls if initialization else 0,
                                   completion_nfes=completion.model_calls if completion else 0)
                    row.update(status='interrupted' if isinstance(failure, KeyboardInterrupt) else 'error',
                               error=f'{type(failure).__name__}: {failure}')
                    raise
                finally:
                    if row is not None:
                        row.update(condition=condition, budget_condition=ctx['condition'], attempt_id=attempt_id,
                                   artifact_directory=str(question_dir), seed=item['seed'],
                                   prompt_ids_hash=item['prompt_ids_hash'],
                                   common_protocol_hash=ctx['common_protocol_hash'])
                        # Match MEDAL's top-level decoded-answer fields while
                        # retaining the detailed development record as well.
                        row.update(row.get('decoded') or {})
                        trace = question_dir / 'trace.jsonl'
                        if trace.is_file():
                            target = directory / 'traces' / f'{condition}_{safe_id}_{attempt_id}.jsonl'
                            shutil.copyfile(trace, target)
                            row['trace_file'] = str(target.relative_to(directory))
                        with (directory / 'predictions.jsonl').open('a') as f:
                            f.write(json.dumps(row, allow_nan=False) + '\n')
                            f.flush()
                            os.fsync(f.fileno())
                        rows.append(row)
                        write_report(directory, rows, config, 'running')
                completed.add((condition, example_id))
                print(f"{condition}: {example_id} correct={row['correct']} NFE={row['total_nfes']}", flush=True)
        status = 'completed'
    except (Exception, KeyboardInterrupt) as failure:
        status = 'interrupted' if isinstance(failure, KeyboardInterrupt) else 'error'
        error = f'{type(failure).__name__}: {failure}'
        raise
    finally:
        write_report(directory, rows, config, status, error)
    return directory
