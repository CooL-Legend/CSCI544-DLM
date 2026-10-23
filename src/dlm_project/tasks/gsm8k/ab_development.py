"""Frozen-input AB development walkthrough; formal E1/E3 runners are separate."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import torch

from dlm_project.decoding.ab_medal import ABMEDALConfig, ABMEDALSearch
from dlm_project.decoding.ab_medal_demo import run_completion
from dlm_project.decoding.ab_batch import run_group, sync
from dlm_project.decoding import search_audit
from dlm_project.experiments.artifacts import save_json
from dlm_project.experiments.manifests import digest
from dlm_project.experiments.paths import create_run_directory
from dlm_project.experiments.sweep_report import write_csv
from dlm_project.tasks.gsm8k.experiment_protocol import load_shared_protocol, get_manifest, select_examples, build_prompt, example_seed
from dlm_project.tasks.gsm8k.e0_tokens import resolve_termination_tokens, decode_output
from dlm_project.tasks.gsm8k.evaluation import score_answer

IMPLEMENTATION_VERSION = 'ab_medal_fixed_budget_v2'


def context(*, stage='dev', condition='A', indices=(0,)):
    """Validate frozen inputs before loading a model. Subsets are diagnostics."""
    common = load_shared_protocol()
    manifest = get_manifest(common=common)
    examples = select_examples(manifest, stage)
    if not indices or len(set(indices)) != len(indices) or list(indices) != sorted(indices):
        raise ValueError('Choose distinct sample indices in manifest order.')
    if any(type(i) is not int or not 0 <= i < len(examples) for i in indices):
        raise ValueError('Sample index is outside the frozen stage.')
    selected = next((c for c in common['search']['conditions'] if c['label'] == condition), None)
    if selected is None:
        raise ValueError('Condition must be A, B, C, or D.')
    config = ABMEDALConfig(**common['decoding']['mcts'], init_tokens=selected['depth'],
                           max_model_calls=selected['budget'])
    return dict(implementation_version=IMPLEMENTATION_VERSION, common=common,
                common_protocol_hash=digest(common), prompt_template_hash=digest(common['prompt']['template']),
                stage=stage, condition=condition, search=asdict(config),
                examples=[examples[i] for i in indices], run_type='development_diagnostic',
                manifest_hash=digest(manifest), setup={})


def validate_model(wrapper, ctx):
    expected = ctx['common']['model']
    identity = wrapper.identity
    if any(identity.get(key) != expected[key] for key in ('family', 'repo_id', 'revision')):
        raise ValueError('Model/tokenizer must match the repository-pinned LLaDA checkpoint.')


def validate_context(ctx):
    """Reject in-memory notebook overrides as well as altered frozen files."""
    common = load_shared_protocol()
    manifest = get_manifest(common=common)
    rows = select_examples(manifest, ctx['stage'])
    indices = []
    for example in ctx['examples']:
        if example not in rows:
            raise ValueError('Development samples must retain their frozen contents and identities.')
        indices.append(rows.index(example))
    expected = context(stage=ctx['stage'], condition=ctx['condition'], indices=tuple(indices))
    for key in ('implementation_version', 'common', 'common_protocol_hash', 'prompt_template_hash',
                'search', 'examples', 'manifest_hash', 'run_type'):
        if ctx[key] != expected[key]:
            raise ValueError(f'Frozen AB development setting changed: {key}.')


def validate_prepared(wrapper, prepared, ctx, *, complete_stage=True):
    validate_context(ctx)
    validate_model(wrapper, ctx)
    expected = ctx['examples'] if complete_stage else [p['example'] for p in prepared]
    if [p['example'] for p in prepared] != expected:
        raise ValueError('Prepared questions differ from the requested manifest order.')
    for item in prepared:
        if item['example'] not in ctx['examples']:
            raise ValueError('Prepared question is outside the frozen diagnostic subset.')
        prompt = build_prompt(item['example']['question'], ctx['common']['prompt'])
        ids = item['ids'].detach().cpu().tolist()
        if (len(ids) != 1 or ids[0] != item['prompt_ids'] or item['prompt_ids_hash'] != digest(ids)
                or item['text'] != prompt or item['prompt_hash'] != digest(prompt)
                or item['prompt_template_hash'] != ctx['prompt_template_hash']
                or item['seed'] != example_seed(item['example']['row_id'])):
            raise ValueError('Prepared prompt, token IDs, hashes or seed changed.')
    if ctx['termination'] != resolve_termination_tokens(wrapper.tokenizer, wrapper.model, wrapper.mask_id):
        raise ValueError('Search/scoring termination token policy changed.')


def load_model(ctx, *, model_path=None, device='cuda', local_files_only=False):
    validate_context(ctx)
    from dlm_project.models.llada import LLaDA
    started = perf_counter()
    wrapper = LLaDA(model_path=model_path or ctx['common']['model']['repo_id'], device=device,
                    dtype=ctx['common']['model']['dtype'], local_files_only=local_files_only)
    sync(wrapper.model)
    validate_model(wrapper, ctx)
    ctx['setup'].update(model_load_count=1, model_loading_seconds=perf_counter()-started)
    return wrapper


def prepare(wrapper, ctx):
    validate_context(ctx)
    validate_model(wrapper, ctx)
    started = perf_counter()
    termination = resolve_termination_tokens(wrapper.tokenizer, wrapper.model, wrapper.mask_id)
    pad = getattr(wrapper.tokenizer, 'pad_token_id', None)
    if pad is None:
        pad = termination['eos_token_id']
    if pad == wrapper.mask_id:
        raise ValueError('Tokenizer padding token cannot be the mask token.')
    prepared = []
    for example in ctx['examples']:
        prompt = build_prompt(example['question'], ctx['common']['prompt'])
        ids = wrapper.encode_prompt(prompt)
        prepared.append(dict(example=example, text=prompt, ids=ids,
            prompt_ids=ids[0].cpu().tolist(), prompt_hash=digest(prompt),
            prompt_ids_hash=digest(ids.cpu().tolist()), seed=example_seed(example['row_id']),
            prompt_template_hash=ctx['prompt_template_hash']))
    sync(wrapper.model)
    ctx['setup']['tokenization_seconds'] = perf_counter()-started
    ctx.update(termination=termination, pad_token_id=pad, model=wrapper.identity)
    return prepared


@torch.inference_mode()
def warmup(wrapper, prepared, ctx, *, batch_size=1):
    validate_prepared(wrapper, prepared, ctx)
    from dlm_project.decoding.ab_batch import forward_states
    ids = [p['ids'].detach().cpu() for p in prepared[:batch_size]]
    states = [torch.cat((p, torch.full((1, 256), wrapper.mask_id, dtype=torch.long)), dim=1) for p in ids]
    sync(wrapper.model)
    started = perf_counter()
    output, _ = forward_states(wrapper.model, states, ctx['pad_token_id'])
    del output
    sync(wrapper.model)
    return dict(model_invocations=1, sequence_nfes=len(states), seconds=perf_counter()-started)


def initialize(wrapper, prepared, ctx, directory=None):
    validate_prepared(wrapper, [prepared], ctx, complete_stage=False)
    from dlm_project.decoding.medal_demo import InitializationResult
    search = ABMEDALSearch(wrapper.model, prepared['ids'], wrapper.mask_id, 256,
        config=ABMEDALConfig(**ctx['search']), seed=prepared['seed'], tokenizer=wrapper.tokenizer,
        search_excluded_token_ids=ctx['termination']['token_ids'])
    sync(wrapper.model)
    started = perf_counter()
    try:
        result = search.run()
        sync(wrapper.model)
        return InitializationResult(search, result, perf_counter()-started)
    except (Exception, KeyboardInterrupt) as error:
        sync(wrapper.model)
        if directory is not None:
            export_question(wrapper, prepared, ctx, search, None, directory,
                search_seconds=perf_counter()-started, completion_seconds=0.,
                status='interrupted' if isinstance(error, KeyboardInterrupt) else 'error',
                error=f'{type(error).__name__}: {error}')
        raise


def complete(wrapper, prepared, initialization, ctx, directory=None):
    validate_prepared(wrapper, [prepared], ctx, complete_stage=False)
    class Counter:
        calls = 0
        def __init__(self):
            self.forwards = []
        def __getattr__(self, name):
            return getattr(wrapper.model, name)
        def __call__(self, *args, **kwargs):
            self.calls += 1
            event = dict(stage='forward', phase='completion', model_calls=self.calls, status='attempted')
            self.forwards.append(event)
            result = wrapper.model(*args, **kwargs)
            event['status'] = 'returned'
            return result
    counter = Counter()
    search = initialization.search
    original = search.model
    search.model = counter
    sync(wrapper.model)
    started = perf_counter()
    try:
        return run_completion(initialization, method='llada', temperature=0., seed=prepared['seed'],
                              tokenizer=wrapper.tokenizer, termination_token_ids=ctx['termination']['token_ids'])
    except (Exception, KeyboardInterrupt) as error:
        sync(wrapper.model)
        if directory is not None:
            export_question(wrapper, prepared, ctx, search, None, directory,
                search_seconds=initialization.wall_seconds, completion_seconds=perf_counter()-started,
                completion_attempts=counter.calls, completion_forward_attempts=counter.forwards,
                status='interrupted' if isinstance(error, KeyboardInterrupt) else 'error',
                error=f'{type(error).__name__}: {error}')
        raise
    finally:
        search.model = original


def _compact(value):
    omitted = {'entropies', 'entropies_before', 'entropies_after', 'state', 'child_state',
               'selected_state', 'candidate_evaluations'}
    if isinstance(value, dict):
        return {k: _compact(v) for k, v in value.items() if k not in omitted}
    if isinstance(value, list):
        return [_compact(v) for v in value]
    return value


def _jsonl(path, rows):
    with Path(path).open('w') as output:
        for row in rows:
            output.write(json.dumps(row, allow_nan=False) + '\n')


def export_question(wrapper, prepared, ctx, search, completion, directory, *,
                    search_seconds, completion_seconds, status='completed', error=None,
                    batch_size=1, group_id=0, completion_attempts=None, completion_forward_attempts=None):
    """Export after timed decoding; reconstruct every node from compact evidence."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    events = [_compact(e) for e in search.trace]
    audit = search_audit.from_events(events, prompt_ids=prepared['prompt_ids'], mask_id=wrapper.mask_id,
        gen_length=256, requested_depth=ctx['search']['init_tokens'],
        requested_budget=ctx['search']['max_model_calls'], search_nfes=search.model_calls,
        adaptive=ctx['search']['search_policy'] == 'gaussian')
    metrics = search_audit.metrics(audit)
    result = search.result() if search.root.analysis is not None else None
    completion_nfes = completion.model_calls if completion is not None else (completion_attempts or 0)
    decoded, score = None, None
    if completion is not None:
        decoded = decode_output(wrapper.tokenizer, completion.token_ids[0, search.prompt_len:].tolist(),
                                ctx['termination']['token_ids'])
        score = score_answer(decoded['scoring_text'], prepared['example']['reference_solution'])
        if completion_nfes != 256 - result.selected.depth:
            raise RuntimeError('Completion NFE accounting failed.')
    if search.model_calls > ctx['search']['max_model_calls']:
        raise RuntimeError('Search exceeded its forward budget.')
    record = dict(**metrics, example_id=prepared['example']['example_id'], condition=ctx['condition'],
        common_protocol_hash=ctx['common_protocol_hash'], prompt_ids_hash=prepared['prompt_ids_hash'],
        prompt_hash=prepared['prompt_hash'], prompt_template_hash=prepared['prompt_template_hash'],
        seed=prepared['seed'], status=status, error=error, batch_size=batch_size, group_id=group_id,
        completion_method='llada', temperature=0., completion_nfes=completion_nfes,
        total_nfes=search.model_calls + completion_nfes,
        search_seconds=search_seconds, completion_seconds=completion_seconds,
        decoding_seconds=search_seconds + completion_seconds,
        timing_scope='per-question latency; overlaps other questions' if batch_size > 1 else 'serial decoding wall time',
        participating_model_invocations=search.model_calls + completion_nfes,
        correct=score['exact_match'] if score else None,
        accuracy=float(score['exact_match']) if score else None,
        parsing_success=score['parse_success'] if score else None,
        parsing_rate=float(score['parse_success']) if score else None,
        score=score, decoded=decoded, artifact_directory=str(directory),
        selected_node_id=result.selected.node_id if result and search.done else None,
        selected_state_ids=result.selected.state[0].tolist() if result and search.done else None)
    prompt_record = {k: v for k, v in prepared.items() if k != 'ids'}
    prompt_record.update(mask_id=wrapper.mask_id, gen_length=256)
    save_json(directory / 'prompt.json', prompt_record)
    save_json(directory / 'config.json', ctx)
    save_json(directory / 'search_audit.json', audit)
    save_json(directory / 'metrics.json', record)
    write_csv(directory / 'metrics.csv', [record])
    _jsonl(directory / 'nodes.jsonl', audit['nodes'])
    _jsonl(directory / 'decisions.jsonl', audit['decisions'] or [])
    _jsonl(directory / 'backups.jsonl', [e for e in events if e['stage'] == 'backpropagation'])
    all_events = events + (completion.trace if completion else completion_forward_attempts or [])
    _jsonl(directory / 'trace.jsonl', all_events)
    _jsonl(directory / 'forwards.jsonl', [e for e in all_events if e['stage'] == 'forward'])
    selection = dict(audit['selection'] or {}, selected_state_ids=record['selected_state_ids'],
                     candidate_statistics=result.stats['candidate_evaluations'] if result and search.done else [])
    save_json(directory / 'selection.json', selection)
    if completion is not None:
        save_json(directory / 'completion.json', dict(**decoded, score=score, token_ids=completion.token_ids.tolist()))
    tree = []
    for row in audit['nodes']:
        node = search.nodes[row['node_id']]
        tree.append({**row, 'children': [c.node_id for c in node.children if c.analysis is not None],
                     'candidate': row['node_id'] in (audit['selection'] or {}).get('candidate_ids', []),
                     'posteriors': {key: asdict(p) for key, p in search.posteriors[row['node_id']].items()},
                     'decisions': search.decision_history[row['node_id']]})
    save_json(directory / 'tree.json', dict(nodes=tree, statistics=metrics))
    return record


def finish_single(wrapper, prepared, ctx, initialization, completion, directory):
    return export_question(wrapper, prepared, ctx, initialization.search, completion, directory,
        search_seconds=initialization.wall_seconds, completion_seconds=completion.wall_seconds)


def show_metrics(record):
    """All requested n=1 metrics; presentation reads artifacts only."""
    groups = {
        'Performance (n=1)': ['correct', 'accuracy', 'parsing_success', 'parsing_rate'],
        'Depth': ['max_depth', 'selected_depth', 'average_search_depth'],
        'Width': ['evaluated_children', 'expanded_parents', 'branching_factor_min', 'branching_factor_mean', 'branching_factor_max'],
        'Candidates': ['candidate_count', 'target_depth_candidate_count', 'shortlist_size'],
        'Exploration': ['unique_evaluated_states', 'duplicate_state_count', 'duplicate_state_percentage',
                        'evaluated_root_subtrees', 'root_subtree_coverage', 'deepest_candidate_root_subtrees', 'shortlist_root_subtrees', 'candidate_diversity'],
        'Rewards': ['negative_ig_count', 'negative_ig_percentage', 'ig_min', 'ig_mean', 'ig_max', 'selected_q'],
        'Allocation': ['nfes_per_depth', 'root_analysis_nfes', 'nfes_per_root_subtree'],
        'Adaptive decisions': ['decision_counts', 'decision_percentages', 'forced_decisions', 'forced_decision_counts', 'forced_decision_percentages', 'forced_decision_percentage'],
        'Efficiency': ['requested_depth_success', 'fallback', 'fallback_rate', 'stop_reason', 'budget_utilization'],
        'Cost': ['search_nfes', 'completion_nfes', 'total_nfes', 'participating_model_invocations',
                 'search_seconds', 'completion_seconds', 'decoding_seconds', 'timing_scope'],
    }
    for group, keys in groups.items():
        print(group)
        for key in keys:
            print(f'  {key}: {record.get(key)}')
    return record


def plot_metrics(directory):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from scripts.run_ab_medal_single import plot_tree
    directory = Path(directory)
    metrics = search_audit.load(directory)
    figure = Figure(figsize=(11, 4), constrained_layout=True)
    axes = figure.subplots(1, 2)
    for axis, key, label in zip(axes, ('nfes_per_depth', 'nfes_per_root_subtree'), ('Evaluated depth', 'Root-child node ID')):
        values = metrics[key]
        axis.bar(list(values), list(values.values()))
        axis.set(xlabel=label, ylabel='Search sequence NFEs')
    figure.suptitle('Search compute allocation (root analysis separate from subtrees)')
    FigureCanvasAgg(figure).print_png(str(directory / 'compute_allocation.png'))
    tree = json.loads((directory / 'tree.json').read_text())
    if tree['nodes']:
        plot_tree(tree, directory / 'ab_medal_tree.png', max_nodes=120, max_depth=10)
    return directory / 'compute_allocation.png'


def _summary(records, groups, requested, status, error=None):
    complete = [r for r in records if r['status'] == 'completed']
    return dict(status=status, error=error, requested=requested, processed=len(records), completed=len(complete),
                accuracy=sum(r['correct'] for r in complete) / len(complete) if complete else None,
                parsing_rate=sum(r['parsing_success'] for r in complete) / len(complete) if complete else None,
                fallback_rate=sum(r['fallback'] for r in complete) / len(complete) if complete else None,
                search_nfes=sum(g['search_nfes'] for g in groups),
                completion_nfes=sum(g['completion_nfes'] for g in groups),
                total_nfes=sum(g['total_nfes'] for g in groups),
                search_seconds=sum(g['search_seconds'] for g in groups),
                completion_seconds=sum(g['completion_seconds'] for g in groups),
                decoding_seconds=sum(g['decoding_seconds'] for g in groups),
                total_model_invocations=sum(g['total_model_invocations'] for g in groups), groups=groups)


def run_samples(wrapper, prepared, ctx, directory, *, batch_size=1):
    """Persist every finished group, including failed/unfinished forward attempts."""
    validate_prepared(wrapper, prepared, ctx)
    if batch_size not in (1, 2, 4):
        raise ValueError('Development batch size must be 1, 2, or 4.')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory / 'config.json', ctx)
    records, groups, status, error = [], [], 'running', None
    save_json(directory / 'summary.json', _summary(records, groups, len(prepared), status))
    try:
        for group_id, offset in enumerate(range(0, len(prepared), batch_size)):
            items = prepared[offset:offset + batch_size]
            run = run_group(wrapper.model, [p['ids'] for p in items], wrapper.mask_id,
                ABMEDALConfig(**ctx['search']), seeds=[p['seed'] for p in items], pad_token_id=ctx['pad_token_id'],
                search_excluded_token_ids=ctx['termination']['token_ids'], tokenizer=wrapper.tokenizer)
            groups.append(run.metrics)
            _jsonl(directory / f'group_{group_id}_forwards.jsonl', run.forwards)
            for index, item in enumerate(items):
                completion = run.completions[index]
                completed = completion is not None
                attempted = sum(index in f['question_indices'] for f in run.forwards if f['phase'] == 'completion')
                record = export_question(wrapper, item, ctx, run.searches[index], completion,
                    directory / item['example']['example_id'].replace(':', '_'),
                    search_seconds=run.search_latencies[index] if run.search_latencies[index] is not None else run.search_seconds,
                    completion_seconds=run.completion_latencies[index] if run.completion_latencies[index] is not None else run.completion_seconds,
                    status='completed' if completed else run.status, error=None if completed else run.error,
                    batch_size=batch_size, group_id=group_id, completion_attempts=attempted,
                    completion_forward_attempts=[dict(f, stage='forward') for f in run.forwards
                        if f['phase'] == 'completion' and index in f['question_indices']])
                records.append(record)
            status, error = ('running' if run.status == 'completed' and offset + len(items) < len(prepared) else run.status), run.error
            _jsonl(directory / 'predictions.jsonl', records)
            write_csv(directory / 'results.csv', records)
            save_json(directory / 'summary.json', _summary(records, groups, len(prepared), status, error))
            if run.status != 'completed':
                break
    except (Exception, KeyboardInterrupt) as failure:
        status = 'interrupted' if isinstance(failure, KeyboardInterrupt) else 'error'
        error = f'{type(failure).__name__}: {failure}'
    finally:
        _jsonl(directory / 'predictions.jsonl', records)
        write_csv(directory / 'results.csv', records)
        save_json(directory / 'summary.json', _summary(records, groups, len(prepared), status, error))
    return dict(directory=str(directory), records=records,
                summary=_summary(records, groups, len(prepared), status, error))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('dev', 'evaluation', 'test'), default='dev')
    parser.add_argument('--condition', choices=('A', 'B', 'C', 'D'), default='A')
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--model-path')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=Path('results/ab_medal/development'))
    parser.add_argument('--from-result', type=Path, help='Reconstruct metrics offline; no model or dataset loading.')
    args = parser.parse_args(argv)
    if args.from_result:
        print(json.dumps(search_audit.load(args.from_result), indent=2))
        return 0
    ctx = context(stage=args.stage, condition=args.condition, indices=tuple(range(args.num_samples)))
    wrapper = load_model(ctx, model_path=args.model_path, device=args.device, local_files_only=args.local_files_only)
    prepared = prepare(wrapper, ctx)
    ctx['setup']['warmup'] = warmup(wrapper, prepared, ctx)
    directory, _ = create_run_directory({'output_dir': str(args.output_dir)})
    result = run_samples(wrapper, prepared, ctx, directory)
    for row in result['records']:
        show_metrics(row)
        plot_metrics(row['artifact_directory'])
    print(json.dumps(result['summary'], indent=2))
    return 0 if result['summary']['status'] == 'completed' else 2
