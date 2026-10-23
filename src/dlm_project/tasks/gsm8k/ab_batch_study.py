"""Validate padding/full-run parity before measuring AB batching speedup."""

import json
import math
from pathlib import Path
from statistics import median

import torch

from dlm_project.decoding.ab_batch import pack_states, sync
from dlm_project.decoding.ab_medal import ABMEDALConfig
from dlm_project.decoding.mcts_decoding import distribution_from_logits, model_device
from dlm_project.experiments.artifacts import save_json
from dlm_project.experiments.sweep_report import write_csv, percentile
from dlm_project.tasks.gsm8k.ab_development import run_samples, warmup
from dlm_project.tasks.gsm8k.experiment_protocol import load_samples

RTOL, ATOL = 1e-4, 1e-5


def compare_runs(reference, candidate):
    """Compare discrete tree/decoder outcomes exactly and IG/Q within tolerance."""
    differences, max_reward, max_q = [], 0., 0.
    if candidate['summary']['status'] != 'completed':
        return dict(passed=False, differences=[candidate['summary']['error'] or candidate['summary']['status']],
                    max_ig_difference=None, max_q_difference=None, rtol=RTOL, atol=ATOL)
    left_rows, right_rows = reference['records'], candidate['records']
    if len(left_rows) != len(right_rows):
        differences.append('Different question counts')
    for left, right in zip(left_rows, right_rows):
        name = left['example_id']
        for key in ('example_id', 'seed', 'prompt_ids_hash', 'selected_node_id', 'selected_state_ids',
                    'selected_depth', 'search_nfes', 'completion_nfes', 'correct'):
            if left[key] != right[key]:
                differences.append(f'{name}: {key}')
        for key in ('full_output_ids', 'scoring_output_ids'):
            if left['decoded'][key] != right['decoded'][key]:
                differences.append(f'{name}: {key}')
        if left['score']['predicted_answer'] != right['score']['predicted_answer']:
            differences.append(f'{name}: parsed answer')
        a = json.loads((Path(left['artifact_directory']) / 'search_audit.json').read_text())
        b = json.loads((Path(right['artifact_directory']) / 'search_audit.json').read_text())
        if len(a['nodes']) != len(b['nodes']):
            differences.append(f'{name}: node count')
        for x, y in zip(a['nodes'], b['nodes']):
            if any(x[k] != y[k] for k in ('node_id', 'parent_id', 'depth', 'evaluation_order', 'state_hash', 'visits')):
                differences.append(f'{name}: node {x["node_id"]} identity/visits')
            for key in ('immediate_reward', 'reward_sum', 'q_value'):
                if x[key] is None or y[key] is None:
                    equal = x[key] is y[key]
                else:
                    delta = abs(x[key] - y[key])
                    if key == 'immediate_reward':
                        max_reward = max(max_reward, delta)
                    if key == 'q_value':
                        max_q = max(max_q, delta)
                    equal = math.isclose(x[key], y[key], rel_tol=RTOL, abs_tol=ATOL)
                if not equal:
                    differences.append(f'{name}: node {x["node_id"]} {key}')
        keys = ('traversal_id', 'decision_order', 'decision_node_id', 'decision_type', 'forced', 'chosen_id', 'child_id')
        decisions = lambda rows: [tuple(d.get(k) for k in keys) for d in rows or []]
        if decisions(a['decisions']) != decisions(b['decisions']):
            differences.append(f'{name}: GEN/CONT traversal decisions')
        if a['selection'] != b['selection']:
            differences.append(f'{name}: final selection')
    return dict(passed=not differences, differences=differences, max_ig_difference=max_reward,
                max_q_difference=max_q, rtol=RTOL, atol=ATOL)


def _logit_difference(left, right, positions):
    maximum, relative, same_argmax, finite_pattern = 0., 0., True, True
    for index in positions.split(32):
        a, b = left[index].float(), right[index].float()
        finite = torch.isfinite(a) & torch.isfinite(b)
        finite_pattern &= bool(torch.equal(torch.isfinite(a), torch.isfinite(b)))
        if finite.any():
            delta = (a[finite] - b[finite]).abs()
            maximum = max(maximum, float(delta.max()))
            relative = max(relative, float((delta / a[finite].abs().clamp_min(ATOL)).max()))
        same_argmax &= bool(torch.equal(a.argmax(-1), b.argmax(-1)))
    return dict(max_abs_logit_difference=maximum, max_relative_logit_difference=relative,
                argmax_equal=same_argmax, finite_pattern_equal=finite_pattern)


@torch.inference_mode()
def padding_probe(wrapper, prepared, ctx, baseline, batch_size):
    """Unpadded vs padded-serial vs batched predictions on identical saved states."""
    rows, calls, nfes = [], 0, 0
    device = model_device(wrapper.model)
    config = ABMEDALConfig(**ctx['search'])

    def forward(ids, attention=None):
        nonlocal calls, nfes
        calls += 1
        nfes += ids.shape[0]
        return wrapper.model(ids) if attention is None else wrapper.model(ids, attention_mask=attention)

    try:
        for phase in ('root', 'selected'):
            for offset in range(0, len(prepared), batch_size):
                group = prepared[offset:offset+batch_size]
                states = []
                for i, item in enumerate(group, offset):
                    tokens = (item['prompt_ids'] + [wrapper.mask_id] * 256 if phase == 'root'
                              else baseline['records'][i]['selected_state_ids'])
                    states.append(torch.tensor([tokens], dtype=torch.long))
                ids, attention, lengths = pack_states(states, ctx['pad_token_id'], device)
                batched = forward(ids, attention)
                for i, (state, item, length) in enumerate(zip(states, group, lengths)):
                    original = forward(state.to(device))
                    padded_serial = forward(ids[i:i+1], attention[i:i+1])
                    prompt_len = len(item['prompt_ids'])
                    positions = torch.nonzero(state[0, prompt_len:] == wrapper.mask_id).flatten().to(device) + prompt_len
                    original_analysis = distribution_from_logits(original.logits[0], state, prompt_len,
                        wrapper.mask_id, config, search_excluded_token_ids=ctx['termination']['token_ids'])[0]
                    for kind, candidate in (('padded_serial', padded_serial.logits[0, :length]),
                                            ('batched', batched.logits[i, :length])):
                        diff = _logit_difference(original.logits[0], candidate, positions)
                        analysis = distribution_from_logits(candidate, state, prompt_len, wrapper.mask_id, config,
                            search_excluded_token_ids=ctx['termination']['token_ids'])[0]
                        actions = lambda value: [(a.output_position, a.token_id) for a in value.actions]
                        diff.update(action_ids_equal=actions(analysis) == actions(original_analysis),
                            entropy_close=bool(torch.allclose(analysis.entropies, original_analysis.entropies, rtol=RTOL, atol=ATOL)))
                        rows.append(dict(example_id=item['example']['example_id'], phase=phase, comparison=kind,
                                         passed=all(diff[k] for k in ('argmax_equal', 'finite_pattern_equal', 'action_ids_equal', 'entropy_close')), **diff))
                    del original, padded_serial
                del batched
        return dict(passed=all(r['passed'] for r in rows), rows=rows, model_invocations=calls,
                    sequence_nfes=nfes, rtol=RTOL, atol=ATOL, error=None)
    except (Exception, KeyboardInterrupt) as error:
        return dict(passed=False, rows=rows, model_invocations=calls, sequence_nfes=nfes,
                    rtol=RTOL, atol=ATOL, error=f'{type(error).__name__}: {error}',
                    interrupted=isinstance(error, KeyboardInterrupt))


def _aggregate(trials, size, reference_seconds):
    good = [t for t in trials if t['batch_size'] == size and t['parity']['passed']]
    times = [t['summary']['decoding_seconds'] for t in good]
    elapsed = median(times) if times else None
    groups = [g for t in good for g in t['summary']['groups']]
    memory = lambda key: max((g[key] for g in groups if g[key] is not None), default=None)
    invocations = sum(g['total_model_invocations'] for g in groups)
    return dict(batch_size=size, successful_repetitions=len(good), median_decoding_seconds=elapsed,
        min_decoding_seconds=min(times) if times else None, max_decoding_seconds=max(times) if times else None,
        runtime_iqr=percentile(times, .75)-percentile(times, .25) if times else None,
        questions_per_second=4 / elapsed if elapsed else None,
        speedup_over_serial=reference_seconds / elapsed if reference_seconds and elapsed else None,
        peak_allocated_bytes=memory('peak_allocated_bytes'), peak_reserved_bytes=memory('peak_reserved_bytes'),
        mean_batch_occupancy=sum(g['total_nfes'] for g in groups) / invocations if invocations else None,
        median_search_nfes=median(t['summary']['search_nfes'] for t in good) if good else None,
        median_completion_nfes=median(t['summary']['completion_nfes'] for t in good) if good else None,
        median_total_nfes=median(t['summary']['total_nfes'] for t in good) if good else None,
        median_model_invocations=median(t['summary']['total_model_invocations'] for t in good) if good else None,
        accuracy=median(t['summary']['accuracy'] for t in good) if good else None,
        parsing_rate=median(t['summary']['parsing_rate'] for t in good) if good else None,
        fallback_rate=median(t['summary']['fallback_rate'] for t in good) if good else None)


def run_study(wrapper, prepared, ctx, directory, *, repetitions=3):
    """Optional extension: require serial, padding, and full-run parity first."""
    expected = [e['example_id'] for e in load_samples('dev')[:4]]
    if ctx['stage'] != 'dev' or [p['example']['example_id'] for p in prepared] != expected:
        raise ValueError('Batch study uses the first four frozen development questions, in order.')
    if type(repetitions) is not int or repetitions < 1:
        raise ValueError('repetitions must be positive.')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory / 'config.json', dict(**ctx, batch_sizes=[1, 2, 4], repetitions=repetitions,
                                            parity_rtol=RTOL, parity_atol=ATOL))
    validation, trials, warmups = {}, [], {}
    def finish(status, error=None):
        serial = _aggregate(trials, 1, None)['median_decoding_seconds']
        summary = [_aggregate(trials, size, serial) for size in (1, 2, 4)]
        report = dict(status=status, error=error, validation=validation,
            warmup=warmups, trials=trials, summary=summary,
            note='Only parity-validated trials contribute to throughput. Formal algorithm comparisons remain serial.')
        save_json(directory / 'validation.json', validation)
        save_json(directory / 'warmup.json', warmups)
        save_json(directory / 'trials.json', trials)
        save_json(directory / 'batch_study.json', report)
        write_csv(directory / 'batch_summary.csv', summary)
        if any(row['successful_repetitions'] for row in summary):
            _plot(summary, directory)
        return report

    try:
        baseline = run_samples(wrapper, prepared, ctx, directory / 'validation_serial', batch_size=1)
        validation['1'] = dict(passed=baseline['summary']['status'] == 'completed', serial=baseline['summary'])
        save_json(directory / 'validation.json', validation)
        if not validation['1']['passed']:
            return finish(baseline['summary']['status'], baseline['summary'].get('error'))
        for size in (2, 4):
            probe = padding_probe(wrapper, prepared, ctx, baseline, size)
            validation[str(size)] = dict(passed=False, padding=probe)
            save_json(directory / 'validation.json', validation)
            if probe.get('interrupted'):
                return finish('interrupted', probe.get('error'))
            if probe['passed']:
                result = run_samples(wrapper, prepared, ctx, directory / f'validation_batch_{size}', batch_size=size)
                comparison = compare_runs(baseline, result)
                validation[str(size)].update(passed=comparison['passed'], full_run=comparison)
                save_json(directory / 'validation.json', validation)
                if result['summary']['status'] == 'interrupted':
                    return finish('interrupted', result['summary'].get('error'))
        eligible = [size for size in (1, 2, 4) if validation[str(size)]['passed']]
        for size in list(eligible):
            try:
                warmups[str(size)] = warmup(wrapper, prepared, ctx, batch_size=size)
            except (Exception, KeyboardInterrupt) as error:
                warmups[str(size)] = dict(error=f'{type(error).__name__}: {error}')
                eligible.remove(size)
                if isinstance(error, KeyboardInterrupt):
                    return finish('interrupted', warmups[str(size)]['error'])
            save_json(directory / 'warmup.json', warmups)
        if not eligible:
            return finish('error', 'No batch size passed both parity validation and warm-up.')
        for repetition in range(repetitions):
            sizes = eligible[repetition % len(eligible):] + eligible[:repetition % len(eligible)] if eligible else []
            for size in sizes:
                result = run_samples(wrapper, prepared, ctx, directory / f'repeat_{repetition+1}_batch_{size}', batch_size=size)
                parity = compare_runs(baseline, result)
                trials.append(dict(repetition=repetition+1, batch_size=size, summary=result['summary'], parity=parity))
                save_json(directory / 'trials.json', trials)
                if not parity['passed']:
                    eligible.remove(size)
                if result['summary']['status'] == 'interrupted':
                    return finish('interrupted', result['summary'].get('error'))
        return finish('completed')
    except (Exception, KeyboardInterrupt) as error:
        return finish('interrupted' if isinstance(error, KeyboardInterrupt) else 'error',
                      f'{type(error).__name__}: {error}')


def _plot(rows, directory):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    good = [r for r in rows if r['successful_repetitions']]
    figure = Figure(figsize=(12, 4), constrained_layout=True)
    axes = figure.subplots(1, 3)
    for axis, key, label in zip(axes, ('median_decoding_seconds', 'questions_per_second', 'peak_allocated_bytes'),
                               ('Median decoding seconds', 'Questions / second', 'Peak allocated GiB')):
        present = [r for r in good if r[key] is not None]
        axis.bar([str(r['batch_size']) for r in present], [r[key] / (2**30 if key.endswith('bytes') else 1) for r in present])
        axis.set(xlabel='Independent questions per batch', ylabel=label)
    figure.suptitle('AB batching: parity-validated trials only')
    FigureCanvasAgg(figure).print_png(str(directory / 'batch_comparison.png'))
