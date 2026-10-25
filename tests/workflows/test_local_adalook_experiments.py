import copy
import importlib.util
import json
from pathlib import Path

import pytest

from dlm_project.tasks.gsm8k import local_experiments as runner, ab_development as dev
from tests.workflows.test_ab_development import wrapper
from tests.decoding.test_ab_batch import BatchModel


def launcher():
    spec = importlib.util.spec_from_file_location('local_launcher', 'scripts/run_gsm8k_experiments.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def settings(tmp_path, mode='adalook_adaptive', method='both', count=1):
    cli = launcher()
    args = cli.parser().parse_args(['--method', method, '--stage', 'dev', '--num-samples', str(count),
        '--search-budget', '5', '--lc', '2', '--reward-mode', mode, '--device', 'cpu'])
    paths = dict(MODEL_CACHE_DIR=tmp_path/'hub', DATASET_DIR=tmp_path/'data', RESULTS_DIR=tmp_path/'results',
                 DATASET_MANIFEST=Path('configs/gsm8k/experiment_samples.json').resolve())
    return cli.build_config(args, paths, None)


@pytest.mark.parametrize('mode', ['ig', 'adalook_one_step', 'adalook_adaptive'])
def test_outputs_and_identical_inputs_then_resume_without_loading(tmp_path, mode):
    config = settings(tmp_path, mode)
    w = wrapper()
    directory = runner.run_experiments(config, wrapper=w)
    rows = runner.read_predictions(directory/'predictions.jsonl')
    assert len(rows) == 2
    assert rows[0]['condition'] == 'confidence_t0'
    assert rows[0]['prompt_ids_hash'] == rows[1]['prompt_ids_hash']
    assert rows[0]['seed'] == rows[1]['seed']
    assert rows[1]['reward_mode'] == mode
    assert rows[1]['search_nfes'] <= 5
    assert sum(r['total_nfes'] for r in rows) == len(w.model.calls)
    for name in ['config.json', 'manifest.json', 'selection.json', 'summary.json',
                 'report/report.md', 'report/summary.csv', 'report/comparison.png']:
        assert (directory/name).is_file()
    before = (directory/'predictions.jsonl').read_bytes()
    runner.run_experiments(config, resume=directory)  # Must skip model loading entirely.
    assert (directory/'predictions.jsonl').read_bytes() == before
    changed = copy.deepcopy(config)
    changed['search']['adalook_alpha'] = .2
    with pytest.raises(ValueError, match='Resume configuration'):
        runner.run_experiments(changed, resume=directory)


def test_interrupted_question_is_saved_and_retried_without_duplicate_success(tmp_path):
    config = settings(tmp_path, method='mcts', count=2)
    w = wrapper(BatchModel(failure=KeyboardInterrupt('test interruption'), fail_at=3))
    with pytest.raises(KeyboardInterrupt):
        runner.run_experiments(config, wrapper=w)
    directory = next((tmp_path/'results').rglob('config.json')).parent
    first = runner.read_predictions(directory/'predictions.jsonl')
    assert first[0]['status'] == 'interrupted' and first[0]['search_nfes'] == 3
    assert json.loads((directory/'summary.json').read_text())['status'] == 'interrupted'
    runner.run_experiments(config, resume=directory, wrapper=wrapper())
    rows = runner.read_predictions(directory/'predictions.jsonl')
    assert len(rows) == 3 and sum(r['status'] == 'completed' for r in rows) == 2
    assert len(list((directory/'traces').glob('*.jsonl'))) == 3


def test_dry_run_is_offline_and_creates_no_results(tmp_path, monkeypatch):
    env = tmp_path/'paths.env'
    env.write_text(f'MODEL_CACHE_DIR={tmp_path}/hub\nDATASET_DIR={tmp_path}/data\nRESULTS_DIR={tmp_path}/results\n')
    monkeypatch.setattr(runner, 'run_experiments', lambda *a, **kw: pytest.fail('dry-run loaded model'))
    cfg = launcher().main(['--env-file', str(env), '--dry-run', '--reward-mode', 'adalook_adaptive'])
    assert cfg['search']['reward_mode'] == 'adalook_adaptive'
    assert not (tmp_path/'results').exists()


def test_demo_configuration_is_validated_and_variant_hash_changes():
    base = dev.context(method='medal')
    variant = dev.context(method='medal', reward_options={'reward_mode':'adalook_adaptive'})
    dev.validate_context(variant)
    assert variant['variant_hash'] != base['variant_hash']
    assert variant['common_protocol_hash'] == base['common_protocol_hash']
    variant['search']['adalook_alpha'] = .7
    with pytest.raises(ValueError, match='Frozen development setting changed'):
        dev.validate_context(variant)


def test_incomplete_last_line_recovery_preserves_completed_rows(tmp_path):
    path = tmp_path/'predictions.jsonl'
    path.write_bytes(b'{"status":"completed"}\n{"status":')
    assert runner.read_predictions(path) == [{'status':'completed'}]
    assert path.read_bytes() == b'{"status":"completed"}\n'
