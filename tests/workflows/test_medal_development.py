"""Paired notebooks share every executable setting except initialization method."""

import json
from pathlib import Path

import nbformat
import pytest
import torch

from dlm_project.decoding import search_audit
from dlm_project.decoding.mcts_decoding import MCTSSearch, MCTSConfig
from dlm_project.decoding.ab_medal import ABMEDALSearch
from dlm_project.tasks.gsm8k import ab_development as dev, ab_batch_study as study
from tests.workflows.test_ab_development import wrapper
from tests.decoding.test_ab_batch import BatchModel


BOOKS = ('AB_MEDAL_MCTS_Colab.ipynb', 'MEDAL_MCTS_Demo.ipynb')


def test_notebooks_have_identical_code_except_initialization_selector():
    ab, medal = [nbformat.read(Path('notebooks/gsm8k') / name, as_version=4) for name in BOOKS]
    for book in (ab, medal):
        nbformat.validate(book)
    assert len(ab.cells) == len(medal.cells)
    for left, right in zip(ab.cells, medal.cells):
        assert (left.id, left.cell_type) == (right.id, right.cell_type)
        if left.cell_type == 'code':
            assert left.source.replace('METHOD = "ab_medal"', 'METHOD = "medal"') == right.source
            assert left.outputs == right.outputs == []
            assert left.execution_count is right.execution_count is None
        elif left.id not in ('ab-introduction', 'ab-search-explanation'):
            assert left.source == right.source


@pytest.mark.parametrize('condition', ['A', 'B', 'C', 'D'])
def test_all_common_inputs_and_configurations_match_ab(condition):
    ab = dev.context(condition=condition)
    medal = dev.context(condition=condition, method='medal')
    for key in ('common', 'common_protocol_hash', 'prompt_template_hash', 'stage', 'condition',
                'examples', 'run_type', 'manifest_hash'):
        assert medal[key] == ab[key]
    for key in MCTSConfig.__dataclass_fields__:
        assert medal['search'][key] == ab['search'][key]
    assert 'search_policy' not in medal['search']
    assert 'prior_m' not in medal['search']
    assert dev.prepare(wrapper(), medal)[0]['prompt_ids'] == dev.prepare(wrapper(), ab)[0]['prompt_ids']


def test_execute_both_actual_notebooks_same_inputs_decoder_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'is_bf16_supported', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda i: 'stub')
    scopes = []
    for name, method in zip(BOOKS, ('ab_medal', 'medal')):
        w, loads = wrapper(), []
        def load(ctx, **kwargs):
            loads.append(ctx)
            return w
        monkeypatch.setattr(dev, 'load_model', load)
        scope = dict(Path=Path, DRIVE_ROOT=tmp_path, display=lambda *a: None)
        book = nbformat.read(Path('notebooks/gsm8k') / name, as_version=4)
        active = False
        for cell in book.cells:
            if cell.cell_type == 'code':
                compile(cell.source, cell.id, 'exec')
                active |= cell.id == 'ab-settings'
                if active:
                    exec(cell.source, scope)
        assert len(loads) == 1 and scope['METHOD'] == method
        assert type(scope['initialization'].search) is (ABMEDALSearch if method == 'ab_medal' else MCTSSearch)
        assert scope['record']['total_nfes'] == len(w.model.calls) - 1
        assert scope['record']['completion_nfes'] == 256 - scope['record']['selected_depth']
        assert scope['record']['completion_method'] == 'llada' and scope['record']['temperature'] == 0
        assert scope['RUN_SERIAL_DEV'] is scope['RUN_BATCH_STUDY'] is False
        assert (scope['SINGLE_OUTPUT'] / f'{method}_tree.png').is_file()
        assert search_audit.load(scope['SINGLE_OUTPUT']) == scope['offline_metrics']
        if method == 'medal':
            assert scope['record']['decision_counts'] is None
            assert (scope['SINGLE_OUTPUT'] / 'decisions.jsonl').read_text() == ''
        scopes.append(scope)
    a, b = scopes
    assert a['item']['text'] == b['item']['text']
    assert a['item']['prompt_ids_hash'] == b['item']['prompt_ids_hash']
    assert a['item']['seed'] == b['item']['seed']
    assert a['item']['example'] == b['item']['example']
    assert a['record'].keys() == b['record'].keys()
    assert a['SINGLE_OUTPUT'] != b['SINGLE_OUTPUT']


def test_medal_development_batch_parity_and_partial_groups(tmp_path):
    w, ctx = wrapper(), dev.context(method='medal', indices=tuple(range(5)))
    prepared = dev.prepare(w, ctx)
    serial = dev.run_samples(w, prepared, ctx, tmp_path / 'serial')
    assert serial['summary']['completed'] == 5
    for size in (2, 4):
        probe = study.padding_probe(w, prepared, ctx, serial, size)
        assert probe['passed']
        batched = dev.run_samples(w, prepared, ctx, tmp_path / str(size), batch_size=size)
        assert study.compare_runs(serial, batched)['passed']
        assert batched['summary']['groups'][-1]['batch_size'] == 1
        assert all(r['decision_counts'] is None for r in batched['records'])
        assert batched['summary']['total_nfes'] == serial['summary']['total_nfes']
        assert batched['summary']['total_model_invocations'] < serial['summary']['total_model_invocations']


def test_medal_failure_exports_partial_metrics(tmp_path):
    w, ctx = wrapper(BatchModel(failure=KeyboardInterrupt('stop'), fail_at=2)), dev.context(method='medal')
    item = dev.prepare(w, ctx)[0]
    with pytest.raises(KeyboardInterrupt):
        dev.initialize(w, item, ctx, tmp_path)
    record = json.loads((tmp_path / 'metrics.json').read_text())
    assert record['status'] == 'interrupted' and record['method'] == 'medal'
    assert record['search_nfes'] == 2 and record['uncompleted_evaluation_nfes'] == 1
    assert record['correct'] is record['decision_counts'] is None
    dev.plot_metrics(tmp_path)
    assert (tmp_path / 'medal_tree.png').is_file()
