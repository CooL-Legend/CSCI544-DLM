"""Pinned demonstrations, Drive publication, and held-out answer isolation."""
import copy
import json
from types import SimpleNamespace
import pytest
from dlm_project.decoding import medal_prompting as mp


def test_training_rows_pinned_published_and_reused_offline(tmp_path, monkeypatch):
    import datasets
    calls=[]
    def load(*args,**kw):
        calls.append((args,kw))
        return [{'question':f'Train {i}: how many apples?', 'answer':'There are 2 apples.\n#### 2'} for i in range(2)]
    monkeypatch.setattr(datasets,'load_dataset',load)
    monkeypatch.setattr(mp.os,'link',lambda *a:pytest.fail('Drive does not support hard links'))
    path=tmp_path/'drive'/'demos.json'
    manifest=mp.get_gsm8k_demonstrations(path,revision='a'*40,cache_dir='cache')
    assert calls==[(('openai/gsm8k','main'),{'split':'train[:2]','revision':'a'*40,'cache_dir':'cache'})]
    monkeypatch.setattr(datasets,'load_dataset',lambda *a,**kw:pytest.fail('Must reuse frozen demos'))
    assert mp.get_gsm8k_demonstrations(path,revision='a'*40)==manifest
    with pytest.raises(ValueError):mp.get_gsm8k_demonstrations(path,revision='b'*40)
    question='Held-out problem?'
    prompt,metadata=mp.gsm8k_prompt('Solve the question.\n'+question,question,manifest)
    assert 'three short subtasks' in prompt and metadata['planning_model_calls']==0
    assert metadata['demonstrations_dataset']['split']=='train'
    assert metadata['demonstration_rows']==[0,1]
    with pytest.raises(ValueError,match='overlaps'):mp.gsm8k_prompt('Train 0','Train 0: how many apples?',manifest)
    changed=copy.deepcopy(manifest);changed['examples'][0]['reference_solution']='#### 1000'
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='hash'):mp.get_gsm8k_demonstrations(path,revision='a'*40)


@pytest.mark.parametrize('size',[4,9])
def test_synthetic_sudoku_demonstrations_and_final_format(size):
    prompt,metadata=mp.sudoku_prompt('Original puzzle; return only the original answer format.',size,'0'*(size*size))
    assert metadata['num_subtasks']==3 and metadata['planning_model_calls']==0
    assert 'synthetic' in metadata['demonstration_source']
    assert ('<answer>' in prompt)==(size==4)
    assert 'Keep the original final-answer format exactly.' in prompt
    assert metadata['effective_prompt_hash']!=metadata['base_prompt_hash']
