"""The common experiment input is a frozen, portable, offline artifact."""

import copy
import json
import runpy
import sys

import pytest

from dlm_project.experiments.manifests import digest
from dlm_project.paths import PROJECT_ROOT
from dlm_project.tasks.gsm8k import experiment_protocol as protocol
from dlm_project.tasks.gsm8k.e0_dataset import build_manifest as build_legacy_manifest


def test_canonical_samples_are_fixed_disjoint_and_available_offline(monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: pytest.fail("Network access"))
    common = protocol.load_shared_protocol()
    first, second = protocol.get_manifest(), protocol.get_manifest()
    assert first == second and first is not second
    assert digest(first) == common["data"]["manifest_hash"]
    pools = {stage: protocol.load_samples(stage) for stage in protocol.STAGE_COUNTS}
    assert {stage: len(rows) for stage, rows in pools.items()} == {"dev": 5, "evaluation": 20, "test": 200}
    assert len({r["example_id"] for rows in pools.values() for r in rows}) == 225
    assert len({protocol.question_key(r["question"]) for rows in pools.values() for r in rows}) == 225
    assert all(r["source_split"] == ("test" if stage == "test" else "train")
               for stage, rows in pools.items() for r in rows)


def test_builder_preserves_old_selection_and_freezes_first_200():
    train = [{"question": f"Train {i}?", "answer": "#### 2"} for i in range(40)]
    test = [{"question": f"Test {i}?", "answer": "#### 3"} for i in range(220)]
    train.append({"question": " test 0? ", "answer": "#### 3"})
    test.append(dict(test[0]))
    kwargs = dict(revision="a" * 40, sample_seed=1)
    old = build_legacy_manifest(train, test, **kwargs)
    new = protocol.build_manifest(train, test, **kwargs)
    assert new == protocol.build_manifest(train, test, **kwargs)
    for stage, count in protocol.STAGE_COUNTS.items():
        expected = [r for r in old["examples"] if r["stage"] == stage][:count]
        assert protocol.select_examples(new, stage) == expected
    with pytest.raises(ValueError):
        protocol.build_manifest(train, test[:100], **kwargs)


@pytest.mark.parametrize("change", ["content", "reorder", "count", "overlap", "revision", "seed"])
def test_official_loader_rejects_changed_manifest(tmp_path, change):
    manifest = protocol.get_manifest()
    rows = manifest["examples"]
    if change == "content":
        rows[0]["question"] += " changed"
        rows[0]["content_hash"] = digest({k: rows[0][k] for k in ("question", "reference_solution")})
    elif change == "reorder":
        rows[0], rows[1] = rows[1], rows[0]
    elif change == "count":
        rows.pop()
    elif change == "overlap":
        rows[5]["question"] = rows[0]["question"]
        rows[5]["content_hash"] = digest({k: rows[5][k] for k in ("question", "reference_solution")})
    elif change == "revision":
        manifest["dataset"]["revision"] = "a" * 40
    else:
        manifest["sample_seed"] = 2
    target = tmp_path / "manifest.json"
    target.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        protocol.get_manifest(target)


def test_prompt_is_literal_and_preparation_verifies_offline(monkeypatch, capsys):
    question = "What are {GSM8K_question} and {items}?"
    prompt = protocol.build_prompt(question)
    assert prompt.endswith("Problem: " + question)
    assert prompt.count("Example 1:") == prompt.count("Example 2:") == 1
    assert "#### 27\n\nExample 2:" in prompt
    assert "#### 16\n\nNow solve" in prompt
    historic = json.loads((PROJECT_ROOT / 'tests/fixtures/e0_v4_run.json').read_text())
    template = protocol.load_shared_protocol()['prompt']['template']
    assert template == historic['config']['prompt']['template']
    guide = (PROJECT_ROOT / 'docs/experiments.md').read_text()
    assert '```text\n' + template + '\n```' in guide
    monkeypatch.setattr(sys, "argv", ["prepare_gsm8k_experiments.py"])
    runpy.run_path(str(PROJECT_ROOT / "scripts/prepare_gsm8k_experiments.py"), run_name="__main__")
    assert "Verified" in capsys.readouterr().out


def test_common_reference_cannot_override_frozen_content(tmp_path):
    import yaml
    changed = copy.deepcopy(protocol.load_shared_protocol())
    changed["generation"]["gen_length"] = 128
    target = tmp_path / "common.yaml"
    target.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="frozen"):
        protocol.load_shared_protocol(target)


def test_preparation_rebuild_uses_only_pinned_source_and_refuses_tampering(tmp_path, monkeypatch):
    import huggingface_hub
    import pyarrow.parquet
    common = protocol.load_shared_protocol()
    frozen = protocol.get_manifest()
    calls = []
    def download(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs["revision"]))
        return filename
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    class Table:
        def to_pylist(self):
            return []
    monkeypatch.setattr(pyarrow.parquet, "read_table", lambda _: Table())
    monkeypatch.setattr(protocol, "build_manifest", lambda *a, **kw: copy.deepcopy(frozen))
    target = tmp_path / "nested" / "copy.json"
    command = ["prepare", "--rebuild", "--output", str(target)]
    monkeypatch.setattr(sys, "argv", command)
    script = str(PROJECT_ROOT / "scripts/prepare_gsm8k_experiments.py")
    runpy.run_path(script, run_name="__main__")
    assert protocol.get_manifest(target) == frozen
    assert len(calls) == 2 and all(c[2] == common["data"]["revision"] for c in calls)
    target.write_text('{}')
    with pytest.raises(ValueError):
        runpy.run_path(script, run_name="__main__")
    assert target.read_text() == '{}'
