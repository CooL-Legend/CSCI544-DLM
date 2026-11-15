"""Exercise the notebook's pre-clone authentication without real credentials."""

import ast
import json
from pathlib import Path
import subprocess
import sys
import traceback
from types import ModuleType, SimpleNamespace

import pytest


NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks/gsm8k/Experiment_2_Eval_IG_KL.ipynb"
REPO_URL = "https://github.com/CooL-Legend/CSCI544-DLM.git"
FAKE_GITHUB_TOKEN = "github_pat_TEST_ONLY"
FAKE_HF_TOKEN = "hf_TEST_ONLY"


@pytest.fixture
def sync_repo(monkeypatch):
    notebook = json.loads(NOTEBOOK.read_text())
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if "def sync_colab_repo(" in "".join(cell["source"]))
    tree = ast.parse(source)
    # Run the actual helper and imports, without cloning into /content.
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    scope = {}
    exec(compile(tree, "notebook_repository_setup", "exec"), scope)
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_GITHUB_TOKEN)
    return scope["sync_colab_repo"]


@pytest.mark.parametrize("invalid", [
    FAKE_GITHUB_TOKEN + "\r\n" + FAKE_HF_TOKEN,
    FAKE_GITHUB_TOKEN + " " + FAKE_HF_TOKEN,
    FAKE_HF_TOKEN,
    '"' + FAKE_GITHUB_TOKEN + '"',
])
def test_mixed_or_wrong_credentials_fail_before_git(sync_repo, monkeypatch, tmp_path, invalid, capsys):
    monkeypatch.setenv("GITHUB_TOKEN", invalid)
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="one GitHub personal access token") as failure:
        sync_repo(tmp_path / "repo", REPO_URL, "sean-1114")
    assert calls == []
    rendered = "".join(traceback.format_exception(failure.value)) + str(capsys.readouterr())
    assert FAKE_GITHUB_TOKEN not in rendered and FAKE_HF_TOKEN not in rendered


def test_clone_uses_secret_free_url_and_temporary_askpass(sync_repo, monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_TOKEN", " \n" + FAKE_GITHUB_TOKEN + "\r\n")
    real_run = subprocess.run
    calls, helper_paths, child_envs = [], [], []

    def git_run(args, **kwargs):
        env = kwargs["env"]
        helper = Path(env["GIT_ASKPASS"])
        helper_paths.append(helper)
        child_envs.append(env)
        calls.append(args)
        assert FAKE_GITHUB_TOKEN not in repr(args)
        assert FAKE_GITHUB_TOKEN not in helper.read_text()
        assert "GITHUB_TOKEN" not in env
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        assert helper.stat().st_mode & 0o777 == 0o700
        for prompt, expected in [("Username for GitHub:", "x-access-token"),
                                 ("Password for GitHub:", FAKE_GITHUB_TOKEN)]:
            reply = real_run([str(helper), prompt], env=env, capture_output=True, text=True, check=True)
            assert reply.stdout.strip() == expected
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", git_run)
    destination = tmp_path / "repo"
    sync_repo(destination, REPO_URL, "sean-1114")
    assert calls == [["git", "-c", "credential.helper=", "clone", "-b", "sean-1114", REPO_URL, str(destination)]]
    assert all(not path.exists() for path in helper_paths)
    assert all("DLM_GITHUB_TOKEN" not in env for env in child_envs)


def test_git_failure_redacts_output_and_cleans_up(sync_repo, monkeypatch, tmp_path, capsys):
    paths, envs = [], []

    def failed_git(args, **kwargs):
        paths.append(Path(kwargs["env"]["GIT_ASKPASS"]))
        envs.append(kwargs["env"])
        return SimpleNamespace(returncode=128, stdout=FAKE_GITHUB_TOKEN,
                               stderr=f"fatal: {FAKE_GITHUB_TOKEN} {FAKE_HF_TOKEN} https://user:password@github.com/repo")

    monkeypatch.setattr(subprocess, "run", failed_git)
    with pytest.raises(RuntimeError, match="Git failed.*128") as failure:
        sync_repo(tmp_path / "repo", REPO_URL, "sean-1114")
    rendered = "".join(traceback.format_exception(failure.value)) + str(capsys.readouterr())
    assert FAKE_GITHUB_TOKEN not in rendered and FAKE_HF_TOKEN not in rendered
    assert "user:password" not in rendered
    assert "[REDACTED]" in str(failure.value)
    assert failure.value.__cause__ is None
    assert all(not path.exists() for path in paths)
    assert all("DLM_GITHUB_TOKEN" not in env for env in envs)


def test_existing_clone_resets_origin_before_authenticated_fetch(sync_repo, monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    calls = []

    def git_run(args, **kwargs):
        calls.append(args[3:])
        assert kwargs["env"]["DLM_GITHUB_TOKEN"] == FAKE_GITHUB_TOKEN
        assert FAKE_GITHUB_TOKEN not in repr(args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", git_run)
    sync_repo(tmp_path, REPO_URL, "sean-1114")
    prefix = ["-C", str(tmp_path)]
    assert calls == [
        prefix + ["remote", "set-url", "origin", REPO_URL],
        prefix + ["fetch", "origin", "sean-1114"],
        prefix + ["checkout", "sean-1114"],
        prefix + ["pull", "--ff-only", "origin", "sean-1114"],
    ]


@pytest.mark.parametrize("secret", [None, FAKE_GITHUB_TOKEN])
def test_colab_secret_or_anonymous_clone(sync_repo, monkeypatch, tmp_path, secret):
    monkeypatch.delenv("GITHUB_TOKEN")
    colab = ModuleType("google.colab")
    secret_reads, calls = [], []

    def get_secret(name):
        secret_reads.append(name)
        if secret is None:
            raise RuntimeError("Secret not configured")
        return secret

    colab.userdata = SimpleNamespace(get=get_secret)
    monkeypatch.setitem(sys.modules, "google.colab", colab)

    def git_run(args, **kwargs):
        calls.append(args)
        assert kwargs["env"]["DLM_GITHUB_TOKEN"] == (secret or "")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", git_run)
    sync_repo(tmp_path / "repo", REPO_URL, "sean-1114")
    assert len(calls) == 1
    assert secret_reads == ["GITHUB_TOKEN"]
