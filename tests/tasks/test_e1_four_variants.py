"""CPU integration coverage for the four-variant E1/E3 protocol.

Also runnable with unittest when pytest is unavailable locally.
Search uses real decoder code and a synthetic model; the unchanged final
completion interface is stubbed to isolate runner/accounting/report behavior.
"""

from dataclasses import asdict
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from dlm_project.decoding.ab_medal import ABMEDALConfig, ABMEDALSearch
from dlm_project.decoding.mcts_decoding import MCTSConfig, MCTSSearch
from dlm_project.tasks.gsm8k import experiment_1 as e1, e3_analysis as e3
from dlm_project.tasks.gsm8k.e1_variants import METHODS, is_adaptive, variant
from tests.decoding.helpers import FakeModel, PROMPT


class Tokenizer:
    eos_token_id = pad_token_id = 4

    def get_vocab(self):
        return {"[MASK]": 0, "one": 1, "two": 2, "three": 3,
                "<|endoftext|>": 4, "<|eot_id|>": 5}

    def convert_ids_to_tokens(self, token):
        return next(k for k, v in self.get_vocab().items() if v == token)

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(self.convert_ids_to_tokens(i) for i in ids
                        if not skip_special_tokens or i not in (0, 4, 5))


def partial_confidence(output):
    rows = [[.001, .4, .35, .247, .001, .001] for _ in output]
    masked = torch.nonzero(output == 0).flatten()
    if len(masked):
        rows[masked[0].item()] = [.001, .7, .2, .097, .001, .001]
    return rows


def wrapper():
    common = e1.context(num_samples=1)["common"]
    return SimpleNamespace(model=FakeModel(partial_confidence), tokenizer=Tokenizer(), mask_id=0,
        identity=common["model"], encode_prompt=lambda text: PROMPT.clone())


def small_context():
    # Diagnostic-only overrides exercise actual rollout/comparison forwards quickly.
    ctx = e1.context(stage="dev", num_samples=1,
                     adalook_options={"adalook_confidence_threshold": .6,
                                      "adalook_variance_threshold": 100.})
    ctx["common"]["generation"]["gen_length"] = 8
    ctx["common"]["decoding"]["mcts"].update(top_k_tokens=1, top_k_actions=2)
    return ctx


class FourVariantTests(unittest.TestCase):
    def setUp(self):
        def completion(initialization, *, method, temperature, **kwargs):
            self.assertEqual(method, "llada")
            self.assertEqual(temperature, 0.0)
            tokens = initialization.state.clone()
            output = tokens[:, initialization.search.prompt_len:]
            output[output == initialization.search.mask_id] = 1
            return SimpleNamespace(token_ids=tokens)

        self.completion = patch.object(e1, "run_completion", side_effect=completion).start()
        self.addCleanup(patch.stopall)

    def test_default_protocol_and_reward_only_configs(self):
        ctx = e1.context(stage="evaluation")
        self.assertEqual(len(ctx["examples"]), 20)
        self.assertEqual(ctx["budgets"], [256, 512])
        self.assertEqual(ctx["max_depth"], 10)
        self.assertEqual(ctx["num_candidates"], 3)
        self.assertEqual(ctx["methods"], list(METHODS))
        for baseline, enhanced in (("medal", "medal_adalook"), ("ab_medal", "ab_medal_adalook")):
            left = e1.search_config(ctx, baseline, (256, 512), 10, 3)
            right = e1.search_config(ctx, enhanced, (256, 512), 10, 3)
            self.assertIs(type(left), type(right))
            original = (ABMEDALConfig if is_adaptive(baseline) else MCTSConfig)(
                init_tokens=10, num_candidates=3, top_k_tokens=3, top_k_actions=5,
                stop_mode="budget", max_model_calls=512)
            self.assertEqual(left, original)
            a, b = asdict(left), asdict(right)
            self.assertEqual(a.pop("reward_mode"), "ig")
            self.assertEqual(b.pop("reward_mode"), "adalook_adaptive")
            self.assertEqual(a, b)
        with self.assertRaises(ValueError):
            e1.context(adalook_options={"exploration_c": 1})

    def test_extended_evaluation_preserves_canonical_inputs(self):
        canonical = e1.context(stage="evaluation")
        extended = e1.context(stage="evaluation", evaluation_set="extended_70")
        self.assertEqual(len(extended["examples"]), 70)
        self.assertEqual(extended["examples"][:20], canonical["examples"])
        self.assertEqual(len({e["example_id"] for e in extended["examples"]}), 70)
        self.assertEqual({e["source_split"] for e in extended["examples"]}, {"train"})
        self.assertNotEqual(extended["manifest_hash"], canonical["manifest_hash"])
        for key in ("common", "adalook_options", "methods", "budgets", "prompt_template_hash"):
            self.assertEqual(extended[key], canonical[key])
        self.assertEqual(e1.context(stage="evaluation", evaluation_set="extended_70",
                                    num_samples=2)["examples"], extended["examples"][:2])
        with self.assertRaisesRegex(ValueError, "requires stage"):
            e1.context(stage="test", evaluation_set="extended_70")
        with self.assertRaisesRegex(ValueError, "Unknown evaluation_set"):
            e1.context(stage="evaluation", evaluation_set="typo")

    def test_adalook_batch_rejected_before_side_effects(self):
        for method in ("medal_adalook", "ab_medal_adalook"):
            with self.subTest(method=method), TemporaryDirectory() as temp:
                w, ctx = wrapper(), small_context()
                out = Path(temp) / "must_not_exist"
                with self.assertRaisesRegex(ValueError, "batch_size=1"):
                    e1.run_e1_stage(w, ctx, out, methods=(method,), batch_size=2)
                self.assertFalse(out.exists())
                self.assertEqual(len(w.model.seen), 0)
                with self.assertRaisesRegex(ValueError, "batch_size=1"):
                    e1.execute_search_batch_checkpointed(w, [], ctx, method)

    def test_extended_evaluation_run_resume_and_old_manifest_rejection(self):
        with TemporaryDirectory() as temp:
            out, w = Path(temp), wrapper()
            ctx = small_context()
            extended = e1.context(stage="evaluation", evaluation_set="extended_70")
            # Exercise one canonical and one newly added question with short decoding.
            extended["examples"] = [extended["examples"][0], extended["examples"][20]]
            extended["common"] = ctx["common"]
            extended["adalook_options"] = ctx["adalook_options"]
            args = dict(budgets=(12, 30), max_depth=3, num_candidates=2)
            rows, aggregates = e1.run_e1_stage(w, extended, out, **args)
            self.assertEqual(len(rows), 16)
            self.assertEqual(len(aggregates), 8)
            self.assertTrue(all(r["sample_count"] == 2 for r in aggregates))
            calls = len(w.model.seen)
            resumed, _ = e1.run_e1_stage(w, extended, out, **args)
            self.assertEqual(json.loads(json.dumps(rows)), resumed)
            self.assertEqual(len(w.model.seen), calls)
            offline, _ = e3.analyze_e3_stage(out / "evaluation", out / "summaries",
                                            stage="evaluation", budgets=(12, 30))
            self.assertEqual(len(offline), 16)
            self.assertEqual(len(w.model.seen), calls)
            extended["manifest_hash"] = e1.context(stage="evaluation")["manifest_hash"]
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                e1.run_e1_stage(w, extended, out, **args)
            self.assertEqual(len(w.model.seen), calls)

    def test_four_variant_artifacts_nfes_and_offline_figures(self):
        with TemporaryDirectory() as temp:
            out = Path(temp)
            w, ctx = wrapper(), small_context()
            rows, aggregates = e1.run_e1_stage(w, ctx, out, budgets=(12, 30), max_depth=3, num_candidates=2)
            self.assertEqual(len(rows), 8)
            self.assertEqual(len(aggregates), 8)
            for row in rows:
                with self.subTest(method=row["method"], budget=row["search_budget"]):
                    self.assertLessEqual(row["search_nfes"], row["search_budget"])
                    self.assertEqual(row["tree_nfes"] + row["temporary_candidate_nfes"] + row["adalook_extra_nfes"], row["search_nfes"])
                    self.assertEqual(row["completion_nfes"], 8 - row["selected_depth"])
                    self.assertEqual(row["total_nfes"], row["search_nfes"] + row["completion_nfes"])
                    directory = Path(row["artifact_directory"])
                    tree = json.loads((directory / "tree.json").read_text())
                    self.assertEqual(len(tree["nodes"]), row["tree_nfes"])
                    self.assertEqual((directory / "posteriors.json").exists(), is_adaptive(row["method"]))
                    for node in tree["nodes"]:
                        self.assertGreaterEqual(node["node_id"], 0)
                        self.assertEqual(sum(t != 0 for t in node["state"][PROMPT.shape[1]:]), node["depth"])
                    if row["reward_mode"] == "ig":
                        self.assertEqual(row["adalook_total_extra_nfes"], 0)
            for method in ("medal_adalook", "ab_medal_adalook"):
                self.assertGreater(sum(r["adalook_extra_nfes"] for r in rows if r["method"] == method), 0)
            self.assertGreater(sum(r["temporary_candidate_nfes"] for r in rows if r["method"] == "ab_medal_adalook"), 0)

            calls = len(w.model.seen)
            resumed, _ = e1.run_e1_stage(w, ctx, out, budgets=(12, 30), max_depth=3, num_candidates=2)
            self.assertEqual(json.loads(json.dumps(rows)), resumed)
            self.assertEqual(len(w.model.seen), calls)
            with patch.object(e3, "_save_fig", wraps=e3._save_fig) as save:
                offline, agg = e3.analyze_e3_stage(out / "dev", out / "summaries", budgets=(12, 30))
                self.assertEqual(save.call_count, 7)
                for call in save.call_args_list:
                    fig, path = call.args
                    self.assertTrue(path.with_suffix(".png").is_file())
                    self.assertTrue(path.with_suffix(".pdf").is_file())
                    if path.name == "fig6_accuracy_vs_budget":
                        self.assertEqual(set(fig.axes[0].get_legend_handles_labels()[1]),
                                         {variant(m)["label"] for m in METHODS})
            self.assertEqual(len(w.model.seen), calls)
            self.assertEqual(len(offline), 8)
            self.assertEqual(len(agg), 8)
            lookup = {(r["method"], r["search_budget"]): r for r in rows}
            for record in offline:
                original = lookup[record["method"], record["search_budget"]]
                for key in ("search_nfes", "total_nfes", "tree_nfes", "adalook_total_extra_nfes", "selected_depth"):
                    self.assertEqual(record[key], original[key])
                self.assertEqual(record["gen_count"] is not None, is_adaptive(record["method"]))

    def test_resume_rejects_changed_reward_or_boundaries_before_forwards(self):
        with TemporaryDirectory() as temp:
            out, w, ctx = Path(temp), wrapper(), small_context()
            e1.run_e1_stage(w, ctx, out, methods=("ab_medal_adalook",), budgets=(12, 30), max_depth=3, num_candidates=2)
            calls = len(w.model.seen)
            manifest = (out / "run_manifest.json").read_text()
            ctx["adalook_options"]["adalook_alpha"] = .2
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                e1.run_e1_stage(w, ctx, out, methods=("ab_medal_adalook",), budgets=(12, 30), max_depth=3, num_candidates=2)
            ctx["adalook_options"]["adalook_alpha"] = .1
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                e1.run_e1_stage(w, ctx, out, methods=("ab_medal_adalook",), budgets=(30,), max_depth=3, num_candidates=2)
            self.assertEqual(len(w.model.seen), calls)
            self.assertEqual((out / "run_manifest.json").read_text(), manifest)

    def test_early_exhaustion_records_actual_cost(self):
        with TemporaryDirectory() as temp:
            w, ctx = wrapper(), small_context()
            rows, _ = e1.run_e1_stage(w, ctx, Path(temp), budgets=(12, 30), max_depth=1, num_candidates=2)
            for row in rows:
                self.assertLess(row["search_nfes"], row["search_budget"])
                self.assertEqual(row["total_nfes"], row["search_nfes"] + row["completion_nfes"])

    def test_all_variants_checkpoint_equivalence_with_same_boundaries(self):
        for method in METHODS:
            with self.subTest(method=method):
                w, ctx = wrapper(), small_context()
                item = e1.prepare(w, ctx)[0]
                continuous, _ = e1.execute_search_checkpointed(w, item, ctx, method, (12, 30), 3, 2)
                self.assertIsInstance(continuous, ABMEDALSearch if is_adaptive(method) else MCTSSearch)
                for budget in (12, 30):
                    standalone, _ = e1.execute_search_checkpointed(w, item, ctx, method,
                        tuple(b for b in (12, 30) if b <= budget), 3, 2)
                    a, b = continuous.checkpoints[budget], standalone.checkpoints[budget]
                    for key in ("model_calls", "selected_state_hash", "shortlist_node_ids", "selected_q"):
                        self.assertEqual(a[key], b[key])
                    self.assertEqual([(n.node_id, n.visits, n.reward_sum) for n in a["nodes"]],
                                     [(n.node_id, n.visits, n.reward_sum) for n in b["nodes"]])
                    self.assertEqual(a.get("posteriors"), b.get("posteriors"))

    def test_notebook_defaults_branches_and_python_syntax(self):
        root = Path(__file__).resolve().parents[2]
        for name in ("Experiment_1_Eval", "Experiment_2_Eval", "MEDAL_MCTS_Demo", "AB_MEDAL_MCTS_Colab"):
            nb = json.loads((root / "notebooks/gsm8k" / f"{name}.ipynb").read_text())
            sources = ["".join(c["source"]) for c in nb["cells"]]
            text = "\n".join(sources)
            self.assertIn('BRANCH = "chong"', text)
            self.assertIn('/blob/chong/', text)
            self.assertNotIn('/blob/adi/', text)
            if name in ("Experiment_1_Eval", "Experiment_2_Eval"):
                for cell, source in zip(nb["cells"], sources):
                    if cell["cell_type"] == "code":
                        compile(source, name, "exec")
                        self.assertFalse(cell["outputs"])
                self.assertIn('STAGE = "evaluation"', text)
                for method in METHODS:
                    self.assertIn(f'"{method}"', text)
            if name == "Experiment_2_Eval":
                self.assertIn('EVALUATION_SET = "extended_70"', text)
                self.assertIn('evaluation_set=EVALUATION_SET', text)
                self.assertIn('NUM_SAMPLES = None', text)
                self.assertIn('BATCH_SIZE = 1', text)
                self.assertIn('batch_size=BATCH_SIZE', text)
                self.assertIn('results/adalook_eval_70', text)
                self.assertNotIn('run_e2_stage', text)
                self.assertNotIn('canonical_20', text)
                self.assertNotIn('extended_50', text)

    def test_extended_notebook_controls_and_smoke_directory_reruns(self):
        root = Path(__file__).resolve().parents[2]
        nb = json.loads((root / "notebooks/gsm8k/Experiment_2_Eval.ipynb").read_text())
        controls = next("".join(c["source"]) for c in nb["cells"]
                        if c["cell_type"] == "code" and 'EVALUATION_SET = ' in "".join(c["source"]))
        with TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            scope = {"DRIVE_ROOT": Path(temp)}
            exec(controls, scope)
            self.assertEqual(len(scope["ctx"]["examples"]), 70)
            self.assertEqual(scope["METHODS"], list(METHODS))
            self.assertEqual(scope["BATCH_SIZE"], 1)
            full_dir = scope["OUTPUT_DIR"]
            smoke = controls.replace("NUM_SAMPLES = None", "NUM_SAMPLES = 2")
            exec(smoke, scope)
            exec(smoke, scope)
            self.assertEqual(scope["OUTPUT_DIR"], full_dir / "smoke_2")
            self.assertEqual(len(scope["ctx"]["examples"]), 2)
            exec(controls, scope)
            self.assertEqual(scope["OUTPUT_DIR"], full_dir)


if __name__ == "__main__":
    unittest.main()
