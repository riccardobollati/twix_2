from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.resume_run import load_resume_state, train_eval_resume_available
from UI.app import Run, RunManager


class ResumeTests(unittest.TestCase):
    def make_training_bundle(self, run):
        template = {"template": [{"type": "table", "node_id": 7, "fields": ["Name"]}]}
        (run.run_dir / "template.json").write_text(json.dumps(template))
        bundle = run.run_dir / "pipeline"
        (bundle / "extraction_scripts").mkdir(parents=True)
        (bundle / "template.json").write_text(json.dumps(template))
        for name in ("pipeline.py", "record_grouping.py", "block_separation.py", "extraction_scripts/7.py"):
            (bundle / name).write_text("# fixture")
        run.logs = ["collect_extraction_scripts: ready", "train_eval_node: generating train ground truth", "402 insufficient credits"]
        return bundle

    def make_run(self, root):
        root = Path(root)
        config = {"pages": 5, "eval": True, "manualGroundTruth": True, "noRouting": True}
        (root / "ui_run.json").write_text(json.dumps({"filename": "sample.pdf", "config": config}))
        (root / "template.json").write_text(json.dumps({"template": [{"node_id": 7}], "metadata": ["footer"]}))
        plan = {"source_start_page": 1, "source_total_pages": 10,
                "train_source_pages": [1, 3], "test_source_pages": [2, 4],
                "first_source_pages": [1], "cluster_sample_pages": [3],
                "cluster_coverage": {"table:1": 3}, "table_cluster_count": 1, "kv_cluster_count": 0}
        folder = root / "artifacts/clustering"
        folder.mkdir(parents=True)
        (folder / "selection.json").write_text(json.dumps(plan))
        train = root / "agent_space/train"
        train.mkdir(parents=True)
        for path in (root / "sample.pdf", root / "ocr_extraction_full.csv", train / "sample.pdf", train / "ocr_extraction.csv"):
            path.write_text("fixture")
        return Run("run-test", root, "sample.pdf", config, status="failed", return_code=1,
                   finished_at="yesterday", logs=["block_separation: failed"])

    def test_restore_retains_template_and_disjoint_physical_page_mapping(self):
        with tempfile.TemporaryDirectory() as root:
            run = self.make_run(root)
            state = load_resume_state(run.run_dir)
            self.assertEqual(state["template"], [{"node_id": 7}])
            self.assertEqual(state["train_source_pages"], [1, 3])
            self.assertEqual(state["test_source_pages"], [2, 4])
            self.assertTrue(state["manual_ground_truth"])
            plan_path = run.run_dir / "artifacts/clustering/selection.json"
            plan = json.loads(plan_path.read_text())
            plan["test_source_pages"] = [1, 4]
            plan_path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, "split is invalid"):
                load_resume_state(run.run_dir)

    def test_ui_resume_reuses_run_and_clears_terminal_status(self):
        with tempfile.TemporaryDirectory() as root:
            run = self.make_run(root)
            manager = object.__new__(RunManager)
            manager._start = Mock()
            manager.resume(run)
            self.assertEqual(run.status, "queued")
            self.assertIsNone(run.return_code)
            self.assertIsNone(run.finished_at)
            manager._start.assert_called_once_with(run, run.run_dir / "sample.pdf", resume=True)
            self.assertEqual(len(list(run.run_dir.glob("ui_run_before_resume_*.json"))), 1)
            with self.assertRaisesRegex(ValueError, "failed or interrupted"):
                manager.resume(run)

    def test_resume_graph_starts_at_separator_without_prior_model_calls(self):
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        from src.agents.graph import build_graph
        with patch("src.agents.graph.block_separation_node", side_effect=RuntimeError("separator reached")) as separator, \
             patch("src.agents.graph.template_inference_node") as template, \
             patch("src.agents.graph.detect_beginning_node") as beginning:
            graph = build_graph(FakeListChatModel(responses=["unused"]), resume_from_block_separation=True)
            with self.assertRaisesRegex(RuntimeError, "separator reached"):
                graph.invoke({"full_pdf_path": "unused", "full_ocr_csv_path": "unused", "doc_name": "unused", "train_pages": 2, "eval_mode": True})
            separator.assert_called_once()
            template.assert_not_called()
            beginning.assert_not_called()

    def test_ui_training_resume_requires_complete_matching_bundle(self):
        with tempfile.TemporaryDirectory() as root:
            run = self.make_run(root)
            bundle = self.make_training_bundle(run)
            self.assertTrue(train_eval_resume_available(run.run_dir))
            self.assertTrue(run.snapshot()["canResume"])
            self.assertEqual(run.snapshot()["resumePhase"], "train_eval")
            manager = object.__new__(RunManager)
            manager._start = Mock()
            manager.resume(run)
            manager._start.assert_called_once_with(run, run.run_dir / "sample.pdf", resume=True)
            self.assertIn("Resuming from train eval", run.logs[-1])
            run.status = "failed"
            (bundle / "extraction_scripts/7.py").unlink()
            self.assertFalse(run.snapshot()["canResume"])
            with self.assertRaisesRegex(ValueError, "Resume requires"):
                manager.resume(run)

    def test_training_resume_rejects_stale_template_and_wrong_failure_stage(self):
        with tempfile.TemporaryDirectory() as root:
            run = self.make_run(root)
            bundle = self.make_training_bundle(run)
            run.logs = ["block_extraction_scripting: failed"]
            self.assertFalse(run.snapshot()["canResume"])
            (bundle / "template.json").write_text('{"template": []}')
            self.assertFalse(train_eval_resume_available(run.run_dir))

    def test_training_resume_command_selects_training_entry_point(self):
        with tempfile.TemporaryDirectory() as root:
            run = self.make_run(root)
            self.make_training_bundle(run)
            manager = object.__new__(RunManager)
            with patch("UI.app.threading.Thread") as thread:
                manager._start(run, run.run_dir / "sample.pdf", resume=True)
            self.assertTrue(run.command.endswith("--from-phase train_eval"))
            thread.return_value.start.assert_called_once()

    def test_resume_graph_starts_at_training_without_regenerating_scripts(self):
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        from src.agents.graph import build_graph
        with patch("src.agents.graph.train_eval_node", side_effect=RuntimeError("training reached")) as training, \
             patch("src.agents.graph.block_separation_node") as separator, \
             patch("src.agents.graph.block_extraction_scripting_node") as scripting, \
             patch("src.agents.graph.template_inference_node") as template:
            graph = build_graph(FakeListChatModel(responses=["unused"]), resume_from_train_eval=True)
            with self.assertRaisesRegex(RuntimeError, "training reached"):
                graph.invoke({"full_pdf_path": "unused", "full_ocr_csv_path": "unused", "doc_name": "unused", "train_pages": 2, "eval_mode": True})
            training.assert_called_once()
            separator.assert_not_called()
            scripting.assert_not_called()
            template.assert_not_called()


if __name__ == "__main__":
    unittest.main()
