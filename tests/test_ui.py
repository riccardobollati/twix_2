from __future__ import annotations

import json
from io import BytesIO
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse
from unittest.mock import Mock, patch

from UI.app import (
    Handler, Run, RunManager, STAGES, _active_model_assignment_ids, _build_run_command,
    _dedupe_log_lines, _effective_model_assignments, _stage_from_logs,
    _sonar_script, _workflow_payload,
)
from src.manual_ground_truth import ManualGroundTruthError


class UiWorkflowTests(unittest.TestCase):
    def test_ui_exposes_seven_pipeline_stages(self) -> None:
        self.assertEqual([stage[0] for stage in STAGES], [
            "clustering", "template", "blocks", "sonar", "record_separation",
            "extraction", "results",
        ])
        html = (Path(__file__).parents[1] / "UI" / "index.html").read_text()
        self.assertIn('data-stage-panel="clustering"', html)
        self.assertIn('id="cluster-list"', html)
        self.assertIn('id="cluster-image"', html)
        self.assertIn('id="manual-ground-truth"', html)
        self.assertIn('id="ground-truth-form"', html)
        self.assertIn('id="model-assignments"', html)
        self.assertIn("What is running what", html)
        self.assertIn('data-stage-panel="record_separation"', html)
        self.assertIn('id="record-image"', html)
        self.assertIn('data-output-view="records"', html)
        self.assertIn('data-output-view="blocks"', html)
        self.assertIn('data-stage-panel="sonar"', html)
        self.assertIn('id="sonar-page-list"', html)
        self.assertIn('id="sonar-run-existing"', html)
        self.assertIn('id="open-run-folder"', html)

    def test_open_folder_targets_selected_run_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "selected-run"
            root.mkdir()
            run = Run("selected-run", root, "input.pdf", {})
            handler = object.__new__(Handler)
            handler._send_json = Mock()
            with patch("UI.app.sys.platform", "darwin"), patch("UI.app.subprocess.run") as open_command:
                handler._open_run_folder(run)
            self.assertEqual(open_command.call_args.args[0], [
                "open", "-a", "Finder", str(root.resolve()),
            ])
            handler._send_json.assert_called_once_with({"opened": True})

            handler.path = "/api/runs/selected-run/open-folder"
            handler._run_from_path = Mock(return_value=run)
            handler._open_run_folder = Mock()
            handler.do_POST()
            handler._open_run_folder.assert_called_once_with(run)

    def test_older_run_can_backfill_sonar_from_saved_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "input.pdf").write_bytes(b"%PDF-1.4")
            (root / "pipeline").mkdir()
            script = root / "pipeline" / "block_separation.py"
            script.write_text("# saved separator\n", encoding="utf-8")
            (root / "metadata.json").write_text(json.dumps({"source_start_page": 7}))
            run = Run("old-run", root, "input.pdf", {"eval": True, "sonarDropPercent": 15}, status="completed")
            self.assertEqual(_sonar_script(root), script)
            workflow = _workflow_payload(run)
            self.assertTrue(workflow["sonarAvailable"])
            self.assertEqual(workflow["sonarStartPage"], 7)

            request = json.dumps({"thresholdPercent": 15}).encode()
            handler = object.__new__(Handler)
            handler.headers = {"Content-Length": str(len(request))}
            handler.rfile = BytesIO(request)
            handler._send_json = Mock()
            report = {"startPage": 7, "endPage": 12, "totalPages": 12, "pages": [], "flaggedPages": []}

            def complete_scan(command, **_kwargs):
                self.assertEqual(command[command.index("--start") + 1], "7")
                self.assertNotIn("--end", command)
                self.assertEqual(Path(command[command.index("src.excluded_blocks_sonar") + 2]), script)
                destination = root / "artifacts" / "excluded_blocks_sonar"
                destination.mkdir(parents=True)
                (destination / "sonar.json").write_text(json.dumps(report))
                return SimpleNamespace(returncode=0, stderr="")

            with patch("UI.app.subprocess.run", side_effect=complete_scan):
                handler._scan_sonar(run)
            handler._send_json.assert_called_once_with(report)

    def test_run_snapshot_exposes_the_terminal_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Run(
                "run-test", Path(tmp), "input.pdf", {"eval": True},
                command="env/bin/python run.py input.pdf --eval",
            )
            self.assertEqual(
                run.snapshot()["command"],
                "env/bin/python run.py input.pdf --eval",
            )

    def test_ui_command_forwards_routed_models_and_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = Run("run-test", root, "input.pdf", {
                "pages": 5,
                "holdoutPages": 0,
                "eval": True,
                "evalPages": 20,
                "trainThreshold": 0.0,
                "sonarDropPercent": 22,
                "trainMaxRetries": 3,
                "noRouting": False,
                "easyModel": "z-ai/glm-5.3-flash",
                "hardModel": "qwen/qwen3.8-flash",
                "referenceModel": "openai/gpt-5.4",
                "templateModel": "qwen/qwen3.8-flash",
            })

            command = _build_run_command(run, root / "input.pdf")

        self.assertIn("--easy-model", command)
        self.assertIn("z-ai/glm-5.3-flash", command)
        self.assertIn("--hard-model", command)
        self.assertIn("qwen/qwen3.8-flash", command)
        self.assertEqual(
            command[command.index("--reference-model") + 1],
            "openai/gpt-5.4",
        )
        self.assertEqual(command[command.index("--template-model") + 1],
                         "qwen/qwen3.8-flash")
        self.assertEqual(command[command.index("--sonar-drop-percent") + 1], "22")

    def test_ui_single_model_mode_uses_only_easy_model_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = Run("run-test", root, "input.pdf", {
                "pages": 5,
                "holdoutPages": 0,
                "eval": True,
                "evalPages": 20,
                "trainThreshold": 0.0,
                "trainMaxRetries": 3,
                "noRouting": True,
                "singleModel": "qwen/qwen3.8-flash",
                "easyModel": "z-ai/glm-5.3-flash",
                "hardModel": "deepseek/deepseek-v4-pro",
                "referenceModel": "z-ai/glm-5.3-flash",
            })

            command = _build_run_command(run, root / "input.pdf")

        self.assertIn("--no-routing", command)
        self.assertEqual(
            command[command.index("--easy-model") + 1],
            "qwen/qwen3.8-flash",
        )
        self.assertNotIn("--hard-model", command)

    def test_ui_command_can_defer_final_ground_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = Run("run-test", root, "input.pdf", {
                "pages": 5, "holdoutPages": 0, "eval": True,
                "evalPages": 20, "trainThreshold": 0.0,
                "manualGroundTruth": True,
            })
            command = _build_run_command(run, root / "input.pdf")
        self.assertIn("--manual-ground-truth", command)

    def test_ui_rejects_manual_ground_truth_without_test_eval(self) -> None:
        payload = b"%PDF-1.4\n"
        handler = object.__new__(Handler)
        handler.headers = {"Content-Length": str(len(payload)), "X-Filename": "input.pdf"}
        handler.rfile = BytesIO(payload)
        handler._send_json = Mock()
        with patch("UI.app.MANAGER.create") as create:
            handler._create_run(urlparse("/api/runs?eval=false&manualGroundTruth=true"))
        create.assert_not_called()
        self.assertIn("requires independent test evaluation",
                      handler._send_json.call_args.args[0]["error"])

    def test_effective_model_assignments_explain_each_model_role(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Run("run-test", Path(tmp), "input.pdf", {
                "noRouting": False,
                "easyModel": "z-ai/glm-5.3-flash",
                "hardModel": "qwen/qwen3.8-flash",
                "referenceModel": "openai/gpt-5.4",
                "templateModel": "qwen/qwen3.8-flash",
            })
            assignments = _effective_model_assignments(run)

        by_id = {assignment["id"]: assignment for assignment in assignments}
        self.assertEqual(by_id["cluster_sampling"]["stage"], "clustering")
        self.assertEqual(by_id["template_inference"]["model"], "qwen/qwen3.8-flash")
        self.assertEqual(by_id["ground_truth"]["model"], "openai/gpt-5.4")
        self.assertEqual(by_id["block_separation"]["model"], "qwen/qwen3.8-flash")
        self.assertEqual(by_id["record_grouping"]["transport"], "No model call")
        self.assertEqual(by_id["difficulty_routing"]["model"], "qwen/qwen3.8-flash")
        self.assertEqual(by_id["easy_extractors"]["model"], "z-ai/glm-5.3-flash")
        self.assertEqual(by_id["hard_extractors"]["model"], "qwen/qwen3.8-flash")
        self.assertEqual(by_id["portable_extraction"]["transport"], "No model call")
        self.assertIn("difficulty_routing", by_id)

    def test_manual_test_ground_truth_has_separate_model_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metadata.json").write_text(json.dumps({
                "test_eval": {"ground_truth_model": "external-model"},
            }))
            run = Run("run-test", root, "input.pdf", {
                "manualGroundTruth": True,
                "referenceModel": "openai/gpt-5.4",
            })
            assignments = _effective_model_assignments(run)
        by_id = {assignment["id"]: assignment for assignment in assignments}
        self.assertEqual(by_id["ground_truth"]["task"], "Train ground truth")
        self.assertEqual(by_id["manual_ground_truth"]["model"], "external-model")

    def test_single_model_assignment_collapses_extractor_roles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Run("run-test", Path(tmp), "input.pdf", {
                "noRouting": True,
                "singleModel": "qwen/qwen3.8-flash",
                "referenceModel": "openai/gpt-5.4",
            })
            assignments = _effective_model_assignments(run)

        by_id = {assignment["id"]: assignment for assignment in assignments}
        self.assertEqual(by_id["all_extractors"]["model"], "qwen/qwen3.8-flash")
        self.assertEqual(by_id["block_separation"]["model"], "qwen/qwen3.8-flash")
        self.assertNotIn("difficulty_routing", by_id)
        self.assertNotIn("easy_extractors", by_id)

    def test_latest_log_marker_selects_the_active_model_role(self) -> None:
        self.assertEqual(
            _active_model_assignment_ids(
                ["block_difficulty_router: done", "block_extraction_scripting | block 2"],
                "extraction",
                False,
            ),
            ["easy_extractors", "hard_extractors"],
        )
        self.assertEqual(
            _active_model_assignment_ids(["train_repair (attempt 1)"], "extraction", False),
            ["hard_extractors"],
        )

    def test_log_markers_advance_through_seven_stages(self) -> None:
        self.assertEqual(_stage_from_logs([], True), "clustering")
        self.assertEqual(_stage_from_logs(["cluster_sampling: scanning pages"], True), "clustering")
        self.assertEqual(_stage_from_logs([
            "cluster_sampling: selected pages", "environment_creation_node: cropped source pages",
        ], True), "template")
        self.assertEqual(_stage_from_logs(["template_inference_node: done"], True), "template")
        self.assertEqual(_stage_from_logs(["block_artifacts_node: done"], True), "blocks")
        self.assertEqual(_stage_from_logs(["excluded_blocks_sonar_node: scanning pages"], True), "sonar")
        self.assertEqual(
            _stage_from_logs([
                "block_artifacts_node: 2 train image(s)",
                "block_artifacts_node: 2 train record image(s)",
            ], True),
            "record_separation",
        )
        self.assertEqual(_stage_from_logs(["train_eval_node: 0.9"], True), "extraction")
        self.assertEqual(_stage_from_logs(["eval_node: complete"], True), "results")
        self.assertEqual(
            _stage_from_logs(["manual_ground_truth: test extraction ready"], True),
            "results",
        )

    def test_awaiting_run_exposes_manual_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "metadata.json").write_text(json.dumps({
                "manual_ground_truth": {"model": "external-model"},
            }))
            run = Run("run-test", root, "input.pdf", {"manualGroundTruth": True},
                      status="awaiting_ground_truth")
            run.logs = ["manual_ground_truth: test extraction ready"]
            payload = _workflow_payload(run)
        self.assertTrue(payload["manualGroundTruth"]["awaiting"])
        self.assertEqual(payload["manualGroundTruth"]["model"], "external-model")
        self.assertIn("$twix-ground-truth", payload["manualGroundTruth"]["skillPrompt"])
        self.assertEqual(payload["stage"], "results")

    def test_manual_submission_keeps_bad_paste_retryable_then_completes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Run("run-test", Path(tmp), "input.pdf",
                      {"manualGroundTruth": True, "eval": True},
                      status="awaiting_ground_truth")
            manager = object.__new__(RunManager)
            with patch("UI.app.load_manual_ground_truth_inputs",
                       side_effect=ManualGroundTruthError("bad fields")):
                with self.assertRaisesRegex(ManualGroundTruthError, "bad fields"):
                    manager.submit_ground_truth(run, "{}", "external-model")
            self.assertEqual(run.status, "awaiting_ground_truth")
            with (patch("UI.app.load_manual_ground_truth_inputs"),
                  patch("UI.app.finalize_manual_ground_truth",
                        return_value={"accuracy": 1.0}) as finalize,
                  patch("UI.app._generate_run_statistics", return_value={"plots": []})):
                result = manager.submit_ground_truth(run, "{}", "external-model")
            self.assertEqual(result["accuracy"], 1.0)
            self.assertEqual(run.status, "completed")
            finalize.assert_called_once()

    def test_successful_manual_pipeline_waits_without_rendering_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run = Run("run-test", Path(tmp), "input.pdf", {
                "pages": 5, "eval": True, "manualGroundTruth": True,
                "trainThreshold": 0.0,
            })
            manager = object.__new__(RunManager)
            with (patch("UI.app.threading.Thread") as thread,
                  patch("UI.app.subprocess.Popen") as popen,
                  patch("UI.app._generate_run_statistics") as stats):
                thread.return_value.start.side_effect = lambda: thread.call_args.kwargs["target"]()
                popen.return_value.stdout = iter(["manual_ground_truth: test extraction ready\n"])
                popen.return_value.wait.return_value = 0
                manager._start(run, Path(tmp) / "input.pdf")
            self.assertEqual(run.status, "awaiting_ground_truth")
            self.assertIsNone(run.finished_at)
            stats.assert_not_called()

    def test_cluster_groups_are_inspectable_in_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "artifacts" / "clustering"
            folder.mkdir(parents=True)
            (root / "input.pdf").write_bytes(b"%PDF-1.4")
            (folder / "page_0013_features.png").write_bytes(b"png")
            (folder / "kv_page_0009.png").write_bytes(b"png")
            (folder / "selection.json").write_text(json.dumps({
                "cluster_coverage": {"table:2": 13, "kv:1": 1},
                "table_cluster_count": 1, "kv_cluster_count": 1,
            }))
            (folder / "clusters.json").write_text(json.dumps({"algorithms": {
                "balanced_complete": {"clusters": {"2": [12, 13, 14]},
                                      "representatives": {"2": 13}},
            }}))
            (folder / "kv_clusters.json").write_text(json.dumps({"algorithms": {
                "kv_balanced": {"clusters": {"1": [1, 2, 9]},
                                "representatives": {"1": 9}},
            }}))
            run = Run("run-test", root, "input.pdf", {"eval": True})
            run.logs = ["cluster_sampling: selected pages"]

            payload = _workflow_payload(run)

        self.assertEqual(payload["stage"], "clustering")
        by_id = {group["id"]: group for group in payload["clusterGroups"]}
        self.assertEqual(by_id["table:2"]["pages"], [12, 13, 14])
        self.assertEqual(by_id["table:2"]["samplePage"], 13)
        self.assertEqual(by_id["table:2"]["overlay"],
                         "artifacts/clustering/page_0013_features.png")
        self.assertEqual(by_id["kv:1"]["samplePage"], 1)
        self.assertEqual(by_id["kv:1"]["overlay"],
                         "artifacts/clustering/kv_page_0009.png")

    def test_terminal_redraw_completions_are_deduplicated(self) -> None:
        line = "block_extraction_scripting | block 2  ✓  89,617 tokens"
        failure = "block_extraction_scripting | block 3  ✗  10,000 tokens"
        self.assertEqual(
            _dedupe_log_lines([line, line, failure, failure, "other output", line]),
            [line, failure, "other output"],
        )

    def test_workflow_prefers_held_out_pdf_and_block_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agent_space" / "train").mkdir(parents=True)
            (root / "agent_space" / "test").mkdir(parents=True)
            (root / "artifacts" / "test_blocks").mkdir(parents=True)
            (root / "artifacts" / "test_record_separation").mkdir(parents=True)
            (root / "artifacts" / "stats").mkdir(parents=True)
            (root / "artifacts" / "excluded_blocks_sonar").mkdir(parents=True)
            (root / "artifacts" / "excluded_blocks_sonar" / "sonar.json").write_text(json.dumps({
                "averageCoverage": 0.9, "flaggedPages": [2], "pages": [],
            }))
            (root / "pipeline" / "train_eval").mkdir(parents=True)
            (root / "input.pdf").write_bytes(b"%PDF-1.4")
            (root / "agent_space" / "train" / "input.pdf").write_bytes(b"%PDF-1.4")
            (root / "agent_space" / "test" / "input.pdf").write_bytes(b"%PDF-1.4")
            (root / "artifacts" / "test_blocks" / "page_2_blocks.png").write_bytes(b"png")
            (root / "artifacts" / "test_blocks" / "page_1_blocks.png").write_bytes(b"png")
            (root / "artifacts" / "test_record_separation" / "page_2_records.png").write_bytes(b"png")
            (root / "artifacts" / "test_record_separation" / "page_1_records.png").write_bytes(b"png")
            (root / "artifacts" / "stats" / "overview.png").write_bytes(b"png")
            (root / "artifacts" / "stats" / "phase_cost.png").write_bytes(b"png")
            (root / "artifacts" / "stats" / "token_breakdown.png").write_bytes(b"png")
            (root / "artifacts" / "stats" / "stats.json").write_text(json.dumps({
                "test_accuracy": 0.91,
                "block_types": 3,
            }))
            (root / "template.json").write_text(json.dumps({"template": []}))
            (root / "pipeline" / "extracted.json").write_text(json.dumps([{"page": 1}]))
            (root / "pipeline" / "grouped.json").write_text(json.dumps([{"record_id": "0", "data": [{"page": 1}]}]))
            (root / "pipeline" / "eval.json").write_text(json.dumps({"accuracy": 0.91}))
            (root / "pipeline" / "train_eval" / "eval.json").write_text(json.dumps({"accuracy": 0.95}))
            run = Run("run-test", root, "input.pdf", {"eval": True})
            run.logs = ["eval_node: complete"]

            payload = _workflow_payload(run)

            self.assertEqual(payload["stage"], "results")
            self.assertEqual(payload["sonar"]["flaggedPages"], [2])
            self.assertEqual(payload["blockSource"], "test")
            self.assertEqual(payload["blockImages"], [
                "artifacts/test_blocks/page_1_blocks.png",
                "artifacts/test_blocks/page_2_blocks.png",
            ])
            self.assertEqual(payload["recordSource"], "test")
            self.assertEqual(payload["recordImages"], [
                "artifacts/test_record_separation/page_1_records.png",
                "artifacts/test_record_separation/page_2_records.png",
            ])
            self.assertGreater(payload["recordRevision"], 0)
            self.assertGreater(payload["blockRevision"], 0)
            self.assertEqual(payload["testPdf"], "agent_space/test/input.pdf")
            self.assertEqual(payload["grouped"][0]["record_id"], "0")
            self.assertEqual(payload["evaluation"]["accuracy"], 0.91)
            self.assertEqual(payload["trainEvaluation"]["accuracy"], 0.95)
            self.assertEqual(payload["statistics"]["block_types"], 3)
            self.assertTrue(payload["modelAssignments"])
            self.assertEqual(payload["activeModelAssignments"], ["evaluation"])
            self.assertEqual(payload["statsImages"], [
                "artifacts/stats/overview.png",
                "artifacts/stats/phase_cost.png",
                "artifacts/stats/token_breakdown.png",
            ])


if __name__ == "__main__":
    unittest.main()
