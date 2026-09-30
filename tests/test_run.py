from __future__ import annotations

import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from run import (
    _api_key_preview, _nonnegative_int, _positive_int, _probability, main,
    resolve_document,
)
from src.run_batch import (
    JobState,
    TerminalDashboard,
    build_command,
    build_forwarded_args,
    parse_args,
)


class ResolveDocumentTests(unittest.TestCase):
    def test_direct_pdf_uses_explicit_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "report.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")

            resolved, output, name = resolve_document(str(pdf), str(root / "artifacts"))

            self.assertEqual(resolved, pdf.resolve())
            self.assertEqual(output, (root / "artifacts").resolve())
            self.assertEqual(name, "report")

    def test_legacy_benchmark_name_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                pdf = Path("data/example/example.pdf")
                pdf.parent.mkdir(parents=True)
                pdf.write_bytes(b"%PDF-1.4\n")

                resolved, output, name = resolve_document("example", None)

                self.assertEqual(resolved, pdf.resolve())
                self.assertEqual(output, pdf.parent.resolve())
                self.assertEqual(name, "example")
            finally:
                os.chdir(previous)

    def test_missing_input_has_actionable_error(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "Pass a PDF path"):
            resolve_document("does-not-exist", None)

    def test_numeric_options_are_validated(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            _positive_int("0")
        with self.assertRaises(argparse.ArgumentTypeError):
            _nonnegative_int("-1")
        with self.assertRaises(argparse.ArgumentTypeError):
            _probability("1.1")

    def test_batch_command_uses_pdf_path_and_isolated_output(self) -> None:
        job = JobState(
            name="example",
            folder=Path("/datasets/example"),
            pdf_path=Path("/datasets/example/example.pdf"),
        )
        args = SimpleNamespace(
            python="python",
            run_script=Path("/repo/run.py"),
        )

        command = build_command(args, ["--eval"], job)

        self.assertIn("/datasets/example/example.pdf", command)
        self.assertEqual(command[-2:], ["--out-dir", "/datasets/example"])

    def test_batch_forwards_no_routing(self) -> None:
        args, unknown = parse_args(["id_60", "--no-routing"])

        forwarded = build_forwarded_args(args, unknown)

        self.assertIn("--no-routing", forwarded)

    def test_batch_forwards_sonar_drop_percent(self) -> None:
        args, unknown = parse_args(["id_60", "--sonar-drop-percent", "22"])
        forwarded = build_forwarded_args(args, unknown)
        self.assertEqual(forwarded[forwarded.index("--sonar-drop-percent") + 1], "22.0")

    def test_batch_forwards_no_eval(self) -> None:
        args, unknown = parse_args(["id_60", "--no-eval"])

        forwarded = build_forwarded_args(args, unknown)

        self.assertIn("--no-eval", forwarded)

    def test_batch_forwards_manual_ground_truth(self) -> None:
        args, unknown = parse_args(["id_60", "--manual-ground-truth"])
        self.assertIn("--manual-ground-truth", build_forwarded_args(args, unknown))

    def test_batch_forwards_model_overrides(self) -> None:
        args, unknown = parse_args([
            "id_60",
            "--easy-model", "z-ai/glm-5.3-flash",
            "--hard-model", "qwen/qwen3.8-flash",
            "--reference-model", "openai/gpt-5.4",
            "--template-model", "deepseek/deepseek-v4.1-flash",
        ])

        forwarded = build_forwarded_args(args, unknown)

        self.assertEqual(
            forwarded,
            [
                "--easy-model", "z-ai/glm-5.3-flash",
                "--hard-model", "qwen/qwen3.8-flash",
                "--reference-model", "openai/gpt-5.4",
                "--template-model", "deepseek/deepseek-v4.1-flash",
            ],
        )

    def test_pipeline_defaults_to_test_eval_without_train_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "input.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            graph = Mock()
            with (
                patch.object(sys, "argv", [
                    "run.py", str(pdf), "--out-dir", str(root / "output"),
                ]),
                patch("run.extract_ocr_full"),
                patch("run.get_llm", return_value=Mock()),
                patch("run.get_template_inference_llm", return_value=Mock()),
                patch("run.get_ground_truth_llm", return_value=Mock()),
                patch("run.build_graph", return_value=graph),
            ):
                main()

            state = graph.invoke.call_args.args[0]
            self.assertTrue(state["eval_mode"])
            self.assertEqual(state["eval_pages"], 20)
            self.assertEqual(state["holdout_pages"], 0)

    def test_manual_ground_truth_flag_is_forwarded_to_graph_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "input.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            graph = Mock()
            with (
                patch.object(sys, "argv", [
                    "run.py", str(pdf), "--out-dir", str(root / "output"),
                    "--manual-ground-truth",
                ]),
                patch("run.extract_ocr_full"),
                patch("run.get_llm", return_value=Mock()),
                patch("run.get_template_inference_llm", return_value=Mock()),
                patch("run.get_ground_truth_llm", return_value=Mock()),
                patch("run.build_graph", return_value=graph),
            ):
                main()
            self.assertTrue(graph.invoke.call_args.args[0]["manual_ground_truth"])

    def test_manual_ground_truth_requires_evaluation(self) -> None:
        with patch.object(sys, "argv", [
            "run.py", "sample.pdf", "--no-eval", "--manual-ground-truth",
        ]), self.assertRaises(SystemExit) as raised:
            main()
        self.assertEqual(raised.exception.code, 2)

    def test_single_model_mode_uses_one_operational_model_and_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "input.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            graph = Mock()
            primary = Mock()
            template_reference = Mock()
            ground_truth_reference = Mock()
            with (
                patch.object(sys, "argv", [
                    "run.py", str(pdf), "--out-dir", str(root / "output"),
                    "--no-eval", "--no-routing",
                    "--easy-model", "z-ai/glm-5.3-flash",
                    "--hard-model", "deepseek/deepseek-v4-pro",
                    "--reference-model", "qwen/qwen3.8-flash",
                    "--template-model", "z-ai/glm-5.3-flash",
                ]),
                patch("run.extract_ocr_full"),
                patch("run.get_llm", return_value=primary) as primary_mock,
                patch(
                    "run.get_template_inference_llm",
                    return_value=template_reference,
                ) as template_mock,
                patch(
                    "run.get_ground_truth_llm",
                    return_value=ground_truth_reference,
                ) as ground_truth_mock,
                patch("run.build_graph", return_value=graph) as build_mock,
            ):
                main()

            primary_mock.assert_called_once_with(model="z-ai/glm-5.3-flash")
            template_mock.assert_called_once_with(model="z-ai/glm-5.3-flash")
            ground_truth_mock.assert_called_once_with(model="qwen/qwen3.8-flash")
            self.assertIs(
                build_mock.call_args.kwargs["template_inference_llm"],
                template_reference,
            )
            self.assertIs(
                build_mock.call_args.kwargs["ground_truth_llm"],
                ground_truth_reference,
            )
            self.assertIs(
                build_mock.call_args.kwargs["difficulty_routing_llm"],
                primary,
            )
            state = graph.invoke.call_args.args[0]
            self.assertEqual(
                state["opencode_model"],
                "openrouter/z-ai/glm-5.3-flash",
            )
            self.assertEqual(state["easy_model"], state["opencode_model"])
            self.assertEqual(state["hard_model"], state["opencode_model"])
            self.assertEqual(state["reference_model"], "qwen/qwen3.8-flash")
            self.assertEqual(state["template_model"], "z-ai/glm-5.3-flash")
            self.assertTrue(state["no_routing"])

    def test_routed_mode_uses_hard_model_for_separator_and_difficulty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf = root / "input.pdf"
            pdf.write_bytes(b"%PDF-1.4\n")
            graph = Mock()
            primary = Mock()
            difficulty = Mock()
            with (
                patch.object(sys, "argv", [
                    "run.py", str(pdf), "--out-dir", str(root / "output"),
                    "--no-eval",
                    "--easy-model", "z-ai/glm-5.3-flash",
                    "--hard-model", "qwen/qwen3.8-flash",
                ]),
                patch("run.extract_ocr_full"),
                patch(
                    "run.get_llm", side_effect=[primary, difficulty]
                ) as llm_mock,
                patch("run.get_template_inference_llm", return_value=Mock()),
                patch("run.get_ground_truth_llm", return_value=Mock()),
                patch("run.build_graph", return_value=graph) as build_mock,
            ):
                main()

            self.assertEqual(
                llm_mock.call_args_list,
                [call(), call(model="qwen/qwen3.8-flash")],
            )
            self.assertIs(
                build_mock.call_args.kwargs["difficulty_routing_llm"],
                difficulty,
            )
            state = graph.invoke.call_args.args[0]
            self.assertEqual(state["easy_model"], "openrouter/z-ai/glm-5.3-flash")
            self.assertEqual(state["hard_model"], "openrouter/qwen/qwen3.8-flash")
            self.assertEqual(state["opencode_model"], state["hard_model"])

    def test_api_key_preview_uses_active_provider(self) -> None:
        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "1234567890secret"},
            clear=True,
        ):
            self.assertEqual(_api_key_preview(), "1234567890")

    def test_api_key_preview_handles_missing_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_api_key_preview(), "<not set>")

    def test_dashboard_displays_captured_key_preview(self) -> None:
        job = JobState(
            name="example",
            folder=Path("/datasets/example"),
            pdf_path=Path("/datasets/example/example.pdf"),
            key_preview="1234567890",
        )

        screen = TerminalDashboard([job])._build_screen()

        self.assertIn("using kwy: 1234567890", screen)


if __name__ == "__main__":
    unittest.main()
