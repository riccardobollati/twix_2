from __future__ import annotations

import argparse
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from run import _api_key_preview, _nonnegative_int, _positive_int, _probability, resolve_document
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
