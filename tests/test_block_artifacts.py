from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.agents.block_artifacts import block_artifacts_node
from src.agents.block_separation import block_separation_node
from src.models.opencode import OPENCODE_BUDGET_EXIT_CODE
from viz_blocks import _parse_targets, render_blocks, run_block_separator
from viz_record_separation import _load_record_blocks, render_record_separation


class BlockArtifactTests(unittest.TestCase):
    @patch("src.agents.block_separation._spinner")
    @patch("src.agents.block_separation._run_script", return_value=None)
    def test_separator_budget_failure_reaches_retry_and_usage_is_accumulated(self, run_script, _spinner):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "agent_space" / "train"
            train.mkdir(parents=True)
            prompts = []

            def generate(prompt, *_args, **_kwargs):
                prompts.append(prompt)
                # A file from a truncated response must not count as success.
                (train / "block_separation.py").touch()
                return (OPENCODE_BUDGET_EXIT_CODE, 10, 2, None) if len(prompts) == 1 else (0, 20, 4, None)

            with patch("src.agents.block_separation.opencode_silent", side_effect=generate):
                result = block_separation_node({
                    "template": [], "ocr_csv_path": str(train / "ocr.csv"),
                    "pdf_path": str(train / "input.pdf"), "agent_space": str(root / "agent_space"),
                })
        self.assertEqual(len(prompts), 2)
        self.assertIn("exhausted its response token budget", prompts[1])
        self.assertIn("finish_reason=length", prompts[1])
        self.assertIn("finish writing the required script", prompts[1])
        run_script.assert_called_once()
        self.assertEqual(result["cost"][0]["input_tokens"], 30)
        self.assertEqual(result["cost"][0]["output_tokens"], 6)

    @patch("src.agents.block_separation._spinner")
    @patch("src.agents.block_separation._run_script", return_value=None)
    @patch("src.agents.block_separation.opencode_silent", return_value=(0, 1, 2, 3))
    def test_separator_prefers_hard_model_over_general_opencode_model(
        self, opencode, _run_script, _spinner
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "agent_space" / "train"
            train.mkdir(parents=True)
            (train / "block_separation.py").touch()
            ocr = train / "ocr.csv"
            ocr.touch()

            block_separation_node({
                "template": [],
                "metadata": [],
                "ocr_csv_path": str(ocr),
                "pdf_path": str(train / "input.pdf"),
                "agent_space": str(root / "agent_space"),
                "opencode_model": "openrouter/vendor/general",
                "hard_model": "openrouter/vendor/hard",
            })

        self.assertEqual(opencode.call_args.kwargs["model"], "openrouter/vendor/hard")

    def test_cli_targets_accept_pdf_then_pages_or_legacy_pages(self) -> None:
        self.assertEqual(
            _parse_targets(["/tmp/input.pdf", "2", "4"]),
            ("/tmp/input.pdf", {2, 4}),
        )
        self.assertEqual(_parse_targets(["2", "4"]), (None, {2, 4}))

    def test_runner_applies_generated_separator_to_supplied_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = root / "pipeline"
            work = root / "work"
            pipeline.mkdir()
            pdf_path = root / "input.pdf"
            Image.new("RGB", (200, 200), "white").save(pdf_path, "PDF")
            script_path = pipeline / "block_separation.py"
            script_path.write_text(
                """import json, os, sys
os.makedirs(sys.argv[2], exist_ok=True)
with open(os.path.join(sys.argv[2], 'block_separation.json'), 'w') as target:
    json.dump({'pages': [{'page': 1, 'blocks': []}]}, target)
""",
                encoding="utf-8",
            )

            result_path = run_block_separator(
                str(pdf_path), str(script_path), str(work)
            )

            self.assertEqual(result_path, str(work / "block_separation.json"))
            self.assertTrue((work / "ocr.csv").is_file())
            self.assertEqual(
                json.loads(Path(result_path).read_text(encoding="utf-8")),
                {"pages": [{"page": 1, "blocks": []}]},
            )

    def test_renderer_writes_blocks_under_requested_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_path = root / "train.pdf"
            blocks_path = root / "block_separation.json"
            output_dir = root / "artifacts" / "blocks"
            Image.new("RGB", (200, 200), "white").save(pdf_path, "PDF")
            blocks_path.write_text(
                json.dumps({
                    "pages": [{
                        "page": 1,
                        "blocks": [{
                            "block_id": "0_0",
                            "type": "kv",
                            "words": [{"x0": 10, "top": 10, "x1": 80, "bottom": 40}],
                        }],
                    }],
                }),
                encoding="utf-8",
            )

            written = render_blocks(
                str(pdf_path), str(blocks_path), str(output_dir), label="sample"
            )

            self.assertEqual(written, [str(output_dir / "page_1_blocks.png")])
            self.assertTrue((output_dir / "page_1_blocks.png").is_file())

    def test_record_renderer_groups_blocks_and_writes_page_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pdf_path = root / "train.pdf"
            blocks_path = root / "block_separation.json"
            output_dir = root / "artifacts" / "record_separation"
            Image.new("RGB", (200, 200), "white").save(pdf_path, "PDF")
            blocks_path.write_text(
                json.dumps({
                    "pages": [{
                        "page": 1,
                        "blocks": [
                            {
                                "block_id": "0_0",
                                "type": "kv",
                                "words": [{
                                    "x0": 10, "top": 10,
                                    "x1": 80, "bottom": 35,
                                }],
                            },
                            {
                                "block_id": "1_0",
                                "type": "table",
                                "words": [{
                                    "x0": 10, "top": 40,
                                    "x1": 150, "bottom": 75,
                                }],
                            },
                            {
                                "block_id": "0_1",
                                "type": "kv",
                                "words": [{
                                    "x0": 10, "top": 90,
                                    "x1": 80, "bottom": 115,
                                }],
                            },
                            {
                                "block_id": "1_1",
                                "type": "table",
                                "words": [{
                                    "x0": 10, "top": 120,
                                    "x1": 150, "bottom": 160,
                                }],
                            },
                        ],
                    }],
                }),
                encoding="utf-8",
            )
            template = [
                {
                    "node_id": 0, "type": "kv", "bid": [0],
                    "complexity": {"repeated": False},
                },
                {
                    "node_id": 1, "type": "table", "bid": [1],
                    "complexity": {
                        "same_row_structure": True, "nested": False,
                    },
                },
            ]

            records = _load_record_blocks(str(blocks_path), template)
            written = render_record_separation(
                str(pdf_path),
                str(blocks_path),
                template,
                str(output_dir),
                label="sample",
            )

            self.assertEqual(len(records), 2)
            self.assertEqual(
                [[b["block_id"] for b in record["data"]] for record in records],
                [["0_0", "1_0"], ["0_1", "1_1"]],
            )
            self.assertEqual(
                written,
                [str(output_dir / "page_1_records.png")],
            )
            self.assertTrue((output_dir / "page_1_records.png").is_file())

    @patch("src.agents.block_artifacts.render_record_separation")
    @patch("src.agents.block_artifacts.render_blocks")
    @patch("src.agents.block_artifacts.run_block_separator")
    def test_graph_node_targets_document_artifacts_folder(
        self, separator, render, render_records
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            document = Path(tmp) / "id_05"
            train = document / "agent_space" / "train"
            train.mkdir(parents=True)
            pdf_path = train / "id_05.pdf"
            pdf_path.touch()
            generated_json = document / "generated" / "block_separation.json"
            separator.return_value = str(generated_json)
            render.return_value = [str(document / "artifacts" / "blocks" / "page_1_blocks.png")]
            render_records.return_value = [
                str(
                    document / "artifacts" / "record_separation"
                    / "page_1_records.png"
                )
            ]

            result = block_artifacts_node({
                "pdf_path": str(pdf_path),
                "agent_space": str(document / "agent_space"),
                "doc_name": "id_05",
            })

            separator.assert_called_once()
            separator_args = separator.call_args.args
            self.assertEqual(separator_args[0], str(pdf_path))
            self.assertEqual(separator_args[1], str(train / "block_separation.py"))
            self.assertTrue(Path(separator_args[2]).name.startswith("block_artifacts_"))
            self.assertFalse(Path(separator_args[2]).exists())
            render.assert_called_once_with(
                str(pdf_path),
                str(generated_json),
                str(document / "artifacts" / "blocks"),
                label="id_05",
            )
            render_records.assert_called_once_with(
                str(pdf_path),
                str(generated_json),
                [],
                str(document / "artifacts" / "record_separation"),
                label="id_05",
            )
            self.assertEqual(result["cost"][0]["phase"], "block_artifacts")

    @patch("src.agents.block_artifacts.render_record_separation")
    @patch("src.agents.block_artifacts.render_blocks")
    @patch("src.agents.block_artifacts.run_block_separator")
    def test_graph_node_renders_held_out_test_pdf(
        self, separator, render, render_records
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            document = Path(tmp) / "id_05"
            train = document / "agent_space" / "train"
            test = document / "agent_space" / "test"
            train.mkdir(parents=True)
            test.mkdir(parents=True)
            train_pdf = train / "id_05.pdf"
            test_pdf = test / "id_05.pdf"
            train_pdf.touch()
            test_pdf.touch()
            separator.side_effect = ["/tmp/train.json", "/tmp/test.json"]
            render.side_effect = [["train.png"], ["test-1.png", "test-2.png"]]
            render_records.side_effect = [
                ["train-record.png"],
                ["test-record-1.png", "test-record-2.png"],
            ]

            block_artifacts_node({
                "pdf_path": str(train_pdf),
                "test_pdf_path": str(test_pdf),
                "agent_space": str(document / "agent_space"),
                "doc_name": "id_05",
            })

            self.assertEqual(separator.call_count, 2)
            self.assertEqual(separator.call_args_list[1].args[:2], (
                str(test_pdf), str(train / "block_separation.py")
            ))
            self.assertEqual(render.call_args_list[1].args, (
                str(test_pdf), "/tmp/test.json",
                str(document / "artifacts" / "test_blocks"),
            ))
            self.assertEqual(
                render.call_args_list[1].kwargs["label"],
                "id_05 · evaluation test",
            )
            self.assertEqual(render_records.call_count, 2)
            self.assertEqual(render_records.call_args_list[1].args, (
                str(test_pdf),
                "/tmp/test.json",
                [],
                str(document / "artifacts" / "test_record_separation"),
            ))
            self.assertEqual(
                render_records.call_args_list[1].kwargs["label"],
                "id_05 · evaluation test",
            )


if __name__ == "__main__":
    unittest.main()
