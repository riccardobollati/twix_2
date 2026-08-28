from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.agents.block_artifacts import block_artifacts_node
from src.agents.block_separation import block_separation_node
from viz_blocks import _parse_targets, render_blocks, run_block_separator


class BlockArtifactTests(unittest.TestCase):
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

    @patch("src.agents.block_artifacts.render_blocks")
    @patch("src.agents.block_artifacts.run_block_separator")
    def test_graph_node_targets_document_artifacts_folder(
        self, separator, render
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
            self.assertEqual(result["cost"][0]["phase"], "block_artifacts")

    @patch("src.agents.block_artifacts.render_blocks")
    @patch("src.agents.block_artifacts.run_block_separator")
    def test_graph_node_renders_held_out_test_pdf(self, separator, render) -> None:
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


if __name__ == "__main__":
    unittest.main()
