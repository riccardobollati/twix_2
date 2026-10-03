from __future__ import annotations

import json
from contextlib import ExitStack
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pypdfium2 as pdfium

from src.agents.graph import _route_after_extraction, build_graph
from src.agents.save_metadata import save_metadata_node
from src.agents.template_inference import ExtractionState
from src.manual_ground_truth import (
    ManualGroundTruthError,
    finalize_manual_ground_truth,
    parse_manual_ground_truth,
)
from src.ground_truth_gen.prompt import build_prompt


TEMPLATE = [
    {
        "node_id": 0,
        "type": "kv",
        "fields": ["Name", "Amount"],
        "bid": [0],
        "child": -1,
        "complexity": {"same_row_structure": True, "nested": False},
    }
]
PREPARE_SCRIPT = (
    Path(__file__).parents[1]
    / ".agents"
    / "skills"
    / "twix-ground-truth"
    / "scripts"
    / "prepare_ground_truth.py"
)


class ManualGroundTruthTests(unittest.TestCase):
    def test_xor_wrapper_is_not_a_ground_truth_block(self) -> None:
        template = TEMPLATE + [
            {"type": "xor-table", "node_id": 1, "includes": [2, 3]},
            {"type": "table", "node_id": 2, "fields": ["Width"], "bid": [1], "complexity": {}},
            {"type": "table", "node_id": 3, "fields": ["Diameter"], "bid": [1], "complexity": {}},
        ]
        prompt = build_prompt(template)
        self.assertIn('"xor-table"', prompt)
        self.assertIn("Never emit an XOR node as a block", prompt)
        self.assertIn('"Width"', prompt)
        raw = json.dumps({"blocks": [{
            "page": 1, "type": "xor-table", "block_id": "1", "data": {},
        }]})
        with self.assertRaisesRegex(ManualGroundTruthError, "not in the inferred template"):
            parse_manual_ground_truth(raw, template, page_count=1)

    def test_parse_accepts_exact_generator_shape(self) -> None:
        raw = json.dumps({
            "blocks": [{
                "page": 1,
                "type": "kv",
                "block_id": "0",
                "data": {"Name": "Example", "Amount": "12.00"},
            }]
        })

        blocks = parse_manual_ground_truth(raw, TEMPLATE, page_count=2)

        self.assertEqual(blocks[0]["page"], 1)
        self.assertEqual(blocks[0]["block_id"], "0")

    def test_parse_rejects_fields_that_do_not_match_template(self) -> None:
        raw = json.dumps({
            "blocks": [{
                "page": 1,
                "type": "kv",
                "block_id": "0",
                "data": {"Name": "Example", "Wrong": "12.00"},
            }]
        })

        with self.assertRaisesRegex(ManualGroundTruthError, "exactly match"):
            parse_manual_ground_truth(raw, TEMPLATE, page_count=1)

    def test_parse_rejects_pages_outside_held_out_pdf(self) -> None:
        raw = json.dumps({
            "blocks": [{
                "page": 3,
                "type": "kv",
                "block_id": "0",
                "data": {"Name": "Example", "Amount": "12.00"},
            }]
        })

        with self.assertRaisesRegex(ManualGroundTruthError, "outside"):
            parse_manual_ground_truth(raw, TEMPLATE, page_count=2)

    def test_finalize_scores_and_logs_manual_model(self) -> None:
        blocks = [{
            "page": 1,
            "type": "kv",
            "block_id": "0",
            "data": {"Name": "Example", "Amount": "12.00"},
        }]
        raw = json.dumps({"blocks": blocks})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = root / "pipeline"
            pipeline.mkdir()
            grouped = [{"record_id": "0", "data": blocks}]
            (pipeline / "grouped.json").write_text(json.dumps(grouped))
            (root / "metadata.json").write_text(json.dumps({
                "test_eval": {
                    "status": "awaiting_manual_ground_truth",
                    "agentic_pipeline_cost": {"phase": "extraction", "duration_s": 1.0},
                },
                "totals": {"duration_s": 2.0},
            }))

            with patch(
                "src.manual_ground_truth.load_manual_ground_truth_inputs",
                return_value=(blocks, TEMPLATE),
            ):
                evaluation = finalize_manual_ground_truth(root, raw, "gpt-6-astra")

            metadata = json.loads((root / "metadata.json").read_text())
            self.assertEqual(evaluation["accuracy"], 1.0)
            self.assertEqual(
                metadata["test_eval"]["ground_truth_model"], "gpt-6-astra"
            )
            self.assertEqual(
                metadata["test_eval"]["ground_truth_source"],
                "manual_ui_submission",
            )
            self.assertEqual(metadata["manual_ground_truth"]["status"], "submitted")
            self.assertTrue((pipeline / "ground_truth_grouped.json").is_file())
            self.assertTrue((pipeline / "eval.txt").is_file())

    def test_metadata_marks_manual_ground_truth_as_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_space = root / "agent_space"
            (root / "pipeline").mkdir()
            agent_space.mkdir()

            save_metadata_node({
                "agent_space": str(agent_space),
                "full_pdf_path": str(root / "input.pdf"),
                "doc_name": "sample",
                "actual_train_pages": 1,
                "eval_mode": True,
                "manual_ground_truth": True,
                "test_start": 2,
                "test_end": 2,
                "cost": [{"phase": "extraction", "duration_s": 1.25}],
            })

            metadata = json.loads((root / "metadata.json").read_text())
            self.assertEqual(
                metadata["test_eval"]["status"],
                "awaiting_manual_ground_truth",
            )
            self.assertEqual(
                metadata["test_eval"]["agentic_pipeline_cost"]["duration_s"],
                1.25,
            )
            self.assertTrue(metadata["run_config"]["manual_ground_truth"])

    def test_graph_defers_only_final_ground_truth(self) -> None:
        self.assertIn("manual_ground_truth", ExtractionState.__annotations__)
        self.assertEqual(
            _route_after_extraction({
                "eval_mode": True,
                "manual_ground_truth": True,
            }),
            "save_metadata",
        )
        self.assertEqual(
            _route_after_extraction({"eval_mode": True}),
            "ground_truth",
        )

    def test_compiled_graph_retains_manual_flag_and_skips_final_model(self) -> None:
        seen = {}
        passthrough = lambda *_args: {"cost": []}
        nodes = (
            "cluster_sampling_node", "environment_creation_node",
            "template_inference_node", "block_separation_node",
            "run_block_separation_node", "sonar_node", "block_artifacts_node",
            "block_samples_node", "block_difficulty_router_node",
            "block_extraction_scripting_node", "collect_extraction_scripts_node",
            "train_eval_node", "extraction_node",
        )
        with ExitStack() as stack:
            for name in nodes:
                stack.enter_context(patch(f"src.agents.graph.{name}", new=passthrough))
            stack.enter_context(patch("src.agents.graph._route_after_train_eval",
                                      return_value="extraction"))
            ground_truth = stack.enter_context(patch("src.agents.graph.ground_truth_node"))
            stack.enter_context(patch("src.agents.graph.save_metadata_node",
                                      side_effect=lambda state: seen.update(state) or {"cost": []}))
            graph = build_graph(Mock())
            graph.invoke({
                "full_pdf_path": "unused.pdf", "full_ocr_csv_path": "unused.csv",
                "doc_name": "example", "train_pages": 3, "eval_mode": True,
                "detected_start_page": 1, "manual_ground_truth": True,
            })
        self.assertTrue(seen["manual_ground_truth"])
        ground_truth.assert_not_called()

    def test_skill_helper_uses_exact_prompt_and_inferred_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_dir = root / "agent_space" / "test"
            test_dir.mkdir(parents=True)
            (root / "ui_run.json").write_text(json.dumps({
                "status": "awaiting_ground_truth",
            }))
            (root / "template.json").write_text(json.dumps({
                "template": TEMPLATE,
                "metadata": [],
            }))
            document = pdfium.PdfDocument.new()
            document.new_page(width=300, height=400)
            document.save(test_dir / "sample.pdf")
            (test_dir / "ocr_extraction.csv").write_text(
                "id,page,x0,top,x1,bottom,text\n"
                "0,1,10,10,40,20,ExactOCR\n"
            )

            result = subprocess.run(
                [sys.executable, "-B", str(PREPARE_SCRIPT), str(root)],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            prepared = root / "manual_ground_truth_input"
            self.assertEqual(
                (prepared / "system_prompt.txt").read_text(),
                build_prompt(TEMPLATE, include_images=True),
            )
            self.assertEqual(
                (prepared / "pages" / "page_0001_ocr.txt").read_text(),
                "ExactOCR",
            )
            self.assertTrue((prepared / "pages" / "page_0001.png").is_file())


if __name__ == "__main__":
    unittest.main()
