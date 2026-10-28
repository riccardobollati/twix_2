from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from src.agents.document_extraction import (
    document_extraction_scripting_node, finalize_output, run_extractor,
)
from src.agents.graph import build_graph
from src.eval.score import score


TEMPLATE = [
    {"type": "kv", "node_id": 7, "fields": ["Width"], "bid": [0]},
    {"type": "kv", "node_id": 9, "fields": ["Width", "Extra"], "bid": [0]},
    {"type": "xor-kv", "node_id": 10, "includes": [7, 9]},
]
SCRIPT = '''import argparse, json
from pathlib import Path
import pdfplumber
p = argparse.ArgumentParser()
p.add_argument("pdf")
p.add_argument("--out-dir", required=True)
a = p.parse_args()
with pdfplumber.open(a.pdf) as pdf:
    blocks = [{"page": i, "type": "kv", "block_id": "7_0001", "uid": f"p{i}_7_0001",
               "data": {"Width": str(int(page.width))}} for i, page in enumerate(pdf.pages, 1)]
out = Path(a.out_dir)
out.mkdir(parents=True, exist_ok=True)
(out / "extracted.json").write_text(json.dumps(blocks))
'''


class DocumentExtractionTests(unittest.TestCase):
    def make_state(self, root):
        source = root / "source.pdf"
        images = [Image.new("RGB", (width, 100), "white") for width in (100, 200, 300)]
        images[0].save(source, "PDF", save_all=True, append_images=images[1:])
        ocr = root / "ocr.csv"
        ocr.write_text("id,page,text,x0,top,x1,bottom\n1,1,train-first,1,1,2,2\n2,2,secret-test,1,1,2,2\n3,3,train-last,1,1,2,2\n")
        return {"full_pdf_path": str(source), "full_ocr_csv_path": str(ocr),
                "output_dir": str(root), "doc_name": "sample", "train_pages": 2,
                "detected_start_page": 1, "eval_mode": True,
                "opencode_model": "openrouter/test/chosen-model"}

    def test_full_graph_uses_one_coding_task_then_scores_unseen_pages(self):
        for evaluation, manual in ((True, False), (True, True), (False, False)):
            with self.subTest(evaluation=evaluation, manual=manual), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                state = self.make_state(root)
                state.update(eval_mode=evaluation, manual_ground_truth=manual)
                def generate(prompt, folder, **kwargs):
                    self.assertEqual(kwargs["model"], state["opencode_model"])
                    self.assertIn('"xor-kv"', prompt)
                    self.assertIn("train-first", prompt)
                    self.assertIn("train-last", prompt)
                    self.assertNotIn("secret-test", prompt)
                    self.assertEqual({p.name for p in Path(folder).iterdir()},
                                     {"training.pdf", "training_ocr.csv", "training_layout.txt", "template.json"})
                    self.assertNotIn("secret-test", (Path(folder) / "training_ocr.csv").read_text())
                    (Path(folder) / "extract.py").write_text(SCRIPT)
                    return 0, 100, 20, None
                plan = {"train_source_pages": [1, 3], "test_source_pages": [2] if evaluation else [],
                        "first_source_pages": [1], "cluster_sample_pages": [3]}
                truth = [{"page": 1, "type": "kv", "block_id": "7", "data": {"Width": "200"}}]
                with patch("src.agents.graph.cluster_sampling_node", return_value=plan), \
                     patch("src.agents.graph.template_inference_node", return_value={"template": TEMPLATE, "metadata": []}), \
                     patch("src.agents.document_extraction.opencode_silent", side_effect=generate) as coding, \
                     patch("src.agents.ground_truth.generate_ground_truth", return_value=(truth, {"input_tokens": 1, "output_tokens": 2, "total_tokens": 0})) as reference:
                    graph = build_graph(Mock(model_name="test/primary"), ground_truth_llm=Mock(model_name="test/reference"))
                    node_names = set(graph.get_graph().nodes)
                    self.assertFalse(node_names & {"block_separation", "excluded_blocks_sonar", "block_samples",
                                                  "train_eval", "opencode_fix", "collect_extraction_scripts"})
                    graph.invoke(state)
                coding.assert_called_once()
                pipeline = root / "pipeline"
                output = json.loads((pipeline / "extracted.json").read_text())
                self.assertEqual([b["data"]["Width"] for b in output], ["200"] if evaluation else ["100", "300"])
                self.assertFalse((pipeline / "pipeline.py").exists())
                self.assertFalse((pipeline / "block_separation.py").exists())
                self.assertFalse((pipeline / "train_eval").exists())
                if evaluation and not manual:
                    self.assertEqual(json.loads((pipeline / "eval.json").read_text())["accuracy"], 1.0)
                    reference.assert_called_once()
                else:
                    reference.assert_not_called()
                    self.assertFalse((pipeline / "eval.json").exists())
                metadata = json.loads((root / "metadata.json").read_text())
                self.assertEqual(metadata["pipeline_variant"], "no_decomposition")
                self.assertFalse(metadata["task_decomposition"])
                self.assertEqual(metadata["pipeline_phases"][0]["input_tokens"], 100)
                self.assertTrue((root / "metadata/cost.json").is_file())

    def test_contract_retries_remove_stale_code_and_accumulate_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = self.make_state(root)
            state.update(agent_space=str(root / "agent_space"), pdf_path=state["full_pdf_path"],
                         ocr_csv_path=state["full_ocr_csv_path"], actual_train_pages=3, template=TEMPLATE)
            def generate(prompt, folder, **kwargs):
                script = Path(folder) / "extract.py"
                self.assertFalse(script.exists())
                if coding.call_count == 1:
                    script.write_text("broken syntax !")
                    return 0, 5, 3, None
                self.assertIn("Previous attempt failed", prompt)
                script.write_text(SCRIPT)
                return 0, 7, 4, None
            with patch("src.agents.document_extraction.opencode_silent", side_effect=generate) as coding:
                result = document_extraction_scripting_node(state)
            self.assertEqual(coding.call_count, 2)
            self.assertEqual(result["cost"][0]["input_tokens"], 12)
            self.assertEqual(result["cost"][0]["output_tokens"], 7)

    def test_invalid_output_contract_is_rejected_and_empty_is_scored(self):
        base = {"page": 1, "type": "kv", "block_id": "7_0001", "uid": "p1-7", "data": {"Width": "100"}}
        invalid = [
            {}, [{**base, "page": 0}], [{**base, "page": 3}],
            [{**base, "block_id": "10_0001", "type": "xor-kv"}],
            [{**base, "data": {"Renamed": "100"}}], [base, base],
            [{**base, "data": {"Width": 100}}], [{**base, "uid": ""}],
            [{**base, "type": "table"}],
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for blocks in invalid:
                with self.subTest(blocks=blocks):
                    (output / "extracted.json").write_text(json.dumps(blocks))
                    with self.assertRaises(ValueError):
                        finalize_output(output, TEMPLATE, 2)
            (output / "extracted.json").write_text("[]")
            finalize_output(output, TEMPLATE, 2)
            self.assertTrue(json.loads((output / "pipeline_health.json").read_text())["degenerate"])
            self.assertEqual(score([base], [], reference="test")["accuracy"], 0.0)

    def test_zero_exit_cannot_reuse_stale_extracted_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = self.make_state(root)
            (root / "extracted.json").write_text("[]")
            with patch("src.agents.document_extraction.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "", "")):
                with self.assertRaises(FileNotFoundError):
                    run_extractor(root / "extract.py", Path(state["full_pdf_path"]), root, TEMPLATE)


if __name__ == "__main__":
    unittest.main()
