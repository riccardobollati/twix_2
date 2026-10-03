from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agents.extraction import extraction_node


class ExtractionArtifactTests(unittest.TestCase):
    @patch("src.agents.extraction.render_record_separation")
    @patch("src.agents.extraction.render_blocks")
    @patch("src.agents.extraction.subprocess.run")
    def test_final_test_overlays_use_finished_pipeline_output(
        self, run, render_blocks, render_records
    ) -> None:
        run.return_value = subprocess.CompletedProcess([], 0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_pdf = root / "agent_space" / "test" / "sample.pdf"
            train_pdf = root / "agent_space" / "train" / "sample.pdf"
            template = [{"type": "kv", "node_id": 0}]
            extraction_node({
                "agent_space": str(root / "agent_space"),
                "pdf_path": str(train_pdf),
                "test_pdf_path": str(test_pdf),
                "template": template,
                "doc_name": "sample",
            })

            self.assertEqual(run.call_args.args[0][2], str(test_pdf))
            blocks_path = str(root / "pipeline" / "block_separation.json")
            render_blocks.assert_called_once_with(
                str(test_pdf), blocks_path,
                str(root / "artifacts" / "test_blocks"),
                label="sample · evaluation test",
            )
            render_records.assert_called_once_with(
                str(test_pdf), blocks_path, template,
                str(root / "artifacts" / "test_record_separation"),
                label="sample · evaluation test",
            )


if __name__ == "__main__":
    unittest.main()
