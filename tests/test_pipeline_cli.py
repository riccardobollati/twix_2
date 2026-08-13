from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


class StandalonePipelineTests(unittest.TestCase):
    def test_generated_pipeline_runs_in_a_fresh_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "bundle"
            output = root / "result"
            extractors = bundle / "extraction_scripts"
            extractors.mkdir(parents=True)
            shutil.copy2(Path(__file__).parents[1] / "src/pipeline.py", bundle / "pipeline.py")

            (bundle / "record_separation.py").write_text(
                """import json, os, sys
os.makedirs(sys.argv[2], exist_ok=True)
with open(os.path.join(sys.argv[2], 'record_separation.json'), 'w') as f:
    json.dump({'records': [{'record_id': '0', 'pages': [
        {'page': 1, 'y_start': 0, 'y_end': 1000}
    ]}]}, f)
""",
                encoding="utf-8",
            )
            (bundle / "block_separation.py").write_text(
                """import json, os, sys
os.makedirs(sys.argv[2], exist_ok=True)
with open(os.path.join(sys.argv[2], 'block_separation.json'), 'w') as f:
    json.dump({'pages': [{'page': 1, 'blocks': [
        {'uid': 'u1', 'block_id': '0_0', 'words': []}
    ]}]}, f)
""",
                encoding="utf-8",
            )
            (extractors / "0.py").write_text(
                "def extract(block):\n"
                "    return {'type': 'kv', 'block_id': block['block_id'], "
                "'data': {'value': 'ok'}}\n",
                encoding="utf-8",
            )

            pdf = root / "input.pdf"
            Image.new("RGB", (100, 100), "white").save(pdf, "PDF")

            result = subprocess.run(
                [
                    sys.executable,
                    str(bundle / "pipeline.py"),
                    str(pdf),
                    "--out-dir",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            grouped = json.loads((output / "grouped.json").read_text(encoding="utf-8"))
            health = json.loads((output / "pipeline_health.json").read_text(encoding="utf-8"))
            self.assertEqual(grouped[0]["data"][0]["data"], {"value": "ok"})
            self.assertFalse(health["degenerate"])


if __name__ == "__main__":
    unittest.main()
