from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from src.agents import block_extraction_scripting


class BlockExtractionScriptingTests(unittest.TestCase):
    def test_limits_parallel_opencode_sessions_to_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            agent_space = output_dir / "agent_space"
            train_dir = agent_space / "train"
            block_samples = train_dir / "block_samples"
            template = [
                {
                    "type": "kv",
                    "fields": [f"Field {node_id}"],
                    "bid": [0],
                    "child": -1,
                    "node_id": node_id,
                    "complexity": {"repeated": False},
                }
                for node_id in range(4)
            ]
            for node_id in range(4):
                (block_samples / str(node_id)).mkdir(parents=True)
            (output_dir / "template.json").write_text(
                json.dumps({"template": template}),
                encoding="utf-8",
            )

            def generate_script(
                _prompt: str,
                sample_dir: str,
                **_kwargs: object,
            ) -> tuple[int, int, int, None]:
                template_id = Path(sample_dir).name
                (Path(sample_dir) / f"{template_id}.py").write_text(
                    "def extract(block):\n    return block\n",
                    encoding="utf-8",
                )
                return 0, 1, 1, None

            with (
                patch.object(
                    block_extraction_scripting,
                    "ThreadPoolExecutor",
                    wraps=ThreadPoolExecutor,
                ) as executor_cls,
                patch.object(
                    block_extraction_scripting,
                    "opencode_silent",
                    side_effect=generate_script,
                ),
            ):
                block_extraction_scripting.block_extraction_scripting_node(
                    {
                        "agent_space": str(agent_space),
                        "pdf_path": str(train_dir / "document.pdf"),
                        "template": template,
                    }
                )

        executor_cls.assert_called_once_with(max_workers=2)


if __name__ == "__main__":
    unittest.main()
