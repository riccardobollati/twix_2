from __future__ import annotations

import io
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from src.agents import block_extraction_scripting


class BlockExtractionScriptingTests(unittest.TestCase):
    def _workspace(self, root: Path, count: int = 1) -> tuple[Path, Path, list[dict]]:
        agent_space = root / "agent_space"
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
            for node_id in range(count)
        ]
        for node_id in range(count):
            (block_samples / str(node_id)).mkdir(parents=True)
        (root / "template.json").write_text(
            json.dumps({"template": template}),
            encoding="utf-8",
        )
        return agent_space, train_dir, template

    def test_limits_parallel_opencode_sessions_to_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            agent_space, train_dir, template = self._workspace(output_dir, count=4)

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

    def test_retries_invalid_script_and_accumulates_attempt_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            agent_space, train_dir, template = self._workspace(output_dir)
            attempts = 0
            prompts: list[str] = []

            def generate_script(
                prompt: str,
                sample_dir: str,
                **_kwargs: object,
            ) -> tuple[int, int, int, None]:
                nonlocal attempts
                attempts += 1
                prompts.append(prompt)
                script_path = Path(sample_dir) / "0.py"
                if attempts == 1:
                    script_path.write_text("def extract(:\n", encoding="utf-8")
                    return 0, 10, 2, None
                script_path.write_text(
                    "def extract(block):\n    return block\n",
                    encoding="utf-8",
                )
                return 0, 20, 4, None

            with patch.object(
                block_extraction_scripting,
                "opencode_silent",
                side_effect=generate_script,
            ):
                result = block_extraction_scripting.block_extraction_scripting_node(
                    {
                        "agent_space": str(agent_space),
                        "pdf_path": str(train_dir / "document.pdf"),
                        "template": template,
                    }
                )

            self.assertEqual(attempts, 2)
            self.assertIn("PREVIOUS ATTEMPT", prompts[1])
            block_cost = result["cost"][0]["blocks"][0]
            self.assertEqual(block_cost["input_tokens"], 30)
            self.assertEqual(block_cost["output_tokens"], 6)

    def test_stale_script_is_not_accepted_and_failure_uses_red_cross(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            agent_space, train_dir, template = self._workspace(output_dir)
            stale_script = train_dir / "block_samples" / "0" / "0.py"
            stale_script.write_text(
                "def extract(block):\n    return block\n",
                encoding="utf-8",
            )
            terminal = io.StringIO()

            with (
                patch.object(
                    block_extraction_scripting,
                    "opencode_silent",
                    return_value=(0, 5, 1, None),
                ) as opencode_mock,
                patch("sys.stdout", terminal),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "exited successfully but did not produce",
                ):
                    block_extraction_scripting.block_extraction_scripting_node(
                        {
                            "agent_space": str(agent_space),
                            "pdf_path": str(train_dir / "document.pdf"),
                            "template": template,
                        }
                    )

            self.assertEqual(opencode_mock.call_count, 3)
            self.assertIn("✗", terminal.getvalue())


if __name__ == "__main__":
    unittest.main()
