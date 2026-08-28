from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agents.train_repair import train_repair_node


class TrainRepairTests(unittest.TestCase):
    def test_repair_prefers_run_hard_model_over_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            document_dir = Path(tmp)
            (document_dir / "agent_space").mkdir()
            (document_dir / "pipeline" / "extraction_scripts").mkdir(parents=True)

            with (
                patch.dict(
                    os.environ,
                    {"OPENCODE_HARD_MODEL": "openrouter/example/env-hard"},
                ),
                patch(
                    "src.agents.train_repair.opencode_silent",
                    return_value=(0, 10, 20, 30),
                ) as run_mock,
            ):
                train_repair_node({
                    "agent_space": str(document_dir / "agent_space"),
                    "hard_model": "openrouter/qwen/qwen3.8-flash",
                })

        self.assertEqual(
            run_mock.call_args.kwargs["model"],
            "openrouter/qwen/qwen3.8-flash",
        )

    def test_repair_uses_hard_model_not_general_opencode_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            document_dir = Path(tmp)
            (document_dir / "agent_space").mkdir()
            (document_dir / "pipeline" / "extraction_scripts").mkdir(parents=True)
            (document_dir / "pipeline" / "extraction_scripts" / "0.py").touch()

            with (
                patch.dict(
                    os.environ,
                    {"OPENCODE_HARD_MODEL": "openrouter/example/hard-model"},
                ),
                patch(
                    "src.agents.train_repair.opencode_silent",
                    return_value=(0, 10, 20, None),
                ) as run_mock,
            ):
                train_repair_node(
                    {
                        "agent_space": str(document_dir / "agent_space"),
                        "opencode_model": "openrouter/example/flash-model",
                    }
                )

        self.assertEqual(run_mock.call_args.kwargs["model"], "openrouter/example/hard-model")


if __name__ == "__main__":
    unittest.main()
