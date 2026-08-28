from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.agents.block_difficulty_router import block_difficulty_router_node


class BlockDifficultyRouterTests(unittest.TestCase):
    def test_no_routing_prefers_run_model_over_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            train_dir = Path(tmp)
            (train_dir / "block_samples" / "0").mkdir(parents=True)
            llm = Mock()

            with patch.dict(
                os.environ,
                {"OPENCODE_EASY_MODEL": "openrouter/example/env-model"},
            ):
                result = block_difficulty_router_node(
                    {
                        "pdf_path": str(train_dir / "document.pdf"),
                        "no_routing": True,
                        "easy_model": "openrouter/z-ai/glm-5.3-flash",
                    },
                    llm,
                )

        self.assertEqual(
            result["block_opencode_models"],
            {"0": "openrouter/z-ai/glm-5.3-flash"},
        )
        llm.invoke.assert_not_called()

    def test_no_routing_assigns_configured_easy_model_without_llm_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            train_dir = Path(tmp)
            (train_dir / "block_samples" / "0").mkdir(parents=True)
            (train_dir / "block_samples" / "2").mkdir(parents=True)
            llm = Mock()

            with patch.dict(
                os.environ,
                {"OPENCODE_EASY_MODEL": "openrouter/example/easy-model"},
            ):
                result = block_difficulty_router_node(
                    {
                        "pdf_path": str(train_dir / "document.pdf"),
                        "no_routing": True,
                    },
                    llm,
                )

        self.assertEqual(
            result["block_opencode_models"],
            {
                "0": "openrouter/example/easy-model",
                "2": "openrouter/example/easy-model",
            },
        )
        self.assertTrue(result["cost"][0]["routing_disabled"])
        llm.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
