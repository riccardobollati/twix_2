from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from src.models.catalog import (
    MODEL_OPTIONS,
    REFERENCE_MODEL_OPTIONS,
    direct_model_id,
    opencode_model_id,
    resolve_easy_model,
    resolve_hard_model,
    resolve_reference_model,
    resolve_template_model,
)


class ModelCatalogTests(unittest.TestCase):
    def test_catalog_contains_supported_flash_models(self) -> None:
        operational = {value for _, value in MODEL_OPTIONS}
        references = {value for _, value in REFERENCE_MODEL_OPTIONS}
        expected = {
            "deepseek/deepseek-v4.1-flash",
            "z-ai/glm-5.3-flash",
            "qwen/qwen3.8-flash",
            "qwen/qwen3.8-27b",
            "qwen/qwen3.8-27b:free",
        }
        self.assertTrue(expected <= operational)
        self.assertTrue(expected <= references)

    def test_catalog_models_are_registered_for_opencode(self) -> None:
        config_path = Path(__file__).parents[1] / "opencode.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        models = config["provider"]["openrouter"]["models"]

        self.assertEqual(
            models["deepseek/deepseek-v4.1-flash"]["name"],
            "DeepSeek V4.1 Flash",
        )
        self.assertEqual(models["qwen/qwen3.8-27b"]["name"], "Qwen3.8 27B")
        self.assertEqual(
            models["qwen/qwen3.8-27b:free"]["name"],
            "Qwen3.8 27B (free)",
        )

    def test_model_ids_are_normalized_for_each_client(self) -> None:
        self.assertEqual(
            opencode_model_id("z-ai/glm-5.3-flash"),
            "openrouter/z-ai/glm-5.3-flash",
        )
        self.assertEqual(
            direct_model_id("openrouter/qwen/qwen3.8-flash"),
            "qwen/qwen3.8-flash",
        )

    def test_explicit_models_override_environment_defaults(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENCODE_EASY_MODEL": "openrouter/example/env-easy",
                "OPENCODE_HARD_MODEL": "openrouter/example/env-hard",
                "OPENROUTER_GROUND_TRUTH_MODEL": "example/env-reference",
                "OPENROUTER_TEMPLATE_INFERENCE_MODEL": "example/env-template",
            },
            clear=True,
        ):
            self.assertEqual(
                resolve_easy_model("z-ai/glm-5.3-flash"),
                "openrouter/z-ai/glm-5.3-flash",
            )
            self.assertEqual(
                resolve_hard_model("qwen/qwen3.8-flash"),
                "openrouter/qwen/qwen3.8-flash",
            )
            self.assertEqual(
                resolve_reference_model("openrouter/openai/gpt-5.4"),
                "openai/gpt-5.4",
            )
            self.assertEqual(resolve_template_model("z-ai/glm-5.3-flash"),
                             "z-ai/glm-5.3-flash")

    def test_empty_models_fall_back_to_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENCODE_EASY_MODEL": "openrouter/example/env-easy",
                "OPENCODE_HARD_MODEL": "example/env-hard",
                "OPENROUTER_GROUND_TRUTH_MODEL": "example/env-reference",
                "OPENROUTER_TEMPLATE_INFERENCE_MODEL": "example/env-template",
            },
            clear=True,
        ):
            self.assertEqual(
                resolve_easy_model(""), "openrouter/example/env-easy"
            )
            self.assertEqual(
                resolve_hard_model(None), "openrouter/example/env-hard"
            )
            self.assertEqual(
                resolve_reference_model(None), "example/env-reference"
            )
            self.assertEqual(resolve_template_model(None), "example/env-template")


if __name__ == "__main__":
    unittest.main()
