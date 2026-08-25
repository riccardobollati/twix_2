from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.models import get_template_inference_llm
from src.models.openrouter import get_ground_truth_model, get_model


class OpenRouterModelTests(unittest.TestCase):
    @patch("src.models.openrouter.ChatOpenAI")
    def test_openrouter_uses_compatible_endpoint_and_json_mode(self, chat_mock) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_MODEL": "deepseek/deepseek-v4-flash",
                "OPENROUTER_REASONING_EFFORT": "xhigh",
                "OPENROUTER_APP_URL": "https://example.test/twix",
                "OPENROUTER_APP_NAME": "TWIX test",
            },
            clear=True,
        ):
            get_model()

        kwargs = chat_mock.call_args.kwargs
        self.assertEqual(kwargs["model"], "deepseek/deepseek-v4-flash")
        self.assertEqual(kwargs["api_key"], "test-key")
        self.assertEqual(kwargs["base_url"], "https://openrouter.ai/api/v1")
        self.assertEqual(
            kwargs["default_headers"],
            {
                "HTTP-Referer": "https://example.test/twix",
                "X-OpenRouter-Title": "TWIX test",
            },
        )
        self.assertEqual(
            kwargs["model_kwargs"]["response_format"], {"type": "json_object"}
        )
        self.assertEqual(
            kwargs["extra_body"]["reasoning"],
            {"effort": "xhigh", "exclude": True},
        )

    def test_invalid_reasoning_effort_fails_before_network_call(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_REASONING_EFFORT": "max",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "OPENROUTER_REASONING_EFFORT"):
                get_model()

    @patch("src.models.openrouter.ChatOpenAI")
    def test_reasoning_effort_override_takes_precedence(self, chat_mock) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_REASONING_EFFORT": "high",
            },
            clear=True,
        ):
            get_model(reasoning_effort="low")

        self.assertEqual(
            chat_mock.call_args.kwargs["extra_body"]["reasoning"],
            {"effort": "low", "exclude": True},
        )

    @patch("src.models.openrouter.ChatOpenAI")
    def test_template_inference_uses_its_own_model_and_reasoning(self, chat_mock) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_MODEL": "deepseek/deepseek-v4-flash",
                "OPENROUTER_TEMPLATE_INFERENCE_MODEL": "openai/gpt-5.4",
                "OPENROUTER_TEMPLATE_INFERENCE_REASONING_EFFORT": "medium",
            },
            clear=True,
        ):
            get_template_inference_llm()

        kwargs = chat_mock.call_args.kwargs
        self.assertEqual(kwargs["model"], "openai/gpt-5.4")
        self.assertEqual(
            kwargs["extra_body"]["reasoning"],
            {"effort": "medium", "exclude": True},
        )

    @patch("src.models.openrouter.ChatOpenAI")
    def test_ground_truth_model_uses_openrouter_and_gpt_5_4(self, chat_mock) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test-key",
                "OPENROUTER_GROUND_TRUTH_MODEL": "openai/gpt-test",
                "OPENROUTER_GROUND_TRUTH_MAX_TOKENS": "12345",
                "OPENROUTER_BASE_URL": "https://router.example/v1",
                "OPENROUTER_APP_NAME": "TWIX test",
            },
            clear=True,
        ):
            get_ground_truth_model()

        kwargs = chat_mock.call_args.kwargs
        self.assertEqual(kwargs["model"], "openai/gpt-test")
        self.assertEqual(kwargs["api_key"], "test-key")
        self.assertEqual(kwargs["base_url"], "https://router.example/v1")
        self.assertEqual(kwargs["max_tokens"], 12345)
        self.assertEqual(
            kwargs["default_headers"], {"X-OpenRouter-Title": "TWIX test"}
        )
        self.assertEqual(
            kwargs["model_kwargs"]["response_format"], {"type": "json_object"}
        )
        self.assertNotIn("extra_body", kwargs)

    @patch("src.models.openrouter.ChatOpenAI")
    def test_ground_truth_model_defaults_to_openai_gpt_5_4(self, chat_mock) -> None:
        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "test-key", "OPENROUTER_APP_NAME": ""},
            clear=True,
        ):
            get_ground_truth_model()

        self.assertEqual(chat_mock.call_args.kwargs["model"], "openai/gpt-5.4")
        self.assertEqual(chat_mock.call_args.kwargs["max_tokens"], 128000)

    def test_ground_truth_max_tokens_must_be_positive_integer(self) -> None:
        for value in ("0", "-1", "not-a-number"):
            with self.subTest(value=value), patch.dict(
                os.environ,
                {
                    "OPENROUTER_API_KEY": "test-key",
                    "OPENROUTER_GROUND_TRUTH_MAX_TOKENS": value,
                },
                clear=True,
            ):
                with self.assertRaisesRegex(
                    ValueError, "OPENROUTER_GROUND_TRUTH_MAX_TOKENS"
                ):
                    get_ground_truth_model()


if __name__ == "__main__":
    unittest.main()
