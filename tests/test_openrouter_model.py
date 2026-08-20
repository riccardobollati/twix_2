from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.models.openrouter import get_model


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


if __name__ == "__main__":
    unittest.main()
