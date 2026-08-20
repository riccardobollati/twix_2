from __future__ import annotations

import unittest
from unittest.mock import patch

from src.models.openai import get_model


class OpenAIModelTests(unittest.TestCase):
    @patch("src.models.openai.ChatOpenAI")
    def test_held_out_ground_truth_model_uses_openai_credentials(self, chat) -> None:
        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-test"},
            clear=False,
        ):
            get_model()

        kwargs = chat.call_args.kwargs
        self.assertEqual(kwargs["model"], "gpt-test")
        self.assertEqual(kwargs["api_key"], "test-key")
        self.assertEqual(kwargs["temperature"], 0)


if __name__ == "__main__":
    unittest.main()
