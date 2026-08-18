from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from src.models.codex import codex


class CodexModelTests(unittest.TestCase):
    @patch("src.models.codex.subprocess.run")
    def test_codex_uses_api_billed_cli(self, run_mock) -> None:
        run_mock.return_value = subprocess.CompletedProcess([], 0)

        with patch("src.models.codex.CODEX_API_HOME", "/tmp/codex-api-home"):
            returncode = codex("write the script", "/tmp/work", "test")

        self.assertEqual(returncode, 0)
        command = run_mock.call_args.args[0]
        self.assertEqual(command[:2], ["codex", "exec"])
        self.assertEqual(
            run_mock.call_args.kwargs["env"]["CODEX_HOME"],
            "/tmp/codex-api-home",
        )


if __name__ == "__main__":
    unittest.main()
