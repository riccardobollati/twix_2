from __future__ import annotations

import json
import subprocess
import unittest
from unittest.mock import patch

from src.models.opencode import opencode, opencode_silent


class OpenCodeModelTests(unittest.TestCase):
    @patch("src.models.opencode.subprocess.run")
    def test_opencode_runs_in_task_directory_with_selected_model(self, run_mock) -> None:
        run_mock.return_value = subprocess.CompletedProcess([], 0)

        returncode = opencode(
            "write the script",
            "/tmp/work",
            "test",
            model="openrouter/deepseek/deepseek-v4-flash",
        )

        self.assertEqual(returncode, 0)
        command = run_mock.call_args.args[0]
        self.assertEqual(
            command[:6],
            [
                "opencode",
                "run",
                "--dir",
                "/tmp/work",
                "--model",
                "openrouter/deepseek/deepseek-v4-flash",
            ],
        )
        self.assertIn("STRICT SCOPE RULE", command[6])
        self.assertEqual(run_mock.call_args.kwargs["cwd"], "/tmp/work")
        self.assertEqual(run_mock.call_args.kwargs["timeout"], 3600)

    @patch("src.models.opencode.subprocess.run")
    def test_jsonl_usage_sums_agent_steps_and_cached_input(self, run_mock) -> None:
        events = [
            {
                "type": "step_finish",
                "part": {
                    "type": "step-finish",
                    "tokens": {
                        "input": 100,
                        "output": 20,
                        "cache": {"read": 30, "write": 5},
                    },
                },
            },
            {
                "type": "step_finish",
                "part": {
                    "type": "step-finish",
                    "tokens": {
                        "input": 40,
                        "output": 10,
                        "cache": {"read": 10, "write": 0},
                    },
                },
            },
        ]
        stdout = "\n".join(json.dumps(event) for event in events)
        run_mock.return_value = subprocess.CompletedProcess(
            [], 0, stdout=stdout, stderr=""
        )

        result = opencode_silent("fix it", "/tmp/work", "test")

        self.assertEqual(result, (0, 185, 30, None))
        command = run_mock.call_args.args[0]
        self.assertIn("--format", command)
        self.assertIn("json", command)


if __name__ == "__main__":
    unittest.main()
