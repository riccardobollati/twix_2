from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from src.models import opencode as opencode_model
from src.models.opencode import (
    OPENCODE_TRANSIENT_EXIT_CODE,
    opencode,
    opencode_silent,
)


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
        isolated_db = run_mock.call_args.kwargs["env"]["OPENCODE_DB"]
        self.assertNotEqual(isolated_db, os.getenv("OPENCODE_DB"))
        self.assertFalse(Path(isolated_db).parent.exists())

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

    @patch("src.models.opencode.subprocess.run")
    def test_each_attempt_uses_a_fresh_temporary_database(self, run_mock) -> None:
        database_paths: list[str] = []

        def capture_database(*_args: object, **kwargs: object) -> subprocess.CompletedProcess:
            database_paths.append(kwargs["env"]["OPENCODE_DB"])  # type: ignore[index]
            return subprocess.CompletedProcess([], 0, stdout="", stderr="")

        run_mock.side_effect = capture_database

        opencode_silent("first", "/tmp/work", "first")
        opencode_silent("second", "/tmp/work", "second")

        self.assertEqual(len(database_paths), 2)
        self.assertNotEqual(database_paths[0], database_paths[1])
        self.assertTrue(all(not Path(path).parent.exists() for path in database_paths))

    @patch("src.models.opencode.subprocess.run")
    def test_timeout_preserves_partial_usage_and_debug_log(self, run_mock) -> None:
        event = json.dumps({
            "type": "step_finish",
            "part": {
                "type": "step-finish",
                "tokens": {"input": 7, "output": 3},
            },
        })
        run_mock.side_effect = subprocess.TimeoutExpired(
            cmd=["opencode"],
            timeout=3600,
            output=event.encode(),
            stderr=b"partial diagnostic",
        )

        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "attempt.log"
            result = opencode_silent(
                "fix it", "/tmp/work", "test", log_path=str(log_path)
            )

            self.assertEqual(result, (124, 7, 3, None))
            self.assertIn(event, log_path.read_text(encoding="utf-8"))
            self.assertIn("partial diagnostic", log_path.read_text(encoding="utf-8"))
            self.assertTrue((Path(tmp) / "attempt.prompt").is_file())

    @patch("src.models.opencode.subprocess.run")
    def test_database_error_is_failure_even_when_opencode_exits_zero(self, run_mock) -> None:
        run_mock.return_value = subprocess.CompletedProcess(
            [], 0, stdout="", stderr="Error: database is locked"
        )

        result = opencode_silent("fix it", "/tmp/work", "test")

        self.assertEqual(result, (OPENCODE_TRANSIENT_EXIT_CODE, None, None, None))

    def test_global_limiter_allows_at_most_two_concurrent_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            release = threading.Event()
            two_entered = threading.Event()
            counter_lock = threading.Lock()
            active = 0
            max_active = 0

            def hold_slot(worker_id: int) -> int:
                nonlocal active, max_active
                with opencode_model._global_opencode_slot(f"worker-{worker_id}"):
                    with counter_lock:
                        active += 1
                        max_active = max(max_active, active)
                        if active == 2:
                            two_entered.set()
                    release.wait(timeout=3)
                    with counter_lock:
                        active -= 1
                return worker_id

            with patch.dict(os.environ, {"TWIX_OPENCODE_LOCK_DIR": tmp}):
                with ThreadPoolExecutor(max_workers=3) as executor:
                    futures = [executor.submit(hold_slot, i) for i in range(3)]
                    self.assertTrue(two_entered.wait(timeout=2))
                    self.assertEqual(max_active, 2)
                    release.set()
                    self.assertEqual(sorted(f.result(timeout=3) for f in futures), [0, 1, 2])

            self.assertEqual(max_active, 2)


if __name__ == "__main__":
    unittest.main()
