from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from typing import IO, Iterator

import fcntl

from src.models.catalog import DEFAULT_EASY_MODEL

DEFAULT_OPENCODE_MODEL = os.getenv(
    "OPENCODE_MODEL", DEFAULT_EASY_MODEL
)
OPENCODE_TIMEOUT_S = 3600
OPENCODE_GLOBAL_SLOTS = 2
OPENCODE_TRANSIENT_EXIT_CODE = 75
_SLOT_POLL_INTERVAL_S = 0.25
_LOCAL_SLOT_LOCK = threading.Lock()
_LOCAL_SLOT_IDS: set[int] = set()
_CAVEMAN_PREFIX = "/caveman full\n\n"
_SCOPE_PREFIX = (
    "STRICT SCOPE RULE: You are running inside a single task folder. "
    "Work ONLY with files in your current working directory. "
    "Do NOT use `..` paths, do NOT navigate to parent directories, "
    "do NOT read or reference any file outside the current folder. "
    "Everything you need for this task is already here.\n\n"
)


def _slot_directory() -> str:
    """Return the shared lock directory used by all TWIX processes."""
    override = os.getenv("TWIX_OPENCODE_LOCK_DIR")
    if override:
        return os.path.abspath(override)
    return os.path.join(
        tempfile.gettempdir(),
        f"twix-opencode-slots-{getattr(os, 'getuid', lambda: 0)()}",
    )


@contextmanager
def _global_opencode_slot(task: str) -> Iterator[int]:
    """Hold one of the process-wide OpenCode slots.

    ``flock`` enforces the limit across separate UI/CLI/batch processes. The
    in-process reservation set also makes the behavior unambiguous across
    platforms whose ``flock`` semantics differ for two handles in one process.
    Locks are released automatically if a worker or its parent process dies.
    """
    slot_dir = _slot_directory()
    os.makedirs(slot_dir, exist_ok=True)
    waiting_reported = False
    handle: IO[str] | None = None
    acquired_slot: int | None = None

    while acquired_slot is None:
        with _LOCAL_SLOT_LOCK:
            for slot_id in range(OPENCODE_GLOBAL_SLOTS):
                if slot_id in _LOCAL_SLOT_IDS:
                    continue
                candidate = open(
                    os.path.join(slot_dir, f"slot-{slot_id}.lock"),
                    "a+",
                    encoding="utf-8",
                )
                try:
                    fcntl.flock(candidate.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    candidate.close()
                    continue
                _LOCAL_SLOT_IDS.add(slot_id)
                acquired_slot = slot_id
                handle = candidate
                break
        if acquired_slot is None:
            if not waiting_reported:
                print(
                    f"[opencode] task {task} waiting for a global execution slot",
                    flush=True,
                )
                waiting_reported = True
            time.sleep(_SLOT_POLL_INTERVAL_S)

    try:
        yield acquired_slot
    finally:
        assert handle is not None
        with _LOCAL_SLOT_LOCK:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            _LOCAL_SLOT_IDS.discard(acquired_slot)
        handle.close()


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _write_debug_log(
    log_path: str | None,
    stdout: str,
    stderr: str,
    prompt: str,
) -> None:
    if not log_path:
        return
    parent = os.path.dirname(log_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as target:
        target.write(stdout)
        if stderr:
            target.write("\n--- stderr ---\n")
            target.write(stderr)
    prompt_path = os.path.splitext(log_path)[0] + ".prompt"
    with open(prompt_path, "w", encoding="utf-8") as target:
        target.write(prompt)


def _contains_database_lock_error(stdout: str, stderr: str) -> bool:
    combined = f"{stdout}\n{stderr}".lower()
    return any(
        marker in combined
        for marker in ("database is locked", "sqlite_busy", "sqlite_locked")
    )


def _opencode_exec(
    prompt: str,
    dir_path: str,
    task: str,
    model: str,
    caveman: bool,
    silent: bool,
    log_path: str | None,
) -> tuple[int, int | None, int | None, int | None]:
    """Run one OpenCode session and return its exit code and token usage.

    OpenCode emits one ``step_finish`` event per agent step. Usage is summed
    across those events because every step is a separately billed model call.
    A timed-out session returns code 124 so callers can retry it.
    """
    if caveman:
        prompt = _CAVEMAN_PREFIX + prompt
    prompt = _SCOPE_PREFIX + prompt

    task_dir = os.path.abspath(dir_path)
    cmd = [
        "opencode",
        "run",
        "--dir",
        task_dir,
        "--model",
        model,
    ]
    if silent:
        cmd.extend(["--format", "json"])
    cmd.append(prompt)

    stdout = stderr = ""
    try:
        # OpenCode normally shares one SQLite database across every CLI process.
        # Give each attempt its own database while retaining the user's normal
        # configuration and auth paths. The outer lock caps all TWIX processes,
        # not only this thread pool.
        with _global_opencode_slot(task):
            with tempfile.TemporaryDirectory(prefix="twix-opencode-db-") as db_dir:
                env = os.environ.copy()
                env["OPENCODE_DB"] = os.path.join(db_dir, "opencode.db")
                result = subprocess.run(
                    cmd,
                    cwd=task_dir,
                    stdout=subprocess.PIPE if silent else None,
                    stderr=subprocess.PIPE if silent else None,
                    text=True,
                    timeout=OPENCODE_TIMEOUT_S,
                    env=env,
                )
    except subprocess.TimeoutExpired as exc:
        stdout = _as_text(exc.stdout)
        stderr = _as_text(exc.stderr)
        _write_debug_log(log_path, stdout, stderr, prompt)
        input_tokens, output_tokens, total_tokens = _parse_tokens(stdout)
        print(
            f"[opencode] task {task} timed out after {OPENCODE_TIMEOUT_S}s",
            flush=True,
        )
        if stderr:
            print(f"[opencode partial stderr]\n{stderr}", flush=True)
        return 124, input_tokens, output_tokens, total_tokens
    except FileNotFoundError:
        _write_debug_log(log_path, stdout, stderr, prompt)
        print(
            "[opencode] executable not found; install OpenCode and ensure it is "
            "on PATH (DeepSeek recommends version >= 1.14.24)",
            flush=True,
        )
        return 127, None, None, None

    if not silent:
        return result.returncode, None, None, None

    stdout = _as_text(result.stdout)
    stderr = _as_text(result.stderr)
    returncode = result.returncode
    if _contains_database_lock_error(stdout, stderr):
        # Some OpenCode versions have emitted a database error while still
        # exiting zero. Give callers a reliable transient-failure signal.
        returncode = OPENCODE_TRANSIENT_EXIT_CODE
        print(
            f"[opencode] task {task} hit a transient SQLite database error",
            flush=True,
        )
    if returncode != 0:
        print(f"[opencode stderr]\n{stderr}", flush=True)
        print(f"[opencode stdout]\n{stdout}", flush=True)
    _write_debug_log(log_path, stdout, stderr, prompt)

    input_tokens, output_tokens, total_tokens = _parse_tokens(stdout)
    return returncode, input_tokens, output_tokens, total_tokens


def opencode(
    prompt: str,
    dir_path: str,
    task: str,
    model: str = DEFAULT_OPENCODE_MODEL,
    caveman: bool = False,
) -> int:
    print(f"Running OpenCode for task: {task}")
    returncode, _, _, _ = _opencode_exec(
        prompt,
        dir_path,
        task,
        model,
        caveman,
        silent=False,
        log_path=None,
    )
    return returncode


def opencode_silent(
    prompt: str,
    dir_path: str,
    task: str,
    model: str = DEFAULT_OPENCODE_MODEL,
    caveman: bool = False,
    log_path: str | None = None,
) -> tuple[int, int | None, int | None, int | None]:
    """Run OpenCode with JSONL output and return exit code plus token usage."""
    return _opencode_exec(
        prompt,
        dir_path,
        task,
        model,
        caveman,
        silent=True,
        log_path=log_path,
    )


def _parse_tokens(jsonl_output: str) -> tuple[int | None, int | None, int | None]:
    """Sum usage from OpenCode ``step_finish`` JSONL events.

    OpenCode reports uncached input separately from cache reads/writes. They
    are combined here so ``input_tokens`` remains comparable with LangChain's
    logical input-token total.
    """
    input_tokens = output_tokens = total_tokens = 0
    found_split = found_total = False

    for raw_line in jsonl_output.splitlines():
        try:
            event = json.loads(raw_line.strip())
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(event, dict):
            continue
        part = event.get("part") if isinstance(event.get("part"), dict) else event
        if event.get("type") != "step_finish" and part.get("type") != "step-finish":
            continue
        usage = part.get("tokens") or part.get("usage")
        if not isinstance(usage, dict):
            continue

        inp = int(usage.get("input") or usage.get("input_tokens") or 0)
        out = int(usage.get("output") or usage.get("output_tokens") or 0)
        total = int(usage.get("total") or usage.get("total_tokens") or 0)
        cache = usage.get("cache") if isinstance(usage.get("cache"), dict) else {}
        inp += int(cache.get("read") or 0) + int(cache.get("write") or 0)
        if inp or out:
            input_tokens += inp
            output_tokens += out
            found_split = True
        elif total:
            total_tokens += total
            found_total = True

    if found_split:
        return input_tokens or None, output_tokens or None, None
    if found_total:
        return None, None, total_tokens or None
    return None, None, None
