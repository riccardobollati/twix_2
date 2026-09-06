from __future__ import annotations

import json
import os
import subprocess

from src.models.catalog import DEFAULT_EASY_MODEL

DEFAULT_OPENCODE_MODEL = os.getenv(
    "OPENCODE_MODEL", DEFAULT_EASY_MODEL
)
OPENCODE_TIMEOUT_S = 3600
_CAVEMAN_PREFIX = "/caveman full\n\n"
_SCOPE_PREFIX = (
    "STRICT SCOPE RULE: You are running inside a single task folder. "
    "Work ONLY with files in your current working directory. "
    "Do NOT use `..` paths, do NOT navigate to parent directories, "
    "do NOT read or reference any file outside the current folder. "
    "Everything you need for this task is already here.\n\n"
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

    try:
        result = subprocess.run(
            cmd,
            cwd=task_dir,
            stdout=subprocess.PIPE if silent else None,
            stderr=subprocess.PIPE if silent else None,
            text=True,
            timeout=OPENCODE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        print(
            f"[opencode] task {task} timed out after {OPENCODE_TIMEOUT_S}s",
            flush=True,
        )
        return 124, None, None, None
    except FileNotFoundError:
        print(
            "[opencode] executable not found; install OpenCode and ensure it is "
            "on PATH (DeepSeek recommends version >= 1.14.24)",
            flush=True,
        )
        return 127, None, None, None

    if not silent:
        return result.returncode, None, None, None

    if result.returncode != 0:
        print(f"[opencode stderr]\n{result.stderr}", flush=True)
        print(f"[opencode stdout]\n{result.stdout}", flush=True)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as target:
            target.write(result.stdout)
            if result.stderr:
                target.write("\n--- stderr ---\n")
                target.write(result.stderr)
        prompt_path = os.path.splitext(log_path)[0] + ".prompt"
        with open(prompt_path, "w", encoding="utf-8") as target:
            target.write(prompt)

    input_tokens, output_tokens, total_tokens = _parse_tokens(result.stdout)
    return result.returncode, input_tokens, output_tokens, total_tokens


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
