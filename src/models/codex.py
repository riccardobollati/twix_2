import json
import os
import subprocess

from src.agents._usage import split_tokens


DEFAULT_CODEX_MODEL = os.getenv("CODEX_MODEL", "gpt-5.4")
CODEX_TIMEOUT_S = 1800  # hard cap for one codex session
_CAVEMAN_PREFIX = "/caveman full\n\n"
_SCOPE_PREFIX = (
    "STRICT SCOPE RULE: You are running inside a single task folder. "
    "Work ONLY with files in your current working directory. "
    "Do NOT use `..` paths, do NOT navigate to parent directories, "
    "do NOT read or reference any file outside the current folder. "
    "Everything you need for this task is already here.\n\n"
)


def _codex_exec(prompt, dir_path, task, model: str, caveman: bool, silent: bool,
                log_path: str | None) -> tuple[int, int | None, int | None, int | None]:
    """Shared codex invocation. Returns (returncode, input_tokens, output_tokens, total_tokens).

    total_tokens is non-None only when the stream reported a total without an
    input/output split. A timed-out session returns returncode 124 (retryable
    by callers with a retry loop).
    """
    if caveman:
        prompt = _CAVEMAN_PREFIX + prompt
    prompt = _SCOPE_PREFIX + prompt

    cmd = [
        "codex", "exec",
        "-m", model,
        "-C", str(dir_path),
        "-s", "workspace-write",
        "--skip-git-repo-check",
    ]
    if silent:
        cmd.append("--json")
    cmd.append(prompt)

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE if silent else None,
            stderr=subprocess.PIPE if silent else None,
            text=True,
            timeout=CODEX_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        print(f"[codex] task {task} timed out after {CODEX_TIMEOUT_S}s", flush=True)
        return 124, None, None, None

    if not silent:
        return result.returncode, None, None, None

    if result.returncode != 0:
        print(f"[codex stderr]\n{result.stderr}", flush=True)
        print(f"[codex stdout]\n{result.stdout}", flush=True)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as _f:
            _f.write(result.stdout)
            if result.stderr:
                _f.write("\n--- stderr ---\n")
                _f.write(result.stderr)
        prompt_path = os.path.splitext(log_path)[0] + ".prompt"
        with open(prompt_path, "w", encoding="utf-8") as _f:
            _f.write(prompt)
    input_tokens, output_tokens, total_tokens = _parse_tokens(result.stdout)
    return result.returncode, input_tokens, output_tokens, total_tokens


def codex(prompt, dir_path, task, model: str = DEFAULT_CODEX_MODEL, caveman: bool = False):
    print(f"Running codex for task: {task}")
    returncode, _, _, _ = _codex_exec(prompt, dir_path, task, model, caveman, silent=False, log_path=None)
    return returncode


def codex_silent(prompt, dir_path, task, model: str = DEFAULT_CODEX_MODEL, caveman: bool = False, log_path: str | None = None) -> tuple[int, int | None, int | None, int | None]:
    """Run codex with --json output, suppress stdout/stderr, return
    (returncode, input_tokens, output_tokens, total_tokens).

    Parses the JSONL event stream for token usage. Returns None for each token
    field if usage information is not found in the output. total_tokens is set
    only when the stream reported just a total (input/output split unknown).
    """
    return _codex_exec(prompt, dir_path, task, model, caveman, silent=True, log_path=log_path)


def _parse_tokens(jsonl_output: str) -> tuple[int | None, int | None, int | None]:
    """Scan JSONL codex output and return (input_tokens, output_tokens, total_tokens).

    Codex emits cumulative usage events, so the max of each field across the
    stream is the session total. total_tokens is populated only when an event
    reported a bare total without an input/output split; it is kept separate
    instead of being booked as output tokens.
    """
    best_inp = 0
    best_out = 0
    best_total = 0
    found = False

    for raw_line in jsonl_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        inp, out, total = _extract_tokens_from(event)
        if inp or out or total:
            best_inp = max(best_inp, inp)
            best_out = max(best_out, out)
            best_total = max(best_total, total)
            found = True

    if not found:
        return None, None, None
    # A split total is redundant once input/output are known
    if best_inp or best_out:
        best_total = 0
    return best_inp or None, best_out or None, best_total or None


def _extract_tokens_from(obj) -> tuple[int, int, int]:
    """Recursively search a parsed JSON object for (input_tokens, output_tokens, total_tokens)."""
    if not isinstance(obj, dict):
        return 0, 0, 0

    inp, out, total = split_tokens(obj)
    if inp or out or total:
        return inp, out, total

    # Recurse into nested dicts/lists
    best_inp, best_out, best_total = 0, 0, 0
    for v in obj.values():
        if isinstance(v, dict):
            i, o, t = _extract_tokens_from(v)
            best_inp, best_out, best_total = max(best_inp, i), max(best_out, o), max(best_total, t)
        elif isinstance(v, list):
            for item in v:
                i, o, t = _extract_tokens_from(item)
                best_inp, best_out, best_total = max(best_inp, i), max(best_out, o), max(best_total, t)
    return best_inp, best_out, best_total
