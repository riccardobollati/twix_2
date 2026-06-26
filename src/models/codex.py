import json
import subprocess


DEFAULT_CODEX_MODEL = "gpt-5.4"
_CAVEMAN_PREFIX = "/caveman full\n\n"
_SCOPE_PREFIX = (
    "STRICT SCOPE RULE: You are running inside a single task folder. "
    "Work ONLY with files in your current working directory. "
    "Do NOT use `..` paths, do NOT navigate to parent directories, "
    "do NOT read or reference any file outside the current folder. "
    "Everything you need for this task is already here.\n\n"
)


def codex(prompt, dir_path, task, model: str = DEFAULT_CODEX_MODEL, caveman: bool = False):
    print(f"Running codex for task: {task}")
    if caveman:
        prompt = _CAVEMAN_PREFIX + prompt
    prompt = _SCOPE_PREFIX + prompt
    result = subprocess.run(
        [
            "codex", "exec",
            "-m", model,
            "-C", str(dir_path),
            "-s", "workspace-write",
            "--skip-git-repo-check",
            prompt,
        ],
    )
    return result.returncode


def codex_silent(prompt, dir_path, task, model: str = DEFAULT_CODEX_MODEL, caveman: bool = False, log_path: str | None = None) -> tuple[int, int | None, int | None]:
    """Run codex with --json output, suppress stdout/stderr, return (returncode, input_tokens, output_tokens).

    Parses the JSONL event stream for token usage. Returns None for each token
    field if usage information is not found in the output.
    """
    if caveman:
        prompt = _CAVEMAN_PREFIX + prompt
    prompt = _SCOPE_PREFIX + prompt
    result = subprocess.run(
        [
            "codex", "exec",
            "-m", model,
            "-C", str(dir_path),
            "-s", "workspace-write",
            "--skip-git-repo-check",
            "--json",
            prompt,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        print(f"[codex stderr]\n{result.stderr}", flush=True)
        print(f"[codex stdout]\n{result.stdout}", flush=True)
    if log_path:
        import os as _os
        _os.makedirs(_os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as _f:
            _f.write(result.stdout)
            if result.stderr:
                _f.write("\n--- stderr ---\n")
                _f.write(result.stderr)
        prompt_path = _os.path.splitext(log_path)[0] + ".prompt"
        with open(prompt_path, "w", encoding="utf-8") as _f:
            _f.write(prompt)
    input_tokens, output_tokens = _parse_tokens(result.stdout)
    return result.returncode, input_tokens, output_tokens


def _parse_tokens(jsonl_output: str) -> tuple[int | None, int | None]:
    """Scan JSONL codex output and return (input_tokens, output_tokens), or (None, None)."""
    best_inp = 0
    best_out = 0
    found = False

    for raw_line in jsonl_output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        inp, out = _extract_tokens_from(event)
        if inp or out:
            best_inp = max(best_inp, inp)
            best_out = max(best_out, out)
            found = True

    return (best_inp or None, best_out or None) if found else (None, None)


def _extract_tokens_from(obj) -> tuple[int, int]:
    """Recursively search a parsed JSON object for (input_tokens, output_tokens)."""
    if not isinstance(obj, dict):
        return 0, 0

    # Prefer explicit split fields
    inp = obj.get("input_tokens") or obj.get("inputTokens") or 0
    out = obj.get("output_tokens") or obj.get("outputTokens") or 0
    if inp or out:
        return inp, out

    # Fall back to total_tokens (split unknown — stored as output by convention)
    total = obj.get("total_tokens") or obj.get("totalTokens") or 0
    if total:
        return 0, total

    # Recurse into nested dicts/lists
    best_inp, best_out = 0, 0
    for v in obj.values():
        if isinstance(v, dict):
            i, o = _extract_tokens_from(v)
            best_inp = max(best_inp, i)
            best_out = max(best_out, o)
        elif isinstance(v, list):
            for item in v:
                i, o = _extract_tokens_from(item)
                best_inp = max(best_inp, i)
                best_out = max(best_out, o)
    return best_inp, best_out
