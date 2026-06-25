import json
import subprocess


def codex(prompt, dir_path, task):
    print(f"Running codex for task: {task}")
    result = subprocess.run(
        [
            "codex", "exec",
            "-C", str(dir_path),
            "-s", "workspace-write",
            "--skip-git-repo-check",
            prompt,
        ],
    )
    return result.returncode


def codex_silent(prompt, dir_path, task) -> tuple[int, int | None, int | None]:
    """Run codex with --json output, suppress stdout/stderr, return (returncode, input_tokens, output_tokens).

    Parses the JSONL event stream for token usage. Returns None for each token
    field if usage information is not found in the output.
    """
    result = subprocess.run(
        [
            "codex", "exec",
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
