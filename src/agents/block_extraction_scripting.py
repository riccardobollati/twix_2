from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from src.agents._ui import FRAMES as _FRAMES, GREEN_TICK as _GREEN_TICK
from src.models.opencode import DEFAULT_OPENCODE_MODEL, opencode_silent
from src.prompts.block_extraction_scripting_prompt import PROMPT_BLOCK_EXTRACTION_SCRIPTING


MAX_ATTEMPTS = 3


# ── animated display ──────────────────────────────────────────────────────────

class _Display:
    """Thread-safe multi-line spinner for parallel OpenCode sessions."""

    def __init__(self, ids: list[str]) -> None:
        self._ids = ids
        self._done: dict[str, bool] = {i: False for i in ids}
        self._tokens: dict[str, int | None] = {i: None for i in ids}
        self._frame = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        # Reserve one line per block
        sys.stdout.write("\n" * len(ids))
        sys.stdout.flush()

    def _redraw(self) -> None:
        n = len(self._ids)
        sys.stdout.write(f"\033[{n}A")  # move cursor up n lines
        for tid in self._ids:
            if self._done[tid]:
                tok = self._tokens[tid]
                suffix = f"  {_GREEN_TICK}  {tok:,} tokens" if tok is not None else f"  {_GREEN_TICK}"
            else:
                suffix = f"  {_FRAMES[self._frame % len(_FRAMES)]}"
            sys.stdout.write(f"\r\033[K  block_extraction_scripting | block {tid}{suffix}\n")
        sys.stdout.flush()

    def mark_done(self, tid: str, tokens: int | None) -> None:
        with self._lock:
            self._done[tid] = True
            self._tokens[tid] = tokens
            self._redraw()

    def _spin(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                if not all(self._done.values()):
                    self._frame += 1
                    self._redraw()
            time.sleep(0.1)

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self._spin, daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self._stop.set()


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_template_node(agent_space: str, template_id: str) -> dict:
    """Return the template node whose node_id matches template_id.

    Block ids follow the "<node_id>_<instance>" convention, so a block-sample
    folder name (template_id) is a node_id. Raises RuntimeError when the node
    cannot be resolved — an extraction prompt without its template node would
    silently degrade the generated script.
    """
    template_path = os.path.join(os.path.dirname(agent_space), "template.json")
    try:
        with open(template_path, encoding="utf-8") as f:
            template = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not load template.json at {template_path}: {exc}") from exc
    try:
        node_id = int(template_id)
    except ValueError as exc:
        raise RuntimeError(f"block sample folder {template_id!r} is not a valid node_id") from exc
    for node in template.get("template", []):
        if node.get("node_id") == node_id:
            return node
    raise RuntimeError(f"no template node with node_id={node_id} in {template_path}")


# ── node ──────────────────────────────────────────────────────────────────────

def block_extraction_scripting_node(state: dict[str, Any]) -> dict[str, Any]:
    """For each block type, run OpenCode (in parallel) inside its sample folder
    to generate a Python extraction script generalizable to all instances of that block.

    Creates agent_space/extraction_scripts/ and writes one <template_id>.py per block type.
    Each OpenCode session is retried up to MAX_ATTEMPTS times when it exits non-zero or
    fails to produce the script file.

    Reads:
        state["agent_space"]  – directory containing block_samples/ and ../template.json
    """
    agent_space = state["agent_space"]
    train_dir   = os.path.dirname(state["pdf_path"])

    block_samples_dir = os.path.join(train_dir, "block_samples")
    extraction_scripts_dir = os.path.join(train_dir, "extraction_scripts")
    os.makedirs(extraction_scripts_dir, exist_ok=True)

    # create the list of ids — restricted to nodes of the CURRENT template so a
    # stale sample folder from an earlier run (different template) can't crash
    # the run or spawn an OpenCode session for a node that no longer exists
    current_ids = {str(n.get("node_id")) for n in (state.get("template") or [])}
    template_ids = []
    for d in sorted(os.listdir(block_samples_dir)):
        if not os.path.isdir(os.path.join(block_samples_dir, d)):
            continue
        if current_ids and d not in current_ids:
            print(f"block_extraction_scripting: skipping stale sample folder {d!r} "
                  "(not a node of the current template)")
            continue
        template_ids.append(d)

    if not template_ids:
        print("block_extraction_scripting: no block sample folders found")
        return {}

    print(
        f"block_extraction_scripting_node: generating {len(template_ids)} "
        "extractor script(s) with OpenCode...",
        flush=True,
    )
    display = _Display(template_ids)
    spinner_thread = display.start()

    block_opencode_models = state.get("block_opencode_models") or {}
    use_caveman = state.get("use_caveman", False)
    debug_dir   = os.path.join(os.path.dirname(agent_space), "debug") if state.get("save_log") else None
    t0 = time.time()

    def _run(template_id: str) -> tuple[str, bool, int | None, int | None, int | None]:
        model = block_opencode_models.get(
            template_id,
            state.get("opencode_model", DEFAULT_OPENCODE_MODEL),
        )
        sample_dir  = os.path.join(block_samples_dir, template_id)
        script_path = os.path.join(sample_dir, f"{template_id}.py")
        template_node = _load_template_node(agent_space, template_id)
        prompt = PROMPT_BLOCK_EXTRACTION_SCRIPTING(template_id, template_node)
        inp = out = tot = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            log_path = os.path.join(debug_dir, f"block_extraction_{template_id}_{attempt}.log") if debug_dir else None
            rc, inp, out, tot = opencode_silent(
                prompt,
                sample_dir,
                task=f"block_extraction_scripting_{template_id}",
                model=model,
                caveman=use_caveman,
                log_path=log_path,
            )
            if rc == 0 and os.path.isfile(script_path):
                return template_id, True, inp, out, tot
        return template_id, False, inp, out, tot

    block_tokens: list[dict] = []
    failed_ids: list[str] = []
    try:
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(_run, tid): tid for tid in template_ids}
            for future in as_completed(futures):
                template_id, ok, inp, out, tot = future.result()
                if not ok:
                    failed_ids.append(template_id)
                    inp = out = tot = None
                shown_t = (inp or 0) + (out or 0) + (tot or 0) or None
                display.mark_done(template_id, shown_t)
                if inp is not None or out is not None or tot is not None:
                    entry = {"id": template_id, "input_tokens": inp or 0, "output_tokens": out or 0}
                    if tot:
                        entry["total_tokens"] = tot
                    block_tokens.append(entry)
    finally:
        display.stop()
        spinner_thread.join(timeout=0.2)

    if failed_ids:
        raise RuntimeError(
            "block_extraction_scripting: no script produced after "
            f"{MAX_ATTEMPTS} attempt(s) for block(s): {', '.join(sorted(failed_ids))}"
        )

    duration = time.time() - t0
    has_tokens = bool(block_tokens)
    total_input  = sum(b["input_tokens"]  for b in block_tokens) if has_tokens else None
    total_output = sum(b["output_tokens"] for b in block_tokens) if has_tokens else None
    total_total  = sum(b.get("total_tokens", 0) for b in block_tokens) if has_tokens else None

    cost_entry: dict = {
        "phase": "block_extraction_scripting",
        "duration_s": round(duration, 3),
    }
    if has_tokens:
        cost_entry["input_tokens"]  = total_input
        cost_entry["output_tokens"] = total_output
        if total_total:
            cost_entry["total_tokens"] = total_total
        cost_entry["blocks"] = sorted(block_tokens, key=lambda b: b["id"])

    return {"cost": [cost_entry]}
