from __future__ import annotations

import os
import threading
import time
from typing import Any

from src.models.codex import codex_silent


_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
_GREEN_TICK = "\033[32m✓\033[0m"

import sys


def _spinner(label: str, stop_event: threading.Event, result: dict) -> None:
    frame = 0
    sys.stdout.write("\n")
    sys.stdout.flush()
    while not stop_event.is_set():
        sys.stdout.write(f"\r\033[K  {label}  {_FRAMES[frame % len(_FRAMES)]}")
        sys.stdout.flush()
        frame += 1
        time.sleep(0.1)
    tok = result.get("tokens")
    suffix = f"  {_GREEN_TICK}  {tok:,} tokens" if tok is not None else f"  {_GREEN_TICK}"
    sys.stdout.write(f"\r\033[K  {label}{suffix}\n")
    sys.stdout.flush()


def _build_prompt(eval_txt: str, extraction_script_ids: list[str]) -> str:
    scripts_list = "\n".join(
        f"  - extraction_scripts/{sid}.py" for sid in sorted(extraction_script_ids)
    )
    return f"""You are debugging a PDF data-extraction pipeline. Fix the Python scripts \
in the current directory so the pipeline output better matches the ground truth.

PIPELINE OVERVIEW
=================
The pipeline runs in this order:
1. record_separation.py <ocr.csv> <out_dir>
   Groups OCR words into logical records (one record = one invoice / form entry).
   Writes: out_dir/record_separation.json

2. block_separation.py <ocr.csv> <out_dir>
   Detects typed blocks (kv, table) within the OCR data.
   Writes: out_dir/block_separation.json

3. extraction_scripts/<id>.py  — each defines  extract(block: dict) -> dict
   Extracts field values from one physical block.
   Called by pipeline.py on every block produced by block_separation.

EVALUATION ERRORS  (from train_eval/eval.txt)
=============================================
{eval_txt}

FILES YOU MAY MODIFY
====================
  - record_separation.py
  - block_separation.py
{scripts_list}

DO NOT modify pipeline.py, template.json, or any other file.

PRIORITY FIXES BY ERROR CODE
=============================
[R-RECORD-MISSING]
  Records are being collapsed together — the pipeline output has too few records.
  Fix record_separation.py: the boundary condition is too strict.
  Read the ALGORITHM comments already in the file and follow them carefully.

[R-RECORD-EXTRA]
  Phantom records are being created — the pipeline has too many records.
  Fix record_separation.py: the boundary condition fires too easily.

[R-NODE-MISSING]
  A block type was not detected on some pages.
  Fix block_separation.py: improve header-row detection or word-assignment logic.

[R-NODE-EXTRA]
  Spurious extra blocks are being emitted.
  Fix block_separation.py or the relevant extraction script — add a guard so the
  rule only fires when the block clearly exists on the page.

[R-CELL-DIFFERS] / [R-KV-VALUE-DIFFERS]
  Field values are extracted incorrectly.
  Fix the relevant extraction_scripts/<id>.py — check column alignment, label/value
  splitting, and text concatenation logic.

ANTI-OVERFITTING RULES  (MANDATORY — violations cause test-set failures)
=========================================================================
1. NEVER hard-code specific text values, names, amounts, dates, or any content from
   the training pages. All logic must be based on positions and field label patterns.
2. NEVER hard-code absolute pixel/coordinate thresholds. Derive all spatial thresholds
   dynamically from the data being processed (medians, percentages, inter-word gaps).
3. NEVER force output counts (records, blocks, rows) to match the training-data counts.
   Counts must emerge from the algorithm — never be hard-coded.
4. NEVER branch on the document name, page-number values, or any fixed metadata.
5. Every modified script must work correctly on pages it has never seen.

Read the existing scripts first to understand the current logic, then make targeted
fixes that address the errors listed above without breaking what already works.
"""


def train_repair_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run a Codex session to fix pipeline scripts based on the training eval report.

    Reads:
        state["agent_space"]           – used to derive pipeline_dir
        state["train_repair_attempts"] – number of repairs already completed

    Side-effect:
        Modifies record_separation.py, block_separation.py, and/or
        extraction_scripts/*.py inside pipeline_dir.
    """
    document_dir   = os.path.dirname(state["agent_space"])
    pipeline_dir   = os.path.join(document_dir, "pipeline")
    train_eval_dir = os.path.join(pipeline_dir, "train_eval")
    eval_txt_path  = os.path.join(train_eval_dir, "eval.txt")

    attempts = state.get("train_repair_attempts", 0) + 1

    # Read the evaluation report
    if os.path.isfile(eval_txt_path):
        with open(eval_txt_path, encoding="utf-8", errors="replace") as f:
            eval_txt = f.read()
    else:
        eval_txt = "(eval.txt not found)"

    # Collect extraction script IDs
    scripts_dir = os.path.join(pipeline_dir, "extraction_scripts")
    extraction_ids: list[str] = []
    if os.path.isdir(scripts_dir):
        extraction_ids = [
            os.path.splitext(name)[0]
            for name in os.listdir(scripts_dir)
            if name.endswith(".py") and not name.startswith("_")
        ]

    prompt = _build_prompt(eval_txt, extraction_ids)

    label = f"train_repair (attempt {attempts})"
    stop_event = threading.Event()
    spinner_result: dict = {}
    spinner_thread = threading.Thread(
        target=_spinner, args=(label, stop_event, spinner_result), daemon=True
    )
    spinner_thread.start()

    t0 = time.time()
    returncode, tokens = codex_silent(prompt, pipeline_dir, task=f"train_repair_{attempts}")
    duration = time.time() - t0

    spinner_result["tokens"] = tokens if returncode == 0 else None
    stop_event.set()
    spinner_thread.join(timeout=0.5)

    if returncode != 0:
        print(f"train_repair_node: codex exited {returncode} on attempt {attempts}")

    cost_entry: dict = {
        "phase": f"train_repair_{attempts}",
        "duration_s": round(duration, 3),
    }
    if tokens is not None:
        cost_entry["tokens"] = tokens

    return {
        "train_repair_attempts": attempts,
        "cost": [cost_entry],
    }
