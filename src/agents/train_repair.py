from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from src.agents._ui import spinner as _spinner
from src.models.catalog import resolve_hard_model
from src.models.opencode import opencode_silent


def _repair_model(state: dict[str, Any] | None = None) -> str:
    """Return the configured hard-model target for every repair session.

    Extraction generation can be routed per block, but repairs operate on the
    whole pipeline and therefore always need the hard target.  Keep the
    fallback chain aligned with ``block_difficulty_router`` for installations
    that configure only ``OPENCODE_MODEL``.
    """
    return resolve_hard_model((state or {}).get("hard_model"))


def _build_prompt(eval_txt: str, extraction_script_ids: list[str], health_problems: list[str]) -> str:
    scripts_list = "\n".join(
        f"  - extraction_scripts/{sid}.py" for sid in sorted(extraction_script_ids)
    )
    health_section = ""
    if health_problems:
        problems = "\n".join(f"  - {p}" for p in health_problems)
        health_section = f"""
PIPELINE-HEALTH VIOLATIONS  (from train_eval/pipeline_health.json)
==================================================================
These are HARD failures — the run is rejected while any of them remains, no
matter how good the accuracy looks:
{problems}
"""
    return f"""You are debugging a PDF data-extraction pipeline. Fix the Python scripts \
in the current directory so the pipeline output better matches the ground truth.
{health_section}

PIPELINE OVERVIEW
=================
The pipeline runs in this order:
1. block_separation.py <ocr.csv> <out_dir>
   Detects typed blocks (kv, table) within the OCR data.
   Writes: out_dir/block_separation.json

2. extraction_scripts/<id>.py  — each defines  extract(block: dict) -> dict
   Extracts field values from one physical block.
   Called by pipeline.py on every block produced by block_separation.

3. record_grouping.py
   Deterministically traverses extracted template node ids and writes grouped.json.
   It is fixed pipeline code and is not generated or repairable.

EVALUATION ERRORS  (from train_eval/eval.txt)
=============================================
{eval_txt}

FILES YOU MAY MODIFY
====================
  - block_separation.py
{scripts_list}

DO NOT modify pipeline.py, template.json, or any other file.

PRIORITY FIXES BY ERROR CODE
=============================
[R-RECORD-MISSING]
  A logical record has no candidate counterpart. Usually a required boundary
  block was missed; fix block_separation.py so the template-node sequence is
  complete and ordered.

[R-RECORD-EXTRA]
  A spurious logical record was formed. Check block_separation.py for a
  false-positive or misordered block that creates an artificial restart.

[R-NODE-MISSING]
  A block type was not detected on some pages.
  Fix block_separation.py: improve header-row detection or word-assignment logic.

[R-NODE-EXTRA]
  Spurious extra blocks are being emitted.
  Fix block_separation.py or the relevant extraction script — add a guard so the
  rule only fires when the block clearly exists on the page.
  NOTE: per-page continuation blocks and consecutive occurrences of the same
  node remain separate by design. NEVER absorb several physical blocks into one.

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
3. NEVER force output counts (blocks or rows) to match the training-data counts.
   Counts must emerge from the algorithm — never be hard-coded.
4. NEVER branch on the document name, page-number values, or any fixed metadata.
5. Every modified script must work correctly on pages it has never seen.
6. Blocks are PAGE-LOCAL (validated after every run; a violation rejects the run):
   a block emitted under page P in block_separation.json may contain ONLY page-P
   words. Table rows continuing onto later pages must be emitted as separate
   continuation blocks of the same node on each page. They remain separate in
   extracted.json. Never absorb later pages' words into one block and never
   synthesize a global cross-page y-axis.
7. NEVER suppress or drop kv blocks based on page position/index or on how many
   blocks of a type were detected on a page.

Read the existing scripts first to understand the current logic, then make targeted
fixes that address the errors listed above without breaking what already works.
"""


def train_repair_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run OpenCode to fix pipeline scripts based on the training eval report.

    Reads:
        state["agent_space"]           – used to derive pipeline_dir
        state["train_repair_attempts"] – number of repairs already completed

    Side-effect:
        Modifies block_separation.py and/or extraction_scripts/*.py inside
        pipeline_dir.
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

    # Read pipeline-health violations (they fail the gate regardless of accuracy)
    health_problems: list[str] = []
    health_path = os.path.join(train_eval_dir, "pipeline_health.json")
    if os.path.isfile(health_path):
        try:
            with open(health_path, encoding="utf-8") as f:
                health = json.load(f)
            if health.get("degenerate"):
                health_problems = health.get("problems", [])
        except (OSError, json.JSONDecodeError):
            pass

    # Collect extraction script IDs
    scripts_dir = os.path.join(pipeline_dir, "extraction_scripts")
    extraction_ids: list[str] = []
    if os.path.isdir(scripts_dir):
        extraction_ids = [
            os.path.splitext(name)[0]
            for name in os.listdir(scripts_dir)
            if name.endswith(".py") and not name.startswith("_")
        ]

    prompt = _build_prompt(eval_txt, extraction_ids, health_problems)

    label = f"train_repair (attempt {attempts})"
    stop_event = threading.Event()
    spinner_result: dict = {}
    spinner_thread = threading.Thread(
        target=_spinner, args=(label, stop_event, spinner_result), daemon=True
    )
    spinner_thread.start()

    t0 = time.time()
    debug_dir = os.path.join(document_dir, "debug")
    log_path  = os.path.join(debug_dir, f"train_repair_{attempts}.log") if state.get("save_log") else None
    returncode, input_t, output_t, total_t = opencode_silent(
        prompt,
        pipeline_dir,
        task=f"train_repair_{attempts}",
        model=_repair_model(state),
        caveman=state.get("use_caveman", False),
        log_path=log_path,
    )
    duration = time.time() - t0

    shown_t = (input_t or 0) + (output_t or 0) + (total_t or 0) or None
    spinner_result["tokens"] = shown_t
    spinner_result["ok"] = returncode == 0
    stop_event.set()
    spinner_thread.join(timeout=0.5)

    if returncode != 0:
        print(f"train_repair_node: OpenCode exited {returncode} on attempt {attempts}")

    cost_entry: dict = {
        "phase": f"train_repair_{attempts}",
        "duration_s": round(duration, 3),
    }
    if input_t is not None:
        cost_entry["input_tokens"] = input_t
    if output_t is not None:
        cost_entry["output_tokens"] = output_t
    if total_t is not None:
        cost_entry["total_tokens"] = total_t

    return {
        "train_repair_attempts": attempts,
        "cost": [cost_entry],
    }
