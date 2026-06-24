from __future__ import annotations

import datetime
import json
import os
import time
from typing import Any

_PIPELINE_PHASES = {
    "template_inference", "record_separation", "run_record_separation",
    "block_separation", "run_block_separation", "block_samples",
    "block_extraction_scripting", "collect_extraction_scripts",
}


def save_metadata_node(state: dict[str, Any]) -> dict[str, Any]:
    """Aggregate run metrics and write metadata.json to the document directory.

    Reads:
        state["cost"]             – accumulated cost entries from all nodes
        state["agent_space"]      – used to derive document_dir
        state["doc_name"]         – document name
        state["actual_train_pages"] – number of train pages used
        state["eval_mode"]        – whether test eval was run
        state["test_start/end"]   – test page range (if eval_mode)

    Writes:
        document_dir/metadata.json
    """
    t0 = time.time()

    document_dir   = os.path.dirname(state["agent_space"])
    pipeline_dir   = os.path.join(document_dir, "pipeline")
    train_eval_dir = os.path.join(pipeline_dir, "train_eval")

    all_costs = state.get("cost", [])

    def _find(phase: str) -> dict:
        return next((c for c in all_costs if c["phase"] == phase), {})

    pipeline_phases = [c for c in all_costs if c["phase"] in _PIPELINE_PHASES]

    # ── train eval ────────────────────────────────────────────────────────────
    train_eval: dict = {}
    train_eval_json = os.path.join(train_eval_dir, "eval.json")
    if os.path.isfile(train_eval_json):
        with open(train_eval_json, encoding="utf-8") as f:
            ev = json.load(f)
        train_eval = {
            "accuracy":               ev.get("accuracy"),
            "exact_match":            ev.get("exact_match"),
            "agentic_pipeline_cost":  _find("train_grouping"),
            "ground_truth_cost":      _find("train_ground_truth"),
        }

    # ── test eval ─────────────────────────────────────────────────────────────
    test_eval: dict = {}
    test_eval_json = os.path.join(pipeline_dir, "eval.json")
    if os.path.isfile(test_eval_json):
        with open(test_eval_json, encoding="utf-8") as f:
            ev = json.load(f)
        test_eval = {
            "accuracy":               ev.get("accuracy"),
            "exact_match":            ev.get("exact_match"),
            "agentic_pipeline_cost":  _find("grouping"),
            "ground_truth_cost":      _find("ground_truth"),
        }

    # ── totals ────────────────────────────────────────────────────────────────
    total_duration = sum(c.get("duration_s", 0) for c in all_costs)
    total_tokens   = sum(c.get("tokens", 0) for c in all_costs) or None

    test_pages = None
    if state.get("test_start") and state.get("test_end"):
        test_pages = state["test_end"] - state["test_start"] + 1

    metadata = {
        "doc_name":        state.get("doc_name"),
        "timestamp":       datetime.datetime.now().isoformat(),
        "train_pages":     state.get("actual_train_pages"),
        "test_pages":      test_pages,
        "eval_mode":       state.get("eval_mode", False),
        "pipeline_phases": pipeline_phases,
        "train_eval":      train_eval,
        "test_eval":       test_eval,
        "totals": {
            "duration_s": round(total_duration, 3),
            "tokens":     total_tokens,
        },
    }

    metadata_path = os.path.join(document_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"save_metadata_node: metadata.json → {metadata_path}")

    duration = time.time() - t0
    return {"cost": [{"phase": "save_metadata", "duration_s": round(duration, 3)}]}
