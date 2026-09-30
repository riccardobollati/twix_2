from __future__ import annotations

import datetime
import json
import os
import time
from typing import Any

_PIPELINE_PHASES = {
    "cluster_sampling", "template_inference", "block_separation", "run_block_separation",
    "block_artifacts", "block_samples",
    "block_difficulty_router", "block_extraction_scripting", "collect_extraction_scripts",
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

    def _find_last(phase: str) -> dict:
        matches = [c for c in all_costs if c["phase"] == phase]
        return matches[-1] if matches else {}

    pipeline_phases = [c for c in all_costs if c["phase"] in _PIPELINE_PHASES]

    # ── train eval ────────────────────────────────────────────────────────────
    train_eval: dict = {}
    train_eval_json = os.path.join(train_eval_dir, "eval.json")
    if os.path.isfile(train_eval_json):
        with open(train_eval_json, encoding="utf-8") as f:
            ev = json.load(f)

        # Accuracy history across loop iterations
        history_path = os.path.join(train_eval_dir, "eval_history.json")
        iterations: list[dict] | None = None
        if os.path.isfile(history_path):
            with open(history_path, encoding="utf-8") as f:
                iterations = json.load(f)

        # Repair cost entries (one per loop iteration)
        repairs = [c for c in all_costs if c["phase"].startswith("train_repair_")]

        train_eval = {
            "accuracy":              ev.get("accuracy"),
            "exact_match":           ev.get("exact_match"),
            "iterations":            iterations,
            "repairs":               repairs if repairs else None,
            "agentic_pipeline_cost": _find_last("train_extraction"),
            "ground_truth_cost":     _find("train_ground_truth"),
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
            "agentic_pipeline_cost":  _find("extraction"),
            "ground_truth_cost":      _find("ground_truth"),
            "ground_truth_model":     state.get("ground_truth_model"),
        }
        # degenerate-output flag from pipeline.py (empty/partial extraction):
        # an accuracy produced by a degenerate run is a pipeline abort, not a
        # model score — keep the flag next to it so charts can tell them apart
        health_path = os.path.join(pipeline_dir, "pipeline_health.json")
        if os.path.isfile(health_path):
            with open(health_path, encoding="utf-8") as f:
                health = json.load(f)
            if health.get("degenerate"):
                test_eval["degenerate_pipeline"] = True
                test_eval["pipeline_problems"] = health.get("problems", [])
    elif state.get("eval_mode") and state.get("manual_ground_truth"):
        test_eval = {
            "status": "awaiting_manual_ground_truth",
            "agentic_pipeline_cost": _find("extraction"),
            "ground_truth_source": "manual_ui_submission",
        }

    # ── totals ────────────────────────────────────────────────────────────────
    total_duration = sum(c.get("duration_s", 0) for c in all_costs)
    total_input    = sum(c.get("input_tokens",  0) for c in all_costs) or None
    total_output   = sum(c.get("output_tokens", 0) for c in all_costs) or None

    test_pages = None
    if state.get("eval_mode"):
        if state.get("test_source_pages") is not None:
            test_pages = len(state["test_source_pages"])
        elif state.get("test_start") and state.get("test_end"):
            test_pages = state["test_end"] - state["test_start"] + 1

    metadata = {
        "doc_name":         state.get("doc_name"),
        "source_file":      os.path.basename(state["full_pdf_path"]),
        "source_sha256":    state.get("source_sha256"),
        "timestamp":        datetime.datetime.now().isoformat(),
        "source_start_page": state.get("detected_start_page", 1),
        "skipped_front_pages": max(0, state.get("detected_start_page", 1) - 1),
        "beginning_detection": {
            "confidence": state.get("beginning_confidence"),
            "reason": state.get("beginning_reason"),
            "cost": _find("detect_beginning"),
        },
        "train_pages":      state.get("actual_train_pages"),
        "test_pages":       test_pages,
        "train_source_pages": state.get("train_source_pages"),
        "build_source_pages": state.get("build_source_pages"),
        "test_source_pages": state.get("test_source_pages"),
        "first_source_pages": state.get("first_source_pages"),
        "cluster_sample_pages": state.get("cluster_sample_pages"),
        "cluster_coverage": state.get("cluster_coverage"),
        "table_cluster_count": state.get("table_cluster_count"),
        "kv_cluster_count": state.get("kv_cluster_count"),
        "eval_mode":        state.get("eval_mode", False),
        "llm_model":        state.get("llm_model"),
        "difficulty_model": state.get("difficulty_model"),
        "ground_truth_model": state.get("ground_truth_model"),
        "opencode_model":   state.get("opencode_model"),
        "easy_model":       state.get("easy_model"),
        "hard_model":       state.get("hard_model"),
        "reference_model":  state.get("reference_model"),
        "template_model":   state.get("template_model"),
        "block_models":     state.get("block_opencode_models"),
        "run_config": {
            "requested_train_pages": state.get("train_pages"),
            "requested_eval_pages": state.get("eval_pages"),
            "holdout_pages": state.get("holdout_pages"),
            "no_routing": state.get("no_routing", False),
            "easy_model": state.get("easy_model"),
            "hard_model": state.get("hard_model"),
            "reference_model": state.get("reference_model"),
            "template_model": state.get("template_model"),
            "train_accuracy_threshold": state.get("train_accuracy_threshold", 0.0),
            "train_repair_max_retries": state.get("train_repair_max_retries", 3),
            "ground_truth_test_window": state.get("gt_test_window", 20),
            "manual_ground_truth": state.get("manual_ground_truth", False),
        },
        "pipeline_phases":  pipeline_phases,
        "train_eval":       train_eval,
        "test_eval":        test_eval,
        "totals": {
            "duration_s":    round(total_duration, 3),
            "input_tokens":  total_input,
            "output_tokens": total_output,
        },
    }

    metadata_path = os.path.join(document_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"save_metadata_node: metadata.json → {metadata_path}")

    duration = time.time() - t0
    return {"cost": [{"phase": "save_metadata", "duration_s": round(duration, 3)}]}
