from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from src.eval.convert import load_json_dedup
from src.eval.score import score as eval_score
from src.eval.render import render_to_file
from src.agents._usage import apply_usage
from src.agents.gt_common import generate_ground_truth
from src.agents.grouping import _maybe_merge_single_block

WINDOW = 5
PIPELINE_TIMEOUT_S = 300  # hard cap for one pipeline.py run


def train_eval_node(state: dict[str, Any], llm: BaseChatModel) -> dict[str, Any]:
    """Run the pipeline on the train PDF and evaluate against generated ground truth.

    Reads:
        state["agent_space"]  – used to derive document_dir/pipeline/
        state["pdf_path"]     – train PDF
        state["ocr_csv_path"] – train OCR CSV
        state["template"]     – inferred template

    Writes:
        document_dir/pipeline/train_eval/grouped.json
        document_dir/pipeline/train_eval/ground_truth.json
        document_dir/pipeline/train_eval/eval.json
        document_dir/pipeline/train_eval/eval.txt
    """
    document_dir   = os.path.dirname(state["agent_space"])
    pipeline_dir   = os.path.join(document_dir, "pipeline")
    train_eval_dir = os.path.join(pipeline_dir, "train_eval")
    os.makedirs(train_eval_dir, exist_ok=True)

    pipeline_py = os.path.join(pipeline_dir, "pipeline.py")
    pdf_path    = state["pdf_path"]
    ocr_path    = state["ocr_csv_path"]
    template    = state["template"]

    cost_entries: list[dict] = []

    # ── 1. Run pipeline on train PDF ──────────────────────────────────────────
    print("train_eval_node: running pipeline on train PDF...")
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, pipeline_py, pdf_path, "--out-dir", train_eval_dir],
            capture_output=True,
            text=True,
            timeout=PIPELINE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"pipeline.py timed out after {PIPELINE_TIMEOUT_S}s on train PDF")
    duration = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(f"pipeline.py failed on train PDF:\n{result.stderr}")
    if result.stdout:
        print(result.stdout, end="")
    cost_entries.append({"phase": "train_grouping", "duration_s": round(duration, 3)})

    grouped_path = os.path.join(train_eval_dir, "grouped.json")
    _maybe_merge_single_block(grouped_path, template)

    # ── 2. Generate ground truth for train pages (only on first run) ─────────
    gt_path = os.path.join(train_eval_dir, "ground_truth.json")
    if os.path.isfile(gt_path):
        print("train_eval_node: reusing existing ground truth (repair retry)")
        gt_duration = 0.0
        gt_usage    = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    else:
        print(f"train_eval_node: generating train ground truth (window={WINDOW})...")
        t0 = time.time()
        records, _, gt_usage = generate_ground_truth(pdf_path, ocr_path, template, llm, WINDOW)
        gt_duration = time.time() - t0
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        _maybe_merge_single_block(gt_path, template)

    gt_cost: dict = {"phase": "train_ground_truth", "duration_s": round(gt_duration, 3)}
    apply_usage(gt_cost, gt_usage["input_tokens"], gt_usage["output_tokens"], gt_usage["total_tokens"])
    cost_entries.append(gt_cost)

    # ── 3. Evaluate ───────────────────────────────────────────────────────────
    t0 = time.time()
    gold = load_json_dedup(gt_path)
    candidate = load_json_dedup(grouped_path)

    eval_result  = eval_score(gold, candidate, reference="model_reference", threshold=0.9)
    eval_duration = time.time() - t0

    eval_json_path = os.path.join(train_eval_dir, "eval.json")
    eval_txt_path  = os.path.join(train_eval_dir, "eval.txt")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)
    render_to_file(eval_result, eval_txt_path)

    print(
        f"train_eval_node: accuracy={eval_result['accuracy']:.3f}  "
        f"exact_match={eval_result['exact_match']}"
    )
    cost_entries.append({"phase": "train_eval", "duration_s": round(eval_duration, 3)})

    # ── 4. Append to accuracy history (tracks each loop iteration) ───────────
    history_path = os.path.join(train_eval_dir, "eval_history.json")
    if os.path.isfile(history_path):
        with open(history_path, encoding="utf-8") as f:
            history: list[dict] = json.load(f)
    else:
        history = []
    history.append({
        "attempt":     state.get("train_repair_attempts", 0),
        "accuracy":    eval_result["accuracy"],
        "exact_match": eval_result["exact_match"],
    })
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    return {"cost": cost_entries}
