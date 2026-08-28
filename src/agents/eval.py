from __future__ import annotations

import json
import os
import time
from typing import Any

from src.eval.convert import load_json_dedup
from src.eval.render import render_to_file
from src.eval.score import score as eval_score


def eval_node(state: dict[str, Any]) -> dict[str, Any]:
    """Score pipeline/extracted.json against pipeline/ground_truth.json.

    Reads:
        state["agent_space"] – used to derive document_dir/pipeline/

    Writes:
        document_dir/pipeline/eval.json
        document_dir/pipeline/eval.txt
    """
    t0 = time.time()

    document_dir = os.path.dirname(state["agent_space"])
    pipeline_dir = os.path.join(document_dir, "pipeline")

    extracted_path = os.path.join(pipeline_dir, "extracted.json")
    gt_path      = os.path.join(pipeline_dir, "ground_truth.json")

    # load with duplicate-key disambiguation so legacy templates with repeated
    # field names don't silently collapse to a single column
    candidate = load_json_dedup(extracted_path)
    gt = load_json_dedup(gt_path)

    eval_result = eval_score(gt, candidate, reference="model_reference", threshold=0.9)

    eval_json_path = os.path.join(pipeline_dir, "eval.json")
    eval_txt_path  = os.path.join(pipeline_dir, "eval.txt")

    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)
    render_to_file(eval_result, eval_txt_path)

    duration = time.time() - t0
    print(f"eval_node: accuracy={eval_result['accuracy']:.3f}  exact_match={eval_result['exact_match']}")
    print(f"  eval.json → {eval_json_path}")
    print(f"  eval.txt  → {eval_txt_path}")

    return {"cost": [{"phase": "eval", "duration_s": round(duration, 3)}]}
