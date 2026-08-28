from __future__ import annotations

import json
import os
import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from src.agents._usage import apply_usage
from src.agents.gt_common import generate_ground_truth


def ground_truth_node(
    state: dict[str, Any],
    llm: BaseChatModel,
    window: int = 3,
    *,
    include_images: bool = False,
) -> dict[str, Any]:
    """Generate ground truth for all test pages and save to pipeline/ground_truth.json.

    Delegates to the shared generator in gt_common (same semantics as the train
    ground truth, only the window size differs).

    Reads:
        state["agent_space"]    – used to derive document_dir/pipeline/
        state["test_pdf_path"]  – test PDF (used when eval_mode=True)
        state["test_ocr_path"]  – test OCR CSV (used as LLM context)
        state["pdf_path"]       – train PDF fallback (used when eval_mode=False)
        state["ocr_csv_path"]   – train OCR fallback
        state["template"]       – inferred template (from template_inference_node)

    Writes:
        document_dir/pipeline/ground_truth.json
    """
    t0 = time.time()

    document_dir = os.path.dirname(state["agent_space"])
    pipeline_dir = os.path.join(document_dir, "pipeline")
    gt_path      = os.path.join(pipeline_dir, "ground_truth.json")
    pdf_path     = state.get("test_pdf_path") or state["pdf_path"]
    ocr_path     = state.get("test_ocr_path") or state["ocr_csv_path"]
    template     = state["template"]

    print(f"ground_truth_node: window={window}")
    blocks, usage = generate_ground_truth(
        pdf_path,
        ocr_path,
        template,
        llm,
        window,
        include_images=include_images,
    )

    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=2, ensure_ascii=False)
    print(f"ground_truth_node: ground_truth.json → {gt_path}")

    duration = time.time() - t0
    cost: dict = {"phase": "ground_truth", "duration_s": round(duration, 3)}
    apply_usage(cost, usage["input_tokens"], usage["output_tokens"], usage["total_tokens"])
    if usage.get("empty_result_fallback"):
        cost["empty_result_fallback"] = True
        cost["initial_window"] = usage["initial_window"]
        cost["effective_window"] = usage["effective_window"]
    return {"cost": [cost]}
