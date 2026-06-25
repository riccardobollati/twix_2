from __future__ import annotations

import base64
import io
import json
import os
import time
from typing import Any

import pdfplumber
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.ground_truth_gen.prompt import build_prompt
from src.ground_truth_gen.ground_truth import _parse_json, _load_page_ocr
from src.agents.grouping import _maybe_merge_single_block


def _render_page_b64(page) -> str:
    buf = io.BytesIO()
    page.to_image(resolution=150).original.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ground_truth_node(state: dict[str, Any], llm: BaseChatModel, window: int = 3) -> dict[str, Any]:
    """Generate ground truth for all test pages and save to pipeline/ground_truth.json.

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

    system_prompt = build_prompt(template)

    records: list[dict] = []
    record_counter = 0
    total_input  = 0
    total_output = 0

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        windows = [list(range(i, min(i + window, total))) for i in range(0, total, window)]
        print(f"ground_truth_node: {total} page(s), {len(windows)} window(s) of up to {window}")

        for win_indices in windows:
            page_nums = [i + 1 for i in win_indices]
            pages_str = ", ".join(str(p) for p in page_nums)

            if len(page_nums) == 1:
                user_text = f"Extract all records from page {page_nums[0]} of the document."
            else:
                user_text = (
                    f"Extract all records from pages {pages_str} of the document. "
                    f"The images are provided in page order ({pages_str})."
                )

            content: list = [{"type": "text", "text": user_text}]
            for idx in win_indices:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{_render_page_b64(pdf.pages[idx])}"},
                })

            if ocr_path and os.path.isfile(ocr_path):
                for page_num in page_nums:
                    ocr_text = _load_page_ocr(ocr_path, page_num)
                    content.append({
                        "type": "text",
                        "text": f"OCR words on page {page_num} (exact character sequences, reading order):\n{ocr_text}",
                    })

            print(f"  pages {pages_str}...", end=" ", flush=True)
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=content),
            ])
            usage = response.usage_metadata or {}
            inp = usage.get("input_tokens") or 0
            out = usage.get("output_tokens") or 0
            if not inp and not out:
                out = usage.get("total_tokens") or 0
            total_input  += inp
            total_output += out
            try:
                parsed    = _parse_json(response.content)
                page_recs = parsed.get("records", [])
            except (json.JSONDecodeError, AttributeError) as exc:
                print(f"WARN: {exc}")
                print(f"  [RAW RESPONSE]\n{response.content}\n  [END RAW]")
                page_recs = []

            n_blocks = 0
            for rec in page_recs:
                blocks = rec.get("data", [])
                for block in blocks:
                    block["page"] = page_nums[0]
                records.append({"record_id": str(record_counter), "data": blocks})
                record_counter += 1
                n_blocks += len(blocks)
            print(f"{len(page_recs)} record(s), {n_blocks} block(s)")
            if not page_recs:
                print(f"  [RAW RESPONSE]\n{response.content}\n  [END RAW]")

    not_merged_path = os.path.join(pipeline_dir, "not_merged_gt.json")
    with open(not_merged_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"ground_truth_node: not_merged_gt.json → {not_merged_path}")

    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    _maybe_merge_single_block(gt_path, template)
    print(f"ground_truth_node: ground_truth.json → {gt_path}")

    duration = time.time() - t0
    cost: dict = {"phase": "ground_truth", "duration_s": round(duration, 3)}
    if total_input:
        cost["input_tokens"] = total_input
    if total_output:
        cost["output_tokens"] = total_output
    return {"cost": [cost]}
