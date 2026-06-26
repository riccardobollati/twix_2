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


def _node_order(template: list[dict]) -> dict[str, int]:
    """Map node_id → 0-based position in the template list."""
    return {str(node.get("node_id", i)): i for i, node in enumerate(template)}


def _should_merge(prev: dict, nxt: dict, order: dict[str, int]) -> bool:
    """True when nxt is a continuation of prev.

    A continuation means the first block of nxt has a template index
    >= the last block of prev (the template node sequence has not reset).
    """
    prev_blocks = prev.get("data", [])
    next_blocks = nxt.get("data",  [])
    if not prev_blocks or not next_blocks:
        return False
    last_idx  = order.get(str(prev_blocks[-1].get("block_id", "")), -1)
    first_idx = order.get(str(next_blocks[0].get("block_id",  "")), -1)
    return last_idx >= 0 and first_idx >= 0 and first_idx >= last_idx


def _merge_records(prev: dict, nxt: dict) -> dict:
    """Merge nxt into prev.

    If both share the same node at the boundary (e.g. a table that spans a
    window cut), their block data is concatenated; otherwise the blocks are
    simply appended.
    """
    prev_blocks = prev.get("data", [])
    next_blocks = nxt.get("data",  [])
    if not prev_blocks or not next_blocks:
        return {**prev, "data": prev_blocks + next_blocks}

    last_bid  = str(prev_blocks[-1].get("block_id", ""))
    first_bid = str(next_blocks[0].get("block_id",  ""))

    if last_bid == first_bid:
        ba, bb = prev_blocks[-1], next_blocks[0]
        if ba.get("type") == "table":
            rows_a = ba["data"] if isinstance(ba.get("data"), list) else []
            rows_b = bb["data"] if isinstance(bb.get("data"), list) else []
            merged_block = {**ba, "data": rows_a + rows_b}
        else:  # kv — keep non-empty values from bb, fall back to ba
            da = ba.get("data") if isinstance(ba.get("data"), dict) else {}
            db = bb.get("data") if isinstance(bb.get("data"), dict) else {}
            merged_block = {**ba, "data": {**da, **{k: v for k, v in db.items() if v}}}
        merged_data = prev_blocks[:-1] + [merged_block] + next_blocks[1:]
    else:
        merged_data = prev_blocks + next_blocks

    return {**prev, "data": merged_data}


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

    order        = _node_order(template)
    total_input  = 0
    total_output = 0
    all_windows: list[tuple[list[int], list[dict]]] = []   # (page_nums, stamped_recs)

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

            # stamp page on every block, then store for boundary-merge pass
            n_blocks = 0
            for rec in page_recs:
                for block in rec.get("data", []):
                    block["page"] = page_nums[0]
                n_blocks += len(rec.get("data", []))
            print(f"{len(page_recs)} record(s), {n_blocks} block(s)")
            if not page_recs:
                print(f"  [RAW RESPONSE]\n{response.content}\n  [END RAW]")

            all_windows.append((page_nums, page_recs))

    # ── boundary-aware merge pass ─────────────────────────────────────────────
    records: list[dict] = []
    for _, page_recs in all_windows:
        if not page_recs:
            continue
        if records and _should_merge(records[-1], page_recs[0], order):
            records[-1] = _merge_records(records[-1], page_recs[0])
            rest = page_recs[1:]
        else:
            rest = page_recs
        for rec in rest:
            records.append({"record_id": str(len(records)), "data": rec.get("data", [])})

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
