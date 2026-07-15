"""Shared LLM ground-truth generator used by both train_eval and ground_truth nodes.

Both nodes generate ground truth the same way: the PDF is split into windows of
consecutive pages, each window is sent to the vision LLM in one call (pages as
images plus per-page OCR text), and records that continue across a window
boundary are merged in a final pass. Only the window size differs between the
train (5) and test (20) callers.
"""

from __future__ import annotations

import base64
import io
import json
from concurrent.futures import ThreadPoolExecutor
from os.path import isfile

import pdfplumber
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents._usage import split_tokens
from src.ground_truth_gen.prompt import build_prompt
from src.ground_truth_gen.ground_truth import _parse_json, _load_page_ocr


_MAX_WORKERS = 4  # parallel LLM calls for independent windows


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
    A record starting at the template's FIRST node (index 0) is a genuinely
    new record, never a continuation.
    """
    prev_blocks = prev.get("data", [])
    next_blocks = nxt.get("data",  [])
    if not prev_blocks or not next_blocks:
        return False
    last_idx  = order.get(str(prev_blocks[-1].get("block_id", "")), -1)
    first_idx = order.get(str(next_blocks[0].get("block_id",  "")), -1)
    return last_idx >= 0 and first_idx > 0 and first_idx >= last_idx


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


def generate_ground_truth(
    pdf_path: str,
    ocr_csv: str | None,
    template: list[dict],
    llm: BaseChatModel,
    window: int,
) -> tuple[list[dict], list[dict], dict]:
    """Generate LLM ground-truth records for every page of pdf_path.

    Windows are sent to the LLM in parallel (up to _MAX_WORKERS at a time) and
    the results are consumed in window order before the boundary-merge pass.

    Returns (records, raw_records, usage):
        records     – merged records (boundary-aware merge pass applied)
        raw_records – per-window records before the merge pass
        usage       – {"input_tokens", "output_tokens", "total_tokens"} sums
                      (total_tokens counts calls whose input/output split was unknown)
    """
    system_prompt = build_prompt(template)
    order = _node_order(template)

    # ── build one payload per window (page rendering is not thread-safe) ──────
    payloads: list[tuple[list[int], list]] = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        windows = [list(range(i, min(i + window, total))) for i in range(0, total, window)]
        print(f"  {total} page(s), {len(windows)} window(s) of up to {window}")

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

            if ocr_csv and isfile(ocr_csv):
                for page_num in page_nums:
                    ocr_text = _load_page_ocr(ocr_csv, page_num)
                    content.append({
                        "type": "text",
                        "text": f"OCR words on page {page_num} (exact character sequences, reading order):\n{ocr_text}",
                    })

            payloads.append((page_nums, content))

    # ── parallel LLM calls, consumed in window order ──────────────────────────
    def _call(content: list):
        return llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=content),
        ])

    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(payloads))) as executor:
        futures = [executor.submit(_call, content) for _, content in payloads]
        responses = [f.result() for f in futures]  # futures are in window order

    input_tokens = output_tokens = total_tokens = 0
    all_windows: list[tuple[list[int], list[dict]]] = []   # (page_nums, stamped_recs)

    for (page_nums, _), response in zip(payloads, responses):
        pages_str = ", ".join(str(p) for p in page_nums)
        inp, out, tot = split_tokens(response.usage_metadata or {})
        input_tokens  += inp
        output_tokens += out
        total_tokens  += tot

        try:
            parsed    = _parse_json(response.content)
            page_recs = parsed.get("records", [])
        except (json.JSONDecodeError, AttributeError) as exc:
            print(f"  pages {pages_str}... WARN: {exc}")
            print(f"  [RAW RESPONSE]\n{response.content}\n  [END RAW]")
            page_recs = []

        # stamp pages on every block, then store for boundary-merge pass
        n_blocks = 0
        for rec in page_recs:
            for block in rec.get("data", []):
                block["page"] = page_nums[0]          # backward compat
                block["window_pages"] = page_nums     # real page range of the window
            n_blocks += len(rec.get("data", []))
        print(f"  pages {pages_str}... {len(page_recs)} record(s), {n_blocks} block(s)")

        all_windows.append((page_nums, page_recs))

    # ── raw per-window records (pre-merge) ────────────────────────────────────
    raw_records: list[dict] = []
    for _, page_recs in all_windows:
        for rec in page_recs:
            raw_records.append({"record_id": str(len(raw_records)), "data": rec.get("data", [])})

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

    usage = {
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "total_tokens":  total_tokens,
    }
    return records, raw_records, usage
