"""Shared LLM ground-truth generator used by train and held-out evaluation.

Both nodes generate ground truth the same way: the PDF is split into windows of
consecutive pages and records that continue across a window boundary are merged
in a final pass. Train evaluation sends layout-preserving OCR to DeepSeek. Final
held-out evaluation sends page images plus exact OCR text to OpenAI.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from os.path import isfile

import pdfplumber
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents._usage import split_tokens
from src.ground_truth_gen.prompt import build_prompt
from src.ground_truth_gen.helpers import (
    load_ocr_pages,
    parse_json_response,
    render_page_b64,
)


_MAX_WORKERS = 4  # parallel LLM calls for independent windows

# GT-volume sanity check: a window whose extracted text is below this fraction
# of its pages' OCR text almost certainly under-extracted (truncated response,
# only-first-page output, JSON parse failure) and is retried once.
_MIN_GT_COVERAGE = 0.2
_MIN_OCR_CHARS = 500  # skip the check on near-empty pages
_EMPTY_GT_FALLBACK_WINDOW = 5


def _node_order(template: list[dict]) -> dict[str, int]:
    """Map node_id → 0-based position in the template list."""
    return {str(node.get("node_id", i)): i for i, node in enumerate(template)}


def _identity_dict(block: dict) -> dict | None:
    """Key→value dict usable as a record identity: a kv block's data, or the
    single row of a one-row table (header nodes are often typed as tables)."""
    data = block.get("data")
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        return data[0]
    return None


def _same_kv_identity(prev: dict, nxt: dict) -> bool:
    """True when nxt's leading block repeats an identity block of prev with the
    same values — the same logical record whose header reprints on every page
    (e.g. a per-page contract header). Requires at least one shared non-empty
    value and no conflicting non-empty values."""
    next_blocks = nxt.get("data", [])
    if not next_blocks:
        return False
    fd = _identity_dict(next_blocks[0])
    if fd is None:
        return False
    first_bid = str(next_blocks[0].get("block_id", ""))
    for block in prev.get("data", []):
        if str(block.get("block_id", "")) != first_bid:
            continue
        bd = _identity_dict(block)
        if bd is None:
            continue
        fd_items = {(k, str(v).strip()) for k, v in fd.items() if v}
        bd_items = {(k, str(v).strip()) for k, v in bd.items() if v}
        if fd_items and fd_items == bd_items:
            return True
    return False


def _should_merge(prev: dict, nxt: dict, order: dict[str, int]) -> bool:
    """True when nxt is a continuation of prev.

    A continuation means the first block of nxt has a template index
    >= the last block of prev (the template node sequence has not reset).
    A record starting at the template's FIRST node (index 0) is a genuinely
    new record — unless its leading kv block repeats the previous record's
    values (same contract/order id), in which case the header merely
    reprinted at the window cut and the record continues.
    """
    prev_blocks = prev.get("data", [])
    next_blocks = nxt.get("data",  [])
    if not prev_blocks or not next_blocks:
        return False
    if _same_kv_identity(prev, nxt):
        return True
    last_idx  = order.get(str(prev_blocks[-1].get("block_id", "")), -1)
    first_idx = order.get(str(next_blocks[0].get("block_id",  "")), -1)
    return last_idx >= 0 and first_idx > 0 and first_idx >= last_idx


def _merge_records(prev: dict, nxt: dict) -> dict:
    """Merge nxt into prev, block by block.

    A next-block whose block_id already exists in prev is the same template
    node seen again across the window cut: table rows are concatenated, kv
    values are unioned (non-empty values from nxt win). Blocks of nodes prev
    has not seen yet are appended. This covers both a table split mid-window
    and a per-page kv header that reprints at the top of every window.
    """
    merged = list(prev.get("data", []))
    by_bid = {str(b.get("block_id", "")): i for i, b in enumerate(merged)}

    for bb in nxt.get("data", []):
        bid = str(bb.get("block_id", ""))
        i = by_bid.get(bid)
        if i is None:
            by_bid[bid] = len(merged)
            merged.append(bb)
            continue
        ba = merged[i]
        if ba.get("type") == "table":
            rows_a = ba["data"] if isinstance(ba.get("data"), list) else []
            rows_b = bb["data"] if isinstance(bb.get("data"), list) else []
            if rows_a == rows_b:
                # identical block reprinted at the window cut (per-page header
                # block) — not new rows
                merged[i] = ba
            else:
                merged[i] = {**ba, "data": rows_a + rows_b}
        else:  # kv — keep non-empty values from bb, fall back to ba
            da = ba.get("data") if isinstance(ba.get("data"), dict) else {}
            db = bb.get("data") if isinstance(bb.get("data"), dict) else {}
            merged[i] = {**ba, "data": {**da, **{k: v for k, v in db.items() if v}}}

    return {**prev, "data": merged}


def _gt_chars(page_recs: list[dict]) -> int:
    """Total characters of extracted values — proxy for how much of the
    window's content the LLM actually returned."""
    total = 0
    for rec in page_recs:
        for block in rec.get("data", []):
            data = block.get("data")
            if isinstance(data, list):
                for row in data:
                    if isinstance(row, dict):
                        total += sum(len(str(v)) for v in row.values() if v)
            elif isinstance(data, dict):
                total += sum(len(str(v)) for v in data.values() if v)
    return total


def _generate_ground_truth_once(
    pdf_path: str,
    ocr_csv: str | None,
    template: list[dict],
    llm: BaseChatModel,
    window: int,
    *,
    include_images: bool = False,
) -> tuple[list[dict], list[dict], dict]:
    """Generate LLM ground-truth records once for every page of pdf_path.

    Windows are sent to the LLM in parallel (up to _MAX_WORKERS at a time) and
    the results are consumed in window order before the boundary-merge pass.

    Returns (records, raw_records, usage):
        records     – merged records (boundary-aware merge pass applied)
        raw_records – per-window records before the merge pass
        usage       – {"input_tokens", "output_tokens", "total_tokens"} sums
                      (total_tokens counts calls whose input/output split was unknown)
    """
    if (not ocr_csv or not isfile(ocr_csv)) and not include_images:
        raise RuntimeError(
            "DeepSeek ground-truth generation requires an OCR CSV because "
            "DeepSeek V4 does not accept page images"
        )

    system_prompt = build_prompt(template, include_images=include_images)
    order = _node_order(template)

    # ── build one text or multimodal payload per window ───────────────────────
    payloads: list[tuple[list[int], str | list[dict], int]] = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        windows = [list(range(i, min(i + window, total))) for i in range(0, total, window)]
        ocr_pages = (
            load_ocr_pages(ocr_csv, range(1, total + 1))
            if ocr_csv and isfile(ocr_csv)
            else {}
        )
        print(f"  {total} page(s), {len(windows)} window(s) of up to {window}")

        for win_indices in windows:
            page_nums = [i + 1 for i in win_indices]
            pages_str = ", ".join(str(p) for p in page_nums)

            if len(page_nums) == 1:
                user_text = f"Extract all records from page {page_nums[0]} of the document."
            else:
                source_description = "images" if include_images else "OCR layouts"
                user_text = (
                    f"Extract all records from pages {pages_str} of the document. "
                    f"The {source_description} are provided in page order ({pages_str})."
                )

            if include_images:
                content: str | list[dict] = [{"type": "text", "text": user_text}]
                for idx in win_indices:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{render_page_b64(pdf.pages[idx])}"
                        },
                    })
            else:
                content = user_text

            ocr_chars = 0
            if ocr_csv and isfile(ocr_csv):
                for page_num in page_nums:
                    ocr_text = ocr_pages[page_num]["text"]
                    ocr_chars += len(ocr_text)
                    if include_images:
                        content.append({
                            "type": "text",
                            "text": (
                                f"OCR words on page {page_num} (exact character "
                                f"sequences, reading order):\n{ocr_text}"
                            ),
                        })
                    else:
                        ocr_layout = ocr_pages[page_num]["layout"]
                        content += (
                            f"\n\nPage {page_num} OCR layout (exact text with "
                            f"PDF-point coordinates):\n{ocr_layout or '[no OCR words]'}"
                        )

            payloads.append((page_nums, content, ocr_chars))

    # ── parallel LLM calls, consumed in window order ──────────────────────────
    def _call(content: str | list[dict]):
        return llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=content),
        ])

    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(payloads))) as executor:
        futures = [executor.submit(_call, content) for _, content, _ in payloads]
        responses = [f.result() for f in futures]  # futures are in window order

    input_tokens = output_tokens = total_tokens = 0

    def _parse_window(response, pages_str: str) -> list[dict]:
        nonlocal input_tokens, output_tokens, total_tokens
        inp, out, tot = split_tokens(response.usage_metadata or {})
        input_tokens  += inp
        output_tokens += out
        total_tokens  += tot
        try:
            return parse_json_response(response.content).get("records", [])
        except (json.JSONDecodeError, AttributeError) as exc:
            print(f"  pages {pages_str}... WARN: {exc}")
            print(f"  [RAW RESPONSE]\n{response.content}\n  [END RAW]")
            return []

    all_windows: list[tuple[list[int], list[dict]]] = []   # (page_nums, stamped_recs)

    for (page_nums, content, ocr_chars), response in zip(payloads, responses):
        pages_str = ", ".join(str(p) for p in page_nums)
        page_recs = _parse_window(response, pages_str)

        # coverage sanity check: retry once when the window returned far less
        # text than its pages contain (truncated/partial responses otherwise
        # become silently incomplete ground truth)
        if ocr_chars >= _MIN_OCR_CHARS:
            coverage = _gt_chars(page_recs) / ocr_chars
            if coverage < _MIN_GT_COVERAGE:
                print(
                    f"  pages {pages_str}... WARN: low GT coverage "
                    f"({coverage:.0%} of {ocr_chars} OCR chars) — retrying window"
                )
                retry_instruction = (
                    "Your previous attempt returned far less data than these pages "
                    "contain. Extract EVERY record and EVERY row from ALL the pages "
                    "listed above — do not stop after the first page."
                )
                if isinstance(content, list):
                    retry_content = content + [{"type": "text", "text": retry_instruction}]
                else:
                    retry_content = content + "\n\n" + retry_instruction
                retry_recs = _parse_window(_call(retry_content), pages_str + " (retry)")
                if _gt_chars(retry_recs) > _gt_chars(page_recs):
                    page_recs = retry_recs
                coverage = _gt_chars(page_recs) / ocr_chars
                if coverage < _MIN_GT_COVERAGE:
                    print(
                        f"  pages {pages_str}... WARN: GT coverage still low after retry "
                        f"({coverage:.0%}) — ground truth for this window is likely incomplete"
                    )

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


def generate_ground_truth(
    pdf_path: str,
    ocr_csv: str | None,
    template: list[dict],
    llm: BaseChatModel,
    window: int,
    *,
    include_images: bool = False,
) -> tuple[list[dict], list[dict], dict]:
    """Generate ground truth, retrying an empty result with 5-page windows.

    Large windows can exhaust the model's output-token budget and yield a
    parseable but empty ``records`` list. Such output is never valid for the
    structured PDFs reaching this phase. Retry the complete generation once
    with smaller windows, aggregate both attempts' usage, and fail explicitly
    if the fallback is still empty instead of producing a misleading score.
    """
    records, raw_records, usage = _generate_ground_truth_once(
        pdf_path, ocr_csv, template, llm, window, include_images=include_images
    )
    if records:
        return records, raw_records, usage

    if window <= _EMPTY_GT_FALLBACK_WINDOW:
        raise RuntimeError(
            f"ground-truth generation returned no records with window={window}"
        )

    print(
        "  WARNING: ground-truth generation returned an empty record list; "
        f"retrying all pages with window={_EMPTY_GT_FALLBACK_WINDOW}"
    )
    retry_records, retry_raw_records, retry_usage = _generate_ground_truth_once(
        pdf_path,
        ocr_csv,
        template,
        llm,
        _EMPTY_GT_FALLBACK_WINDOW,
        include_images=include_images,
    )
    combined_usage = {
        key: (usage.get(key, 0) or 0) + (retry_usage.get(key, 0) or 0)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    combined_usage.update({
        "empty_result_fallback": True,
        "initial_window": window,
        "effective_window": _EMPTY_GT_FALLBACK_WINDOW,
    })

    if not retry_records:
        raise RuntimeError(
            "ground-truth generation returned no records after retrying with "
            f"window={_EMPTY_GT_FALLBACK_WINDOW}"
        )
    return retry_records, retry_raw_records, combined_usage
