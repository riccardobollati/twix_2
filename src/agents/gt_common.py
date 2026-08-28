"""Shared flat, page-local ground-truth generation for train and test eval."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from os.path import isfile

import pdfplumber
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents._usage import split_tokens
from src.ground_truth_gen.helpers import load_ocr_pages, parse_json_response, render_page_b64
from src.ground_truth_gen.prompt import build_prompt

_MAX_WORKERS = 4
_MIN_GT_COVERAGE = 0.2
_MIN_OCR_CHARS = 500
_EMPTY_GT_FALLBACK_WINDOW = 5


def _gt_chars(blocks: list[dict]) -> int:
    total = 0
    for block in blocks:
        data = block.get("data")
        if isinstance(data, list):
            for row in data:
                if isinstance(row, dict):
                    total += sum(len(str(value)) for value in row.values() if value)
        elif isinstance(data, dict):
            total += sum(len(str(value)) for value in data.values() if value)
    return total


def _normalize_window_blocks(raw_blocks: list, page_nums: list[int]) -> list[dict]:
    """Keep valid blocks and enforce an unambiguous page-local page number."""
    if not isinstance(raw_blocks, list):
        return []
    blocks: list[dict] = []
    allowed_pages = set(page_nums)
    for raw in raw_blocks:
        if not isinstance(raw, dict):
            continue
        try:
            page = int(raw.get("page"))
        except (TypeError, ValueError):
            if len(page_nums) == 1:
                page = page_nums[0]
            else:
                print("  WARN: skipping GT block without a valid page in a multi-page window")
                continue
        if page not in allowed_pages:
            print(f"  WARN: skipping GT block assigned to page {page} outside window {page_nums}")
            continue
        blocks.append({**raw, "page": page})
    return blocks


def _generate_ground_truth_once(
    pdf_path: str,
    ocr_csv: str | None,
    template: list[dict],
    llm: BaseChatModel,
    window: int,
    *,
    include_images: bool = False,
) -> tuple[list[dict], dict]:
    """Generate flat blocks once for every page, preserving window order."""
    if (not ocr_csv or not isfile(ocr_csv)) and not include_images:
        raise RuntimeError("text-only ground-truth generation requires an OCR CSV")

    system_prompt = build_prompt(template, include_images=include_images)
    payloads: list[tuple[list[int], str | list[dict], int]] = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        windows = [list(range(i, min(i + window, total))) for i in range(0, total, window)]
        page_numbers = [index + 1 for index in range(total)]
        ocr_pages = load_ocr_pages(ocr_csv, page_numbers) if ocr_csv and isfile(ocr_csv) else {}
        print(f"  {total} page(s), {len(windows)} window(s) of up to {window}")
        for win_indices in windows:
            page_nums = [index + 1 for index in win_indices]
            pages_str = ", ".join(str(page) for page in page_nums)
            user_text = (
                f"Extract every physical block from page {page_nums[0]}."
                if len(page_nums) == 1
                else f"Extract every physical block from pages {pages_str}; keep blocks page-local."
            )
            if include_images:
                content: str | list[dict] = [{"type": "text", "text": user_text}]
                for index in win_indices:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{render_page_b64(pdf.pages[index])}"},
                    })
            else:
                content = user_text

            ocr_chars = 0
            for page_num in page_nums:
                page_ocr = ocr_pages.get(page_num, {})
                ocr_text = page_ocr.get("text", "")
                ocr_chars += len(ocr_text)
                if include_images:
                    content.append({
                        "type": "text",
                        "text": f"OCR words on page {page_num} (exact character sequences, reading order):\n{ocr_text}",
                    })
                else:
                    layout = page_ocr.get("layout", "")
                    content += f"\n\nPage {page_num} OCR layout (exact text with PDF-point coordinates):\n{layout or '[no OCR words]'}"
            payloads.append((page_nums, content, ocr_chars))

    def call(content: str | list[dict]):
        return llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=content)])

    with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(payloads))) as executor:
        responses = [executor.submit(call, content) for _, content, _ in payloads]
        responses = [future.result() for future in responses]

    input_tokens = output_tokens = total_tokens = 0

    def parse_window(response, page_nums: list[int], label: str) -> list[dict]:
        nonlocal input_tokens, output_tokens, total_tokens
        input_count, output_count, total_count = split_tokens(response.usage_metadata or {})
        input_tokens += input_count
        output_tokens += output_count
        total_tokens += total_count
        try:
            return _normalize_window_blocks(parse_json_response(response.content).get("blocks", []), page_nums)
        except (json.JSONDecodeError, AttributeError) as exc:
            print(f"  pages {label}... WARN: {exc}")
            print(f"  [RAW RESPONSE]\n{response.content}\n  [END RAW]")
            return []

    blocks: list[dict] = []
    for (page_nums, content, ocr_chars), response in zip(payloads, responses):
        label = ", ".join(str(page) for page in page_nums)
        window_blocks = parse_window(response, page_nums, label)
        if ocr_chars >= _MIN_OCR_CHARS and _gt_chars(window_blocks) / ocr_chars < _MIN_GT_COVERAGE:
            print(f"  pages {label}... WARN: low GT coverage — retrying window")
            instruction = (
                "Your previous attempt returned far less data than these pages contain. "
                "Extract EVERY physical block and EVERY table row from ALL pages listed above, "
                "preserving separate page-local blocks."
            )
            retry_content = content + ([{"type": "text", "text": instruction}] if isinstance(content, list) else f"\n\n{instruction}")
            retry_blocks = parse_window(call(retry_content), page_nums, label + " (retry)")
            if _gt_chars(retry_blocks) > _gt_chars(window_blocks):
                window_blocks = retry_blocks
        print(f"  pages {label}... {len(window_blocks)} block(s)")
        blocks.extend(window_blocks)

    return blocks, {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}


def generate_ground_truth(
    pdf_path: str,
    ocr_csv: str | None,
    template: list[dict],
    llm: BaseChatModel,
    window: int,
    *,
    include_images: bool = False,
) -> tuple[list[dict], dict]:
    """Generate flat GT, retrying an empty large-window result at window five."""
    blocks, usage = _generate_ground_truth_once(pdf_path, ocr_csv, template, llm, window, include_images=include_images)
    if blocks:
        return blocks, usage
    if window <= _EMPTY_GT_FALLBACK_WINDOW:
        raise RuntimeError(f"ground-truth generation returned no blocks with window={window}")

    print(f"  WARNING: ground-truth generation returned an empty block list; retrying all pages with window={_EMPTY_GT_FALLBACK_WINDOW}")
    retry_blocks, retry_usage = _generate_ground_truth_once(
        pdf_path, ocr_csv, template, llm, _EMPTY_GT_FALLBACK_WINDOW, include_images=include_images
    )
    combined = {
        key: (usage.get(key, 0) or 0) + (retry_usage.get(key, 0) or 0)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    combined.update({"empty_result_fallback": True, "initial_window": window, "effective_window": _EMPTY_GT_FALLBACK_WINDOW})
    if not retry_blocks:
        raise RuntimeError(f"ground-truth generation returned no blocks after retrying with window={_EMPTY_GT_FALLBACK_WINDOW}")
    return retry_blocks, combined
