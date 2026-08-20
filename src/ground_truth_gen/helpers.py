"""Shared rendering and parsing helpers for ground-truth generation."""

from __future__ import annotations

import base64
import csv
import io
import json
import re
from typing import Any


def render_page_b64(page: Any) -> str:
    """Render a PDF page to a 150-DPI PNG and return its base64 payload."""
    buffer = io.BytesIO()
    page.to_image(resolution=150).original.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def parse_json_response(text: str) -> dict:
    """Parse the first JSON object while preserving duplicate keys."""
    from src.eval.convert import loads_json_dedup

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise json.JSONDecodeError("no JSON object found", text, 0)
    return loads_json_dedup(match.group())


def load_page_ocr(csv_path: str, page_num: int) -> str:
    """Return page OCR in top-to-bottom, left-to-right order."""
    return load_ocr_pages(csv_path, [page_num])[page_num]["text"]


def format_words_layout(words: list[dict[str, Any]]) -> str:
    """Render OCR words as compact text lines while preserving page geometry.

    DeepSeek V4 is text-only. Grouping words by their vertical centers and
    retaining each word's x-span gives it the layout evidence previously
    carried by page images without changing the OCR character sequences.
    """
    normalized: list[dict[str, Any]] = []
    for word in words:
        try:
            top = float(word["top"])
            bottom = float(word["bottom"])
            x0 = float(word["x0"])
            x1 = float(word["x1"])
        except (KeyError, TypeError, ValueError):
            continue
        normalized.append({
            "top": top,
            "bottom": bottom,
            "x0": x0,
            "x1": x1,
            "text": str(word.get("text", "")),
        })

    normalized.sort(key=lambda item: ((item["top"] + item["bottom"]) / 2, item["x0"]))
    lines: list[list[dict[str, Any]]] = []
    line_centers: list[float] = []
    for word in normalized:
        center = (word["top"] + word["bottom"]) / 2
        height = max(1.0, word["bottom"] - word["top"])
        if lines and abs(center - line_centers[-1]) <= max(2.0, height * 0.35):
            lines[-1].append(word)
            line_centers[-1] = sum(
                (item["top"] + item["bottom"]) / 2 for item in lines[-1]
            ) / len(lines[-1])
        else:
            lines.append([word])
            line_centers.append(center)

    rendered: list[str] = []
    for index, line in enumerate(lines, start=1):
        line.sort(key=lambda item: item["x0"])
        y0 = min(item["top"] for item in line)
        y1 = max(item["bottom"] for item in line)
        cells = " | ".join(
            f'x={item["x0"]:.1f}-{item["x1"]:.1f} '
            f'{json.dumps(item["text"], ensure_ascii=False)}'
            for item in line
        )
        rendered.append(f"line={index} y={y0:.1f}-{y1:.1f}: {cells}")
    return "\n".join(rendered)


def load_page_ocr_layout(csv_path: str, page_num: int) -> str:
    """Return one page's OCR as layout-preserving text."""
    return load_ocr_pages(csv_path, [page_num])[page_num]["layout"]


def load_ocr_pages(
    csv_path: str, page_nums: list[int] | range
) -> dict[int, dict[str, str]]:
    """Load plain and layout OCR for several pages with one CSV scan."""
    requested = list(page_nums)
    wanted = set(requested)
    by_page: dict[int, list[dict[str, Any]]] = {page: [] for page in requested}
    with open(csv_path, encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            try:
                page = int(row["page"])
            except (KeyError, ValueError):
                continue
            if page in wanted:
                by_page[page].append(row)

    result: dict[int, dict[str, str]] = {}
    for page in requested:
        words = by_page[page]
        ordered: list[tuple[float, float, str]] = []
        for word in words:
            try:
                ordered.append(
                    (float(word["top"]), float(word["x0"]), str(word.get("text", "")))
                )
            except (KeyError, TypeError, ValueError):
                continue
        ordered.sort()
        result[page] = {
            "text": " ".join(item[2] for item in ordered),
            "layout": format_words_layout(words),
        }
    return result
