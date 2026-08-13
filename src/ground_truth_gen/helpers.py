"""Shared rendering and parsing helpers for ground-truth generation."""

from __future__ import annotations

import base64
import csv
import io
import json
import re
from typing import Any


def render_page_b64(page: Any) -> str:
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
    words: list[tuple[float, float, str]] = []
    with open(csv_path, encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if int(row["page"]) == page_num:
                words.append((float(row["top"]), float(row["x0"]), row["text"]))
    words.sort()
    return " ".join(word[2] for word in words)
