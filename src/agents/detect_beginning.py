from __future__ import annotations

import json
import re
import threading
import time
from typing import Any

import pdfplumber
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.agents._ui import spinner as _spinner
from src.agents._usage import apply_usage, split_tokens
from src.agents.template_inference import ExtractionState
from src.ground_truth_gen.helpers import load_ocr_pages
from src.prompts.detect_beginning_prompt import PROMPT_DETECT_BEGINNING


_BATCH_PAGES = 6
_MAX_PARSE_ATTEMPTS = 2


def _parse_response(text: str, batch_start: int, batch_end: int) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        text = fenced.group(1)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("response must be a JSON object")

    start_page = parsed.get("start_page")
    if isinstance(start_page, bool) or (
        start_page is not None and not isinstance(start_page, int)
    ):
        raise ValueError("start_page must be an integer or null")
    if start_page is not None and not batch_start <= start_page <= batch_end:
        raise ValueError(
            f"start_page must be between {batch_start} and {batch_end}, or null"
        )

    confidence = parsed.get("confidence")
    if confidence not in {"high", "medium", "low"}:
        raise ValueError("confidence must be high, medium, or low")
    reason = parsed.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    return {
        "start_page": start_page,
        "confidence": confidence,
        "reason": reason.strip(),
    }


def detect_beginning_node(
    state: ExtractionState, llm: BaseChatModel
) -> dict[str, Any]:
    """Find the first physical PDF page containing the structured template.

    Pages are inspected chronologically in small OCR-layout batches. A batch with
    no structured data advances the scan; the first reported candidate stops
    it. If the model cannot identify structured content anywhere, page 1 is a
    conservative fallback that preserves the pipeline's previous behavior.
    """
    responses: list[BaseMessage] = []
    input_tokens = output_tokens = total_tokens = 0
    detected: dict[str, Any] | None = None

    stop_event = threading.Event()
    spinner_result: dict[str, Any] = {}
    spinner_thread = threading.Thread(
        target=_spinner,
        args=("detect_beginning", stop_event, spinner_result),
        daemon=True,
    )
    spinner_thread.start()

    t0 = time.time()
    try:
        with pdfplumber.open(state["full_pdf_path"]) as pdf:
            total_pages_in_pdf = len(pdf.pages)
            if total_pages_in_pdf == 0:
                raise RuntimeError("cannot detect a beginning in an empty PDF")
            ocr_pages = load_ocr_pages(
                state["full_ocr_csv_path"], range(1, total_pages_in_pdf + 1)
            )

            for batch_start in range(1, total_pages_in_pdf + 1, _BATCH_PAGES):
                batch_end = min(batch_start + _BATCH_PAGES - 1, total_pages_in_pdf)
                content = [
                    f"Inspect physical PDF pages {batch_start} through {batch_end}. "
                    "Choose the earliest actual structured-record page, if any."
                ]
                for page_number in range(batch_start, batch_end + 1):
                    layout = ocr_pages[page_number]["layout"]
                    content.append(
                        f"Physical PDF page {page_number} OCR layout "
                        "(coordinates are PDF points):\n"
                        f"{layout or '[no OCR words]'}"
                    )

                messages: list[BaseMessage] = [
                    SystemMessage(content=PROMPT_DETECT_BEGINNING),
                    HumanMessage(content="\n\n".join(content)),
                ]
                batch_result: dict[str, Any] | None = None
                last_error = ""
                for attempt in range(1, _MAX_PARSE_ATTEMPTS + 1):
                    response = llm.invoke(messages)
                    responses.append(response)
                    inp, out, tot = split_tokens(response.usage_metadata or {})
                    input_tokens += inp
                    output_tokens += out
                    total_tokens += tot
                    try:
                        batch_result = _parse_response(
                            response.content, batch_start, batch_end
                        )
                        break
                    except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
                        last_error = str(exc)
                        if attempt < _MAX_PARSE_ATTEMPTS:
                            messages = messages + [
                                response,
                                HumanMessage(content=(
                                    f"Invalid response: {exc}. Return only the required "
                                    "JSON object using a page from this batch or null."
                                )),
                            ]

                if batch_result is None:
                    raise RuntimeError(
                        "detect_beginning received invalid model output after "
                        f"{_MAX_PARSE_ATTEMPTS} attempts: {last_error}"
                    )
                if batch_result["start_page"] is not None:
                    detected = batch_result
                    break

        if detected is None:
            detected = {
                "start_page": 1,
                "confidence": "low",
                "reason": (
                    "No structured-record page was identified; falling back to page 1."
                ),
            }
            print("  warning: detect_beginning found no structured page; using page 1")
    finally:
        duration = time.time() - t0
        spinner_result["tokens"] = input_tokens + output_tokens + total_tokens
        stop_event.set()
        spinner_thread.join(timeout=0.5)

    start_page = detected["start_page"]
    skipped = start_page - 1
    print(
        f"  Structured data starts at physical page {start_page} "
        f"({detected['confidence']} confidence); skipped {skipped} front-matter page(s)."
    )

    cost_entry: dict[str, Any] = {
        "phase": "detect_beginning",
        "duration_s": round(duration, 3),
    }
    apply_usage(cost_entry, input_tokens, output_tokens, total_tokens)
    return {
        "messages": responses,
        "detected_start_page": start_page,
        "beginning_confidence": detected["confidence"],
        "beginning_reason": detected["reason"],
        "cost": [cost_entry],
    }
