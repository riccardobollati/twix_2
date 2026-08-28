from __future__ import annotations

import csv
import os
from typing import Any

import pdfplumber
import pypdfium2 as pdfium

from src.agents.template_inference import ExtractionState


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_page_count(pdf_path: str) -> int:
    with pdfplumber.open(pdf_path) as pdf:
        return len(pdf.pages)


def _crop_ocr_csv(src: str, dst: str, max_pages: int) -> None:
    with open(src, encoding="utf-8") as fin, \
         open(dst, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if int(row["page"]) <= max_pages:
                writer.writerow(row)


def _extract_page_range_pdf(src: str, dst: str, start: int, end: int) -> None:
    doc = pdfium.PdfDocument(src)
    new_doc = pdfium.PdfDocument.new()
    indices = [p - 1 for p in range(start, end + 1) if 0 <= p - 1 < len(doc)]
    new_doc.import_pages(doc, indices)
    new_doc.save(dst)


def _extract_page_range_ocr(
    src: str, dst: str, start: int, end: int, remap_offset: int = 0
) -> None:
    with open(src, encoding="utf-8") as fin, \
         open(dst, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            p = int(row["page"])
            if start <= p <= end:
                row = dict(row)
                row["page"] = str(p - remap_offset)
                writer.writerow(row)


# ── node ──────────────────────────────────────────────────────────────────────

def environment_creation_node(state: ExtractionState) -> dict[str, Any]:
    full_pdf_path     = state["full_pdf_path"]
    full_ocr_csv_path = state["full_ocr_csv_path"]
    doc_name          = state["doc_name"]
    train_pages_req   = state["train_pages"]
    eval_mode         = state["eval_mode"]
    eval_pages_req    = state.get("eval_pages")

    total_pages = _get_page_count(full_pdf_path)
    requested_start = state.get("detected_start_page", 1)
    if not 1 <= requested_start <= total_pages:
        raise ValueError(
            f"start page {requested_start} is outside the PDF's 1–{total_pages} page range"
        )
    source_start = requested_start
    usable_pages = total_pages - source_start + 1
    data_dir = state.get("output_dir") or os.path.dirname(full_pdf_path)
    os.makedirs(data_dir, exist_ok=True)
    # create the agent_space dir
    agent_space = os.path.join(data_dir, "agent_space")
    os.makedirs(agent_space, exist_ok=True)

    if eval_mode:
        if usable_pages < 2:
            raise RuntimeError(
                "eval_mode requires at least 2 structured-data pages to split "
                f"into train and test, but only {usable_pages} remain from "
                f"physical page {source_start} in {full_pdf_path}"
            )
        actual_train_pages = max(1, min(train_pages_req, usable_pages // 2))
        train_start = source_start
        train_end = train_start + actual_train_pages - 1
        test_start = train_end + 1
        test_end = (
            min(test_start + eval_pages_req - 1, total_pages)
            if eval_pages_req is not None else total_pages
        )

        # create test dir and train dir
        train_dir = os.path.join(agent_space, "train")
        test_dir  = os.path.join(agent_space, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        train_pdf = os.path.join(train_dir, f"{doc_name}.pdf")
        train_ocr = os.path.join(train_dir, "ocr_extraction.csv")
        test_pdf  = os.path.join(test_dir,  f"{doc_name}.pdf")
        test_ocr  = os.path.join(test_dir,  "ocr_extraction.csv")

        # create training data
        _extract_page_range_pdf(full_pdf_path, train_pdf, train_start, train_end)
        _extract_page_range_ocr(
            full_ocr_csv_path, train_ocr, train_start, train_end,
            remap_offset=train_start - 1,
        )
        # create test data
        _extract_page_range_pdf(full_pdf_path, test_pdf, test_start, test_end)
        _extract_page_range_ocr(
            full_ocr_csv_path, test_ocr, test_start, test_end,
            remap_offset=test_start - 1,
        )
        print(
            f"  Train: source pages {train_start}–{train_end} → {train_dir}\n"
            f"  Test:  source pages {test_start}–{test_end} → {test_dir}"
        )

        ws_pdf = train_pdf
        ws_ocr = train_ocr

        out: dict[str, Any] = {
            "test_pdf_path": test_pdf,
            "test_ocr_path": test_ocr,
            "test_start":    test_start,
            "test_end":      test_end,
        }
    else:
        actual_train_pages = max(1, min(train_pages_req, usable_pages))
        train_start = source_start
        train_end = train_start + actual_train_pages - 1
        ws_pdf = os.path.join(agent_space, f"{doc_name}.pdf")
        ws_ocr = os.path.join(agent_space, "ocr_extraction.csv")
        _extract_page_range_pdf(full_pdf_path, ws_pdf, train_start, train_end)
        _extract_page_range_ocr(
            full_ocr_csv_path, ws_ocr, train_start, train_end,
            remap_offset=train_start - 1,
        )
        print(
            f"  Cropped source pages {train_start}–{train_end} "
            f"({actual_train_pages} page(s)) → {agent_space}"
        )
        out = {}

    # ── held-out validation split ─────────────────────────────────────────────
    # Separation/extraction scripts are generated against the first
    # (train − holdout) pages only; train_eval still runs on ALL train pages.
    # Scripts that overfit the pages they were built on then fail at train
    # time — where the repair loop can see and fix it — instead of on test.
    holdout_req = state.get("holdout_pages", 0)
    holdout = max(0, min(holdout_req, actual_train_pages - 2))
    build_pages = actual_train_pages - holdout

    if holdout > 0:
        build_ocr = os.path.join(os.path.dirname(ws_ocr), "ocr_extraction_build.csv")
        _crop_ocr_csv(ws_ocr, build_ocr, build_pages)
        print(
            f"  Build: pages 1–{build_pages} (script generation), "
            f"held out for train eval: pages {build_pages + 1}–{actual_train_pages}"
        )
    else:
        build_ocr = ws_ocr

    return {
        **out,
        "agent_space":        agent_space,
        "pdf_path":           ws_pdf,
        "ocr_csv_path":       ws_ocr,
        "build_ocr_csv_path": build_ocr,
        "actual_train_pages": actual_train_pages,
    }
