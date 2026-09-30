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


def _crop_ocr_csv(src: str, dst: str, local_pages: set[int]) -> None:
    with open(src, encoding="utf-8") as fin, \
         open(dst, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if int(row["page"]) in local_pages:
                writer.writerow(row)


def _extract_pages_pdf(src: str, dst: str, pages: list[int]) -> None:
    doc = pdfium.PdfDocument(src)
    new_doc = pdfium.PdfDocument.new()
    indices = [p - 1 for p in pages]
    new_doc.import_pages(doc, indices)
    new_doc.save(dst)


def _extract_pages_ocr(src: str, dst: str, pages: list[int]) -> None:
    local_page = {source: index for index, source in enumerate(pages, 1)}
    with open(src, encoding="utf-8") as fin, \
         open(dst, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            p = int(row["page"])
            if p in local_page:
                row = dict(row)
                row["page"] = str(local_page[p])
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
    planned_train = state.get("train_source_pages")
    if planned_train is not None:
        train_pages = list(planned_train)
        if not train_pages or train_pages != sorted(set(train_pages)) or any(
            page < source_start or page > total_pages for page in train_pages
        ):
            raise ValueError("train_source_pages must be sorted, distinct structured source pages")
    else:
        legacy_count = max(1, min(train_pages_req, usable_pages // 2 if eval_mode else usable_pages))
        train_pages = list(range(source_start, source_start + legacy_count))
    planned_test = state.get("test_source_pages")
    if eval_mode:
        if planned_test is not None:
            test_pages = list(planned_test)
        else:
            remaining = [page for page in range(source_start, total_pages + 1) if page not in train_pages]
            test_pages = remaining[:eval_pages_req] if eval_pages_req is not None else remaining
        if (not test_pages or test_pages != sorted(set(test_pages))
                or any(page < source_start or page > total_pages for page in test_pages)
                or set(train_pages) & set(test_pages)):
            raise ValueError("test_source_pages must be sorted, distinct, and disjoint from training")
    else:
        test_pages = []
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
        actual_train_pages = len(train_pages)

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
        _extract_pages_pdf(full_pdf_path, train_pdf, train_pages)
        _extract_pages_ocr(full_ocr_csv_path, train_ocr, train_pages)
        # create test data
        _extract_pages_pdf(full_pdf_path, test_pdf, test_pages)
        _extract_pages_ocr(full_ocr_csv_path, test_ocr, test_pages)
        print(
            f"  Train: source pages {train_pages} → {train_dir}\n"
            f"  Test:  source pages {test_pages} → {test_dir}"
        )

        ws_pdf = train_pdf
        ws_ocr = train_ocr

        out: dict[str, Any] = {
            "test_pdf_path": test_pdf,
            "test_ocr_path": test_ocr,
            "test_start":    test_pages[0],
            "test_end":      test_pages[-1],
        }
    else:
        actual_train_pages = len(train_pages)
        ws_pdf = os.path.join(agent_space, f"{doc_name}.pdf")
        ws_ocr = os.path.join(agent_space, "ocr_extraction.csv")
        _extract_pages_pdf(full_pdf_path, ws_pdf, train_pages)
        _extract_pages_ocr(full_ocr_csv_path, ws_ocr, train_pages)
        print(
            f"  Cropped source pages {train_pages} "
            f"({actual_train_pages} page(s)) → {agent_space}"
        )
        out = {}

    # ── held-out validation split ─────────────────────────────────────────────
    # Separation/extraction scripts are generated against the first
    # (train − holdout) pages only; train_eval still runs on ALL train pages.
    # Scripts that overfit the pages they were built on then fail at train
    # time — where the repair loop can see and fix it — instead of on test.
    holdout_req = state.get("holdout_pages", 0)
    required = set(state.get("first_source_pages", [])) | set(state.get("cluster_sample_pages", []))
    optional = [page for page in train_pages if page not in required]
    holdout = max(0, min(holdout_req, len(optional), actual_train_pages - 2))
    held_out = set(optional[-holdout:]) if holdout else set()
    build_pages = [page for page in train_pages if page not in held_out]

    if holdout > 0:
        build_ocr = os.path.join(os.path.dirname(ws_ocr), "ocr_extraction_build.csv")
        build_local_pages = {index for index, source in enumerate(train_pages, 1) if source not in held_out}
        _crop_ocr_csv(ws_ocr, build_ocr, build_local_pages)
        print(
            f"  Build: source pages {build_pages} (script generation), "
            f"held out for train eval: {sorted(held_out)}"
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
        "train_source_pages": train_pages,
        "test_source_pages": test_pages,
        "build_source_pages": build_pages,
    }
