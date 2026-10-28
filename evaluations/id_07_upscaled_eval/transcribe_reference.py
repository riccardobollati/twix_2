#!/usr/bin/env python3
"""Transcribe selected PDF pages into canonical evaluation truth.

This evaluation-only helper reads the PDF directly.  It does not load baseline
predictions, adapted candidate output, saved references, scores, or the frozen
extractor.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pdfplumber


HEADERS = [
    "Identification", "Name", "Lateral limits", "Upper limit", "Lower limit",
    "Type of activity", "Remarks", "Validity",
]
ID_RE = re.compile(r"^\s*(LI\s*P\s*\d+[A-Z]?)\s*[-–—]\s*(.*)$")
LATERAL_START_RE = re.compile(r"^(Cerchio|Circle|Linea|line|Arco|Arc|Area|Settore|Sector|Poligono|Polygon)\b", re.I)
COORD_RE = re.compile(r"^\d{1,2}\s*°")
PAGE_LABEL_RE = re.compile(r"ENR\s+5\.1\.1-(\d+)")


def norm(value: object) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def lines(value: object) -> list[str]:
    return [norm(line) for line in ("" if value is None else str(value)).splitlines() if norm(line)]


def parse_first_cell(value: object) -> tuple[str, str, str]:
    cell_lines = lines(value)
    if not cell_lines:
        raise ValueError("empty identification cell")
    match = ID_RE.match(cell_lines[0])
    if not match:
        raise ValueError(f"unrecognized identification line: {cell_lines[0]!r}")
    identification = re.sub(r"\s+", " ", match.group(1)).replace("P ", "P")
    name_parts = [norm(match.group(2))] if norm(match.group(2)) else []
    rest = cell_lines[1:]
    while rest and not (LATERAL_START_RE.match(rest[0]) or COORD_RE.match(rest[0])):
        name_parts.append(rest.pop(0))
    return identification, norm(" ".join(name_parts)), norm(" ".join(rest))


def parse_row(row: list[object]) -> tuple[dict[str, str], list[str]]:
    cells = [(row[i] if i < len(row) else "") for i in range(5)]
    identification, name, lateral = parse_first_cell(cells[0])
    limit_lines = lines(cells[1])
    canonical = {
        "Identification": identification,
        "Name": name,
        "Lateral limits": lateral,
        "Upper limit": limit_lines[0] if limit_lines else "",
        "Lower limit": limit_lines[1] if len(limit_lines) > 1 else "",
        "Type of activity": norm(cells[2]),
        "Remarks": norm(cells[3]),
        "Validity": norm(cells[4]),
    }
    return canonical, [norm(cell) for cell in cells]


def canonical_node(source_page: int, page_rows: list[dict[str, str]]) -> dict:
    rows = [
        [{"key": header, "value": row[header]} for header in HEADERS]
        for row in page_rows
    ]
    return {
        "record_id": f"page_{source_page}",
        "nodes": [{
            "id": f"areas_page_{source_page}",
            "type": "table",
            "content": {"headers": HEADERS, "rows": rows},
            "relationship": {"parent_id": None},
        }],
    }


def transcribe(pdf_path: Path, start_page: int, end_page: int) -> tuple[dict, dict, dict]:
    canonical_records = []
    raw_pages = []
    inventory_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for source_page in range(start_page, end_page + 1):
            page = pdf.pages[source_page - 1]
            page_text = page.extract_text() or ""
            label_match = PAGE_LABEL_RE.search(page_text)
            printed_page = int(label_match.group(1)) if label_match else None
            tables = sorted(page.find_tables(), key=lambda table: (table.bbox[1], table.bbox[0]))
            page_rows: list[dict[str, str]] = []
            raw_rows = []
            parse_errors = []
            for table_index, table in enumerate(tables):
                for row_index, row in enumerate(table.extract()):
                    first_lines = lines(row[0] if row else "")
                    if not first_lines or not ID_RE.match(first_lines[0]):
                        continue
                    try:
                        canonical, raw_cells = parse_row(row)
                    except Exception as exc:
                        parse_errors.append({
                            "table_index": table_index,
                            "row_index": row_index,
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                        continue
                    page_rows.append(canonical)
                    raw_rows.append({
                        "table_index": table_index,
                        "row_index": row_index,
                        "cells": raw_cells,
                        "transcription": canonical,
                    })
            canonical_records.append(canonical_node(source_page, page_rows))
            raw_pages.append({"source_page": source_page, "rows": raw_rows})
            inventory_pages.append({
                "source_page": source_page,
                "printed_page_label": printed_page,
                "table_count": len(tables),
                "substantive_structures": [{
                    "type": "table",
                    "name": "Prohibited areas",
                    "fields": HEADERS,
                    "row_count": len(page_rows),
                }],
                "identifiers": [row["Identification"] for row in page_rows],
                "parse_errors": parse_errors,
                "visual_review": "completed_against_rendered_source_page",
            })
    canonical = {"records": canonical_records}
    raw = {
        "source": "Codex-generated local PDF reference",
        "method": "direct pdfplumber table-cell transcription with visual page review",
        "pages": raw_pages,
    }
    inventory = {
        "source": "Codex-generated local PDF reference; not human-verified truth",
        "page_range": [start_page, end_page],
        "template_additions": [],
        "pages": inventory_pages,
    }
    return canonical, raw, inventory


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("canonical_json", type=Path)
    parser.add_argument("raw_json", type=Path)
    parser.add_argument("inventory_json", type=Path)
    parser.add_argument("--start-page", type=int, default=6)
    parser.add_argument("--end-page", type=int, default=25)
    args = parser.parse_args()
    canonical, raw, inventory = transcribe(args.pdf, args.start_page, args.end_page)
    write_json(args.canonical_json, canonical)
    write_json(args.raw_json, raw)
    write_json(args.inventory_json, inventory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
