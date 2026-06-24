"""Visualise OCR word bounding boxes drawn on PDF pages.

Usage:
    python viz_bb.py <folder> <page> [<page> ...]

Arguments:
    folder      Document folder name inside data/ (e.g. id_20).
    page        One or more page numbers to render.

Output:
    data/<folder>/viz/bb/page_<N>_bb.png  for each requested page.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import pdfplumber

_FILL   = (80, 160, 255, 40)
_STROKE = (0,  80,  200, 200)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw OCR word bounding boxes on PDF pages.")
    parser.add_argument("folder", help="Document folder name inside data/ (e.g. id_20).")
    parser.add_argument("pages", nargs="+", type=int, help="Page number(s) to render.")
    args = parser.parse_args()

    data_dir = os.path.join("data", args.folder)
    pdf_path = os.path.join(data_dir, f"{args.folder}.pdf")
    ocr_path = os.path.join(data_dir, "ocr_extraction_full.csv")
    viz_dir  = os.path.join(data_dir, "viz", "bb")

    for path, label in [(pdf_path, "PDF"), (ocr_path, "OCR CSV")]:
        if not os.path.isfile(path):
            print(f"Error: {label} not found at {path}", file=sys.stderr)
            sys.exit(1)

    os.makedirs(viz_dir, exist_ok=True)

    requested = set(args.pages)
    words_by_page: dict[int, list[dict]] = defaultdict(list)
    with open(ocr_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            page = int(row["page"])
            if page in requested:
                words_by_page[page].append({
                    "x0":     float(row["x0"]),
                    "top":    float(row["top"]),
                    "x1":     float(row["x1"]),
                    "bottom": float(row["bottom"]),
                    "text":   row["text"],
                })

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for page_num in sorted(requested):
            if page_num < 1 or page_num > total:
                print(f"Warning: page {page_num} out of range (1–{total}), skipping.")
                continue
            page  = pdf.pages[page_num - 1]
            img   = page.to_image(resolution=150)
            words = words_by_page.get(page_num, [])
            for w in words:
                img.draw_rect(
                    (w["x0"], w["top"], w["x1"], w["bottom"]),
                    fill=_FILL,
                    stroke=_STROKE,
                    stroke_width=1,
                )
            out_path = os.path.join(viz_dir, f"page_{page_num}_bb.png")
            img.save(out_path, format="PNG", quantize=False)
            print(f"Page {page_num}: {len(words)} word(s) drawn → {out_path}")


if __name__ == "__main__":
    main()
