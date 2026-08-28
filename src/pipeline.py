#!/usr/bin/env python3
"""Standalone extraction pipeline.

Usage:
    python pipeline.py <pdf_path> [--out-dir DIR]

Steps:
  1. OCR extraction with pdfplumber
  2. Block separation (agent-generated block_separation.py)
  3. Coordinate relativisation + per-block extraction
  4. Save block_separation.json and extracted.json in out_dir
"""

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import pdfplumber


SCRIPT_TIMEOUT_S = 300  # hard cap for one generated-script execution


# ── OCR ───────────────────────────────────────────────────────────────────────

def _extract_ocr(pdf_path: str) -> list[dict]:
    words = []
    word_id = 0
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for word in (page.extract_words() or []):
                words.append({
                    "id":     word_id,
                    "page":   page_num,
                    "x0":     word["x0"],
                    "top":    word["top"],
                    "x1":     word["x1"],
                    "bottom": word["bottom"],
                    "text":   word["text"],
                })
                word_id += 1
    return words


def _save_ocr_csv(words: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "page", "x0", "top", "x1", "bottom", "text"]
        )
        writer.writeheader()
        writer.writerows(words)


# ── coordinate helpers ────────────────────────────────────────────────────────

def _block_bbox(words: list[dict]):
    if not words:
        return None
    return (
        min(w["x0"]     for w in words),
        min(w["top"]    for w in words),
        max(w["x1"]     for w in words),
        max(w["bottom"] for w in words),
    )


def _relativize(words: list[dict]) -> list[dict]:
    bbox = _block_bbox(words)
    if bbox is None:
        return words
    ox, oy = bbox[0], bbox[1]
    return [
        {**w,
         "x0":     w["x0"]     - ox,
         "x1":     w["x1"]     - ox,
         "top":    w["top"]    - oy,
         "bottom": w["bottom"] - oy}
        for w in words
    ]


# ── block separation ──────────────────────────────────────────────────────────

def _run_block_separation(pipeline_dir: str, ocr_csv_path: str, out_dir: str) -> dict:
    script_path = os.path.join(pipeline_dir, "block_separation.py")
    try:
        result = subprocess.run(
            [sys.executable, script_path, ocr_csv_path, out_dir],
            cwd=pipeline_dir,
            timeout=SCRIPT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        print(f"block_separation timed out after {SCRIPT_TIMEOUT_S}s", file=sys.stderr)
        sys.exit(1)
    if result.returncode != 0:
        print("block_separation failed", file=sys.stderr)
        sys.exit(1)
    with open(os.path.join(out_dir, "block_separation.json"), encoding="utf-8") as f:
        return json.load(f)


# ── extraction scripts ────────────────────────────────────────────────────────

def _load_extractors(pipeline_dir: str) -> dict:
    """Import every <template_id>.py from extraction_scripts/ and return {id: extract_fn}."""
    scripts_dir = os.path.join(pipeline_dir, "extraction_scripts")
    extractors = {}
    for fname in sorted(os.listdir(scripts_dir)):
        if not fname.endswith(".py"):
            continue
        template_id = fname[:-3]
        spec = importlib.util.spec_from_file_location(
            f"extractor_{template_id}",
            os.path.join(scripts_dir, fname),
        )
        if spec is None or spec.loader is None:
            print(f"  warning: extractor {fname} has no import loader, skipping")
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            extractors[template_id] = module.extract
        except Exception as exc:
            print(f"  warning: extractor {fname} failed to load, skipping: {exc}")
    return extractors


# ── health check ──────────────────────────────────────────────────────────────

def _health_check(
    words: list[dict],
    separation: dict,
    extracted: list[dict],
) -> dict:
    """Detect degenerate pipeline output (silent total failures).

    A separation/extraction stack that fits the train pages can still produce
    nothing on new pages (unmatched header tokens or dropped blocks). Scoring an
    empty extracted.json against a real ground truth reads as a model failure
    when it is actually a detectable pipeline abort — flag it instead.
    """
    total_pages = len({w["page"] for w in words})
    pages_with_blocks = len({
        p["page"] for p in separation.get("pages", []) if p.get("blocks")
    })
    # page-locality: a block on page P holding other pages' words means the
    # separation script globbed the document into a mega-block — downstream
    # extraction then mixes coordinate spaces across pages
    word_page = {w["id"]: w["page"] for w in words}
    cross_page_blocks = []
    for page_entry in separation.get("pages", []):
        page_num = page_entry.get("page")
        for block in page_entry.get("blocks", []):
            n_foreign = sum(
                1 for w in block.get("words", [])
                if word_page.get(w.get("id"), page_num) != page_num
            )
            if n_foreign:
                cross_page_blocks.append(
                    f"{block.get('block_id')}@p{page_num} ({n_foreign} foreign words)"
                )
    problems = []
    if not extracted:
        problems.append("no blocks extracted on any page")
    elif total_pages and pages_with_blocks / total_pages < 0.5:
        problems.append(
            f"blocks found on only {pages_with_blocks}/{total_pages} pages"
        )
    if cross_page_blocks:
        problems.append(
            f"{len(cross_page_blocks)} block(s) contain words from other pages "
            f"(blocks must be page-local): {cross_page_blocks[:5]}"
        )
    health = {
        "degenerate":             bool(problems),
        "problems":               problems,
        "total_pages":            total_pages,
        "pages_with_blocks":      pages_with_blocks,
        "blocks_extracted":       len(extracted),
        "cross_page_blocks":      cross_page_blocks,
    }
    for problem in problems:
        print(f"  WARNING: degenerate pipeline output — {problem}", file=sys.stderr)
    return health


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the extraction pipeline on a PDF.")
    parser.add_argument("pdf_path", help="Path to the PDF file to process.")
    parser.add_argument("--out-dir", default=None, metavar="DIR",
                        help="Directory for output files (default: current working directory).")
    args = parser.parse_args()

    pdf_path     = os.path.abspath(args.pdf_path)
    pipeline_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir      = os.path.abspath(args.out_dir) if args.out_dir else os.getcwd()

    if not os.path.isfile(pdf_path):
        parser.error(f"PDF not found: {pdf_path}")
    required_paths = [
        os.path.join(pipeline_dir, "block_separation.py"),
        os.path.join(pipeline_dir, "extraction_scripts"),
    ]
    missing = [path for path in required_paths if not os.path.exists(path)]
    if missing:
        parser.error("incomplete pipeline; missing: " + ", ".join(missing))
    os.makedirs(out_dir, exist_ok=True)

    # 1. OCR
    print("Extracting OCR...")
    words = _extract_ocr(pdf_path)
    ocr_csv_path = os.path.join(out_dir, "ocr.csv")
    _save_ocr_csv(words, ocr_csv_path)
    print(f"  {len(words)} words extracted → ocr.csv")

    # 2. Block separation → block_separation.json written to out_dir
    print("Running block separation...")
    separation = _run_block_separation(pipeline_dir, ocr_csv_path, out_dir)

    # 3. Load extractors
    extractors = _load_extractors(pipeline_dir)

    # 4. Extract each block
    print("Extracting data from blocks...")
    results = []
    for page_entry in separation.get("pages", []):
        page_num = page_entry.get("page")
        for block in page_entry.get("blocks", []):
            block_id = block.get("block_id", "")
            if not block_id:
                continue
            template_id = block_id.rsplit("_", 1)[0]
            if template_id not in extractors:
                print(f"  warning: no extractor for block {block_id}, skipping")
                continue
            relative_block = {**block, "words": _relativize(block.get("words", []))}
            try:
                extracted = extractors[template_id](relative_block)
                extracted["page"] = page_num
                extracted["uid"]  = block.get("uid")
                results.append(extracted)
            except Exception as exc:
                print(f"  warning: extraction failed for {block_id}: {exc}")

    # 5. Save extracted.json. This flat, page-local block list is the final
    # pipeline output; repeated node ids remain separate physical blocks.
    output_path = os.path.join(out_dir, "extracted.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  {len(results)} block(s) extracted → {output_path}")

    # 6. Health check → pipeline_health.json
    health = _health_check(words, separation, results)
    health_path = os.path.join(out_dir, "pipeline_health.json")
    with open(health_path, "w", encoding="utf-8") as f:
        json.dump(health, f, indent=2, ensure_ascii=False)
    if health["degenerate"]:
        print(f"WARNING: degenerate output, see {health_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
