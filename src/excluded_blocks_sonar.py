"""Measure words missed by a generated block separator on a PDF page range."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
import pypdfium2 as pdfium

from viz_blocks import render_blocks, run_block_separator


DEFAULT_PAGES = 100
DEFAULT_DROP_PERCENT = 15.0


def page_count(pdf_path: str | Path) -> int:
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        return len(document)
    finally:
        document.close()


def _crop_pdf(source: Path, destination: Path, start: int, end: int) -> None:
    document = pdfium.PdfDocument(str(source))
    cropped = pdfium.PdfDocument.new()
    try:
        cropped.import_pages(document, list(range(start - 1, end)))
        cropped.save(str(destination))
    finally:
        cropped.close()
        document.close()


def measure_coverage(ocr_csv: str | Path, blocks_json: str | Path) -> list[dict]:
    """Compare exact OCR IDs from one separator invocation, page by page."""
    word_pages: dict[str, int] = {}
    totals: dict[int, set[str]] = {}
    with Path(ocr_csv).open(encoding="utf-8") as source:
        for row in csv.DictReader(source):
            page = int(row["page"])
            word_id = str(row["id"])
            if word_id in word_pages:
                raise ValueError(f"duplicate OCR word ID {word_id}")
            word_pages[word_id] = page
            totals.setdefault(page, set()).add(word_id)

    separation = json.loads(Path(blocks_json).read_text(encoding="utf-8"))
    inside: dict[int, set[str]] = {}
    for entry in separation["pages"]:
        page = int(entry["page"])
        for block in entry.get("blocks", []):
            for word in block.get("words", []):
                word_id = str(word["id"])
                if word_pages.get(word_id) != page:
                    raise ValueError(f"block on page {page} contains unknown or foreign word {word_id}")
                inside.setdefault(page, set()).add(word_id)

    result = []
    for page in sorted(set(totals) | set(inside)):
        total = len(totals.get(page, ()))
        covered = len(inside.get(page, ()))
        result.append({
            "localPage": page,
            "totalWords": total,
            "insideBlockWords": covered,
            "outsideBlockWords": total - covered,
            "coverage": covered / total if total else None,
        })
    return result


def flag_pages(pages: list[dict], threshold_percent: float) -> tuple[float | None, list[int]]:
    """Flag pages whose coverage falls N percent below mean page coverage."""
    if not math.isfinite(threshold_percent) or not 0 <= threshold_percent <= 100:
        raise ValueError("drop threshold must be between 0 and 100 percent")
    values = [row["coverage"] for row in pages if row["coverage"] is not None]
    average = sum(values) / len(values) if values else None
    if average == 0:
        return average, [int(row["page"]) for row in pages if row["totalWords"] > 0]
    flagged = [
        int(row["page"])
        for row in pages
        if average is not None and average > 0 and row["coverage"] is not None
        and row["coverage"] < average * (1 - threshold_percent / 100)
    ]
    return average, flagged


def _save_plot(pages: list[dict], average: float | None, destination: Path) -> None:
    figure = Figure(figsize=(max(13, min(28, len(pages) * 0.13)), 6), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    x = [row["page"] for row in pages]
    inside = [row["insideBlockWords"] for row in pages]
    outside = [row["outsideBlockWords"] for row in pages]
    axis.bar(x, inside, width=0.9, color="#2563EB", label="Words inside blocks")
    axis.bar(x, outside, bottom=inside, width=0.9, color="#CBD5E1", label="Words outside blocks")
    axis.set(xlabel="Source PDF page", ylabel="Word count", title="Excluded blocks sonar · word coverage")
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)
    right = axis.twinx()
    right.plot(x, [row["coverage"] for row in pages], color="#DC2626", linewidth=1.8, label="Coverage")
    if average is not None:
        right.axhline(average, color="#A94F43", linestyle="--", linewidth=1, label="Average coverage")
    right.set_ylim(0, 1.05)
    right.set_ylabel("Words inside blocks / total page words")
    right.yaxis.set_major_formatter(PercentFormatter(1.0))
    handles1, labels1 = axis.get_legend_handles_labels()
    handles2, labels2 = right.get_legend_handles_labels()
    axis.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    figure.savefig(destination, dpi=160)


def scan(
    source_pdf: str | Path,
    script_path: str | Path,
    output_dir: str | Path,
    *,
    start_page: int = 1,
    end_page: int | None = None,
    threshold_percent: float = DEFAULT_DROP_PERCENT,
) -> dict:
    """Run the generated separator on a source-page range and persist sonar data."""
    source_pdf = Path(source_pdf).resolve()
    script_path = Path(script_path).resolve()
    output_dir = Path(output_dir).resolve()
    total_pages = page_count(source_pdf)
    if end_page is None:
        end_page = min(total_pages, start_page + DEFAULT_PAGES - 1)
    if not (1 <= start_page <= end_page <= total_pages):
        raise ValueError(f"page range must be within 1–{total_pages}")
    if not math.isfinite(threshold_percent) or not 0 <= threshold_percent <= 100:
        raise ValueError("drop threshold must be between 0 and 100 percent")

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sonar_", dir=output_dir.parent) as temporary:
        work = Path(temporary)
        cropped_pdf = work / "scan.pdf"
        _crop_pdf(source_pdf, cropped_pdf, start_page, end_page)
        blocks_json = Path(run_block_separator(str(cropped_pdf), str(script_path), str(work)))
        measured = measure_coverage(work / "ocr.csv", blocks_json)
        by_local_page = {row["localPage"]: row for row in measured}
        pages = []
        for local_page in range(1, end_page - start_page + 2):
            row = by_local_page.get(local_page, {
                "localPage": local_page, "totalWords": 0,
                "insideBlockWords": 0, "outsideBlockWords": 0, "coverage": None,
            })
            pages.append({**row, "page": start_page + local_page - 1})
        average, flagged = flag_pages(pages, threshold_percent)
        report = {
            "startPage": start_page,
            "endPage": end_page,
            "totalPages": total_pages,
            "thresholdPercent": threshold_percent,
            "averageCoverage": average,
            "flaggedPages": flagged,
            "pages": pages,
        }
        (work / "sonar.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        _save_plot(pages, average, work / "word_coverage.png")
        for filename in ("scan.pdf", "block_separation.json", "sonar.json", "word_coverage.png"):
            os.replace(work / filename, output_dir / filename)
    shutil.rmtree(output_dir / "previews", ignore_errors=True)
    return report


def render_page(output_dir: str | Path, source_page: int) -> Path:
    """Render a single saved sonar page with the existing block visualizer."""
    output_dir = Path(output_dir)
    report = json.loads((output_dir / "sonar.json").read_text(encoding="utf-8"))
    if not report["startPage"] <= source_page <= report["endPage"]:
        raise ValueError("page is outside the scanned range")
    local_page = source_page - report["startPage"] + 1
    preview_dir = output_dir / "previews" / f"page_{source_page}"
    preview = preview_dir / f"page_{local_page}_blocks.png"
    if not preview.is_file():
        render_blocks(
            str(output_dir / "scan.pdf"), str(output_dir / "block_separation.json"),
            str(preview_dir), requested_pages={local_page}, label=f"sonar page {source_page}",
        )
    return preview


def sonar_node(state: dict) -> dict:
    """Graph phase after separator generation; sonar failures do not stop extraction."""
    started = time.time()
    output_dir = Path(state.get("output_dir") or Path(state["full_pdf_path"]).parent)
    script = Path(state["pdf_path"]).parent / "block_separation.py"
    destination = output_dir / "artifacts" / "excluded_blocks_sonar"
    print("excluded_blocks_sonar_node: scanning pages", flush=True)
    try:
        report = scan(
            state["full_pdf_path"], script, destination,
            start_page=state.get("detected_start_page", 1),
            threshold_percent=state.get("sonar_drop_percent", DEFAULT_DROP_PERCENT),
        )
        print(
            f"excluded_blocks_sonar_node: scanned source pages "
            f"{report['startPage']}–{report['endPage']}; "
            f"{len(report['flaggedPages'])} flagged at {report['thresholdPercent']:g}%",
            flush=True,
        )
    except Exception as exc:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "error.json").write_text(json.dumps({"error": str(exc)}), encoding="utf-8")
        print(f"excluded_blocks_sonar_node: warning: {exc}", flush=True)
    return {"cost": [{"phase": "excluded_blocks_sonar", "duration_s": round(time.time() - started, 3)}]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run excluded-blocks sonar on a PDF range")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("script", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int)
    parser.add_argument("--threshold", type=float, default=DEFAULT_DROP_PERCENT)
    args = parser.parse_args()
    report = scan(
        args.pdf, args.script, args.output,
        start_page=args.start, end_page=args.end, threshold_percent=args.threshold,
    )
    print(
        f"Scanned pages {report['startPage']}–{report['endPage']}: "
        f"{len(report['flaggedPages'])} flagged; saved to {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
