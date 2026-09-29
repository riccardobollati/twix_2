"""Audit every source PDF against generated clustering JSON and HTML files."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from build_html import METHODS
from run_all import HERE, ROOT, sources, valid_result


def validate_report(report: dict, folder: Path, total: int, max_pages: int) -> list[str]:
    errors = []
    expected = set(range(1, min(total, max_pages) + 1))
    page_entries = report.get("pages", [])
    observed = [page.get("page") for page in page_entries]
    if len(observed) != len(expected) or set(observed) != expected:
        errors.append("page metadata does not cover the scanned range exactly")
    eligible = {page["page"] for page in page_entries if page.get("eligible")}
    if set(report.get("skipped_pages", [])) != expected - eligible:
        errors.append("skipped pages do not match pages without table evidence")
    overlays = set(report.get("overlays", []))
    for name in overlays:
        if not (folder / name).is_file():
            errors.append(f"missing overlay {name}")
    for method in METHODS:
        outcome = report.get("algorithms", {}).get(method)
        if outcome is None:
            errors.append(f"missing method {method}")
            continue
        clusters = outcome.get("clusters", {})
        assigned = [number for group in clusters.values() for number in group]
        if set(assigned) != eligible or any(count != 1 for count in Counter(assigned).values()):
            errors.append(f"{method} does not partition eligible pages exactly")
        if len(clusters) != outcome.get("cluster_count"):
            errors.append(f"{method} cluster count disagrees with assignments")
        representatives = outcome.get("representatives", {})
        if set(representatives) != set(clusters):
            errors.append(f"{method} representative keys disagree with clusters")
        for key, number in representatives.items():
            if number not in clusters.get(key, []):
                errors.append(f"{method} representative {number} is outside cluster {key}")
            if f"page_{number:04d}_features.png" not in overlays:
                errors.append(f"{method} representative {number} has no overlay")
    if not (folder / "index.html").is_file():
        errors.append("missing document HTML")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=HERE)
    parser.add_argument("--max-pages", type=int, default=500)
    args = parser.parse_args()
    discovered = sources(args.data_dir.resolve())
    problems = []
    total_scanned = 0
    for name, pdf_path, total in discovered:
        folder = args.output_root / name
        result_path = folder / "clusters.json"
        if not valid_result(result_path, pdf_path, total, args.max_pages):
            problems.append(f"{name}: missing, stale, or incomplete clusters.json")
            continue
        report = json.loads(result_path.read_text(encoding="utf-8"))
        problems.extend(f"{name}: {message}" for message in validate_report(report, folder, total, args.max_pages))
        total_scanned += report["page_count"]
    for filename in ("index.html", "summary.json", "report.md"):
        if not (args.output_root / filename).is_file():
            problems.append(f"missing {filename}")
    summary_path = args.output_root / "summary.json"
    if summary_path.is_file():
        rows = json.loads(summary_path.read_text(encoding="utf-8"))
        if len(rows) != len(discovered) or any(row.get("status") != "complete" for row in rows):
            problems.append("summary.json does not mark every source PDF complete")
    if problems:
        for problem in problems:
            print(problem)
        print(f"FAILED: {len(problems)} issue(s) across {len(discovered)} PDFs")
        return 1
    print(f"Verified {len(discovered)} PDFs, {total_scanned} scanned pages, all cluster partitions, "
          "representative PNGs, and HTML indexes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
