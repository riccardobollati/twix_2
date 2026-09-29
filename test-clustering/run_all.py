"""Run deterministic table clustering on every source PDF in data/.

Examples:
    env/bin/python test-clustering/run_all.py --list
    env/bin/python test-clustering/run_all.py
    env/bin/python test-clustering/run_all.py --only new_04

The run resumes from valid clusters.json files. Use --force to regenerate.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import pdfplumber

from build_html import build


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CLUSTER_SCRIPT = ROOT / "test-sorter-lines" / "cluster_pages.py"


def sources(data_dir: Path) -> list[tuple[str, Path, int]]:
    paths = sorted([*data_dir.glob("*.pdf"), *data_dir.glob("*/*.pdf")])
    found = []
    used = set()
    for path in paths:
        name = path.stem if path.parent == data_dir else path.parent.name
        if name in used:
            raise ValueError(f"duplicate result name {name!r}; rename one source directory")
        used.add(name)
        with pdfplumber.open(path) as pdf:
            found.append((name, path.resolve(), len(pdf.pages)))
    return found


def valid_result(path: Path, pdf_path: Path, total: int, max_pages: int) -> bool:
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        return (result.get("version") == 1 and result.get("source_pdf") == str(pdf_path)
                and result.get("source_total_pages") == total
                and result.get("source_mtime_ns") == pdf_path.stat().st_mtime_ns
                and result.get("page_count") == min(total, max_pages)
                and all((path.parent / name).is_file() for name in result.get("overlays", [])))
    except (OSError, ValueError, TypeError):
        return False


def write_report(output_root: Path, selected: list[tuple[str, Path, int]]) -> None:
    sys.path.insert(0, str(CLUSTER_SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("cluster_pages_batch_report", CLUSTER_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reports = []
    for name, path, _ in selected:
        result_path = output_root / name / "clusters.json"
        if result_path.is_file():
            report = json.loads(result_path.read_text(encoding="utf-8"))
            if report.get("source_pdf") == str(path):
                reports.append(report)
    if reports:
        module.write_comparison(reports, output_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=HERE)
    parser.add_argument("--max-pages", type=int, default=500)
    parser.add_argument("--overlay-limit", type=int, default=500,
                        help="maximum medoid images per PDF (default: 500)")
    parser.add_argument("--only", nargs="*", help="document directory names to run")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_pages <= 500 or args.overlay_limit < 0:
        parser.error("max-pages must be in 1..500 and overlay-limit nonnegative")
    discovered = sources(args.data_dir.resolve())
    if not discovered:
        parser.error(f"no source PDFs in {args.data_dir}")
    selected = [item for item in discovered if args.only is None or item[0] in args.only]
    if args.only is not None and len(selected) != len(set(args.only)):
        parser.error("some --only document names were not found")
    if args.list:
        for name, pdf_path, total in selected:
            print(f"{name:24} {min(total,args.max_pages):4}/{total:<4} {pdf_path}")
        print(f"{len(selected)} PDF(s), {sum(min(item[2],args.max_pages) for item in selected)} scanned pages")
        return 0
    failures = []
    for index, (name, pdf_path, total) in enumerate(selected, 1):
        folder = args.output_root / name
        result = folder / "clusters.json"
        if not args.force and valid_result(result, pdf_path, total, args.max_pages):
            print(f"[{index}/{len(selected)}] {name}: already complete", flush=True)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(CLUSTER_SCRIPT), str(pdf_path), "--output-root", str(args.output_root),
               "--result-name", name, "--max-pages", str(args.max_pages),
               "--overlay-limit", str(args.overlay_limit), "--representatives-only"]
        print(f"[{index}/{len(selected)}] {name}: clustering {min(total,args.max_pages)}/{total} pages", flush=True)
        with (folder / "run.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            assert process.stdout
            for line in process.stdout:
                log.write(line)
                if re.search(r"page (?:1|\d*[05])/(?:\d+):", line):
                    print(f"  {name}: {line.strip()}", flush=True)
            code = process.wait()
        if code != 0 or not valid_result(result, pdf_path, total, args.max_pages):
            failures.append(name)
            print(f"  {name}: FAILED (see {folder / 'run.log'})", flush=True)
        else:
            print(f"  {name}: saved {result}", flush=True)
        build(args.output_root, discovered)
    rows = build(args.output_root, discovered)
    write_report(args.output_root, discovered)
    completed = sum(row["status"] == "complete" for row in rows)
    print(f"Processed {completed}/{len(discovered)} source PDFs; failures: {failures or 'none'}", flush=True)
    print(f"Open {args.output_root / 'index.html'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
