"""Build the second header/structure clustering comparison for every source PDF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from build_html import build
from build_kv_html import build as build_kv, write_comparison as write_kv_comparison
from run_all import ROOT, sources
from cluster_pages_v2 import METHODS, TABLE_ALGORITHM_REVISION, write_comparison
from cluster_kv import METHODS as KV_METHODS


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "cluster_pages_v2.py"
KV_SCRIPT = HERE / "cluster_kv.py"
TITLE = "Clustering viewing 2"


def valid_result(path: Path, pdf_path: Path, total: int, max_pages: int) -> bool:
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        return (result.get("version") == 2
                and result.get("algorithm_revision") == TABLE_ALGORITHM_REVISION
                and result.get("source_pdf") == str(pdf_path)
                and result.get("source_total_pages") == total
                and result.get("source_mtime_ns") == pdf_path.stat().st_mtime_ns
                and result.get("page_count") == min(total, max_pages)
                and set(result.get("algorithms", {})) == set(METHODS)
                and all((path.parent / name).is_file() for name in result.get("overlays", [])))
    except (OSError, ValueError, TypeError):
        return False


def valid_kv_result(path: Path, pdf_path: Path, total: int, max_pages: int) -> bool:
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        return (result.get("version") == 1 and result.get("kind") == "kv"
                and result.get("source_pdf") == str(pdf_path)
                and result.get("source_total_pages") == total
                and result.get("source_mtime_ns") == pdf_path.stat().st_mtime_ns
                and result.get("page_count") == min(total, max_pages)
                and set(result.get("algorithms", {})) == set(KV_METHODS)
                and all((path.parent / name).is_file() for name in result.get("overlays", [])))
    except (OSError, ValueError, TypeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-root", type=Path, default=HERE / "viewing-2")
    parser.add_argument("--max-pages", type=int, default=500)
    parser.add_argument("--overlay-limit", type=int, default=500)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-kv", action="store_true", help="rebuild KV results while reusing table results")
    args = parser.parse_args()
    if not 1 <= args.max_pages <= 500 or args.overlay_limit < 0:
        parser.error("max-pages must be in 1..500 and overlay-limit nonnegative")
    discovered = sources(args.data_dir.resolve())
    selected = [item for item in discovered if args.only is None or item[0] in args.only]
    if args.only is not None and len(selected) != len(set(args.only)):
        parser.error("some --only document names were not found")
    if args.list:
        for name, path, total in selected:
            print(f"{name:24} {min(total, args.max_pages):4}/{total:<4} {path}")
        return 0
    failures = []
    for index, (name, path, total) in enumerate(selected, 1):
        folder = args.output_root / name
        result_path = folder / "clusters.json"
        folder.mkdir(parents=True, exist_ok=True)
        if not args.force and valid_result(result_path, path, total, args.max_pages):
            print(f"[{index}/{len(selected)}] {name}: table results already complete", flush=True)
        else:
            cmd = [sys.executable, str(SCRIPT), str(path), "--output-root", str(args.output_root),
                   "--result-name", name, "--max-pages", str(args.max_pages),
                   "--overlay-limit", str(args.overlay_limit)]
            print(f"[{index}/{len(selected)}] {name}: table clustering {min(total,args.max_pages)}/{total} pages", flush=True)
            with (folder / "run.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                assert process.stdout
                for line in process.stdout:
                    log.write(line)
                    if re.search(r"page (?:1|\d*[05])/(?:\d+):", line):
                        print(f"  {name}: {line.strip()}", flush=True)
                code = process.wait()
            if code or not valid_result(result_path, path, total, args.max_pages):
                failures.append(f"{name}: tables")
                print(f"  {name}: table clustering FAILED (see {folder / 'run.log'})", flush=True)
            else:
                print(f"  {name}: table results saved", flush=True)
        kv_path = folder / "kv_clusters.json"
        if not (args.force or args.force_kv) and valid_kv_result(kv_path, path, total, args.max_pages):
            print(f"  {name}: KV results already complete", flush=True)
        else:
            cmd = [sys.executable, str(KV_SCRIPT), str(path), "--output-root", str(args.output_root),
                   "--result-name", name, "--max-pages", str(args.max_pages),
                   "--overlay-limit", str(args.overlay_limit)]
            print(f"  {name}: KV clustering", flush=True)
            with (folder / "kv_run.log").open("w", encoding="utf-8") as log:
                process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
            if process.returncode or not valid_kv_result(kv_path, path, total, args.max_pages):
                failures.append(f"{name}: KV")
                print(f"  {name}: KV clustering FAILED (see {folder / 'kv_run.log'})", flush=True)
            else:
                print(f"  {name}: KV results saved", flush=True)
        if valid_kv_result(kv_path, path, total, args.max_pages):
            build_kv(folder, name, path, json.loads(kv_path.read_text(encoding="utf-8")))
        build(args.output_root, discovered, list(METHODS), TITLE)
    rows = build(args.output_root, discovered, list(METHODS), TITLE)
    reports = []
    for name, path, total in discovered:
        result = args.output_root / name / "clusters.json"
        if valid_result(result, path, total, args.max_pages):
            reports.append(json.loads(result.read_text(encoding="utf-8")))
    if reports:
        write_comparison(reports, args.output_root)
    kv_reports = []
    for name, path, total in discovered:
        result = args.output_root / name / "kv_clusters.json"
        if valid_kv_result(result, path, total, args.max_pages):
            kv_reports.append(json.loads(result.read_text(encoding="utf-8")))
    if kv_reports:
        write_kv_comparison(kv_reports, args.output_root)
    print(f"Completed {sum(row['status']=='complete' for row in rows)}/{len(rows)} PDFs; failures: {failures or 'none'}", flush=True)
    print(f"Open {args.output_root / 'index.html'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
