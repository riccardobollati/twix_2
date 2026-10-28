#!/usr/bin/env python3
"""Evaluation-only runner for the frozen baseline extractor."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("extractor", type=Path)
    parser.add_argument("input_pdf", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("log_json", type=Path)
    args = parser.parse_args()

    command = [sys.executable, str(args.extractor.resolve()), str(args.input_pdf.resolve()),
               str(args.output_json.resolve())]
    started = time.monotonic()
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    duration = time.monotonic() - started
    payload = {
        "command": command,
        "input_pdf": str(args.input_pdf.resolve()),
        "output_json": str(args.output_json.resolve()),
        "exit_code": proc.returncode,
        "duration_seconds": round(duration, 6),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "output_exists": args.output_json.is_file(),
    }
    if args.output_json.is_file():
        try:
            raw = json.loads(args.output_json.read_text(encoding="utf-8"))
            payload["rows_or_items"] = len(raw.get("areas", [])) if isinstance(raw, dict) else None
        except Exception as exc:  # preserve malformed raw output as evidence
            payload["output_parse_error"] = f"{type(exc).__name__}: {exc}"
    atomic_json(args.log_json, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
