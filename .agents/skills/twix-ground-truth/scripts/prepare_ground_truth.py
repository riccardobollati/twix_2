#!/usr/bin/env python3
"""Prepare exact TWIX held-out ground-truth inputs for Codex inspection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pdfplumber


PROJECT_DIR = Path(__file__).resolve().parents[4]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.ground_truth_gen.helpers import load_ocr_pages
from src.ground_truth_gen.prompt import build_prompt


def _awaiting_run() -> Path:
    candidates: list[Path] = []
    for state_path in sorted((PROJECT_DIR / "UI" / "runs").glob("run-*/ui_run.json")):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if state.get("status") == "awaiting_ground_truth":
            candidates.append(state_path.parent)
    if len(candidates) != 1:
        raise SystemExit(
            "Pass a run directory: expected exactly one awaiting TWIX UI run, "
            f"found {len(candidates)}."
        )
    return candidates[0]


def prepare(run_dir: Path) -> Path:
    run_dir = run_dir.expanduser().resolve()
    state_path = run_dir / "ui_run.json"
    template_path = run_dir / "template.json"
    test_dir = run_dir / "agent_space" / "test"
    test_pdfs = sorted(test_dir.glob("*.pdf"))
    ocr_path = test_dir / "ocr_extraction.csv"

    if not state_path.is_file():
        raise SystemExit(f"Not a TWIX UI run directory: {run_dir}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "awaiting_ground_truth":
        raise SystemExit(
            f"Run status is {state.get('status')!r}, not 'awaiting_ground_truth'."
        )
    if len(test_pdfs) != 1 or not ocr_path.is_file() or not template_path.is_file():
        raise SystemExit("Run is missing its inferred template, test PDF, or test OCR.")

    template_payload = json.loads(template_path.read_text(encoding="utf-8"))
    template = template_payload.get("template")
    if not isinstance(template, list) or not template:
        raise SystemExit("template.json does not contain a non-empty inferred template.")

    output_dir = run_dir / "manual_ground_truth_input"
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "system_prompt.txt").write_text(
        build_prompt(template, include_images=True), encoding="utf-8"
    )

    manifest_pages: list[dict[str, object]] = []
    with pdfplumber.open(test_pdfs[0]) as pdf:
        page_numbers = list(range(1, len(pdf.pages) + 1))
        ocr_pages = load_ocr_pages(str(ocr_path), page_numbers)
        for page_number, page in zip(page_numbers, pdf.pages):
            image_path = pages_dir / f"page_{page_number:04d}.png"
            ocr_text_path = pages_dir / f"page_{page_number:04d}_ocr.txt"
            page.to_image(resolution=150).original.save(image_path, format="PNG")
            ocr_text_path.write_text(
                ocr_pages[page_number]["text"], encoding="utf-8"
            )
            manifest_pages.append({
                "page": page_number,
                "image": str(image_path),
                "ocr": str(ocr_text_path),
            })

    manifest = {
        "run_directory": str(run_dir),
        "template": str(template_path),
        "test_pdf": str(test_pdfs[0]),
        "test_ocr": str(ocr_path),
        "system_prompt": str(output_dir / "system_prompt.txt"),
        "page_count": len(manifest_pages),
        "pages": manifest_pages,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare exact inputs for the TWIX manual ground-truth skill."
    )
    parser.add_argument("run_directory", nargs="?", type=Path)
    args = parser.parse_args()
    prepare(args.run_directory or _awaiting_run())


if __name__ == "__main__":
    main()
