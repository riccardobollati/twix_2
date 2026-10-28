"""Run a generated block separator and draw its boundaries on PDF pages.

Usage:
    python -m inspect_results.viz_blocks <folder> [pdf_path] [page ...]
    python -m inspect_results.viz_blocks <folder> --pdf-path PATH [page ...]
    python -m inspect_results.viz_blocks <folder> --pdf-path PATH --out-dir DIR

Arguments:
    folder   Document folder name inside data/ (e.g. id_05_routing).
    pdf      PDF to process; defaults to data/<folder>/agent_space/train/<folder>.pdf.
    page     One or more page numbers to render (default: all pages).

Output:
    data/<folder>/viz/blocks/page_<N>_blocks.png by default, or DIR when
    --out-dir is supplied.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile

import pdfplumber
from PIL import ImageDraw, ImageFont

# One (fill, stroke) colour pair per block type — cycles if more than 8 types
_COLOURS = [
    ((255,  80,  80,  55), (200,   0,   0, 220)),  # red
    (( 60, 130, 255,  55), (  0,  70, 200, 220)),  # blue
    (( 60, 200,  90,  55), (  0, 130,  40, 220)),  # green
    ((255, 185,  30,  55), (190, 110,   0, 220)),  # amber
    ((190,  70, 255,  55), (110,   0, 200, 220)),  # purple
    (( 50, 210, 210,  55), (  0, 145, 145, 220)),  # teal
    ((255, 120,  30,  55), (190,  65,   0, 220)),  # orange
    ((220, 220,  50,  55), (150, 150,   0, 220)),  # yellow
]

_SCRIPT_TIMEOUT_S = 300


def _block_bbox(words: list[dict]) -> tuple[float, float, float, float] | None:
    if not words:
        return None
    return (
        min(w["x0"]     for w in words),
        min(w["top"]    for w in words),
        max(w["x1"]     for w in words),
        max(w["bottom"] for w in words),
    )


def _default_pdf_path(doc_dir: str, folder: str) -> str:
    """Return the conventional eval-mode train PDF path.

    Non-eval runs use ``agent_space/<folder>.pdf``; retain that layout as a
    fallback so the visualizer remains useful for both run modes.
    """
    train_pdf = os.path.join(doc_dir, "agent_space", "train", f"{folder}.pdf")
    if os.path.isfile(train_pdf):
        return train_pdf
    return os.path.join(doc_dir, "agent_space", f"{folder}.pdf")


def _parse_targets(targets: list[str]) -> tuple[str | None, set[int] | None]:
    """Interpret optional positional PDF path followed by page numbers.

    A numeric first value preserves the original ``<folder> 1 2`` page-only
    syntax. A non-numeric first value is treated as the PDF path.
    """
    if not targets:
        return None, None
    positional_pdf = None
    page_values = targets
    try:
        int(targets[0])
    except ValueError:
        positional_pdf = targets[0]
        page_values = targets[1:]

    try:
        pages = {int(value) for value in page_values}
    except ValueError as exc:
        raise ValueError("page selections must be integers") from exc
    if any(page < 1 for page in pages):
        raise ValueError("page selections must be positive integers")
    return positional_pdf, pages or None


def _extract_ocr_csv(pdf_path: str, output_path: str) -> None:
    """Extract word-level OCR in the generated separator's CSV format."""
    fields = ["id", "page", "x0", "top", "x1", "bottom", "text"]
    with pdfplumber.open(pdf_path) as pdf, open(
        output_path, "w", newline="", encoding="utf-8"
    ) as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        word_id = 0
        for page_num, page in enumerate(pdf.pages, start=1):
            for word in page.extract_words() or []:
                writer.writerow({
                    "id": word_id,
                    "page": page_num,
                    "x0": word["x0"],
                    "top": word["top"],
                    "x1": word["x1"],
                    "bottom": word["bottom"],
                    "text": word["text"],
                })
                word_id += 1


def run_block_separator(
    pdf_path: str,
    script_path: str,
    work_dir: str,
) -> str:
    """OCR ``pdf_path``, run the generated separator, and return its JSON path."""
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"block-separation script not found: {script_path}")

    os.makedirs(work_dir, exist_ok=True)
    ocr_path = os.path.join(work_dir, "ocr.csv")
    _extract_ocr_csv(pdf_path, ocr_path)

    try:
        result = subprocess.run(
            [sys.executable, script_path, ocr_path, work_dir],
            cwd=os.path.dirname(script_path),
            capture_output=True,
            text=True,
            timeout=_SCRIPT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"block_separation.py timed out after {_SCRIPT_TIMEOUT_S}s"
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no error output"
        raise RuntimeError(
            f"block_separation.py exited {result.returncode}: {detail}"
        )

    blocks_json_path = os.path.join(work_dir, "block_separation.json")
    if not os.path.isfile(blocks_json_path):
        raise RuntimeError(
            "block_separation.py completed without writing block_separation.json"
        )
    return blocks_json_path


def render_blocks(
    pdf_path: str,
    blocks_json_path: str,
    output_dir: str,
    *,
    label: str = "document",
    requested_pages: set[int] | None = None,
    metadata_path: str | None = None,
) -> list[str]:
    """Render block overlays and return the PNG paths written.

    This is the reusable entry point used by the pipeline graph and the CLI.
    Existing generated page images in ``output_dir`` are replaced so reruns do
    not leave stale artifacts from a previous page selection or split.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not os.path.isfile(blocks_json_path):
        raise FileNotFoundError(f"block-separation JSON not found: {blocks_json_path}")

    with open(blocks_json_path, encoding="utf-8") as f:
        separation = json.load(f)

    block_models: dict[str, str] = {}
    if metadata_path and os.path.isfile(metadata_path):
        with open(metadata_path, encoding="utf-8") as f:
            meta = json.load(f)
        block_models = meta.get("block_models") or {}

    all_block_ids: list[str] = []
    for page_entry in separation.get("pages", []):
        for block in page_entry.get("blocks", []):
            bid = str(block.get("block_id", ""))
            if bid and bid not in all_block_ids:
                all_block_ids.append(bid)
    all_block_ids.sort(key=lambda value: (0, int(value)) if value.isdigit() else (1, value))

    colour_map = {
        bid: _COLOURS[i % len(_COLOURS)]
        for i, bid in enumerate(all_block_ids)
    }

    pages_blocks: dict[int, list[tuple[str, str, tuple]]] = {}
    for page_entry in separation.get("pages", []):
        page_num = page_entry["page"]
        for block in page_entry.get("blocks", []):
            bid = str(block.get("block_id", ""))
            btype = block.get("type", "")
            bbox = _block_bbox(block.get("words", []))
            if not bid or bbox is None:
                continue
            pages_blocks.setdefault(page_num, []).append((bid, btype, bbox))

    os.makedirs(output_dir, exist_ok=True)
    for name in os.listdir(output_dir):
        if name.startswith("page_") and name.endswith("_blocks.png"):
            os.remove(os.path.join(output_dir, name))

    print(f"\nBlock types for {label}:")
    for bid in all_block_ids:
        model = block_models.get(bid, "")
        model_tag = f"  [{model}]" if model else ""
        print(f"  block {bid}{model_tag}")
    print()

    written: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        pages_to_render = sorted(
            requested_pages if requested_pages else set(range(1, total_pages + 1))
        )

        for page_num in pages_to_render:
            if page_num < 1 or page_num > total_pages:
                print(f"  skip: page {page_num} out of range (1–{total_pages})")
                continue

            page = pdf.pages[page_num - 1]
            img = page.to_image(resolution=150)
            drawn = pages_blocks.get(page_num, [])

            for bid, _btype, bbox in drawn:
                fill, stroke = colour_map[bid]
                img.draw_rect(bbox, fill=fill, stroke=stroke, stroke_width=2)

            pil_img = img.annotated.convert("RGBA")
            draw = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
            except OSError:
                font = ImageFont.load_default()

            scale = pil_img.width / page.width
            for bid, btype, bbox in drawn:
                _, stroke = colour_map[bid]
                label_text = f"{bid}" + (f" ({btype})" if btype else "")
                x0 = bbox[0] * scale
                top = bbox[1] * scale
                tw, th = draw.textlength(label_text, font=font), 12
                draw.rectangle(
                    [x0, max(0, top - th - 2), x0 + tw + 4, max(0, top)],
                    fill=(*stroke[:3], 220),
                )
                draw.text(
                    (x0 + 2, max(0, top - th - 1)),
                    label_text,
                    font=font,
                    fill=(255, 255, 255),
                )

            out_path = os.path.join(output_dir, f"page_{page_num}_blocks.png")
            pil_img.convert("RGB").save(out_path, format="PNG")
            written.append(out_path)
            print(f"Page {page_num}: {len(drawn)} block instance(s) → {out_path}")

    print()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw template block boundaries on PDF pages."
    )
    parser.add_argument("folder", help="Document folder name inside data/ (e.g. id_05_routing).")
    parser.add_argument(
        "target",
        nargs="*",
        help="Optional PDF path followed by page numbers; default PDF is the train split.",
    )
    parser.add_argument(
        "--pdf-path",
        help=(
            "PDF to process (default: "
            "data/<folder>/agent_space/train/<folder>.pdf)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        help="Folder for rendered PNGs (default: data/<folder>/viz/blocks).",
    )
    args = parser.parse_args()

    try:
        positional_pdf, requested = _parse_targets(args.target)
    except ValueError as exc:
        parser.error(str(exc))
    if positional_pdf and args.pdf_path:
        parser.error("provide the PDF path positionally or with --pdf-path, not both")

    data_root = os.path.join(os.path.dirname(__file__), "data")
    doc_dir   = os.path.join(data_root, args.folder)
    if not os.path.isdir(doc_dir):
        print(f"Error: folder not found: {doc_dir}", file=sys.stderr)
        sys.exit(1)

    pdf_path = (
        os.path.abspath(args.pdf_path or positional_pdf)
        if args.pdf_path or positional_pdf
        else _default_pdf_path(doc_dir, args.folder)
    )
    script_path = os.path.join(doc_dir, "pipeline", "block_separation.py")
    if not os.path.isfile(pdf_path):
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(script_path):
        print(f"Error: block-separation script not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    meta_path = os.path.join(doc_dir, "metadata.json")
    output_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(doc_dir, "viz", "blocks")
    try:
        with tempfile.TemporaryDirectory(prefix=f"viz_blocks_{args.folder}_") as work_dir:
            blocks_json_path = run_block_separator(pdf_path, script_path, work_dir)
            render_blocks(
                pdf_path,
                blocks_json_path,
                output_dir,
                label=args.folder,
                requested_pages=requested,
                metadata_path=meta_path,
            )
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
