"""Visualise template block boundaries drawn on PDF pages.

Usage:
    python viz_blocks.py <folder> [page ...]
    python viz_blocks.py <folder> --pdf-path PATH --blocks-json PATH [page ...]

Arguments:
    folder   Document folder name inside data/ (e.g. id_05_routing).
    page     One or more page numbers to render (default: all pages).

Output:
    data/<folder>/viz/blocks/page_<N>_blocks.png  for each rendered page.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

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


def _block_bbox(words: list[dict]) -> tuple[float, float, float, float] | None:
    if not words:
        return None
    return (
        min(w["x0"]     for w in words),
        min(w["top"]    for w in words),
        max(w["x1"]     for w in words),
        max(w["bottom"] for w in words),
    )


def _find_train_dir(doc_dir: str) -> str:
    candidate = os.path.join(doc_dir, "agent_space", "train")
    if os.path.isdir(candidate):
        return candidate
    # older layout: pipeline/
    candidate2 = os.path.join(doc_dir, "pipeline")
    if os.path.isdir(candidate2):
        return candidate2
    print(f"Error: cannot find train directory under {doc_dir}", file=sys.stderr)
    sys.exit(1)


def _find_pdf(train_dir: str) -> str:
    for f in os.listdir(train_dir):
        if f.endswith(".pdf"):
            return os.path.join(train_dir, f)
    print(f"Error: no PDF found in {train_dir}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw template block boundaries on PDF pages."
    )
    parser.add_argument("folder", help="Document folder name inside data/ (e.g. id_05_routing).")
    parser.add_argument("page", nargs="*", type=int, help="Page numbers to render (default: all).")
    parser.add_argument(
        "--pdf-path",
        help="PDF to render (default: the train PDF under the document folder).",
    )
    parser.add_argument(
        "--blocks-json",
        help="Block-separation JSON to draw (default: the train output).",
    )
    args = parser.parse_args()

    data_root = os.path.join(os.path.dirname(__file__), "data")
    doc_dir   = os.path.join(data_root, args.folder)
    if not os.path.isdir(doc_dir):
        print(f"Error: folder not found: {doc_dir}", file=sys.stderr)
        sys.exit(1)

    train_dir = None
    if not args.pdf_path or not args.blocks_json:
        train_dir = _find_train_dir(doc_dir)

    pdf_path = (
        os.path.abspath(args.pdf_path)
        if args.pdf_path else _find_pdf(train_dir)
    )
    blocks_json_path = (
        os.path.abspath(args.blocks_json)
        if args.blocks_json else os.path.join(train_dir, "block_separation.json")
    )
    if not os.path.isfile(pdf_path):
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(blocks_json_path):
        print(f"Error: block-separation JSON not found: {blocks_json_path}", file=sys.stderr)
        sys.exit(1)

    with open(blocks_json_path, encoding="utf-8") as f:
        separation = json.load(f)

    # Optional: load block_models from metadata for legend annotations
    block_models: dict[str, str] = {}
    meta_path = os.path.join(doc_dir, "metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        block_models = meta.get("block_models") or {}

    # Collect all known block_ids and assign colours
    all_block_ids: list[str] = []
    for page_entry in separation.get("pages", []):
        for block in page_entry.get("blocks", []):
            bid = str(block.get("block_id", ""))
            if bid and bid not in all_block_ids:
                all_block_ids.append(bid)
    all_block_ids.sort(key=lambda x: int(x) if x.isdigit() else x)

    colour_map = {
        bid: _COLOURS[i % len(_COLOURS)]
        for i, bid in enumerate(all_block_ids)
    }

    # Build per-page block list
    pages_blocks: dict[int, list[tuple[str, str, tuple]]] = {}
    for page_entry in separation.get("pages", []):
        page_num = page_entry["page"]
        for block in page_entry.get("blocks", []):
            bid   = str(block.get("block_id", ""))
            btype = block.get("type", "")
            bbox  = _block_bbox(block.get("words", []))
            if not bid or bbox is None:
                continue
            pages_blocks.setdefault(page_num, []).append((bid, btype, bbox))

    requested = set(args.page) if args.page else None

    viz_dir = os.path.join(doc_dir, "viz", "blocks")
    os.makedirs(viz_dir, exist_ok=True)

    # Print legend
    print(f"\nBlock types for {args.folder}:")
    for bid in all_block_ids:
        model = block_models.get(bid, "")
        model_tag = f"  [{model}]" if model else ""
        print(f"  block {bid}{model_tag}")
    print()

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        pages_to_render = sorted(
            requested if requested else set(range(1, total_pages + 1))
        )

        for page_num in pages_to_render:
            if page_num < 1 or page_num > total_pages:
                print(f"  skip: page {page_num} out of range (1–{total_pages})")
                continue

            page  = pdf.pages[page_num - 1]
            img   = page.to_image(resolution=150)
            drawn = pages_blocks.get(page_num, [])

            for bid, btype, bbox in drawn:
                fill, stroke = colour_map[bid]
                img.draw_rect(bbox, fill=fill, stroke=stroke, stroke_width=2)

            # draw labels via PIL on the annotated image
            pil_img = img.annotated.convert("RGBA")
            draw    = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
            except OSError:
                font = ImageFont.load_default()

            scale = pil_img.width / page.width

            for bid, btype, bbox in drawn:
                _, stroke = colour_map[bid]
                label     = f"{bid}" + (f" ({btype})" if btype else "")
                x0  = bbox[0] * scale
                top = bbox[1] * scale
                tw, th = draw.textlength(label, font=font), 12
                draw.rectangle(
                    [x0, max(0, top - th - 2), x0 + tw + 4, max(0, top)],
                    fill=(*stroke[:3], 220),
                )
                draw.text((x0 + 2, max(0, top - th - 1)), label, font=font, fill=(255, 255, 255))

            out_path = os.path.join(viz_dir, f"page_{page_num}_blocks.png")
            pil_img.convert("RGB").save(out_path, format="PNG")
            print(f"Page {page_num}: {len(drawn)} block instance(s) → {out_path}")

    print()


if __name__ == "__main__":
    main()
