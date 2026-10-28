"""Run block separation, infer records deterministically, and render overlays.

Usage:
    python -m inspect_results.viz_record_separation <folder> [pdf_path] [page ...]
    python -m inspect_results.viz_record_separation <folder> --pdf-path PATH [page ...]
    python -m inspect_results.viz_record_separation <folder> --pdf-path PATH --out-dir DIR

The generated ``block_separation.py`` supplies page-local physical blocks.
``src.record_grouping`` then assigns those blocks to logical records using the
inferred template.  No model call or document-specific record script is used.

Output defaults to:
    data/<folder>/artifacts/record_separation/page_<N>_records.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any

import pdfplumber
from PIL import Image, ImageDraw, ImageFont

from src.record_grouping import group_blocks_into_records
from inspect_results.viz_blocks import (
    _COLOURS,
    _block_bbox,
    _default_pdf_path,
    _parse_targets,
    run_block_separator,
)


def _load_template(template: str | os.PathLike | dict | list[dict]) -> list[dict]:
    """Normalize a template path, root object, or node list to a node list."""
    payload: Any = template
    if isinstance(template, (str, os.PathLike)):
        template_path = os.fspath(template)
        if not os.path.isfile(template_path):
            raise FileNotFoundError(f"template JSON not found: {template_path}")
        with open(template_path, encoding="utf-8") as f:
            payload = json.load(f)
    if isinstance(payload, dict):
        payload = payload.get("template")
    if not isinstance(payload, list) or not payload:
        raise ValueError("template must be a non-empty node list or template.json root")
    if not all(isinstance(node, dict) for node in payload):
        raise ValueError("every template node must be an object")
    return payload


def _load_record_blocks(
    blocks_json_path: str,
    template: list[dict],
) -> list[dict]:
    """Load physical blocks, stamp their pages, and group them into records."""
    if not os.path.isfile(blocks_json_path):
        raise FileNotFoundError(
            f"block-separation JSON not found: {blocks_json_path}"
        )
    with open(blocks_json_path, encoding="utf-8") as f:
        separation = json.load(f)

    blocks: list[dict] = []
    for page_entry in separation.get("pages", []):
        page_num = page_entry.get("page")
        for block in page_entry.get("blocks", []):
            if isinstance(block, dict):
                blocks.append({**block, "page": page_num})
    return group_blocks_into_records(blocks, template)


def render_record_separation(
    pdf_path: str,
    blocks_json_path: str,
    template: str | os.PathLike | dict | list[dict],
    output_dir: str,
    *,
    label: str = "document",
    requested_pages: set[int] | None = None,
) -> list[str]:
    """Render record boundaries and return the PNG paths written.

    ``template`` may be a template node list, the root ``template.json``
    object, or a filesystem path.  This is the reusable entry point used by
    both the graph artifact phase and the standalone CLI.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    template_nodes = _load_template(template)
    records = _load_record_blocks(blocks_json_path, template_nodes)
    colour_index_by_record = {
        str(record.get("record_id", "")): index
        for index, record in enumerate(records)
    }

    # page -> [(record id, block id, block type, block bbox)]
    pages_records: dict[int, list[tuple[str, str, str, tuple]]] = {}
    record_blocks: dict[str, list[str]] = {}
    for record in records:
        record_id = str(record.get("record_id", ""))
        for block in record.get("data", []):
            try:
                page_num = int(block.get("page"))
            except (TypeError, ValueError):
                continue
            block_id = str(block.get("block_id", ""))
            bbox = _block_bbox(block.get("words", []))
            if not block_id or bbox is None:
                continue
            pages_records.setdefault(page_num, []).append(
                (record_id, block_id, str(block.get("type", "")), bbox)
            )
            record_blocks.setdefault(record_id, []).append(block_id)

    os.makedirs(output_dir, exist_ok=True)
    for name in os.listdir(output_dir):
        if name.startswith("page_") and name.endswith("_records.png"):
            os.remove(os.path.join(output_dir, name))

    print(f"\nRecords for {label}:")
    for record in records:
        record_id = str(record.get("record_id", ""))
        block_ids = record_blocks.get(record_id, [])
        print(f"  record {record_id}: {', '.join(block_ids) or '[no drawable blocks]'}")
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
            page_image = page.to_image(resolution=150)
            pil_image = page_image.original.convert("RGBA")
            overlay = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay, "RGBA")
            scale = pil_image.width / page.width
            drawn = pages_records.get(page_num, [])

            # A record spanning pages gets one envelope per physical page.
            by_record: dict[str, list[tuple[str, str, tuple]]] = {}
            for record_id, block_id, block_type, bbox in drawn:
                by_record.setdefault(record_id, []).append(
                    (block_id, block_type, bbox)
                )

            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
            except OSError:
                font = ImageFont.load_default()

            for record_id, record_page_blocks in by_record.items():
                colour_index = colour_index_by_record[record_id]
                fill, stroke = _COLOURS[colour_index % len(_COLOURS)]
                scaled_boxes = [
                    tuple(value * scale for value in bbox)
                    for _, _, bbox in record_page_blocks
                ]
                envelope = (
                    min(box[0] for box in scaled_boxes),
                    min(box[1] for box in scaled_boxes),
                    max(box[2] for box in scaled_boxes),
                    max(box[3] for box in scaled_boxes),
                )

                draw.rectangle(envelope, fill=fill, outline=stroke, width=4)
                for (_block_id, _block_type, _bbox), box in zip(
                    record_page_blocks, scaled_boxes
                ):
                    draw.rectangle(box, outline=stroke, width=2)

                node_labels = ", ".join(
                    f"{block_id}" + (f" ({block_type})" if block_type else "")
                    for block_id, block_type, _ in record_page_blocks
                )
                label_text = f"record {record_id}: {node_labels}"
                text_width = draw.textlength(label_text, font=font)
                text_height = 12
                label_top = max(0, envelope[1] - text_height - 3)
                draw.rectangle(
                    [
                        envelope[0],
                        label_top,
                        envelope[0] + text_width + 5,
                        label_top + text_height + 2,
                    ],
                    fill=(*stroke[:3], 235),
                )
                draw.text(
                    (envelope[0] + 2, label_top + 1),
                    label_text,
                    font=font,
                    fill=(255, 255, 255, 255),
                )

            out_path = os.path.join(output_dir, f"page_{page_num}_records.png")
            composited = Image.alpha_composite(pil_image, overlay)
            composited.convert("RGB").save(out_path, format="PNG")
            written.append(out_path)
            print(
                f"Page {page_num}: {len(by_record)} record segment(s), "
                f"{len(drawn)} block(s) → {out_path}"
            )

    print()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw deterministic record boundaries on PDF pages."
    )
    parser.add_argument(
        "folder",
        help="Document folder name inside data/ (e.g. id_05_routing).",
    )
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
        "--template-path",
        help="Template JSON (default: data/<folder>/template.json).",
    )
    parser.add_argument(
        "--script-path",
        help=(
            "Generated block separator "
            "(default: data/<folder>/pipeline/block_separation.py)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        help=(
            "Folder for rendered PNGs "
            "(default: data/<folder>/artifacts/record_separation)."
        ),
    )
    args = parser.parse_args()

    try:
        positional_pdf, requested = _parse_targets(args.target)
    except ValueError as exc:
        parser.error(str(exc))
    if positional_pdf and args.pdf_path:
        parser.error("provide the PDF path positionally or with --pdf-path, not both")

    data_root = os.path.join(os.path.dirname(__file__), "data")
    doc_dir = os.path.join(data_root, args.folder)
    if not os.path.isdir(doc_dir):
        print(f"Error: folder not found: {doc_dir}", file=sys.stderr)
        sys.exit(1)

    pdf_path = (
        os.path.abspath(args.pdf_path or positional_pdf)
        if args.pdf_path or positional_pdf
        else _default_pdf_path(doc_dir, args.folder)
    )
    script_path = (
        os.path.abspath(args.script_path)
        if args.script_path
        else os.path.join(doc_dir, "pipeline", "block_separation.py")
    )
    template_path = (
        os.path.abspath(args.template_path)
        if args.template_path
        else os.path.join(doc_dir, "template.json")
    )
    output_dir = (
        os.path.abspath(args.out_dir)
        if args.out_dir
        else os.path.join(doc_dir, "artifacts", "record_separation")
    )

    try:
        with tempfile.TemporaryDirectory(
            prefix=f"viz_record_separation_{args.folder}_"
        ) as work_dir:
            blocks_json_path = run_block_separator(
                pdf_path, script_path, work_dir
            )
            render_record_separation(
                pdf_path,
                blocks_json_path,
                template_path,
                output_dir,
                label=args.folder,
                requested_pages=requested,
            )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
