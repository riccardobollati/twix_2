"""Validate annotations and create class-index masks plus visual review pages."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
CLASSES = {"table": 1, "key_value": 2}
COLORS = {"table": (20, 175, 65), "key_value": (30, 95, 230)}


def build_one(entry: dict, review_dir: Path, dataset_root: Path = DATASET) -> tuple[str, dict[str, int]]:
    split, stem = entry["split"], entry["id"]
    image_path = ROOT / entry["image"]
    annotation_path = dataset_root / split / "annotations" / f"{stem}.json"
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing annotation: {annotation_path}")
    image = Image.open(image_path).convert("RGB")
    annotation = json.loads(annotation_path.read_text())
    if annotation.get("image") != image_path.name:
        raise ValueError(f"Wrong image name in {annotation_path}")
    if (annotation.get("width"), annotation.get("height")) != image.size:
        raise ValueError(f"Wrong dimensions in {annotation_path}: expected {image.size}")
    if not isinstance(annotation.get("regions"), list):
        raise ValueError(f"Missing regions list in {annotation_path}")

    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    counts = {name: 0 for name in CLASSES}
    boxes = []
    for i, region in enumerate(annotation["regions"]):
        kind = region.get("class")
        box = region.get("box")
        if kind not in CLASSES or not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"Bad region {i} in {annotation_path}")
        if any(type(v) is not int for v in box):
            raise ValueError(f"Non-integer box {i} in {annotation_path}")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
            raise ValueError(f"Out-of-bounds box {i} in {annotation_path}: {box}")
        for previous in boxes:
            if max(x0, previous[0]) < min(x1, previous[2]) and max(y0, previous[1]) < min(y1, previous[3]):
                raise ValueError(f"Overlapping boxes in {annotation_path}: {box} and {previous}")
        boxes.append(box)
        counts[kind] += 1
        color = COLORS[kind]
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=CLASSES[kind])
        overlay_draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=(*color, 68), outline=(*color, 240), width=3)
        overlay_draw.text((x0 + 4, y0 + 4), kind.replace("_", " "), fill=(0, 0, 0, 255), stroke_width=2, stroke_fill=(255, 255, 255, 255))

    mask_path = dataset_root / split / "masks" / f"{stem}.png"
    mask.save(mask_path)
    preview = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    preview_path = review_dir / f"{stem}.jpg"
    preview.save(preview_path, quality=90)
    return preview_path.name, counts


def main() -> None:
    manifest = json.loads((DATASET / "manifest.json").read_text())
    review_dir = ROOT / "review"
    review_dir.mkdir(exist_ok=True)
    rows = []
    totals = {split: {name: 0 for name in CLASSES} for split in ("train", "val", "test")}
    for entry in manifest["pages"]:
        preview, counts = build_one(entry, review_dir)
        split, stem = entry["split"], entry["id"]
        for name in CLASSES:
            totals[split][name] += counts[name]
        source = f"../dataset/{split}/images/{stem}.png"
        annotation = f"../dataset/{split}/annotations/{stem}.json"
        rows.append(f'<section><h2>{html.escape(stem)} <small>{split}</small></h2>'
                    f'<p>{counts["table"]} table(s), {counts["key_value"]} key–value block(s) · '
                    f'<a href="{annotation}">annotation JSON</a></p>'
                    f'<div class="pair"><img src="{source}" alt="Original {stem}">'
                    f'<img src="{preview}" alt="Labeled {stem}"></div></section>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>CNN label review</title>
<style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:0 20px;background:#f7f8fa;color:#17202a}
section{background:white;border:1px solid #ddd;border-radius:12px;padding:20px;margin:24px 0}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}.pair img{width:100%;height:auto;object-fit:contain;border:1px solid #ddd}
small{color:#667}a{color:#174ca5}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>
<h1>Draft label review</h1><p>Left: source page. Right: mask overlay. Green = table; blue = key–value block.
Boxes include whitespace inside the labeled region. Review these drafts before trusting training metrics.</p>
'''+"\n".join(rows)+"</html>\n"
    (review_dir / "index.html").write_text(page)
    (review_dir / "summary.json").write_text(json.dumps(totals, indent=2) + "\n")
    for split in totals:
        entries = [entry for entry in manifest["pages"] if entry["split"] == split]
        columns = 3 if split == "train" else 2
        tile_width, tile_height = 420, 390
        sheet = Image.new("RGB", (columns * tile_width, math.ceil(len(entries) / columns) * tile_height), "white")
        sheet_draw = ImageDraw.Draw(sheet)
        for i, entry in enumerate(entries):
            preview = Image.open(review_dir / f'{entry["id"]}.jpg')
            preview.thumbnail((tile_width - 16, tile_height - 35))
            left = (i % columns) * tile_width + (tile_width - preview.width) // 2
            top = (i // columns) * tile_height + 28
            sheet.paste(preview, (left, top))
            sheet_draw.text(((i % columns) * tile_width + 8, (i // columns) * tile_height + 7),
                            entry["id"], fill="black")
        sheet.save(review_dir / f"contact_{split}.jpg", quality=88)
    print(f"Built {len(rows)} masks and review/index.html; regions: {totals}")


if __name__ == "__main__":
    main()
