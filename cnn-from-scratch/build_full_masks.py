"""Validate expanded annotations and build masks plus per-document visual review."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from build_masks import ROOT, build_one


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset_full")
    parser.add_argument("--review-dir", type=Path, default=ROOT / "review_full")
    parser.add_argument("--allow-missing", action="store_true", help="Build a partial review while labeling is in progress")
    args = parser.parse_args()
    dataset = args.dataset_root.resolve()
    review = args.review_dir.resolve()
    manifest = json.loads((dataset / "manifest.json").read_text())
    review.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    totals = defaultdict(Counter)
    missing = []
    invalid = {}
    for entry in manifest["pages"]:
        doc, split, stem = entry["document"], entry["split"], entry["id"]
        annotation = dataset / split / "annotations" / f"{stem}.json"
        if not annotation.exists():
            missing.append(stem)
            continue
        try:
            preview, counts = build_one(entry, review, dataset)
        except (ValueError, OSError) as exc:
            invalid[stem] = str(exc)
            continue
        totals[split].update(counts)
        grouped[doc].append((entry, preview, counts, annotation))

    if (missing or invalid) and not args.allow_missing:
        raise ValueError(f"Missing {len(missing)} annotations and invalid {len(invalid)} annotations; "
                         f"first missing: {missing[:5]}; first invalid: {list(invalid.items())[:5]}")
    style = '''<style>body{font:16px system-ui;max-width:1600px;margin:30px auto;padding:0 20px;background:#f7f8fa;color:#17202a}
section{background:#fff;border:1px solid #ddd;border-radius:10px;padding:16px;margin:20px 0}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}.pair img{width:100%;height:auto;border:1px solid #ddd}
small{color:#667}a{color:#174ca5}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>'''
    links = []
    for doc in sorted(set(entry["document"] for entry in manifest["pages"])):
        rows = grouped[doc]
        links.append(f'<li><a href="{html.escape(doc)}.html">{html.escape(doc)}</a> — {len(rows)} / '
                     f'{sum(e["document"] == doc for e in manifest["pages"])} labeled · '
                     f'<a href="contact_{html.escape(doc)}.jpg">contact sheet</a></li>')
        sections = []
        for entry, preview, counts, annotation in rows:
            source = Path(entry["image"]).resolve() if Path(entry["image"]).is_absolute() else ROOT / entry["image"]
            sections.append(
                f'<section id="{html.escape(entry["id"])}"><h2>{html.escape(entry["id"])} <small>{entry["split"]}</small></h2>'
                f'<p>{counts["table"]} table(s), {counts["key_value"]} key–value block(s) · '
                f'<a href="{annotation.as_uri()}">annotation JSON</a></p>'
                f'<div class="pair"><img loading="lazy" src="{source.as_uri()}">'
                f'<img loading="lazy" src="{(review / preview).as_uri()}"></div></section>')
        (review / f"{doc}.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
            f'<title>{html.escape(doc)} label review</title>{style}<h1>{html.escape(doc)}</h1>'
            '<p><a href="index.html">All documents</a> · Green = table; blue = key–value.</p>'
            + '\n'.join(sections) + '</html>\n')
        if rows:
            tile_w, tile_h, columns = 320, 330, 5
            sheet = Image.new("RGB", (tile_w * columns, tile_h * ((len(rows) + columns - 1) // columns)), "white")
            pen = ImageDraw.Draw(sheet)
            for i, (entry, preview, _, _) in enumerate(rows):
                thumb = Image.open(review / preview)
                thumb.thumbnail((tile_w - 12, tile_h - 33))
                x = (i % columns) * tile_w + (tile_w - thumb.width) // 2
                y = (i // columns) * tile_h + 27
                sheet.paste(thumb, (x, y))
                pen.text(((i % columns) * tile_w + 7, (i // columns) * tile_h + 5),
                         f'{entry["id"]} {entry["split"]}', fill="black")
            sheet.save(review / f"contact_{doc}.jpg", quality=86)
    (review / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        f'<title>Expanded CNN label review</title>{style}<h1>Expanded CNN label review</h1>'
        f'<p>{sum(map(len, grouped.values()))} / {len(manifest["pages"])} pages labeled. '
        'Review every original and overlay before training. Green = table; blue = key–value.</p>'
        '<ul>' + '\n'.join(links) + '</ul></html>\n')
    (review / "summary.json").write_text(json.dumps({
        "labeled": sum(map(len, grouped.values())), "expected": len(manifest["pages"]),
        "missing": missing, "invalid": invalid,
        "regions_by_split": {k: dict(v) for k, v in totals.items()},
    }, indent=2) + '\n')
    print(f"Built masks/review for {sum(map(len, grouped.values()))}/{len(manifest['pages'])} pages; "
          f"missing={len(missing)} invalid={len(invalid)}")


if __name__ == "__main__":
    main()
