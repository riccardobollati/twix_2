"""Build a visual comparison of test labels and a trained checkpoint's predictions."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from data import PageDataset, ROOT
from model import model_from_checkpoint
from train import choose_device, confusion_update, metrics


COLORS = {1: (20, 175, 65, 112), 2: (30, 95, 230, 112)}


def overlay(source: Image.Image, ids: Image.Image) -> Image.Image:
    colors = np.zeros((source.height, source.width, 4), dtype=np.uint8)
    values = np.asarray(ids)
    for class_id, color in COLORS.items():
        colors[values == class_id] = color
    return Image.alpha_composite(source.convert("RGBA"), Image.fromarray(colors, "RGBA")).convert("RGB")


def restore_size(mask: Image.Image, source_size: tuple[int, int], square_size: int) -> Image.Image:
    width, height = source_size
    ratio = min(square_size / width, square_size / height)
    resized_width = max(1, round(width * ratio))
    resized_height = max(1, round(height * ratio))
    left = (square_size - resized_width) // 2
    top = (square_size - resized_height) // 2
    return mask.crop((left, top, left + resized_width, top + resized_height)).resize(
        source_size, Image.Resampling.NEAREST
    )


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "first" / "best.pt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--label-review-dir", type=Path, default=ROOT / "review")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    output_dir = args.output_dir or args.checkpoint.parent / "test_review"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = model_from_checkpoint(checkpoint).to(device)
    model.eval()
    dataset = PageDataset("test", checkpoint["image_size"], args.dataset_root)
    total_confusion = torch.zeros((3, 3), dtype=torch.long)
    document_confusions: dict[str, torch.Tensor] = {}
    first_page_by_document: dict[str, str] = {}
    pages = []
    cards = []
    overview_rows = []

    with torch.no_grad():
        for index, entry in enumerate(dataset.items):
            image_tensor, target, stem = dataset[index]
            prediction = model(image_tensor.unsqueeze(0).to(device)).argmax(dim=1).squeeze(0).cpu()
            confusion = torch.zeros((3, 3), dtype=torch.long)
            confusion_update(confusion, prediction, target)
            total_confusion += confusion
            document = entry["document"]
            document_confusions.setdefault(document, torch.zeros((3, 3), dtype=torch.long))
            document_confusions[document] += confusion
            first_page_by_document.setdefault(document, stem)
            page_metrics = metrics(confusion)
            pages.append({"id": stem, "metrics": page_metrics})

            source = Image.open(ROOT / entry["image"]).convert("RGB")
            restored = restore_size(Image.fromarray(prediction.numpy().astype(np.uint8)),
                                    source.size, checkpoint["image_size"])
            preview_name = f"{stem}_prediction.jpg"
            overlay(source, restored).save(output_dir / preview_name, quality=90)
            source_path = (ROOT / entry["image"]).relative_to(ROOT)
            label_path = args.label_review_dir / f"{stem}.jpg"
            source_href = Path(os.path.relpath(ROOT / source_path, output_dir)).as_posix()
            label_href = Path(os.path.relpath(label_path, output_dir)).as_posix()
            overview_rows.append((document, stem, ROOT / source_path, label_path, output_dir / preview_name))
            cards.append(
                f'<section id="{html.escape(stem)}"><h2>{html.escape(stem)}</h2>'
                f'<p>Table IoU: {percent(page_metrics["iou_table"])} · '
                f'KV IoU: {percent(page_metrics["iou_key_value"])} · '
                f'Background IoU: {percent(page_metrics["iou_background"])}</p>'
                f'<div class="grid"><figure><figcaption>Source</figcaption><img loading="lazy" src="{source_href}"></figure>'
                f'<figure><figcaption>Ground truth</figcaption><img loading="lazy" src="{label_href}"></figure>'
                f'<figure><figcaption>Prediction</figcaption><img loading="lazy" src="{preview_name}"></figure></div></section>'
            )

    overall = metrics(total_confusion)
    document_metrics = {name: metrics(confusion) for name, confusion in document_confusions.items()}
    report = {"checkpoint": str(args.checkpoint), "best_epoch": checkpoint["epoch"],
              "first_kernel": checkpoint.get("first_kernel", 3),
              "overall": overall, "documents": document_metrics, "pages": pages}
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    document_rows = ''.join(
        f'<tr><td><a href="#{html.escape(first_page_by_document[name])}">{html.escape(name)}</a> '
        f'(<a href="contact_{html.escape(name)}.jpg">contact sheet</a>)</td>'
        f'<td>{percent(values["iou_table"])}</td><td>{percent(values["iou_key_value"])}</td>'
        f'<td>{percent(values["mean_foreground_iou"])}</td></tr>'
        for name, values in document_metrics.items()
    )
    kernel = checkpoint.get("first_kernel", 3)
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Test predictions</title>
<style>body{{font:16px system-ui;max-width:1800px;margin:30px auto;padding:0 20px;background:#f7f8fa;color:#17202a}}
section{{background:white;border:1px solid #ddd;border-radius:12px;padding:20px;margin:24px 0}}
.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}}
figure{{margin:0}}figcaption{{font-weight:600;margin:0 0 8px}}img{{width:100%;border:1px solid #ddd}}
table{{border-collapse:collapse;background:white}}td,th{{padding:7px 12px;border:1px solid #ddd;text-align:left}}
@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}</style>
<h1>Held-out test predictions</h1><p>Checkpoint epoch {checkpoint['epoch']}; first kernel {kernel}×{kernel}.
Green = table, blue = key–value block.
IoU compares the model prediction with the labeled pixels; higher is better.</p>
<p><strong>Overall:</strong> table {percent(overall['iou_table'])} · KV {percent(overall['iou_key_value'])}
· mean foreground {percent(overall['mean_foreground_iou'])} · background {percent(overall['iou_background'])}.</p>
<p><a href="overview.jpg">All test pages contact sheet</a></p>
<h2>By test document</h2><table><thead><tr><th>Document</th><th>Table IoU</th><th>KV IoU</th>
<th>Mean foreground IoU</th></tr></thead><tbody>{document_rows}</tbody></table>
{''.join(cards)}</html>\n'''
    (output_dir / "index.html").write_text(body)
    tile_width, tile_height = 420, 360
    overview = Image.new("RGB", (tile_width * 3, tile_height * len(overview_rows)), "white")
    overview_draw = ImageDraw.Draw(overview)
    for row, (_, stem, source_path, label_path, prediction_path) in enumerate(overview_rows):
        for column, (caption, path) in enumerate((
            ("source", source_path), ("ground truth", label_path), ("prediction", prediction_path)
        )):
            thumbnail = Image.open(path).convert("RGB")
            thumbnail.thumbnail((tile_width - 16, tile_height - 42))
            left = column * tile_width + (tile_width - thumbnail.width) // 2
            top = row * tile_height + 36
            overview.paste(thumbnail, (left, top))
            overview_draw.text((column * tile_width + 8, row * tile_height + 8),
                               f"{stem} · {caption}", fill="black")
    overview.save(output_dir / "overview.jpg", quality=88)
    for document in document_metrics:
        rows = [row for row in overview_rows if row[0] == document]
        sheet = Image.new("RGB", (tile_width * 3, tile_height * len(rows)), "white")
        pen = ImageDraw.Draw(sheet)
        for row_index, (_, stem, source_path, label_path, prediction_path) in enumerate(rows):
            for column, (caption, path) in enumerate((
                ("source", source_path), ("ground truth", label_path), ("prediction", prediction_path)
            )):
                thumbnail = Image.open(path).convert("RGB")
                thumbnail.thumbnail((tile_width - 16, tile_height - 42))
                left = column * tile_width + (tile_width - thumbnail.width) // 2
                top = row_index * tile_height + 36
                sheet.paste(thumbnail, (left, top))
                pen.text((column * tile_width + 8, row_index * tile_height + 8),
                         f"{stem} · {caption}", fill="black")
        sheet.save(output_dir / f"contact_{document}.jpg", quality=88)
    print(f"Created {output_dir / 'index.html'}")
    print(f"Overall: {overall}")


if __name__ == "__main__":
    main()
