"""Run the trained model on pages from PDFs absent from every dataset split."""

from __future__ import annotations

import argparse
import html
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from data import ROOT, resize_and_pad
from model import UNet
from review_test import overlay, restore_size
from train import choose_device


REPO = ROOT.parent
DOCUMENTS = (
    "id_01", "id_03", "new_01", "new_02", "new_06",
    "id_36_upscaled", "id_37_upscaled", "id_40_upscaled",
)
PAGES = (3, 4)


def render(pdf: Path, page: int, output: Path) -> None:
    if output.exists():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["pdftoppm", "-f", str(page), "-l", str(page), "-scale-to", "1280",
         "-singlefile", "-png", str(pdf), str(output.with_suffix(""))],
        check=True, capture_output=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs" / "first" / "best.pt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    output_dir = args.output_dir or args.checkpoint.parent / "unseen_review"
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((ROOT / "dataset" / "manifest.json").read_text())
    used = {(REPO / item["source_pdf"]).resolve() for item in manifest["pages"]}
    sources = {}
    for document in DOCUMENTS:
        pdfs = sorted((REPO / "data" / document).glob("*.pdf"))
        if len(pdfs) != 1:
            raise RuntimeError(f"Expected one top-level PDF for {document}, found {len(pdfs)}")
        pdf = pdfs[0].resolve()
        if pdf in used:
            raise ValueError(f"Source PDF already appears in the dataset: {pdf}")
        sources[document] = pdf

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = UNet(base_channels=checkpoint["base_channels"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    size = checkpoint["image_size"]
    records = []
    sections = []

    with torch.no_grad():
        for document, pdf in sources.items():
            cards = []
            for page in PAGES:
                stem = f"{document}_p{page:04d}"
                original_path = image_dir / f"{stem}.png"
                prediction_path = image_dir / f"{stem}_prediction.jpg"
                render(pdf, page, original_path)
                source = Image.open(original_path).convert("RGB")
                prepared = resize_and_pad(source, size)
                tensor = torch.from_numpy(np.asarray(prepared).copy()).permute(2, 0, 1).float()
                prediction = model((tensor / 255.0).unsqueeze(0).to(device)).argmax(dim=1)
                square_mask = Image.fromarray(prediction.squeeze(0).cpu().numpy().astype(np.uint8))
                restored = restore_size(square_mask, source.size, size)
                overlay(source, restored).save(prediction_path, quality=90)
                values, counts = np.unique(np.asarray(restored), return_counts=True)
                coverage = {int(value): int(count) / (source.width * source.height)
                            for value, count in zip(values, counts)}
                records.append({
                    "document": document, "source_pdf": str(pdf.relative_to(REPO)),
                    "page": page, "image": str(original_path.relative_to(output_dir)),
                    "prediction": str(prediction_path.relative_to(output_dir)),
                    "predicted_table_fraction": coverage.get(1, 0.0),
                    "predicted_key_value_fraction": coverage.get(2, 0.0),
                })
                cards.append(
                    f'<article><h3>PDF page {page}</h3><div class="pair">'
                    f'<figure><figcaption>Source</figcaption><img src="images/{stem}.png"></figure>'
                    f'<figure><figcaption>Prediction</figcaption><img src="images/{stem}_prediction.jpg"></figure>'
                    f'</div></article>'
                )
            sections.append(f'<section><h2>{html.escape(document)}</h2><p>{html.escape(str(pdf.relative_to(REPO)))}</p>'
                            + "".join(cards) + "</section>")

    (output_dir / "predictions.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "checkpoint_epoch": checkpoint["epoch"],
        "dataset_pdf_overlap": False, "pages": records,
    }, indent=2) + "\n")
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Unseen PDF predictions</title>
<style>body{{font:16px system-ui;max-width:1500px;margin:30px auto;padding:0 20px;background:#f7f8fa;color:#17202a}}
section{{background:white;border:1px solid #ddd;border-radius:12px;padding:20px;margin:24px 0}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}figure{{margin:0}}figcaption{{font-weight:600;margin-bottom:8px}}
img{{width:100%;border:1px solid #ddd}}article{{margin:24px 0}}@media(max-width:800px){{.pair{{grid-template-columns:1fr}}}}</style>
<h1>Predictions on entirely unseen PDFs</h1><p>Checkpoint epoch {checkpoint['epoch']}; pages 3–4 from eight source PDFs absent from
train, validation, and test. Green = predicted table; blue = predicted key–value block. These PDFs have no pixel labels,
so this gallery is a visual check, not an accuracy measurement.</p>{''.join(sections)}</html>\n'''
    (output_dir / "index.html").write_text(page)

    tile_width, tile_height = 380, 325
    overview = Image.new("RGB", (tile_width * 2, tile_height * len(DOCUMENTS)), "white")
    draw = ImageDraw.Draw(overview)
    for row, document in enumerate(DOCUMENTS):
        stem = f"{document}_p{PAGES[0]:04d}"
        for column, (label, path) in enumerate((
            ("source", image_dir / f"{stem}.png"),
            ("prediction", image_dir / f"{stem}_prediction.jpg"),
        )):
            image = Image.open(path).convert("RGB")
            image.thumbnail((tile_width - 12, tile_height - 36))
            x = column * tile_width + (tile_width - image.width) // 2
            y = row * tile_height + 30
            overview.paste(image, (x, y))
            draw.text((column * tile_width + 8, row * tile_height + 6),
                      f"{document} · {label}", fill="black")
    overview.save(output_dir / "overview.jpg", quality=88)
    print(f"Reviewed {len(records)} pages from {len(sources)} unseen PDFs: {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
