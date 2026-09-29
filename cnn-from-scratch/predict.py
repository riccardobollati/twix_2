"""Color an unseen page image using a trained U-Net checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data import resize_and_pad
from model import model_from_checkpoint
from train import choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Overlay PNG (default: <image>.overlay.png)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = model_from_checkpoint(checkpoint).to(device)
    model.eval()
    source = Image.open(args.image).convert("RGB")
    size = checkpoint["image_size"]
    prepared = resize_and_pad(source, size)
    tensor = torch.from_numpy(np.asarray(prepared).copy()).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    with torch.no_grad():
        mask = Image.fromarray(model(tensor).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8))
    ratio = min(size / source.width, size / source.height)
    resized = (max(1, round(source.width * ratio)), max(1, round(source.height * ratio)))
    left, top = (size - resized[0]) // 2, (size - resized[1]) // 2
    mask = mask.crop((left, top, left + resized[0], top + resized[1])).resize(source.size, Image.Resampling.NEAREST)
    palette = np.zeros((source.height, source.width, 4), dtype=np.uint8)
    ids = np.asarray(mask)
    palette[ids == 1] = (20, 175, 65, 110)
    palette[ids == 2] = (30, 95, 230, 110)
    result = Image.alpha_composite(source.convert("RGBA"), Image.fromarray(palette, "RGBA"))
    output = args.output or args.image.with_name(args.image.stem + ".overlay.png")
    result.save(output)
    print(output)


if __name__ == "__main__":
    main()
