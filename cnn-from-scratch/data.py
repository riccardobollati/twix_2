"""Dataset and image preprocessing for page segmentation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parent
LABELS = {0: "background", 1: "table", 2: "key_value"}


def resize_and_pad(image: Image.Image, size: int, *, mask: bool = False) -> Image.Image:
    ratio = min(size / image.width, size / image.height)
    dimensions = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
    resampling = Image.Resampling.NEAREST if mask else Image.Resampling.BILINEAR
    resized = image.resize(dimensions, resampling)
    canvas = Image.new("L" if mask else "RGB", (size, size), 0 if mask else (255, 255, 255))
    canvas.paste(resized, ((size - dimensions[0]) // 2, (size - dimensions[1]) // 2))
    return canvas


class PageDataset(Dataset):
    def __init__(self, split: str, size: int = 768, dataset_root: Path | None = None):
        if split not in {"train", "val", "test"}:
            raise ValueError(split)
        if size < 256 or size % 16:
            raise ValueError("size must be at least 256 and divisible by 16")
        self.size = size
        self.dataset_root = Path(dataset_root) if dataset_root is not None else ROOT / "dataset"
        self.items = [entry for entry in json.loads((self.dataset_root / "manifest.json").read_text())["pages"]
                      if entry["split"] == split]
        if not self.items:
            raise ValueError(f"No {split} pages")
        for entry in self.items:
            mask = self.dataset_root / split / "masks" / f'{entry["id"]}.png'
            if not mask.exists():
                raise FileNotFoundError(f"Missing {mask}; run build_masks.py first")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        entry = self.items[index]
        image = Image.open(ROOT / entry["image"]).convert("RGB")
        mask = Image.open(self.dataset_root / entry["split"] / "masks" / f'{entry["id"]}.png').convert("L")
        if image.size != mask.size:
            raise ValueError(f"Image/mask size mismatch: {entry['id']}")
        image = resize_and_pad(image, self.size)
        mask = resize_and_pad(mask, self.size, mask=True)
        image_tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(np.asarray(mask).copy()).long()
        if mask_tensor.max() > 2:
            raise ValueError(f"Unknown class ID in mask {entry['id']}")
        return image_tensor, mask_tensor, entry["id"]
