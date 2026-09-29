"""Train a randomly initialized, three-class page segmentation U-Net."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from data import LABELS, PageDataset, ROOT
from model import UNet, WindowUNet


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def class_weights(dataset: PageDataset, device: torch.device) -> torch.Tensor:
    counts = np.zeros(3, dtype=np.int64)
    for entry in dataset.items:
        path = dataset.dataset_root / "train" / "masks" / f'{entry["id"]}.png'
        counts += np.bincount(np.asarray(Image.open(path), dtype=np.uint8).ravel(), minlength=3)[:3]
    if np.any(counts == 0):
        raise ValueError(f"Every class must occur in train masks; pixel counts: {counts.tolist()}")
    frequencies = counts / counts.sum()
    values = 1 / np.sqrt(frequencies)
    values /= values[0]
    return torch.tensor(values, dtype=torch.float32, device=device)


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = logits.softmax(dim=1)
    one_hot = torch.nn.functional.one_hot(target, 3).permute(0, 3, 1, 2)
    intersection = (probabilities[:, 1:] * one_hot[:, 1:]).sum(dim=(0, 2, 3))
    denominator = probabilities[:, 1:].sum(dim=(0, 2, 3)) + one_hot[:, 1:].sum(dim=(0, 2, 3))
    return 1 - ((2 * intersection + 1) / (denominator + 1)).mean()


def confusion_update(matrix: torch.Tensor, prediction: torch.Tensor, target: torch.Tensor) -> None:
    flat = (target * 3 + prediction).reshape(-1)
    matrix += torch.bincount(flat, minlength=9).reshape(3, 3).cpu()


def metrics(matrix: torch.Tensor) -> dict:
    result = {}
    foreground = []
    for class_id, name in LABELS.items():
        true_positive = matrix[class_id, class_id].item()
        union = matrix[class_id, :].sum().item() + matrix[:, class_id].sum().item() - true_positive
        value = true_positive / union if union else None
        result[f"iou_{name}"] = value
        if class_id and value is not None:
            foreground.append(value)
    result["mean_foreground_iou"] = sum(foreground) / len(foreground) if foreground else None
    result["pixel_accuracy"] = matrix.diag().sum().item() / max(1, matrix.sum().item())
    return result


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    matrix = torch.zeros((3, 3), dtype=torch.long)
    for image, mask, _ in loader:
        prediction = model(image.to(device)).argmax(dim=1).cpu()
        confusion_update(matrix, prediction, mask)
    return metrics(matrix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=0,
                        help="Stop after this many epochs without validation improvement; 0 disables")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--first-kernel", type=int, choices=(3, 16), default=3,
                        help="3 keeps the original U-Net; 16 uses a 16×16 first kernel with stride 16")
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "first")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.patience < 0:
        parser.error("epochs, batch size, and learning rate must be positive")
    if args.first_kernel == 16 and args.image_size % 256:
        parser.error("the 16×16 first-kernel model needs image-size divisible by 256")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    train_set = PageDataset("train", args.image_size, args.dataset_root)
    val_set = PageDataset("val", args.image_size, args.dataset_root)
    test_set = PageDataset("test", args.image_size, args.dataset_root)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    model_class = WindowUNet if args.first_kernel == 16 else UNet
    model = model_class(base_channels=args.base_channels).to(device)
    weights = class_weights(train_set, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_score = -1.0
    best_epoch = 0
    best_path = args.output_dir / "best.pt"
    print(f"device={device} parameters={sum(p.numel() for p in model.parameters()):,} "
          f"pages={len(train_set)}/{len(val_set)}/{len(test_set)} weights={weights.tolist()}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks) + 0.5 * dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * images.shape[0]
        val = evaluate(model, val_loader, device)
        score = val["mean_foreground_iou"] or 0.0
        record = {"epoch": epoch, "train_loss": loss_sum / len(train_set), "validation": val}
        history.append(record)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save({"state_dict": model.state_dict(), "image_size": args.image_size,
                        "first_kernel": args.first_kernel,
                        "base_channels": args.base_channels, "epoch": epoch,
                        "validation": val, "classes": LABELS}, best_path)
        print(f"epoch {epoch:03d} loss={record['train_loss']:.4f} "
              f"val_foreground_IoU={score:.4f} best={best_score:.4f}", flush=True)
        if args.patience and epoch - best_epoch >= args.patience:
            print(f"Stopped after {args.patience} epochs without validation improvement", flush=True)
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    report = {"best_epoch": checkpoint["epoch"], "validation": checkpoint["validation"],
              "test": evaluate(model, test_loader, device), "history": history,
              "train_pages": len(train_set), "val_pages": len(val_set), "test_pages": len(test_set)}
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Best epoch {report['best_epoch']}; test={report['test']}", flush=True)
    print(f"Saved {best_path} and {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
