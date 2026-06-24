"""Generate batch result plots from all documents in a data folder.

Usage:
    python viz_batch_results.py <folder>

Outputs saved to <folder>/results/:
    accuracy.png    – train vs test accuracy per document
    train_cost.png  – train-phase tokens and duration: ground truth vs code gen
    test_cost.png   – test-phase latency (pipeline vs GT) and GT token usage
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── data loading ──────────────────────────────────────────────────────────────

def _load_all(folder: str) -> list[dict]:
    docs = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name, "metadata.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            d["_name"] = name
            docs.append(d)
    return docs


def _pipeline_totals(phases: list[dict]) -> tuple[float, float]:
    """Sum tokens and duration across all pipeline (code-generation) phases."""
    tokens   = sum(p.get("tokens", 0)      for p in phases)
    duration = sum(p.get("duration_s", 0)  for p in phases)
    return tokens, duration


# ── plot 1: accuracy ──────────────────────────────────────────────────────────

def plot_accuracy(docs: list[dict], out_path: str) -> None:
    names      = [d["_name"] for d in docs]
    train_acc  = [d.get("train_eval", {}).get("accuracy") for d in docs]
    test_acc   = [d.get("test_eval",  {}).get("accuracy") for d in docs]

    x     = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.9), 5))

    bars_train = ax.bar(x - width / 2, train_acc, width, label="Train", color="#4C72B0")
    bars_test  = ax.bar(x + width / 2, test_acc,  width, label="Test",  color="#DD8452")

    for bar in (*bars_train, *bars_test):
        h = bar.get_height()
        if h is not None:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.2f}", ha="center", va="bottom", fontsize=7,
            )

    ax.set_title("Train vs Test Accuracy per Document")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ── plot 2: train-phase cost ──────────────────────────────────────────────────

def plot_train_cost(docs: list[dict], out_path: str) -> None:
    docs = [d for d in docs if "train_eval" in d and "pipeline_phases" in d]
    if not docs:
        return

    names = [d["_name"] for d in docs]

    gt_tokens   = [d["train_eval"].get("ground_truth_cost", {}).get("tokens",      0) for d in docs]
    gt_time     = [d["train_eval"].get("ground_truth_cost", {}).get("duration_s",  0) for d in docs]
    code_tokens = [_pipeline_totals(d["pipeline_phases"])[0] for d in docs]
    code_time   = [_pipeline_totals(d["pipeline_phases"])[1] for d in docs]

    x     = np.arange(len(names))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(14, len(names) * 1.3), 5))

    # — tokens —
    ax1.bar(x - width / 2, [t / 1_000 for t in gt_tokens],   width, label="Ground truth",    color="#4C72B0")
    ax1.bar(x + width / 2, [t / 1_000 for t in code_tokens], width, label="Code generation", color="#DD8452")
    ax1.set_title("Tokens")
    ax1.set_ylabel("Tokens (k)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # — duration —
    ax2.bar(x - width / 2, gt_time,   width, label="Ground truth",    color="#4C72B0")
    ax2.bar(x + width / 2, code_time, width, label="Code generation", color="#DD8452")
    ax2.set_title("Duration")
    ax2.set_ylabel("Seconds")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Train Phase Cost: Ground Truth vs Code Generation", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ── plot 3: test-phase cost ───────────────────────────────────────────────────

def plot_test_cost(docs: list[dict], out_path: str) -> None:
    docs = [d for d in docs if "test_eval" in d]
    if not docs:
        return

    names         = [d["_name"] for d in docs]
    pipeline_time = [d["test_eval"].get("agentic_pipeline_cost", {}).get("duration_s", 0) for d in docs]
    gt_time       = [d["test_eval"].get("ground_truth_cost",     {}).get("duration_s", 0) for d in docs]
    gt_tokens     = [d["test_eval"].get("ground_truth_cost",     {}).get("tokens",     0) for d in docs]

    x     = np.arange(len(names))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(14, len(names) * 1.3), 5))

    # — latency —
    ax1.bar(x - width / 2, pipeline_time, width, label="Pipeline (generated code)", color="#4C72B0")
    ax1.bar(x + width / 2, gt_time,       width, label="Ground truth generation",   color="#DD8452")
    ax1.set_title("Latency")
    ax1.set_ylabel("Seconds")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # — GT tokens —
    ax2.bar(x, [t / 1_000 for t in gt_tokens], color="#4C72B0")
    ax2.set_title("Ground Truth Tokens")
    ax2.set_ylabel("Tokens (k)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Test Phase Cost", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate batch result plots from a data folder.")
    parser.add_argument("folder", help="Root data folder (e.g. 'data').")
    args = parser.parse_args()

    docs = _load_all(args.folder)
    if not docs:
        print(f"No metadata.json files found under {args.folder!r}.")
        return
    print(f"Found {len(docs)} document(s): {[d['_name'] for d in docs]}")

    out_dir = os.path.join(args.folder, "results")
    os.makedirs(out_dir, exist_ok=True)

    plot_accuracy(   docs, os.path.join(out_dir, "accuracy.png"))
    plot_train_cost( docs, os.path.join(out_dir, "train_cost.png"))
    plot_test_cost(  docs, os.path.join(out_dir, "test_cost.png"))


if __name__ == "__main__":
    main()
