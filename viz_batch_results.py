"""Generate batch result plots from all documents in a data folder.

Usage:
    python viz_batch_results.py <folder>

Outputs saved to <folder>/results/:
    accuracy.png    – train (per loop iteration) vs test accuracy per document
    train_cost.png  – train-phase tokens and duration: ground truth vs code gen
    test_cost.png   – test-phase latency (pipeline vs GT) and GT token usage
    phase_cost_overall.png – mean phase spend share and variance across documents
    phase_token_distributions.png – token-cost histograms for the five phases
    <name>/phase_cost.png   – USD spend by generation/fixing phase for one document
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ── token pricing (USD per million tokens) ───────────────────────────────────
PRICE_INPUT_PER_M       = 2.50   # hard model (gpt-5.4)
PRICE_OUTPUT_PER_M      = 15.00  # hard model (gpt-5.4)
PRICE_INPUT_EASY_PER_M  = 0.75   # easy model (gpt-5.4-mini)
PRICE_OUTPUT_EASY_PER_M = 4.50   # easy model (gpt-5.4-mini)


PHASE_COST_LABELS = {
    "template_inference": "Infer template",
    "record_separation": "Record separation",
    "block_separation": "Block separation",
    "data_extraction": "Data extraction",
    "fixing_loops": "Fixing loops",
}

PHASE_COST_COLORS = {
    "template_inference": "#4C72B0",
    "record_separation": "#55A868",
    "block_separation": "#C44E52",
    "data_extraction": "#8172B2",
    "fixing_loops": "#CCB974",
}


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


def _cost_usd(
    cost: dict,
    inp_price: float = PRICE_INPUT_PER_M,
    out_price: float = PRICE_OUTPUT_PER_M,
) -> float:
    """Convert a cost entry's token counts to USD."""
    inp, out = _split_tokens(cost)
    return (inp * inp_price + out * out_price) / 1_000_000


def _pipeline_cost_usd(phases: list[dict], block_models: dict | None) -> float:
    """Total USD cost for all pipeline phases.

    For the block_extraction_scripting phase, per-block model pricing is applied
    when per-block token data and a block_models mapping are available; otherwise
    falls back to hard-model pricing for the whole phase.
    """
    total = 0.0
    for p in phases:
        if p.get("phase") == "block_extraction_scripting" and block_models:
            blocks = p.get("blocks")
            if blocks:
                for b in blocks:
                    model = block_models.get(str(b.get("id", "")), "")
                    if "mini" in model:
                        total += _cost_usd(b, PRICE_INPUT_EASY_PER_M, PRICE_OUTPUT_EASY_PER_M)
                    else:
                        total += _cost_usd(b)
                continue  # per-block totals already accumulated; skip phase-level sum
        total += _cost_usd(p)
    return total


def _extraction_cost_usd(d: dict) -> float:
    """Return extraction USD cost, combining routing and all block scripts."""
    total = 0.0
    block_models = d.get("block_models")
    for phase in d.get("pipeline_phases", []):
        phase_name = phase.get("phase")
        if phase_name == "block_difficulty_router":
            total += _cost_usd(phase)
        elif phase_name == "block_extraction_scripting":
            total += _pipeline_cost_usd([phase], block_models)
    return total


def _phase_costs_usd(d: dict) -> dict[str, float]:
    """Return the requested generation/fixing spend buckets in USD."""
    pipeline_costs: dict[str, float] = {}
    for phase in d.get("pipeline_phases", []):
        phase_name = phase.get("phase")
        if phase_name in {"template_inference", "record_separation", "block_separation"}:
            pipeline_costs[phase_name] = (
                pipeline_costs.get(phase_name, 0.0) + _cost_usd(phase)
            )

    repairs = d.get("train_eval", {}).get("repairs") or []
    return {
        "template_inference": pipeline_costs.get("template_inference", 0.0),
        "record_separation": pipeline_costs.get("record_separation", 0.0),
        "block_separation": pipeline_costs.get("block_separation", 0.0),
        "data_extraction": _extraction_cost_usd(d),
        "fixing_loops": sum(_cost_usd(repair) for repair in repairs),
    }


def _split_tokens(cost: dict) -> tuple[float, float]:
    """Return (input_tokens, output_tokens) from a cost entry.

    Falls back to treating a legacy ``tokens`` total as output when the split
    fields are absent (backward-compatible with older metadata files).
    """
    inp = cost.get("input_tokens", 0)
    out = cost.get("output_tokens", 0)
    if not inp and not out:
        out = cost.get("tokens", 0)
    return inp, out


def _total_tokens(cost: dict) -> float:
    """Return total input plus output tokens for one cost entry."""
    inp, out = _split_tokens(cost)
    return inp + out


def _phase_tokens(d: dict) -> dict[str, float]:
    """Return total token usage for each generation/fixing phase."""
    totals = {key: 0.0 for key in PHASE_COST_LABELS}
    for phase in d.get("pipeline_phases", []):
        phase_name = phase.get("phase")
        if phase_name in {"template_inference", "record_separation", "block_separation"}:
            totals[phase_name] += _total_tokens(phase)
        elif phase_name in {"block_difficulty_router", "block_extraction_scripting"}:
            # The scripting entry already aggregates every extraction block.
            totals["data_extraction"] += _total_tokens(phase)

    repairs = d.get("train_eval", {}).get("repairs") or []
    totals["fixing_loops"] = sum(_total_tokens(repair) for repair in repairs)
    return totals


def _pipeline_totals(phases: list[dict]) -> tuple[float, float, float]:
    """Sum input/output tokens and duration across all pipeline phases."""
    input_tokens  = sum(_split_tokens(p)[0] for p in phases)
    output_tokens = sum(_split_tokens(p)[1] for p in phases)
    duration      = sum(p.get("duration_s", 0) for p in phases)
    return input_tokens, output_tokens, duration


# ── plot 1: accuracy ──────────────────────────────────────────────────────────

def plot_accuracy(docs: list[dict], out_path: str) -> None:
    # Find max training iterations across all documents
    max_iters = 1
    for d in docs:
        iters = d.get("train_eval", {}).get("iterations")
        if iters:
            max_iters = max(max_iters, len(iters))

    names = [d["_name"] for d in docs]
    n     = len(names)

    # Per-document per-iteration train accuracy (padded to max_iters)
    train_accs: list[list[float | None]] = []
    for d in docs:
        iters = d.get("train_eval", {}).get("iterations") or []
        row   = [it.get("accuracy") for it in iters]
        if not row:
            row = [d.get("train_eval", {}).get("accuracy")]
        row += [None] * (max_iters - len(row))
        train_accs.append(row)

    test_acc = [d.get("test_eval", {}).get("accuracy") for d in docs]

    bars_per_doc = max_iters + 1  # train iters + test
    group_width  = 0.8
    bar_width    = group_width / bars_per_doc
    x            = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(10, n * max(1.2, bars_per_doc * 0.35)), 5))

    train_cmap   = plt.colormaps["Blues"]
    train_colors = [train_cmap(0.35 + 0.5 * i / max(max_iters - 1, 1)) for i in range(max_iters)]

    all_bars: list[tuple] = []

    for i in range(max_iters):
        vals    = [row[i] for row in train_accs]
        label   = "Train" if max_iters == 1 else f"Train iter {i}"
        offsets = x - group_width / 2 + (i + 0.5) * bar_width
        bars    = ax.bar(offsets, [v or 0 for v in vals], bar_width,
                         label=label, color=train_colors[i])
        all_bars.append((bars, vals))

    # Test bar
    offsets   = x - group_width / 2 + (max_iters + 0.5) * bar_width
    bars_test = ax.bar(offsets, [v or 0 for v in test_acc], bar_width,
                       label="Test", color="#DD8452")
    all_bars.append((bars_test, test_acc))

    for bars, vals in all_bars:
        for bar, v in zip(bars, vals):
            if v is not None and v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7,
                )

    valid_test = [v for v in test_acc if v is not None]
    if valid_test:
        avg_test = sum(valid_test) / len(valid_test)
        ax.axhline(avg_test, color="#DD8452", linestyle="--", linewidth=1.2,
                   label=f"Avg test {avg_test:.2f}")
        ax.text(0.99, avg_test, f"{avg_test:.2f}", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8, color="#DD8452")

    title = ("Train vs Test Accuracy per Document" if max_iters == 1
             else "Training Loop & Test Accuracy per Document")
    ax.set_title(title)
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

# ── helpers for stacked token bars ───────────────────────────────────────────

def _stacked_token_bar(
    ax,
    positions,
    input_vals: list[float],
    output_vals: list[float],
    width: float,
    input_color: str,
    output_color: str,
    label_prefix: str,
) -> None:
    """Draw a pair of stacked bars (input bottom, output top) at `positions`."""
    ax.bar(positions, input_vals,  width, label=f"{label_prefix} input",
           color=input_color)
    ax.bar(positions, output_vals, width, label=f"{label_prefix} output",
           color=output_color, bottom=input_vals)


# ── plot 2: train-phase cost ──────────────────────────────────────────────────

def plot_train_cost(docs: list[dict], out_path: str) -> None:
    docs = [d for d in docs if "train_eval" in d and "pipeline_phases" in d]
    if not docs:
        return

    names = [d["_name"] for d in docs]

    gt_tokens_split = [_split_tokens(d["train_eval"].get("ground_truth_cost", {})) for d in docs]
    gt_input    = [s[0] for s in gt_tokens_split]
    gt_output   = [s[1] for s in gt_tokens_split]
    gt_time     = [d["train_eval"].get("ground_truth_cost", {}).get("duration_s", 0) for d in docs]

    pipeline_results = [_pipeline_totals(d["pipeline_phases"]) for d in docs]
    code_input  = [r[0] for r in pipeline_results]
    code_output = [r[1] for r in pipeline_results]
    code_time   = [r[2] for r in pipeline_results]

    x     = np.arange(len(names))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(14, len(names) * 1.3), 5))

    # — tokens (stacked: input bottom, output top) —
    _stacked_token_bar(
        ax1, x - width / 2,
        [i / 1_000 for i in gt_input],   [o / 1_000 for o in gt_output],
        width, "#2C5282", "#4C72B0", "GT",
    )
    _stacked_token_bar(
        ax1, x + width / 2,
        [i / 1_000 for i in code_input],  [o / 1_000 for o in code_output],
        width, "#8B3A00", "#DD8452", "Code",
    )
    ax1.set_title("Tokens")
    ax1.set_ylabel("Tokens (k)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.legend(fontsize=8)
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
    gt_tokens_split = [_split_tokens(d["test_eval"].get("ground_truth_cost", {})) for d in docs]
    gt_input      = [s[0] for s in gt_tokens_split]
    gt_output     = [s[1] for s in gt_tokens_split]

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

    # — GT tokens (stacked: input bottom, output top) —
    ax2.bar(x, [i / 1_000 for i in gt_input],  color="#2C5282", label="Input tokens")
    ax2.bar(x, [o / 1_000 for o in gt_output], color="#4C72B0", label="Output tokens",
            bottom=[i / 1_000 for i in gt_input])
    ax2.set_title("Ground Truth Tokens")
    ax2.set_ylabel("Tokens (k)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Test Phase Cost", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ── break-even computation ────────────────────────────────────────────────────

def _break_even_pages(d: dict) -> float | None:
    """Return break-even pages for one document, or None if not computable.

    Fixed cost  = code generation (pipeline_phases) + train GT + train grouping + repairs.
    LLM cost/pg = test ground-truth tokens / test_pages  (full LLM-per-page baseline).
    Run cost/pg  = test agentic_pipeline_cost / test_pages  (grouping LLM still runs).
    Break-even   = fixed_cost / (llm_cost_per_page - run_cost_per_page).
    """
    train_pages = d.get("train_pages") or 0
    test_pages  = d.get("test_pages")  or 0
    train_eval  = d.get("train_eval", {})
    test_eval   = d.get("test_eval",  {})
    train_gt    = train_eval.get("ground_truth_cost", {})
    test_gt     = test_eval.get("ground_truth_cost",  {})

    if test_gt and test_pages:
        llm_cost_per_page = _cost_usd(test_gt) / test_pages
    elif train_gt and train_pages:
        llm_cost_per_page = _cost_usd(train_gt) / train_pages
    else:
        return None
    if llm_cost_per_page <= 0:
        return None

    pipe_run_per_page = 0.0
    if test_pages:
        pipe_run_per_page = _cost_usd(test_eval.get("agentic_pipeline_cost", {})) / test_pages

    repairs    = train_eval.get("repairs") or []
    block_models = d.get("block_models")
    fixed_cost = (
        _pipeline_cost_usd(d.get("pipeline_phases", []), block_models)
        + _cost_usd(train_gt)
        + _cost_usd(train_eval.get("agentic_pipeline_cost", {}))
        + sum(_cost_usd(r) for r in repairs)
    )

    savings = llm_cost_per_page - pipe_run_per_page
    return fixed_cost / savings if savings > 0 else None


# ── plot 4: break-even analysis ──────────────────────────────────────────────

def plot_break_even(docs: list[dict], out_path: str) -> None:
    usable = []
    for d in docs:
        be = _break_even_pages(d)
        if be is None:
            continue
        usable.append({
            "name":       d["_name"],
            "break_even": be,
        })

    if not usable:
        return

    n     = len(usable)
    names = [u["name"] for u in usable]

    fig, ax = plt.subplots(figsize=(max(8, n * 1.1), 5))

    x    = np.arange(n)
    bars = ax.bar(x, [u["break_even"] for u in usable], 0.6,
                  color="#4C72B0", label="Break-even pages")
    for bar, u in zip(bars, usable):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{u['break_even']:.0f}", ha="center", va="bottom", fontsize=8,
        )

    avg_be = sum(u["break_even"] for u in usable) / len(usable)
    ax.axhline(avg_be, color="#4C72B0", linestyle="--", linewidth=1.2)
    ax.text(0.99, avg_be, f"{avg_be:.0f}", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color="#4C72B0")

    ax.set_title(
        f"Break-even pages  —  ${PRICE_INPUT_PER_M}/M input · ${PRICE_OUTPUT_PER_M}/M output"
    )
    ax.set_ylabel("Pages")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ── shared helper ────────────────────────────────────────────────────────────

def _block_count(d: dict) -> int:
    if d.get("block_models"):
        return len(d["block_models"])
    for p in d.get("pipeline_phases", []):
        if p.get("phase") == "block_extraction_scripting" and p.get("blocks"):
            return len(p["blocks"])
    return 0


# ── plot 5: accuracy + break-even combined ───────────────────────────────────

BE_COLOR    = "#9467bd"   # purple — break-even
BLOCK_COLOR = "#2ca02c"   # green  — block count

def plot_combined(docs: list[dict], out_path: str) -> None:
    """Accuracy bars (left axis) + break-even bar (right axis) + block count bar (far-right axis)."""
    max_iters = 1
    for d in docs:
        iters = d.get("train_eval", {}).get("iterations")
        if iters:
            max_iters = max(max_iters, len(iters))

    names = [d["_name"] for d in docs]
    n     = len(names)

    train_accs: list[list[float | None]] = []
    for d in docs:
        iters = d.get("train_eval", {}).get("iterations") or []
        row   = [it.get("accuracy") for it in iters]
        if not row:
            row = [d.get("train_eval", {}).get("accuracy")]
        row += [None] * (max_iters - len(row))
        train_accs.append(row)

    test_acc    = [d.get("test_eval", {}).get("accuracy") for d in docs]
    break_even  = [_break_even_pages(d) for d in docs]
    block_count = [_block_count(d) for d in docs]

    # slots: train iters + test + break-even + block count
    acc_slots   = max_iters + 1
    total_slots = acc_slots + 2
    group_width = 0.8
    bar_width   = group_width / total_slots
    x           = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(10, n * max(1.4, total_slots * 0.35)), 5))
    ax2 = ax.twinx()
    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("outward", 55))

    # ── accuracy bars on ax (left) ────────────────────────────────────────────
    train_cmap   = plt.colormaps["Blues"]
    train_colors = [train_cmap(0.35 + 0.5 * i / max(max_iters - 1, 1)) for i in range(max_iters)]
    all_bars: list[tuple] = []

    for i in range(max_iters):
        vals    = [row[i] for row in train_accs]
        label   = "Train" if max_iters == 1 else f"Train iter {i}"
        offsets = x - group_width / 2 + (i + 0.5) * bar_width
        bars    = ax.bar(offsets, [v or 0 for v in vals], bar_width,
                         label=label, color=train_colors[i])
        all_bars.append((bars, vals))

    test_offsets = x - group_width / 2 + (max_iters + 0.5) * bar_width
    bars_test    = ax.bar(test_offsets, [v or 0 for v in test_acc], bar_width,
                          label="Test", color="#DD8452")
    all_bars.append((bars_test, test_acc))

    for bars, vals in all_bars:
        for bar, v in zip(bars, vals):
            if v is not None and v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7,
                )

    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    # ── break-even bars on ax2 (right) ───────────────────────────────────────
    be_offsets = x - group_width / 2 + (acc_slots + 0.5) * bar_width
    be_vals    = [be or 0 for be in break_even]
    bars_be    = ax2.bar(be_offsets, be_vals, bar_width,
                         label="Break-even pages", color=BE_COLOR, alpha=0.85)

    for bar, be in zip(bars_be, break_even):
        if be is not None and be > 0:
            ax2.text(
                bar.get_x() + bar.get_width() / 2, be + 0.5,
                f"{be:.0f}", ha="center", va="bottom", fontsize=7, color=BE_COLOR,
            )

    ax2.set_ylabel("Break-even (pages)", color=BE_COLOR)
    ax2.tick_params(axis="y", labelcolor=BE_COLOR)
    ax2.spines["right"].set_color(BE_COLOR)

    # ── block count bars on ax3 (far right) ──────────────────────────────────
    bc_offsets = x - group_width / 2 + (acc_slots + 1.5) * bar_width
    bars_bc    = ax3.bar(bc_offsets, block_count, bar_width,
                         label="Block types", color=BLOCK_COLOR, alpha=0.75)

    for bar, bc in zip(bars_bc, block_count):
        if bc > 0:
            ax3.text(
                bar.get_x() + bar.get_width() / 2, bc + 0.05,
                str(bc), ha="center", va="bottom", fontsize=7, color=BLOCK_COLOR,
            )

    ax3.set_ylabel("Block types", color=BLOCK_COLOR)
    ax3.tick_params(axis="y", labelcolor=BLOCK_COLOR)
    ax3.spines["right"].set_color(BLOCK_COLOR)
    ax3.set_ylim(0, max(block_count or [1]) * 1.4)
    ax3.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # ── avg lines ────────────────────────────────────────────────────────────
    valid_test = [v for v in test_acc if v is not None]
    if valid_test:
        avg_test = sum(valid_test) / len(valid_test)
        ax.axhline(avg_test, color="#DD8452", linestyle="--", linewidth=1.2,
                   label=f"Avg test {avg_test:.2f}")
        ax.text(0.99, avg_test, f"{avg_test:.2f}", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8, color="#DD8452")

    valid_be = [be for be in break_even if be is not None]
    if valid_be:
        avg_be = sum(valid_be) / len(valid_be)
        ax2.axhline(avg_be, color=BE_COLOR, linestyle="--", linewidth=1.2,
                    label=f"Avg break-even {avg_be:.0f}")
        ax2.text(0.99, avg_be, f"{avg_be:.0f}", transform=ax2.get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=8, color=BE_COLOR)

    # ── shared x-axis & legend ────────────────────────────────────────────────
    title = ("Train vs Test Accuracy + Break-even + Block types" if max_iters == 1
             else "Training Loop & Test Accuracy + Break-even + Block types")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles3, labels3 = ax3.get_legend_handles_labels()
    ax.legend(handles1 + handles2 + handles3, labels1 + labels2 + labels3, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ── plot 6: break-even vs block count scatter ────────────────────────────────

def plot_cost_vs_blocks(docs: list[dict], out_path: str) -> None:
    """Scatter: number of block types (x) vs break-even pages (y), one point per document."""
    points = []
    for d in docs:
        be = _break_even_pages(d)
        bc = _block_count(d)
        if be is not None and bc > 0:
            points.append({"name": d["_name"], "blocks": bc, "break_even": be})

    if not points:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    xs = [p["blocks"]     for p in points]
    ys = [p["break_even"] for p in points]

    ax.scatter(xs, ys, color="#4C72B0", s=80, zorder=3)

    for p in points:
        ax.annotate(
            p["name"],
            (p["blocks"], p["break_even"]),
            textcoords="offset points",
            xytext=(6, 3),
            fontsize=8,
            color="dimgray",
        )

    ax.set_xlabel("Number of block types")
    ax.set_ylabel("Break-even (pages)")
    ax.set_title("Break-even pages vs number of block types")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ── plot 7: per-document overview ────────────────────────────────────────────

def _plot_phase_pie(
    costs: dict[str, float],
    out_path: str,
    title: str,
) -> None:
    """Plot one document's non-zero phase costs as a USD pie chart."""
    phase_keys = [key for key in PHASE_COST_LABELS if costs.get(key, 0) > 0]
    if not phase_keys:
        return

    values = [costs[key] for key in phase_keys]
    colors = [PHASE_COST_COLORS[key] for key in phase_keys]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, _, autotexts = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 2 else "",
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    for text in autotexts:
        text.set_fontsize(8)

    legend_labels = [
        f"{PHASE_COST_LABELS[key]} — ${costs[key]:.4f}"
        for key in phase_keys
    ]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(f"{title}\nTotal: ${total:.4f}")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_overall_phase_cost(docs: list[dict], out_path: str) -> None:
    """Plot mean per-document phase percentages and population variance."""
    percentages = []
    for d in docs:
        costs = _phase_costs_usd(d)
        total = sum(costs.values())
        if total > 0:
            percentages.append({key: value / total * 100 for key, value in costs.items()})

    if not percentages:
        return

    phase_keys = [
        key for key in PHASE_COST_LABELS
        if any(row[key] > 0 for row in percentages)
    ]
    means = [float(np.mean([row[key] for row in percentages])) for key in phase_keys]
    variances = [float(np.var([row[key] for row in percentages])) for key in phase_keys]
    colors = [PHASE_COST_COLORS[key] for key in phase_keys]

    fig, ax = plt.subplots(figsize=(9, 6))
    wedges, _, autotexts = ax.pie(
        means,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=lambda pct: f"{pct:.1f}%" if pct >= 2 else "",
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    for text in autotexts:
        text.set_fontsize(8)

    legend_labels = [
        f"{PHASE_COST_LABELS[key]} — mean {mean:.1f}% · variance {variance:.1f} pp²"
        for key, mean, variance in zip(phase_keys, means, variances)
    ]
    ax.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_title(f"Average Phase Spend Share Across {len(percentages)} Documents")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_phase_token_distributions(docs: list[dict], out_path: str) -> None:
    """Plot one token-cost histogram per phase across all documents."""
    if not docs:
        return

    per_doc = [_phase_tokens(d) for d in docs]
    layout = [
        ["template_inference", "record_separation", "block_separation"],
        ["data_extraction", "fixing_loops", "."],
    ]
    fig, axes = plt.subplot_mosaic(layout, figsize=(15, 9))

    for phase_key, ax in axes.items():
        values = np.asarray([row[phase_key] for row in per_doc], dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values))

        ax.hist(
            values,
            bins="auto",
            color=PHASE_COST_COLORS[phase_key],
            edgecolor="white",
            alpha=0.9,
        )
        ax.axvline(
            mean,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean {mean:,.0f}\nStd dev {std:,.0f}",
        )
        ax.set_title(PHASE_COST_LABELS[phase_key])
        ax.set_xlabel("Tokens")
        ax.set_ylabel("Documents")
        ax.xaxis.set_major_formatter(mticker.EngFormatter())
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Phase Token Cost Distributions Across {len(docs)} Documents",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_document(d: dict, out_dir: str) -> None:
    """Two-panel overview for a single document: accuracy (left) + break-even & blocks (right)."""
    name    = d["_name"]
    doc_dir = os.path.join(out_dir, name)
    os.makedirs(doc_dir, exist_ok=True)

    # accuracy
    iters      = d.get("train_eval", {}).get("iterations") or []
    train_accs = [it.get("accuracy") for it in iters]
    if not train_accs:
        train_accs = [d.get("train_eval", {}).get("accuracy")]
    test_acc = d.get("test_eval", {}).get("accuracy")

    n_train      = len(train_accs)
    acc_labels   = ([f"Train iter {i}" for i in range(n_train)] if n_train > 1 else ["Train"]) + ["Test"]
    train_cmap   = plt.colormaps["Blues"]
    train_colors = [train_cmap(0.35 + 0.5 * i / max(n_train - 1, 1)) for i in range(n_train)]
    acc_colors   = train_colors + ["#DD8452"]
    acc_raw      = list(train_accs) + [test_acc]
    acc_vals     = [v or 0 for v in acc_raw]

    # break-even + block count
    be = _break_even_pages(d)
    bc = _block_count(d)

    fig, (ax_acc, ax_be) = plt.subplots(1, 2, figsize=(12, 5))

    # ── left: accuracy ────────────────────────────────────────────────────────
    x    = np.arange(len(acc_labels))
    bars = ax_acc.bar(x, acc_vals, 0.5, color=acc_colors)
    for bar, v in zip(bars, acc_raw):
        if v is not None and v > 0:
            ax_acc.text(
                bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9,
            )
    ax_acc.set_title("Accuracy")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(acc_labels)
    ax_acc.set_ylim(0, 1.15)
    ax_acc.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax_acc.grid(axis="y", alpha=0.3)

    # ── right: break-even (left axis) + block count (right axis) ─────────────
    ax_bc = ax_be.twinx()

    if be is not None:
        ax_be.bar([0], [be], 0.4, color=BE_COLOR, alpha=0.85, label="Break-even pages")
        ax_be.text(0, be + 0.3, f"{be:.0f}", ha="center", va="bottom",
                   fontsize=10, color=BE_COLOR, fontweight="bold")

    ax_bc.bar([1], [bc], 0.4, color=BLOCK_COLOR, alpha=0.75, label="Block types")
    ax_bc.text(1, bc + 0.05, str(bc), ha="center", va="bottom",
               fontsize=10, color=BLOCK_COLOR, fontweight="bold")

    ax_be.set_ylabel("Break-even (pages)", color=BE_COLOR)
    ax_be.tick_params(axis="y", labelcolor=BE_COLOR)
    ax_be.spines["right"].set_color(BE_COLOR)
    if be is not None:
        ax_be.set_ylim(0, be * 1.2)

    ax_bc.set_ylabel("Block types", color=BLOCK_COLOR)
    ax_bc.tick_params(axis="y", labelcolor=BLOCK_COLOR)
    ax_bc.spines["right"].set_color(BLOCK_COLOR)
    ax_bc.set_ylim(0, max(bc, 1) * 1.5)
    ax_bc.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax_be.set_xlim(-0.6, 1.6)
    ax_be.set_xticks([0, 1])
    ax_be.set_xticklabels(["Break-even", "Block types"])
    ax_be.set_title("Break-even & Block types")
    ax_be.grid(axis="y", alpha=0.3)

    fig.suptitle(name, fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(doc_dir, "overview.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")

    _plot_phase_pie(
        _phase_costs_usd(d),
        os.path.join(doc_dir, "phase_cost.png"),
        f"{name}: Generation & Fixing Spend",
    )


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

    plot_accuracy(      docs, os.path.join(out_dir, "accuracy.png"))
    plot_train_cost(    docs, os.path.join(out_dir, "train_cost.png"))
    plot_test_cost(     docs, os.path.join(out_dir, "test_cost.png"))
    plot_break_even(    docs, os.path.join(out_dir, "break_even.png"))
    plot_combined(      docs, os.path.join(out_dir, "combined.png"))
    plot_cost_vs_blocks(docs, os.path.join(out_dir, "cost_vs_blocks.png"))
    plot_overall_phase_cost(docs, os.path.join(out_dir, "phase_cost_overall.png"))
    plot_phase_token_distributions(
        docs, os.path.join(out_dir, "phase_token_distributions.png")
    )

    for d in docs:
        plot_document(d, out_dir)


if __name__ == "__main__":
    main()
