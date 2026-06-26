#!/usr/bin/env python3
"""Test the block difficulty router on an existing block_samples folder.

Usage:
    python test_routing.py data/<name>/          # looks for agent_space/train/block_samples
    python test_routing.py data/<name>/agent_space/train/block_samples
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def _find_block_samples_dir(path: Path) -> Path:
    """Accept either the document root or the block_samples dir directly."""
    if (path / "block_samples").is_dir():
        # given agent_space/train/
        return path / "block_samples"
    if path.name == "block_samples" and path.is_dir():
        return path
    candidate = path / "agent_space" / "train" / "block_samples"
    if candidate.is_dir():
        return candidate
    console.print(f"[red]Cannot find block_samples under {path}[/red]")
    sys.exit(1)


def _sample_counts(block_samples_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in sorted(block_samples_dir.iterdir()):
        if d.is_dir():
            counts[d.name] = len(list(d.glob("*.png")))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the block difficulty router and show classification results."
    )
    parser.add_argument(
        "path",
        help="Document folder (data/<name>/) or agent_space/train/ or block_samples/ directly.",
    )
    args = parser.parse_args()

    input_path = Path(args.path).resolve()
    block_samples_dir = _find_block_samples_dir(input_path)
    train_dir = block_samples_dir.parent

    console.print()
    console.print(Panel(
        Text(str(block_samples_dir), style="bold white"),
        title="[bold]Block Difficulty Router — test run[/]",
        border_style="bright_white",
        box=box.DOUBLE_EDGE,
        padding=(0, 2),
    ))

    # Show sample counts first
    counts = _sample_counts(block_samples_dir)
    if not counts:
        console.print("[red]No block subdirectories found.[/red]")
        sys.exit(1)

    t_pre = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
    t_pre.add_column("Block", style="bold")
    t_pre.add_column("Samples", justify="right", style="dim")
    for block_id, n in counts.items():
        t_pre.add_row(block_id, str(n))
    console.print(t_pre)
    console.print()

    # Build minimal state (router only needs pdf_path to derive train_dir)
    state = {"pdf_path": str(train_dir / "dummy.pdf")}

    # Lazy import so the script fails fast on missing samples rather than missing deps
    from src.agents.block_difficulty_router import MODEL_EASY, MODEL_HARD
    from src.agents.block_difficulty_router import block_difficulty_router_node
    from src.models import get_llm

    console.print("[dim]Initialising LLM…[/dim]")
    llm = get_llm()

    console.print("[dim]Running router…[/dim]\n")
    t0 = time.time()
    result = block_difficulty_router_node(state, llm)
    elapsed = time.time() - t0

    assignments: dict[str, str] = result.get("block_codex_models", {})

    # Results table
    t = Table(
        title="[bold]Routing Results[/]",
        box=box.ROUNDED,
        border_style="bright_white",
        show_lines=True,
    )
    t.add_column("Block",      style="bold white", width=8)
    t.add_column("Samples",    justify="right",    style="dim", width=9)
    t.add_column("Difficulty", width=10)
    t.add_column("Model",      style="bold")

    for block_id in sorted(assignments):
        model  = assignments[block_id]
        is_easy = model == MODEL_EASY
        difficulty_text = Text("easy", style="bold green") if is_easy else Text("hard", style="bold red")
        t.add_row(
            block_id,
            str(counts.get(block_id, "?")),
            difficulty_text,
            model,
        )

    console.print()
    console.print(t)
    console.print(f"\n[dim]Completed in {elapsed:.1f}s[/dim]")
    console.print()


if __name__ == "__main__":
    main()
