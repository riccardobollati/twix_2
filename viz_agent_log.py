#!/usr/bin/env python3
"""Visualize Codex session JSONL logs from the debug/ folder.

Usage:
    python viz_agent_log.py data/<name>/debug/           # interactive picker
    python viz_agent_log.py data/<name>/debug/block_separation.log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import IntPrompt
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

console = Console(highlight=False)

# ── truncation limits ─────────────────────────────────────────────────────────
_HEAD_LINES = 40
_TAIL_LINES = 8


# ── log parsing ───────────────────────────────────────────────────────────────

def _parse_log(path: Path) -> tuple[list[dict], dict | None, str]:
    """Return (events, usage, stderr_text) for a Codex JSONL log file.

    - events:      every parsed JSON object from stdout
    - usage:       dict from the turn.completed event, or None
    - stderr_text: content after the '--- stderr ---' separator, if any
    """
    raw = path.read_text(encoding="utf-8", errors="replace")

    stderr_text = ""
    if "\n--- stderr ---\n" in raw:
        stdout_part, stderr_text = raw.split("\n--- stderr ---\n", 1)
    else:
        stdout_part = raw

    events: list[dict] = []
    usage: dict | None = None
    for line in stdout_part.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        events.append(obj)
        if obj.get("type") == "turn.completed" and "usage" in obj:
            usage = obj["usage"]

    return events, usage, stderr_text


def _count_items(events: list[dict]) -> tuple[int, int]:
    """Return (n_messages, n_commands) from completed item events."""
    msgs = cmds = 0
    for ev in events:
        if ev.get("type") != "item.completed":
            continue
        t = ev.get("item", {}).get("type", "")
        if t == "agent_message":
            msgs += 1
        elif t == "command_execution":
            cmds += 1
    return msgs, cmds


# ── event rendering ───────────────────────────────────────────────────────────

def _truncate(text: str) -> tuple[str, int]:
    """Return (display_text, n_omitted). Folds long output."""
    lines = text.splitlines()
    if len(lines) <= _HEAD_LINES + _TAIL_LINES:
        return text, 0
    omitted = len(lines) - _HEAD_LINES - _TAIL_LINES
    kept = lines[:_HEAD_LINES] + [f"… {omitted:,} lines omitted …"] + lines[-_TAIL_LINES:]
    return "\n".join(kept), omitted


def _strip_shell_wrapper(cmd: str) -> str:
    """Remove /bin/zsh -lc '…' wrapper for cleaner display."""
    for prefix in ("/bin/zsh -lc ", "/bin/bash -lc "):
        if cmd.startswith(prefix):
            inner = cmd[len(prefix):].strip()
            if len(inner) >= 2 and inner[0] in ('"', "'") and inner[-1] == inner[0]:
                inner = inner[1:-1]
            return inner
    return cmd


def _render_agent_message(item: dict, seq: int) -> None:
    text = item.get("text", "").strip()
    if not text:
        return
    console.print(Panel(
        Text(text, overflow="fold"),
        title=f"[bold bright_green]🤖  Assistant[/]  [dim]#{seq}[/]",
        title_align="left",
        border_style="bright_green",
        box=box.ROUNDED,
        padding=(0, 1),
    ))


def _render_command(item: dict, seq: int) -> None:
    raw_cmd = item.get("command", "")
    output  = (item.get("aggregated_output") or "").rstrip()
    code    = item.get("exit_code")
    status  = item.get("status", "")

    success = code == 0
    failed  = (code is not None and code != 0) or status == "failed"

    if success:
        border, badge = "steel_blue1", "[bold green]✓ exit 0[/]"
    elif failed:
        border, badge = "red", f"[bold red]✗ exit {code}[/]"
    else:
        border, badge = "dim", "[dim]in progress[/]"

    cmd_display = _strip_shell_wrapper(raw_cmd)
    parts: list = [Syntax(cmd_display, "bash", theme="monokai", word_wrap=True)]

    if output:
        display_out, omitted = _truncate(output)
        parts.append(Rule(style="dim"))
        parts.append(Text(display_out, style="dim white", overflow="fold"))

    console.print(Panel(
        Group(*parts),
        title=f"[bold {'steel_blue1' if success else 'red' if failed else 'dim'}]$  Shell[/]  {badge}  [dim]#{seq}[/]",
        title_align="left",
        border_style=border,
        box=box.ROUNDED,
        padding=(0, 1),
    ))


def _render_usage(usage: dict) -> None:
    inp    = usage.get("input_tokens",            0)
    cached = usage.get("cached_input_tokens",     0)
    out    = usage.get("output_tokens",           0)
    reason = usage.get("reasoning_output_tokens", 0)
    billed = inp - cached

    cache_pct = f"  {cached / inp * 100:.1f}% cache hit" if inp else ""

    t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    t.add_column(style="dim", width=26)
    t.add_column(style="bold white", justify="right", width=12)
    t.add_column(style="dim italic")

    t.add_row("Input tokens",        f"{inp:,}",    "")
    t.add_row("  ↳ cached",          f"{cached:,}", cache_pct)
    t.add_row("  ↳ billed (new)",    f"{billed:,}", "")
    t.add_row("Output tokens",       f"{out:,}",    "")
    if reason:
        t.add_row("  ↳ reasoning",   f"{reason:,}", "")
    t.add_row("Total tokens",        f"{inp + out:,}", "")

    console.print(Panel(
        t,
        title="[bold magenta]📊  Token Usage[/]",
        border_style="magenta",
        box=box.ROUNDED,
    ))


def render_log(path: Path) -> None:
    events, usage, stderr_text = _parse_log(path)
    msgs, cmds = _count_items(events)

    # ── header ────────────────────────────────────────────────────────────────
    header = Text()
    header.append(path.name, style="bold white")
    header.append(f"  ·  {msgs} messages, {cmds} commands", style="dim")
    if usage:
        inp = usage.get("input_tokens", 0)
        out = usage.get("output_tokens", 0)
        header.append("  ·  ", style="dim")
        header.append(f"{inp:,}↑  {out:,}↓", style="bold magenta")
        header.append(" tokens", style="dim")

    console.print()
    console.print(Panel(
        header,
        box=box.DOUBLE_EDGE,
        border_style="bright_white",
        padding=(0, 2),
    ))
    console.print()

    # ── prompt (saved alongside the log by codex_silent) ──────────────────────
    prompt_path = path.with_suffix(".prompt")
    if prompt_path.exists():
        prompt_text = prompt_path.read_text(encoding="utf-8", errors="replace").strip()
        if prompt_text:
            console.print(Panel(
                Text(prompt_text, overflow="fold"),
                title="[bold yellow]📋  Task Prompt[/]  [dim]#0[/]",
                title_align="left",
                border_style="yellow",
                box=box.ROUNDED,
                padding=(0, 1),
            ))

    # ── events ────────────────────────────────────────────────────────────────
    seq = 0
    for ev in events:
        t = ev.get("type", "")

        # skip boilerplate / in-progress duplicates
        if t in ("thread.started", "turn.started", "turn.completed", "item.started"):
            continue

        if t == "item.completed":
            item      = ev.get("item", {})
            item_type = item.get("type", "")
            seq += 1
            if item_type == "agent_message":
                _render_agent_message(item, seq)
            elif item_type == "command_execution":
                _render_command(item, seq)
            else:
                console.print(Panel(
                    Syntax(json.dumps(item, indent=2), "json", theme="monokai"),
                    title=f"[dim]{item_type}  #{seq}[/]",
                    border_style="dim",
                    box=box.ROUNDED,
                ))

    # ── token summary ─────────────────────────────────────────────────────────
    console.print()
    if usage:
        _render_usage(usage)
    else:
        console.print("[dim]No token usage data found in this log.[/dim]")

    # ── stderr (if any) ───────────────────────────────────────────────────────
    if stderr_text.strip():
        console.print()
        console.print(Panel(
            Text(stderr_text.strip(), style="red", overflow="fold"),
            title="[bold red]stderr[/]",
            border_style="red",
            box=box.ROUNDED,
        ))

    console.print()


# ── file picker ───────────────────────────────────────────────────────────────

def _quick_stats(path: Path) -> tuple[int, int, dict | None]:
    try:
        events, usage, _ = _parse_log(path)
    except Exception:
        return 0, 0, None
    msgs, cmds = _count_items(events)
    return msgs, cmds, usage


def pick_and_render(log_dir: Path) -> None:
    files = sorted(log_dir.glob("*.log"))
    if not files:
        console.print(f"[red]No .log files found in {log_dir}[/red]")
        sys.exit(1)

    t = Table(
        title=f"[bold]Codex Session Logs[/]  [dim]{log_dir}[/]",
        box=box.ROUNDED,
        border_style="bright_white",
        show_lines=True,
    )
    t.add_column("#",        style="dim bold",     width=4,  justify="right")
    t.add_column("File",     style="bold white")
    t.add_column("Messages", justify="right",      style="bright_green")
    t.add_column("Commands", justify="right",      style="steel_blue1")
    t.add_column("Input ↑",  justify="right",      style="magenta")
    t.add_column("Cached",   justify="right",      style="dim magenta")
    t.add_column("Output ↓", justify="right",      style="magenta")

    for i, f in enumerate(files, 1):
        msgs, cmds, usage = _quick_stats(f)
        inp    = f"{usage.get('input_tokens',          0):,}" if usage else "—"
        cached = f"{usage.get('cached_input_tokens',   0):,}" if usage else "—"
        out    = f"{usage.get('output_tokens',         0):,}" if usage else "—"
        t.add_row(str(i), f.name, str(msgs), str(cmds), inp, cached, out)

    console.print()
    console.print(t)
    console.print()

    idx = IntPrompt.ask(
        "Select file",
        choices=[str(i) for i in range(1, len(files) + 1)],
        show_choices=False,
    )
    console.print()
    with console.pager(styles=True):
        render_log(files[idx - 1])


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize Codex session JSONL logs from the debug/ folder."
    )
    parser.add_argument(
        "path",
        help="Path to a .log file or a directory containing .log files.",
    )
    args = parser.parse_args()

    p = Path(args.path)
    if not p.exists():
        console.print(f"[red]Not found: {p}[/red]")
        sys.exit(1)

    if p.is_dir():
        pick_and_render(p)
    elif p.suffix == ".log":
        with console.pager(styles=True):
            render_log(p)
    else:
        console.print(f"[red]Expected a .log file or directory, got: {p}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
