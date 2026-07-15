from __future__ import annotations

import sys
import threading
import time


FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
GREEN_TICK = "\033[32m✓\033[0m"


def spinner(label: str, stop_event: threading.Event, result: dict) -> None:
    """Animate a single-line spinner until stop_event is set.

    When stopped, prints a green tick followed by ``result["tokens"]`` (if the
    caller stored a token count there before setting the event).
    """
    frame = 0
    sys.stdout.write("\n")
    sys.stdout.flush()
    while not stop_event.is_set():
        sys.stdout.write(f"\r\033[K  {label}  {FRAMES[frame % len(FRAMES)]}")
        sys.stdout.flush()
        frame += 1
        time.sleep(0.1)
    tok = result.get("tokens")
    suffix = f"  {GREEN_TICK}  {tok:,} tokens" if tok is not None else f"  {GREEN_TICK}"
    sys.stdout.write(f"\r\033[K  {label}{suffix}\n")
    sys.stdout.flush()
