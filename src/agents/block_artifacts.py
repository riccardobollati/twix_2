from __future__ import annotations

import os
import tempfile
import time
from typing import Any

from viz_blocks import render_blocks, run_block_separator


def block_artifacts_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run separation on the full train PDF and render its block overlays.

    Initial script generation may use a reduced build OCR that excludes held-out
    train pages. Reapplying the completed script here ensures the artifacts cover
    every train page while keeping the generated JSON temporary.
    """
    t0 = time.time()
    train_dir = os.path.dirname(state["pdf_path"])
    document_dir = os.path.dirname(state["agent_space"])
    script_path = os.path.join(train_dir, "block_separation.py")
    output_dir = os.path.join(document_dir, "artifacts", "blocks")

    with tempfile.TemporaryDirectory(prefix="block_artifacts_") as work_dir:
        blocks_json_path = run_block_separator(
            state["pdf_path"], script_path, work_dir
        )
        written = render_blocks(
            state["pdf_path"],
            blocks_json_path,
            output_dir,
            label=state.get("doc_name", "document"),
        )
    print(f"block_artifacts_node: {len(written)} image(s) → {output_dir}")

    return {
        "cost": [{
            "phase": "block_artifacts",
            "duration_s": round(time.time() - t0, 3),
        }]
    }
