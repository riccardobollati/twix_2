from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Any

from inspect_results.viz_blocks import render_blocks
from inspect_results.viz_record_separation import render_record_separation


PIPELINE_TIMEOUT_S = 300


def extraction_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run the assembled pipeline on the test PDF, or train PDF without eval.

    The standalone runner keeps flat physical blocks in ``extracted.json`` and
    deterministically groups their template-node sequence into ``grouped.json``.
    """
    document_dir = os.path.dirname(state["agent_space"])
    pipeline_dir = os.path.join(document_dir, "pipeline")
    pipeline_py = os.path.join(pipeline_dir, "pipeline.py")
    pdf_path = state.get("test_pdf_path") or state["pdf_path"]

    phase = "evaluation test PDF" if state.get("test_pdf_path") else "train PDF"
    print(f"extraction_node: running the finished pipeline on the {phase}...", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, pipeline_py, pdf_path, "--out-dir", pipeline_dir],
            capture_output=True,
            text=True,
            timeout=PIPELINE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"pipeline.py timed out after {PIPELINE_TIMEOUT_S}s")
    duration = time.time() - t0

    if result.returncode != 0:
        raise RuntimeError(
            f"pipeline.py failed (exit {result.returncode}):\n{result.stderr}"
        )
    if result.stdout:
        print(result.stdout, end="")

    extracted_path = os.path.join(pipeline_dir, "extracted.json")
    print(f"extraction_node: extracted.json → {extracted_path}")
    print(f"extraction_node: grouped.json → {os.path.join(pipeline_dir, 'grouped.json')}")
    if state.get("test_pdf_path"):
        # Earlier previews use the separator before train repair. Replace them
        # with overlays from the final validation run's own physical blocks.
        blocks_path = os.path.join(pipeline_dir, "block_separation.json")
        label = f"{state.get('doc_name', 'document')} · evaluation test"
        render_blocks(
            pdf_path, blocks_path,
            os.path.join(document_dir, "artifacts", "test_blocks"),
            label=label,
        )
        render_record_separation(
            pdf_path, blocks_path, state.get("template") or [],
            os.path.join(document_dir, "artifacts", "test_record_separation"),
            label=label,
        )
    return {"cost": [{"phase": "extraction", "duration_s": round(duration, 3)}]}
