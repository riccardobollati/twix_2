from __future__ import annotations

import os
import tempfile
import time
from typing import Any

from inspect_results.viz_blocks import render_blocks, run_block_separator
from inspect_results.viz_record_separation import render_record_separation


def block_artifacts_node(state: dict[str, Any]) -> dict[str, Any]:
    """Render train and held-out block plus deterministic record overlays.

    Initial script generation may use a reduced build OCR that excludes held-out
    train pages. Reapplying the completed script here ensures the artifacts cover
    every train page while keeping the generated JSON temporary. Record overlays
    reuse that JSON and the inferred template; no additional model or generated
    script is involved.
    """
    t0 = time.time()
    train_dir = os.path.dirname(state["pdf_path"])
    document_dir = os.path.dirname(state["agent_space"])
    script_path = os.path.join(train_dir, "block_separation.py")
    output_dir = os.path.join(document_dir, "artifacts", "blocks")
    record_output_dir = os.path.join(
        document_dir, "artifacts", "record_separation"
    )

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
        record_written = render_record_separation(
            state["pdf_path"],
            blocks_json_path,
            state.get("template") or [],
            record_output_dir,
            label=state.get("doc_name", "document"),
        )
    print(f"block_artifacts_node: {len(written)} train image(s) → {output_dir}")
    print(
        f"block_artifacts_node: {len(record_written)} train record image(s) "
        f"→ {record_output_dir}"
    )

    test_pdf_path = state.get("test_pdf_path")
    if test_pdf_path:
        test_output_dir = os.path.join(document_dir, "artifacts", "test_blocks")
        test_record_output_dir = os.path.join(
            document_dir, "artifacts", "test_record_separation"
        )
        with tempfile.TemporaryDirectory(prefix="test_block_artifacts_") as work_dir:
            test_blocks_json_path = run_block_separator(
                test_pdf_path, script_path, work_dir
            )
            test_written = render_blocks(
                test_pdf_path,
                test_blocks_json_path,
                test_output_dir,
                label=f"{state.get('doc_name', 'document')} · evaluation test",
            )
            test_record_written = render_record_separation(
                test_pdf_path,
                test_blocks_json_path,
                state.get("template") or [],
                test_record_output_dir,
                label=f"{state.get('doc_name', 'document')} · evaluation test",
            )
        print(
            f"block_artifacts_node: {len(test_written)} evaluation test image(s) "
            f"→ {test_output_dir}"
        )
        print(
            f"block_artifacts_node: {len(test_record_written)} evaluation test "
            f"record image(s) → {test_record_output_dir}"
        )

    return {
        "cost": [{
            "phase": "block_artifacts",
            "duration_s": round(time.time() - t0, 3),
        }]
    }
