from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any


PIPELINE_TIMEOUT_S = 300  # hard cap for one pipeline.py run


def _maybe_merge_single_block(path: str, template: list) -> None:
    """Merge records and/or blocks based on template structure.

    Pass 1 — single-node template only:
      - table: collapse all records + all blocks into 1 record / 1 block (rows concatenated)
      - kv:    collapse all records into 1 record, blocks kept distinct

    Pass 2 — any template:
      Within each record, merge ALL blocks that share the same node and are of
      type table into one block (rows concatenated in block order). Page-local
      separation emits one continuation block per page; when a record's tables
      interleave across pages (6,7,8 / 6,7,8 …) those continuations are never
      adjacent, so a consecutive-only merge leaves them apart and the eval
      counts every continuation as an extra node — which rewards scripts that
      glob whole documents into one cross-page mega-block instead.
    """
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if not records:
        return

    changed = False

    # ── pass 1: single-node collapse ─────────────────────────────────────────
    if len(template) == 1:
        n_records  = len(records)
        all_blocks = [b for r in records for b in r.get("data", [])]
        node_type  = template[0].get("type")

        if node_type == "table" and len(all_blocks) > 1:
            first       = all_blocks[0]
            merged_rows = [row for b in all_blocks for row in b.get("data", [])]
            records     = [{"record_id": "0", "data": [
                {"type": first["type"], "block_id": first["block_id"], "data": merged_rows}
            ]}]
            print(f"  merged {n_records} record(s), {len(all_blocks)} block(s) → 1 record, 1 block ({len(merged_rows)} row(s))")
            changed = True
        elif node_type != "table" and n_records > 1:
            records = [{"record_id": "0", "data": all_blocks}]
            print(f"  merged {n_records} record(s) → 1 record, {len(all_blocks)} block(s)")
            changed = True

    # ── pass 2: merge same-node table blocks within each record ───────────────
    # block_id follows "<node_id>_<instance>", so the prefix is a node_id.
    # Blocks arrive in page order, so concatenating in block order preserves
    # row order across the record's pages.
    node_type_by_id = {
        str(node.get("node_id", i)): node.get("type")
        for i, node in enumerate(template)
    }
    new_records = []
    for record in records:
        new_blocks: list[dict] = []
        merged_at: dict[str, int] = {}  # node_id → index of its block in new_blocks
        n_merged: dict[str, int] = {}
        for block in record.get("data", []):
            node_id = str(block.get("block_id", "")).rsplit("_", 1)[0]
            if node_type_by_id.get(node_id) != "table":
                new_blocks.append(block)
                continue
            if node_id not in merged_at:
                merged_at[node_id] = len(new_blocks)
                new_blocks.append(block)
                continue
            target = new_blocks[merged_at[node_id]]
            rows_a = target.get("data")
            rows_b = block.get("data")
            rows_a = rows_a if isinstance(rows_a, list) else []
            rows_b = rows_b if isinstance(rows_b, list) else []
            new_blocks[merged_at[node_id]] = {**target, "data": rows_a + rows_b}
            n_merged[node_id] = n_merged.get(node_id, 1) + 1
            changed = True
        for node_id, count in n_merged.items():
            total_rows = len(new_blocks[merged_at[node_id]].get("data", []))
            print(f"  merged {count} table block(s) for node_id={node_id} → {total_rows} rows")
        new_records.append({**record, "data": new_blocks})

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(new_records, f, indent=2, ensure_ascii=False)


def grouping_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run the assembled pipeline.py on the test PDF (or train if no test) to produce grouped.json.

    Reads:
        state["agent_space"]    – used to derive document_dir/pipeline/
        state["test_pdf_path"]  – test PDF (used when eval_mode=True)
        state["pdf_path"]       – train PDF fallback (used when eval_mode=False)

    Side-effect:
        Runs pipeline.py, which writes record_separation.json, block_separation.json,
        extracted.json and grouped.json into document_dir/pipeline/.
    """
    document_dir = os.path.dirname(state["agent_space"])
    pipeline_dir = os.path.join(document_dir, "pipeline")
    pipeline_py  = os.path.join(pipeline_dir, "pipeline.py")
    pdf_path     = state.get("test_pdf_path") or state["pdf_path"]

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

    grouped_path = os.path.join(pipeline_dir, "grouped.json")
    print(f"grouping_node: grouped.json → {grouped_path}")

    return {"cost": [{"phase": "grouping", "duration_s": round(duration, 3)}]}


def merge_records_node(state: dict[str, Any]) -> dict[str, Any]:
    """Merge all records into one when the template has exactly one node.

    Reads/writes document_dir/pipeline/grouped.json in-place.
    """
    document_dir = os.path.dirname(state["agent_space"])
    pipeline_dir = os.path.join(document_dir, "pipeline")
    grouped_path = os.path.join(pipeline_dir, "grouped.json")

    _maybe_merge_single_block(grouped_path, state.get("template", []))
    return {}
