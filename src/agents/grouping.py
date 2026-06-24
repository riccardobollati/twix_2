from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any


def _maybe_merge_single_block(path: str, template: list) -> None:
    """Merge records and/or blocks based on template structure.

    Pass 1 — single-node template only:
      - table: collapse all records + all blocks into 1 record / 1 block (rows concatenated)
      - kv:    collapse all records into 1 record, blocks kept distinct

    Pass 2 — any template:
      Within each record, merge consecutive blocks that share the same bid and are
      of type table (handles a table that spans multiple pages).
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

    # ── pass 2: merge consecutive same-bid table blocks within each record ────
    bid_to_type = {
        str(b): node.get("type")
        for node in template
        for b in (node["bid"] if isinstance(node.get("bid"), list) else [node.get("bid", "")])
    }
    new_records = []
    for record in records:
        blocks     = record.get("data", [])
        new_blocks = []
        i = 0
        while i < len(blocks):
            block = blocks[i]
            bid   = str(block.get("block_id", "")).rsplit("_", 1)[0]
            if bid_to_type.get(bid) == "table":
                group = [block]
                j = i + 1
                while j < len(blocks) and str(blocks[j].get("block_id", "")).rsplit("_", 1)[0] == bid:
                    group.append(blocks[j])
                    j += 1
                if len(group) > 1:
                    first       = group[0]
                    merged_rows = [row for b in group for row in b.get("data", [])]
                    new_blocks.append({**first, "data": merged_rows})
                    print(f"  merged {len(group)} consecutive table block(s) for bid={bid} → {len(merged_rows)} rows")
                    changed = True
                else:
                    new_blocks.append(block)
                i = j
            else:
                new_blocks.append(block)
                i += 1
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
    result = subprocess.run(
        [sys.executable, pipeline_py, pdf_path, "--out-dir", pipeline_dir],
        capture_output=True,
        text=True,
    )
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
