"""Convert pipeline / ground-truth output to the eval records/nodes format.

Four input shapes are recognised automatically:

  Grouped (ground_truth.json / grouped.json):
    [{"record_id": "0", "data": [{"type": ..., "block_id": ..., "data": ..., "page": 1}, ...]}, ...]

  Ground-truth generator dict (legacy):
    {"doc_name": ..., "pages": [{"page": 1, "blocks": [{...}, ...]}, ...]}

  Legacy ground-truth list:
    [{"page": 1, "blocks": [{"type": ..., "block_id": ..., "data": ...}, ...]}, ...]

  Pipeline flat (extracted.json):
    [{"type": "table"|"kv", "block_id": "0_0001", "data": ..., "page": 1}, ...]

All are converted to the internal eval shape:
  {
    "records": [
      {
        "record_id": "<id>",
        "nodes": [
          {
            "id": "<block_id>",
            "type": "table" | "key_value",
            "content": <converted>,
            "relationship": {"parent_id": null}
          },
          ...
        ]
      },
      ...
    ]
  }

Type name mapping : "kv"  →  "key_value"  (eval uses "key_value")

Content conversions:
  table   : [{"col1": "v1", "col2": "v2"}, ...]
            → {"headers": ["col1","col2"],
               "rows": [[{"key":"col1","value":"v1"}, {"key":"col2","value":"v2"}], ...]}

  key_value: {"field1": "v1", ...}
             → [{"key": "field1", "value": "v1"}, ...]
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


# ── duplicate-key-safe JSON loading ───────────────────────────────────────────

def dedup_object_pairs(pairs: list) -> dict:
    """``object_pairs_hook`` that keeps duplicate JSON keys instead of last-wins.

    Repeated keys are disambiguated deterministically as ``key``, ``key#2``,
    ``key#3``, … so duplicate table columns / kv fields survive ingestion.
    """
    out: dict = {}
    counts: dict = {}
    for k, v in pairs:
        if k not in out:
            out[k] = v
            continue
        n = counts.get(k, 1) + 1
        new_key = f"{k}#{n}"
        while new_key in out:
            n += 1
            new_key = f"{k}#{n}"
        counts[k] = n
        out[new_key] = v
    return out


def loads_json_dedup(text: str):
    """json.loads with duplicate-key disambiguation (apply to gold AND candidate)."""
    return json.loads(text, object_pairs_hook=dedup_object_pairs)


def load_json_dedup(path):
    """Load a JSON file with duplicate-key disambiguation."""
    return loads_json_dedup(Path(path).read_text(encoding="utf-8"))


# ── public entry point ────────────────────────────────────────────────────────

def to_eval_format(data) -> dict:
    """Return *data* converted to the eval records/nodes dict.

    If *data* is already in that shape (has a ``"records"`` key at the root)
    it is returned as-is.
    """
    if isinstance(data, dict) and "records" in data:
        return data
    # Ground-truth generator format: {"pages": [{page, blocks:[...]}, ...], ...}
    if isinstance(data, dict) and "pages" in data:
        return _from_ground_truth(data["pages"] or [])
    if not isinstance(data, list) or not data:
        return {"records": []}
    first = data[0]
    if isinstance(first, dict) and "record_id" in first and "data" in first:
        return _from_grouped(data)
    if isinstance(first, dict) and "blocks" in first:
        return _from_ground_truth(data)
    if isinstance(first, dict) and "block_id" in first:
        return _from_pipeline(data)
    return {"records": []}


# ── format-specific converters ────────────────────────────────────────────────

def _from_grouped(data: list) -> dict:
    """[{record_id, data:[{type, block_id, data, page}]}]  →  {records:[…]}"""
    records = []
    for entry in data:
        record_id = str(entry.get("record_id", ""))
        nodes = [_block_to_node(b) for b in (entry.get("data") or [])]
        records.append({"record_id": record_id, "nodes": nodes})
    return {"records": records}


def _from_ground_truth(data: list) -> dict:
    """[{page, blocks:[{type, block_id, data}]}]  →  {records:[…]}"""
    records = []
    for entry in data:
        page = entry.get("page", 0)
        nodes = [_block_to_node(b) for b in (entry.get("blocks") or [])]
        records.append({"record_id": f"page_{page}", "nodes": nodes})
    return {"records": records}


def _from_pipeline(data: list) -> dict:
    """[{type, block_id, data, page}]  →  {records:[…]}  (grouped by page)"""
    by_page: dict = defaultdict(list)
    for block in data:
        by_page[block.get("page", 0)].append(block)
    records = []
    for page in sorted(by_page):
        nodes = [_block_to_node(b) for b in by_page[page]]
        records.append({"record_id": f"page_{page}", "nodes": nodes})
    return {"records": records}


# ── block → node ──────────────────────────────────────────────────────────────

def _block_to_node(block: dict) -> dict:
    raw_type = block.get("type", "")
    node_type = "key_value" if raw_type == "kv" else raw_type
    data = block.get("data")

    if node_type == "table":
        content = _table_content(data)
    elif node_type == "key_value":
        content = _kv_content(data)
    else:
        content = data

    return {
        "id": block.get("block_id", ""),
        "type": node_type,
        "content": content,
        "relationship": {"parent_id": None},
    }


# ── content converters ────────────────────────────────────────────────────────

def _table_content(data) -> dict:
    """[{"col1": "v1", ...}, ...]  →  {headers:[…], rows:[[{key,value},…],…]}"""
    if not isinstance(data, list) or not data:
        return {"headers": [], "rows": []}
    # Union of keys across ALL rows, first-seen order (row 0 may miss columns
    # or not be a dict at all).
    headers: list = []
    seen: set = set()
    for row in data:
        if not isinstance(row, dict):
            continue
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                headers.append(k)
    rows = [
        [{"key": h, "value": str(row.get(h, ""))} for h in headers]
        for row in data
        if isinstance(row, dict)
    ]
    return {"headers": headers, "rows": rows}


def _kv_content(data) -> list:
    """{"k": "v", ...}  →  [{"key": "k", "value": "v"}, ...]"""
    if isinstance(data, dict):
        return [{"key": str(k), "value": str(v)} for k, v in data.items()]
    if isinstance(data, list):
        out = []
        for item in data:
            if not isinstance(item, dict):
                continue
            if "key" in item and "value" in item:
                out.append({"key": str(item["key"]), "value": str(item["value"])})
            else:
                for k, v in item.items():
                    out.append({"key": str(k), "value": str(v)})
        return out
    return []
