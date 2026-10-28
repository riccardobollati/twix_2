#!/usr/bin/env python3
"""Adapt the frozen baseline JSON to TWIX canonical records/nodes.

The adapter scores the eight fields printed in the source table.  It is
deterministic and uses only the supplied raw JSON.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


HEADERS = [
    "Identification",
    "Name",
    "Lateral limits",
    "Upper limit",
    "Lower limit",
    "Type of activity",
    "Remarks",
    "Validity",
]


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _printed_page(page: dict) -> int:
    label = _text(page.get("page_label"))
    match = re.search(r"-(\d+)\s*$", label)
    if match:
        return int(match.group(1))
    return int(page.get("page_number", 0))


def _row(area: dict) -> dict[str, str]:
    lateral = area.get("lateral_limits") or {}
    upper = area.get("upper_limit") or {}
    lower = area.get("lower_limit") or {}
    activity = area.get("type_of_activity") or {}
    remarks = area.get("remarks") or {}
    return {
        "Identification": _text(area.get("identification")),
        "Name": _text(area.get("name")),
        "Lateral limits": _text(lateral.get("text")),
        "Upper limit": _text(upper.get("text")),
        "Lower limit": _text(lower.get("text")),
        "Type of activity": _text(activity.get("text")),
        "Remarks": _text(remarks.get("text")),
        "Validity": _text(area.get("validity")),
    }


def adapt(raw: dict) -> dict:
    """Return canonical TWIX scorer records, grouped by source physical page."""
    page_map = {
        int(page.get("page_number", 0)): _printed_page(page)
        for page in (raw.get("pages") or [])
        if isinstance(page, dict)
    }
    by_page: dict[int, list[dict[str, str]]] = defaultdict(list)
    for area in raw.get("areas") or []:
        if not isinstance(area, dict):
            continue
        local_pages = area.get("pages") or []
        local_page = int(local_pages[0]) if local_pages else 0
        source_page = page_map.get(local_page, local_page)
        by_page[source_page].append(_row(area))

    records = []
    for page in sorted(by_page):
        rows = [
            [{"key": header, "value": row[header]} for header in HEADERS]
            for row in by_page[page]
        ]
        records.append({
            "record_id": f"page_{page}",
            "nodes": [{
                "id": f"areas_page_{page}",
                "type": "table",
                "content": {"headers": HEADERS, "rows": rows},
                "relationship": {"parent_id": None},
            }],
        })
    return {"records": records}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_json", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    raw = json.loads(args.raw_json.read_text(encoding="utf-8"))
    result = adapt(raw)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
