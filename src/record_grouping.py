"""Deterministically group document-order blocks into logical records.

The generated block separator already identifies the template node for every
physical block. Record boundaries can therefore be inferred without another
model call: traverse those node ids in document order and start a new record
when the template restarts. Virtual XOR nodes make alternative physical nodes
count as one required template slot.
"""

from __future__ import annotations

from typing import Any


def _node_id(block: dict, known_ids: set[str]) -> str | None:
    """Recover a template node id from either ``N`` or ``N_instance``."""
    raw = str(block.get("block_id", ""))
    if not raw:
        return None
    if raw in known_ids:
        return raw
    prefix = raw.rsplit("_", 1)[0]
    return prefix if prefix in known_ids else None


def _page(block: dict) -> int | None:
    try:
        return int(block.get("page"))
    except (TypeError, ValueError):
        return None


def group_blocks_into_records(
    blocks: list[dict],
    template: list[dict[str, Any]],
) -> list[dict]:
    """Return ``[{record_id, data}]`` from a document-order block stream.

    Boundary rules are deliberately small and deterministic:

    * after every required template slot has been visited, the next non-continuation
      block starts a record;
    * before completion, a second visit to a non-repeatable node starts a
      record (so ``0,1,0,1,2`` becomes ``[0,1]`` and ``[0,1,2]``);
    * ``complexity.repeated`` and nodes with multiple ``bid`` positions may be
      visited repeatedly inside an incomplete record;
    * identical table nodes on consecutive pages may be adjacent physical
      continuation blocks and stay in the same record;
    * a page gap forces a boundary, which keeps sampled-page evaluation honest.

    An XOR slot counts as visited when any included physical node appears.
    Unknown node ids are preserved but do not count toward template completion.
    """
    if not blocks:
        return []

    specs: dict[str, dict] = {}
    ordered_ids: list[str] = []
    member_to_slot: dict[str, str] = {}
    for index, node in enumerate(template):
        node_id = str(node.get("node_id", index))
        if node.get("type") in ("xor-table", "xor-kv"):
            for member in node.get("includes", []):
                member_to_slot[str(member)] = node_id
        else:
            specs[node_id] = node
    for index, node in enumerate(template):
        node_id = str(node.get("node_id", index))
        if node.get("type") in ("xor-table", "xor-kv"):
            ordered_ids.append(node_id)
        elif node_id not in member_to_slot:
            ordered_ids.append(node_id)
    known_ids = set(specs)
    required_slots = set(ordered_ids)

    # Without a usable template there is no principled boundary signal. Keep
    # every block in one lossless record instead of inventing page records.
    if not known_ids:
        return [{"record_id": "0", "data": list(blocks)}]

    repeatable = {
        node_id
        for node_id, node in specs.items()
        if bool((node.get("complexity") or {}).get("repeated"))
        or len(node.get("bid") or []) > 1
    }
    first_node_id = ordered_ids[0]

    records: list[dict] = []
    current: list[dict] = []
    visited: set[str] = set()
    previous_node_id: str | None = None
    previous_page: int | None = None

    def flush() -> None:
        nonlocal current, visited, previous_node_id, previous_page
        if current:
            records.append({"record_id": str(len(records)), "data": current})
        current = []
        visited = set()
        previous_node_id = None
        previous_page = None

    for block in blocks:
        node_id = _node_id(block, known_ids)
        page = _page(block)

        page_gap = (
            current
            and previous_page is not None
            and page is not None
            and page > previous_page + 1
        )
        same_node = node_id is not None and node_id == previous_node_id
        consecutive_page_table = (
            same_node
            and specs[node_id].get("type") == "table"
            and previous_page is not None
            and page == previous_page + 1
        )
        slot_id = member_to_slot.get(node_id, node_id) if node_id is not None else None
        repeated_visit = slot_id is not None and slot_id in visited
        complete = required_slots.issubset(visited)

        starts_new = bool(page_gap)
        if current and not starts_new and node_id is not None:
            if complete:
                # Repetitions at the tail of a completed record remain legal
                # only while they are consecutive. Any template restart closes
                # the completed record.
                starts_new = not (
                    consecutive_page_table
                    or (same_node and node_id in repeatable)
                )
            elif repeated_visit and (node_id not in repeatable or node_id != previous_node_id):
                starts_new = not consecutive_page_table
            elif (
                slot_id == first_node_id
                and first_node_id not in visited
                and bool(visited)
            ):
                # The document may begin with a continuation of a record. The
                # first template node is still a reliable anchor for the next.
                starts_new = True

        if starts_new:
            flush()

        current.append(block)
        if slot_id is not None:
            visited.add(slot_id)
        previous_node_id = node_id
        previous_page = page

    flush()
    return records
