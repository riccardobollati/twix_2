"""eval.md §6 — parent-edge agreement and path agreement."""

from __future__ import annotations

from .normalize import normalize


def _parent_id(node: dict):
    rel = node.get("relationship") or {}
    return rel.get("parent_id")


def _id_to_idx(nodes: list[dict]) -> dict:
    return {n.get("id"): i for i, n in enumerate(nodes) if n.get("id") is not None}


def structure_score(gold_nodes: list[dict], cand_nodes: list[dict], alignment: dict) -> dict:
    """Compute edge_accuracy, path_accuracy, note_agreement_rate, and structure_diffs."""
    g_id_to_idx = _id_to_idx(gold_nodes)
    c_id_to_idx = _id_to_idx(cand_nodes)
    g_to_c = alignment["g_to_c"]

    matched_g_idxs = list(g_to_c.keys())

    edge_correct = 0
    structure_diffs = []

    for g_idx in matched_g_idxs:
        c_idx = g_to_c[g_idx]
        g_node = gold_nodes[g_idx]
        c_node = cand_nodes[c_idx]
        g_pid = _parent_id(g_node)
        c_pid = _parent_id(c_node)

        if g_pid is None and c_pid is None:
            edge_correct += 1
            continue
        if g_pid is None or c_pid is None:
            structure_diffs.append({
                "gold_id": g_node.get("id"),
                "candidate_id": c_node.get("id"),
                "gold_parent": g_pid,
                "candidate_parent": c_pid,
                "reason": "R-PARENT-DIFFERS",
            })
            continue

        gp_idx = g_id_to_idx.get(g_pid)
        cp_idx = c_id_to_idx.get(c_pid)
        if gp_idx is None or cp_idx is None:
            structure_diffs.append({
                "gold_id": g_node.get("id"),
                "candidate_id": c_node.get("id"),
                "gold_parent": g_pid,
                "candidate_parent": c_pid,
                "reason": "R-PARENT-DIFFERS",
            })
            continue
        # Check if these parents are aligned.
        if g_to_c.get(gp_idx) == cp_idx:
            edge_correct += 1
        else:
            structure_diffs.append({
                "gold_id": g_node.get("id"),
                "candidate_id": c_node.get("id"),
                "gold_parent": g_pid,
                "candidate_parent": c_pid,
                "reason": "R-PARENT-DIFFERS",
            })

    if matched_g_idxs:
        edge_accuracy = edge_correct / len(matched_g_idxs)
    else:
        edge_accuracy = 1.0 if not gold_nodes else 0.0

    # Path accuracy.
    def ancestors_g(idx: int) -> set:
        seen = set()
        cur = idx
        while True:
            pid = _parent_id(gold_nodes[cur])
            if pid is None:
                break
            p_idx = g_id_to_idx.get(pid)
            if p_idx is None or p_idx in seen:
                break
            seen.add(p_idx)
            cur = p_idx
        return seen

    def ancestors_c(idx: int) -> set:
        seen = set()
        cur = idx
        while True:
            pid = _parent_id(cand_nodes[cur])
            if pid is None:
                break
            p_idx = c_id_to_idx.get(pid)
            if p_idx is None or p_idx in seen:
                break
            seen.add(p_idx)
            cur = p_idx
        return seen

    pairs_correct = 0
    pairs_total = 0
    for i, gi in enumerate(matched_g_idxs):
        for gj in matched_g_idxs[i + 1:]:
            ci = g_to_c[gi]
            cj = g_to_c[gj]
            g_anc_i = ancestors_g(gi)
            g_anc_j = ancestors_g(gj)
            c_anc_i = ancestors_c(ci)
            c_anc_j = ancestors_c(cj)

            g_rel = "i_in_j" if gj in g_anc_i else ("j_in_i" if gi in g_anc_j else "neither")
            c_rel = "i_in_j" if cj in c_anc_i else ("j_in_i" if ci in c_anc_j else "neither")
            pairs_total += 1
            if g_rel == c_rel:
                pairs_correct += 1
    path_accuracy = (pairs_correct / pairs_total) if pairs_total else 1.0

    # Note agreement rate (diagnostic only).
    note_total = 0
    note_match = 0
    for g_idx in matched_g_idxs:
        c_idx = g_to_c[g_idx]
        g_note = (gold_nodes[g_idx].get("relationship") or {}).get("note", "")
        c_note = (cand_nodes[c_idx].get("relationship") or {}).get("note", "")
        if not g_note and not c_note:
            continue
        note_total += 1
        if normalize(g_note) == normalize(c_note):
            note_match += 1
    note_agreement_rate = (note_match / note_total) if note_total else 1.0

    return {
        "edge_accuracy": edge_accuracy,
        "path_accuracy": path_accuracy,
        "note_agreement_rate": note_agreement_rate,
        "structure_diffs": structure_diffs,
        # Raw counts for cross-record aggregation in score.py
        "_edge_correct": edge_correct,
        "_edge_total": len(matched_g_idxs),
        "_pairs_correct": pairs_correct,
        "_pairs_total": pairs_total,
        "_note_match": note_match,
        "_note_total": note_total,
    }
