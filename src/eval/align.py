"""eval.md §3 — node alignment and record alignment via Hungarian assignment."""

from __future__ import annotations

from scipy.optimize import linear_sum_assignment

from .content import content_similarity


SIM_THRESHOLD = 0.4       # §3.2 node-level threshold
RECORD_SIM_THRESHOLD = 0.3  # §3.0 record-level threshold

W_TYPE = 0.30
W_CONTENT = 0.50
W_STRUCT = 0.20


def _build_id_to_idx(nodes: list[dict]) -> dict:
    return {n.get("id"): i for i, n in enumerate(nodes) if n.get("id") is not None}


def _pair_sim(g_node: dict, c_node: dict, structure_term: float) -> tuple[float, float, float, dict]:
    """Return (sim, type_match, content_sim, content_diag)."""
    type_match = 1.0 if g_node.get("type") == c_node.get("type") else 0.0
    content_sim, content_diag = content_similarity(g_node, c_node)
    sim = (
        W_TYPE * type_match
        + W_CONTENT * content_sim
        + W_STRUCT * structure_term
    )
    return sim, type_match, content_sim, content_diag


def _assignment(matrix: list[list[float]]) -> list[tuple[int, int]]:
    if not matrix or not matrix[0]:
        return []
    n_g = len(matrix)
    n_c = len(matrix[0])
    size = max(n_g, n_c)
    cost = [[1.0] * size for _ in range(size)]
    for i in range(n_g):
        for j in range(n_c):
            cost[i][j] = 1.0 - matrix[i][j]
    row_ind, col_ind = linear_sum_assignment(cost)
    out = []
    for i, j in zip(row_ind, col_ind):
        if i < n_g and j < n_c:
            out.append((int(i), int(j)))
    return out


def align(gold_nodes: list[dict], cand_nodes: list[dict]) -> dict:
    """Two-pass Hungarian alignment per §3.

    Returns a dict with:
      pairs:           list[(g_idx, c_idx, sim, type_match, content_sim, content_diag)]
      missing_gold:    list of g_idx
      extra_candidate: list of c_idx
      g_to_c:          dict mapping gold idx -> candidate idx (only for matches above τ)
      c_to_g:          dict mapping candidate idx -> gold idx
      closest_for_missing: dict g_idx -> (best_c_idx, best_sim)
    """
    n_g = len(gold_nodes)
    n_c = len(cand_nodes)

    sim_matrix = [[0.0] * n_c for _ in range(n_g)]
    type_matrix = [[0.0] * n_c for _ in range(n_g)]
    content_matrix = [[0.0] * n_c for _ in range(n_g)]
    diag_matrix = [[None] * n_c for _ in range(n_g)]

    for i in range(n_g):
        for j in range(n_c):
            s, tm, cs, dg = _pair_sim(gold_nodes[i], cand_nodes[j], structure_term=0.0)
            sim_matrix[i][j] = s
            type_matrix[i][j] = tm
            content_matrix[i][j] = cs
            diag_matrix[i][j] = dg

    # Pass 1.
    pairs = _assignment(sim_matrix)

    def filter_pairs(prs):
        kept = [
            (i, j, sim_matrix[i][j], type_matrix[i][j], content_matrix[i][j], diag_matrix[i][j])
            for (i, j) in prs
            if sim_matrix[i][j] >= SIM_THRESHOLD
        ]
        kept.sort(key=lambda t: (
            str(gold_nodes[t[0]].get("id")), str(cand_nodes[t[1]].get("id"))
        ))
        return kept

    kept = filter_pairs(pairs)

    # Pass 2: recompute with structure_term informed by parent matches.
    g_to_c = {i: j for i, j, *_ in kept}
    g_id_to_idx = _build_id_to_idx(gold_nodes)
    c_id_to_idx = _build_id_to_idx(cand_nodes)

    def parent_term(g_idx: int, c_idx: int) -> float:
        g_pid = ((gold_nodes[g_idx].get("relationship") or {}).get("parent_id"))
        c_pid = ((cand_nodes[c_idx].get("relationship") or {}).get("parent_id"))
        if g_pid is None and c_pid is None:
            return 1.0
        if g_pid is None or c_pid is None:
            return 0.0
        gp_idx = g_id_to_idx.get(g_pid)
        cp_idx = c_id_to_idx.get(c_pid)
        if gp_idx is None or cp_idx is None:
            return 0.0
        if g_to_c.get(gp_idx) == cp_idx:
            return 1.0
        return 0.0

    sim_matrix2 = [[0.0] * n_c for _ in range(n_g)]
    for i in range(n_g):
        for j in range(n_c):
            pt = parent_term(i, j)
            sim_matrix2[i][j] = (
                W_TYPE * type_matrix[i][j]
                + W_CONTENT * content_matrix[i][j]
                + W_STRUCT * pt
            )

    pairs2 = _assignment(sim_matrix2)
    kept2 = [
        (i, j, sim_matrix2[i][j], type_matrix[i][j], content_matrix[i][j], diag_matrix[i][j])
        for (i, j) in pairs2
        if sim_matrix2[i][j] >= SIM_THRESHOLD
    ]
    kept2.sort(key=lambda t: (
        str(gold_nodes[t[0]].get("id")), str(cand_nodes[t[1]].get("id"))
    ))

    matched_g = {t[0] for t in kept2}
    matched_c = {t[1] for t in kept2}
    missing_gold = sorted(i for i in range(n_g) if i not in matched_g)
    extra_cand = sorted(j for j in range(n_c) if j not in matched_c)

    g_to_c2 = {t[0]: t[1] for t in kept2}
    c_to_g2 = {t[1]: t[0] for t in kept2}

    closest_for_missing = {}
    for i in missing_gold:
        best_j = -1
        best_s = 0.0
        for j in range(n_c):
            if sim_matrix2[i][j] > best_s:
                best_s = sim_matrix2[i][j]
                best_j = j
        closest_for_missing[i] = (best_j, best_s)

    return {
        "pairs": kept2,
        "missing_gold": missing_gold,
        "extra_candidate": extra_cand,
        "g_to_c": g_to_c2,
        "c_to_g": c_to_g2,
        "closest_for_missing": closest_for_missing,
    }


# ---------------------------------------------------------------------------
# Record-level alignment  (eval.md §3.0)
# ---------------------------------------------------------------------------

def _type_multiset_jaccard(nodes_g: list[dict], nodes_c: list[dict]) -> float:
    from collections import Counter
    cg: dict = Counter(n.get("type") for n in nodes_g)
    cc: dict = Counter(n.get("type") for n in nodes_c)
    types = set(cg) | set(cc)
    intersection = sum(min(cg.get(t, 0), cc.get(t, 0)) for t in types)
    union = sum(max(cg.get(t, 0), cc.get(t, 0)) for t in types)
    return intersection / union if union else 1.0


def _size_similarity(n_g: int, n_c: int) -> float:
    mx = max(n_g, n_c, 1)
    return 1.0 - abs(n_g - n_c) / mx


def _order_prior(i: int, j: int, n_g: int, n_c: int) -> float:
    g_pos = i / max(n_g - 1, 1)
    c_pos = j / max(n_c - 1, 1)
    return 1.0 - abs(g_pos - c_pos)


def _mean_pairwise_node_sim(nodes_g: list[dict], nodes_c: list[dict]) -> float:
    """Mean of best-matching candidate sim for each gold node (structure_sim=0)."""
    if not nodes_g or not nodes_c:
        return 0.0
    sims = []
    for g in nodes_g:
        best = 0.0
        for c in nodes_c:
            type_match = 1.0 if g.get("type") == c.get("type") else 0.0
            cs, _ = content_similarity(g, c)
            s = W_TYPE * type_match + W_CONTENT * cs
            if s > best:
                best = s
        sims.append(best)
    return sum(sims) / len(sims)


def _record_sim(
    g_record: dict,
    c_record: dict,
    g_idx: int,
    c_idx: int,
    n_g_records: int,
    n_c_records: int,
) -> float:
    nodes_g = list(g_record.get("nodes") or [])
    nodes_c = list(c_record.get("nodes") or [])
    s1 = _mean_pairwise_node_sim(nodes_g, nodes_c)
    s2 = _type_multiset_jaccard(nodes_g, nodes_c)
    s3 = _size_similarity(len(nodes_g), len(nodes_c))
    s4 = _order_prior(g_idx, c_idx, n_g_records, n_c_records)
    return 0.50 * s1 + 0.20 * s2 + 0.10 * s3 + 0.20 * s4


def align_records(gold_records: list[dict], cand_records: list[dict]) -> dict:
    """Hungarian alignment at the record level per eval.md §3.0.

    Returns:
      pairs:           list[(g_idx, c_idx, record_sim)]  — matched record pairs above τ_record
      missing_gold:    list of g_idx for unmatched gold records
      extra_candidate: list of c_idx for unmatched candidate records
    """
    n_g = len(gold_records)
    n_c = len(cand_records)

    if n_g == 0 and n_c == 0:
        return {"pairs": [], "missing_gold": [], "extra_candidate": []}

    sim_matrix = [
        [_record_sim(gold_records[i], cand_records[j], i, j, n_g, n_c) for j in range(n_c)]
        for i in range(n_g)
    ]

    raw_pairs = _assignment(sim_matrix)
    kept = [
        (i, j, sim_matrix[i][j])
        for (i, j) in raw_pairs
        if sim_matrix[i][j] >= RECORD_SIM_THRESHOLD
    ]
    # Deterministic tie-break: sort by gold record_id, then candidate record_id.
    kept.sort(key=lambda t: (
        str(gold_records[t[0]].get("record_id", t[0])),
        str(cand_records[t[1]].get("record_id", t[1])),
    ))

    matched_g = {i for i, _, _ in kept}
    matched_c = {j for _, j, _ in kept}
    missing_gold = sorted(i for i in range(n_g) if i not in matched_g)
    extra_cand = sorted(j for j in range(n_c) if j not in matched_c)

    return {
        "pairs": kept,
        "missing_gold": missing_gold,
        "extra_candidate": extra_cand,
    }
