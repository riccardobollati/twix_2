"""eval.md §5 — per-type content similarity (table / key_value / metadata)."""

from __future__ import annotations

from typing import Iterable

from scipy.optimize import linear_sum_assignment

from .normalize import edit_distance_norm, equal_strings, lcs_len, normalize


def _safe_list(x) -> list:
    return list(x) if isinstance(x, list) else []


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #

def _row_keyvals(row) -> list[tuple[str, str]]:
    out = []
    if not isinstance(row, list):
        return out
    for cell in row:
        if isinstance(cell, dict):
            out.append((str(cell.get("key", "")), str(cell.get("value", ""))))
        else:
            out.append(("", str(cell)))
    return out


def _row_similarity(g_row, c_row) -> float:
    g = _row_keyvals(g_row)
    c = _row_keyvals(c_row)
    if not g and not c:
        return 1.0
    if not g or not c:
        return 0.0
    used = [False] * len(c)
    matches = 0
    for gk, gv in g:
        for j, (ck, cv) in enumerate(c):
            if used[j]:
                continue
            if equal_strings(gk, ck) and equal_strings(gv, cv):
                used[j] = True
                matches += 1
                break
    denom = max(len(g), len(c))
    return matches / denom if denom else 0.0


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def table_score(g_content: dict, c_content: dict) -> tuple[float, dict]:
    """Return (table_sim, diagnostics).

    diagnostics:
      header_score, cell_f1, header_f1, header_lcs_penalty,
      header_diffs (list of (gold, candidate, reason)),
      row_alignment (list of (gold_idx_or_None, candidate_idx_or_None, sim)),
      cell_diffs (list of (g_row_idx, c_row_idx, header, gold_value, candidate_value, reason)),
      missing_rows (list of g_row_idx),
      extra_rows  (list of c_row_idx).
    """
    g_headers = _safe_list((g_content or {}).get("headers"))
    c_headers = _safe_list((c_content or {}).get("headers"))
    g_rows = _safe_list((g_content or {}).get("rows"))
    c_rows = _safe_list((c_content or {}).get("rows"))

    # Header F1.
    g_h_norm = [normalize(h) for h in g_headers]
    c_h_norm = [normalize(h) for h in c_headers]

    g_h_set = set(g_h_norm)
    c_h_set = set(c_h_norm)
    h_tp = len(g_h_set & c_h_set)
    h_fp = len(c_h_set - g_h_set)
    h_fn = len(g_h_set - c_h_set)
    header_f1 = _f1(h_tp, h_fp, h_fn)

    if g_headers or c_headers:
        denom = max(len(g_headers), len(c_headers))
        lcs_penalty = lcs_len(g_headers, c_headers) / denom if denom else 0.0
    else:
        lcs_penalty = 1.0
    header_score = header_f1 * lcs_penalty

    header_diffs = []
    for h in g_headers:
        if normalize(h) not in c_h_set:
            header_diffs.append({"gold": h, "candidate": None, "reason": "R-HEADER-DIFFERS"})
    for h in c_headers:
        if normalize(h) not in g_h_set:
            header_diffs.append({"gold": None, "candidate": h, "reason": "R-HEADER-DIFFERS"})
    if (
        not header_diffs
        and g_h_set == c_h_set
        and g_h_norm != c_h_norm
        and g_headers
        and c_headers
    ):
        header_diffs.append({
            "gold": list(g_headers),
            "candidate": list(c_headers),
            "reason": "R-HEADER-ORDER",
        })

    # Row alignment via Hungarian on row-similarity matrix.
    row_alignment = []
    cell_diffs = []
    missing_rows = []
    extra_rows = []
    cell_tp = cell_fp = cell_fn = 0

    if not g_rows and not c_rows:
        cell_f1 = 1.0
    elif not g_rows:
        cell_f1 = 0.0
        for j, c_row in enumerate(c_rows):
            extra_rows.append(j)
            cell_fp += sum(1 for _ in _row_keyvals(c_row))
    elif not c_rows:
        cell_f1 = 0.0
        for i, g_row in enumerate(g_rows):
            missing_rows.append(i)
            cell_fn += sum(1 for _ in _row_keyvals(g_row))
    else:
        n_g, n_c = len(g_rows), len(c_rows)
        size = max(n_g, n_c)
        # Build padded similarity matrix (cost = -sim).
        cost = [[1.0] * size for _ in range(size)]
        for i in range(n_g):
            for j in range(n_c):
                cost[i][j] = 1.0 - _row_similarity(g_rows[i], c_rows[j])
        try:
            row_ind, col_ind = linear_sum_assignment(cost)
        except Exception:
            row_ind, col_ind = list(range(size)), list(range(size))

        matched_g = set()
        matched_c = set()
        for i, j in zip(row_ind, col_ind):
            if i < n_g and j < n_c:
                sim = 1.0 - cost[i][j]
                if sim > 0.0:
                    row_alignment.append((int(i), int(j), float(sim)))
                    matched_g.add(int(i))
                    matched_c.add(int(j))
        for i in range(n_g):
            if i not in matched_g:
                missing_rows.append(i)
                cell_fn += sum(1 for _ in _row_keyvals(g_rows[i]))
        for j in range(n_c):
            if j not in matched_c:
                extra_rows.append(j)
                cell_fp += sum(1 for _ in _row_keyvals(c_rows[j]))

        for i, j, _ in row_alignment:
            g_kv = _row_keyvals(g_rows[i])
            c_kv = _row_keyvals(c_rows[j])
            used = [False] * len(c_kv)
            for gk, gv in g_kv:
                hit = None
                for k, (ck, cv) in enumerate(c_kv):
                    if used[k]:
                        continue
                    if equal_strings(gk, ck):
                        hit = k
                        break
                if hit is None:
                    cell_fn += 1
                    cell_diffs.append({
                        "g_row": i, "c_row": j, "header": gk,
                        "gold": gv, "candidate": None, "reason": "R-CELL-DIFFERS",
                    })
                else:
                    used[hit] = True
                    ck, cv = c_kv[hit]
                    if equal_strings(gv, cv):
                        cell_tp += 1
                    else:
                        cell_fn += 1  # gold value not satisfied
                        cell_fp += 1  # candidate value spurious
                        cell_diffs.append({
                            "g_row": i, "c_row": j, "header": gk,
                            "gold": gv, "candidate": cv,
                            "reason": _norm_residual_reason(gv, cv),
                        })
            for k, (ck, cv) in enumerate(c_kv):
                if not used[k]:
                    cell_fp += 1
                    cell_diffs.append({
                        "g_row": i, "c_row": j, "header": ck,
                        "gold": None, "candidate": cv, "reason": "R-CELL-DIFFERS",
                    })
        cell_f1 = _f1(cell_tp, cell_fp, cell_fn)

    table_sim = 0.30 * header_score + 0.70 * cell_f1
    diag = {
        "header_score": header_score,
        "header_f1": header_f1,
        "header_lcs_penalty": lcs_penalty,
        "cell_f1": cell_f1,
        "header_diffs": header_diffs,
        "row_alignment": row_alignment,
        "cell_diffs": cell_diffs,
        "missing_rows": missing_rows,
        "extra_rows": extra_rows,
    }
    return table_sim, diag


# --------------------------------------------------------------------------- #
# key_value
# --------------------------------------------------------------------------- #

def _kv_pairs(content) -> list[tuple[str, str]]:
    out = []
    if not isinstance(content, list):
        return out
    for entry in content:
        if isinstance(entry, dict):
            out.append((str(entry.get("key", "")), str(entry.get("value", ""))))
    return out


def kv_score(g_content, c_content) -> tuple[float, dict]:
    g_pairs = _kv_pairs(g_content)
    c_pairs = _kv_pairs(c_content)

    tp = fp = fn = 0
    diffs = []
    used = [False] * len(c_pairs)
    g_keys = [normalize(k) for k, _ in g_pairs]
    c_keys = [normalize(k) for k, _ in c_pairs]

    # Match by key first.
    for i, (gk, gv) in enumerate(g_pairs):
        gkn = g_keys[i]
        hit = None
        for j, (ck, cv) in enumerate(c_pairs):
            if used[j]:
                continue
            if c_keys[j] == gkn:
                hit = j
                break
        if hit is None:
            fn += 1
            diffs.append({
                "field": gk, "gold": gv, "candidate": None,
                "reason": "R-KV-KEY-MISSING",
            })
        else:
            used[hit] = True
            ck, cv = c_pairs[hit]
            if equal_strings(gv, cv):
                tp += 1
            else:
                fn += 1
                fp += 1
                diffs.append({
                    "field": gk, "gold": gv, "candidate": cv,
                    "reason": _norm_residual_reason(gv, cv, fallback="R-KV-VALUE-DIFFERS"),
                })
    for j, (ck, cv) in enumerate(c_pairs):
        if not used[j]:
            fp += 1
            diffs.append({
                "field": ck, "gold": None, "candidate": cv,
                "reason": "R-KV-KEY-EXTRA",
            })

    f1 = _f1(tp, fp, fn)
    g_keyset = set(g_keys)
    c_keyset = set(c_keys)
    key_only_tp = len(g_keyset & c_keyset)
    key_only_fp = len(c_keyset - g_keyset)
    key_only_fn = len(g_keyset - c_keyset)
    key_only_f1 = _f1(key_only_tp, key_only_fp, key_only_fn)
    return f1, {"f1": f1, "key_only_f1": key_only_f1, "diffs": diffs}


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #

def metadata_score(g_content, c_content, ordered: bool = False) -> tuple[float, dict]:
    g_strings = list(g_content) if isinstance(g_content, list) else []
    c_strings = list(c_content) if isinstance(c_content, list) else []
    g_strings = [str(s) for s in g_strings]
    c_strings = [str(s) for s in c_strings]

    if ordered and (g_strings or c_strings):
        sim = edit_distance_norm("\n".join(g_strings), "\n".join(c_strings))
        diffs = []
        if normalize("\n".join(g_strings)) != normalize("\n".join(c_strings)):
            diffs.append({
                "field": "ordered_sequence",
                "gold": g_strings, "candidate": c_strings,
                "reason": "R-METADATA-STRING-MISSING",
            })
        return sim, {"f1": sim, "ordered": True, "diffs": diffs}

    g_norm = [normalize(s) for s in g_strings]
    c_norm = [normalize(s) for s in c_strings]
    used = [False] * len(c_norm)
    tp = fp = fn = 0
    diffs = []
    for i, gn in enumerate(g_norm):
        hit = None
        for j, cn in enumerate(c_norm):
            if not used[j] and gn == cn:
                hit = j
                break
        if hit is None:
            fn += 1
            diffs.append({
                "field": f"item[{i}]",
                "gold": g_strings[i], "candidate": None,
                "reason": "R-METADATA-STRING-MISSING",
            })
        else:
            used[hit] = True
            tp += 1
    for j, cn in enumerate(c_norm):
        if not used[j]:
            fp += 1
            diffs.append({
                "field": f"item[{j}]",
                "gold": None, "candidate": c_strings[j],
                "reason": "R-METADATA-STRING-EXTRA",
            })
    if not g_strings and not c_strings:
        f1 = 1.0
    else:
        f1 = _f1(tp, fp, fn)
    return f1, {"f1": f1, "ordered": False, "diffs": diffs}


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #

def content_similarity(g_node: dict, c_node: dict) -> tuple[float, dict]:
    g_type = g_node.get("type")
    c_type = c_node.get("type")
    if g_type != c_type:
        return 0.0, {"reason": "R-TYPE-MISMATCH", "gold_type": g_type, "candidate_type": c_type}
    if g_type == "table":
        return table_score(g_node.get("content") or {}, c_node.get("content") or {})
    if g_type == "key_value":
        return kv_score(g_node.get("content"), c_node.get("content"))
    if g_type == "metadata":
        return metadata_score(g_node.get("content"), c_node.get("content"))
    return 0.0, {"reason": "R-TYPE-MISMATCH"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _norm_residual_reason(g, c, fallback: str = "R-CELL-DIFFERS") -> str:
    """Decide if a value mismatch is purely §3.4 normalization residual."""
    gs = str(g) if g is not None else ""
    cs = str(c) if c is not None else ""
    if normalize(gs) == normalize(cs) and gs != cs:
        return "R-NORMALIZATION-RESIDUAL"
    return fallback
