"""eval.md §5 — per-type content similarity (table / key_value / metadata)."""

from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher

from .normalize import equal_normalized, equal_strings, lcs_len, normalize


MIN_ROW_SIM = 0.4  # §5.1 minimum row similarity for a row pair to count as aligned
# a pair below MIN_ROW_SIM still aligns when it shares an exact non-empty cell
# (e.g. the same "Period" value) — systematic wrong-value rows then surface as
# per-cell diffs instead of a wall of missing/extra rows
MIN_ROW_SIM_ANCHORED = 0.2


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


def _normalized_rows(rows: list) -> list[tuple[tuple[str, str], ...]]:
    """Normalize each row into a tuple of (norm_key, norm_value) pairs.

    The tuples are hashable, giving an O(1) whole-row exact-equality fast path
    before any pairwise row alignment. Not cached on the rows-list identity:
    callers pass transient copies (see ``_safe_list``), so an ``id(rows)`` key
    both never hits usefully and — once a freed list's address is reused —
    returns a stale, wrong-length result. The per-node-pair
    ``content_similarity`` memoization already avoids recomputation.
    """
    return [
        tuple((normalize(k), normalize(v)) for k, v in _row_keyvals(row))
        for row in rows
    ]


def _row_similarity_norm(g_kv: tuple, c_kv: tuple) -> float:
    """Row similarity over pre-normalized (key, value) tuples."""
    if not g_kv and not c_kv:
        return 1.0
    if not g_kv or not c_kv:
        return 0.0
    if g_kv == c_kv:  # whole-row exact-equality fast path
        return 1.0
    used = [False] * len(c_kv)
    matches = 0
    for gk, gv in g_kv:
        for j, (ck, cv) in enumerate(c_kv):
            if used[j]:
                continue
            if equal_normalized(gk, ck) and equal_normalized(gv, cv):
                used[j] = True
                matches += 1
                break
    denom = max(len(g_kv), len(c_kv))
    return matches / denom if denom else 0.0


def _shares_anchor(g_kv: tuple, c_kv: tuple) -> bool:
    """True when the rows share at least one exact (key, value) cell with a
    non-empty value — evidence they are the same logical row."""
    c_set = {(ck, cv) for ck, cv in c_kv if cv}
    return any(gv and (gk, gv) in c_set for gk, gv in g_kv)


def _dp_align_gap(
    g_norm: list, c_norm: list,
    g_lo: int, g_hi: int, c_lo: int, c_hi: int,
) -> list[tuple[int, int, float]]:
    """Order-preserving alignment of one gap via banded Needleman–Wunsch.

    Maximizes total row similarity with zero gap penalty, O(n × band).
    A pair below MIN_ROW_SIM is never matched (gap instead), so weak overlaps
    surface as missing/extra rows rather than cascading per-cell diffs.
    """
    m = g_hi - g_lo
    n = c_hi - c_lo
    if m <= 0 or n <= 0:
        return []
    band = max(15, max(m, n) // 20) + abs(m - n)
    NEG = float("-inf")

    def bounds(i: int) -> tuple[int, int]:
        center = (i * n) // m
        return max(0, center - band), min(n, center + band)

    los: list[int] = []
    scores: list[list[float]] = []
    backs: list[list[str]] = []
    for i in range(m + 1):
        lo, hi = bounds(i)
        los.append(lo)
        scores.append([NEG] * (hi - lo + 1))
        backs.append([""] * (hi - lo + 1))

    def get(i: int, j: int) -> float:
        lo = los[i]
        row = scores[i]
        if j < lo or j - lo >= len(row):
            return NEG
        return row[j - lo]

    diag_sims: dict[tuple[int, int], float] = {}

    for j in range(los[0], los[0] + len(scores[0])):
        scores[0][j - los[0]] = 0.0
        backs[0][j - los[0]] = "l" if j > 0 else ""
    for i in range(1, m + 1):
        lo = los[i]
        for j in range(lo, lo + len(scores[i])):
            best, move = get(i - 1, j), "u"      # gold row i-1 unmatched
            left = get(i, j - 1)                  # candidate row j-1 unmatched
            if left > best:
                best, move = left, "l"
            if j > 0:
                diag = get(i - 1, j - 1)
                if diag > NEG:
                    sim = _row_similarity_norm(
                        g_norm[g_lo + i - 1], c_norm[c_lo + j - 1]
                    )
                    alignable = sim >= MIN_ROW_SIM or (
                        sim >= MIN_ROW_SIM_ANCHORED
                        and _shares_anchor(g_norm[g_lo + i - 1], c_norm[c_lo + j - 1])
                    )
                    if alignable and diag + sim > best:
                        best, move = diag + sim, "d"
                        diag_sims[(i, j)] = sim
            scores[i][j - lo] = best
            backs[i][j - lo] = move

    # Traceback from (m, n).
    out: list[tuple[int, int, float]] = []
    i, j = m, n
    while i > 0 or j > 0:
        lo = los[i]
        move = backs[i][j - lo] if lo <= j < lo + len(backs[i]) else ""
        if move == "d" and (i, j) in diag_sims:
            out.append((g_lo + i - 1, c_lo + j - 1, diag_sims[(i, j)]))
            i, j = i - 1, j - 1
        elif move == "l" and j > 0:
            j -= 1
        elif i > 0:
            i -= 1
        else:
            j -= 1
    out.reverse()
    return out


def _align_rows(g_norm: list, c_norm: list) -> list[tuple[int, int, float]]:
    """Order-preserving row alignment (eval.md §5.1).

    Exact matches are anchored first with SequenceMatcher on the hashed
    normalized rows; the gaps between anchors are then locally aligned by best
    row similarity (banded Needleman–Wunsch) while preserving order.
    """
    if not g_norm or not c_norm:
        return []
    sm = SequenceMatcher(None, g_norm, c_norm, autojunk=False)
    alignment: list[tuple[int, int, float]] = []
    prev_g = prev_c = 0
    for a, b, size in sm.get_matching_blocks():  # final block is (m, n, 0)
        alignment.extend(_dp_align_gap(g_norm, c_norm, prev_g, a, prev_c, b))
        for k in range(size):
            alignment.append((a + k, b + k, 1.0))
        prev_g, prev_c = a + size, b + size
    return alignment


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0:
        return 0.0
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _wrong_column_target(gk: str, gv: str, cv: str, g_kv: list, c_kv: list):
    """Detect wrong-column extraction for a mismatched aligned cell.

    Returns the *other* gold header whose value the candidate emitted here,
    "" when only the candidate-side check hits (gold value appears under a
    different candidate header), or None for a plain cell diff.
    """
    for gk2, gv2 in g_kv:
        if gk2 != gk and equal_strings(gv2, cv):
            return gk2
    for ck2, cv2 in c_kv:
        if ck2 != gk and equal_strings(gv, cv2):
            return ""
    return None


def table_score(g_content: dict, c_content: dict) -> tuple[float, dict]:
    """Return (table_sim, diagnostics).

    diagnostics:
      header_score, cell_f1, header_f1, header_lcs_penalty,
      header_diffs (list of (gold, candidate, reason)),
      row_alignment (list of (gold_idx_or_None, candidate_idx_or_None, sim)),
      cell_diffs (list of (g_row_idx, c_row_idx, header, gold_value, candidate_value, reason)),
      missing_rows (list of g_row_idx),
      extra_rows  (list of c_row_idx),
      column_swaps (per-column systematic wrong-column summaries).
    """
    g_headers = _safe_list((g_content or {}).get("headers"))
    c_headers = _safe_list((c_content or {}).get("headers"))
    g_rows = _safe_list((g_content or {}).get("rows"))
    c_rows = _safe_list((c_content or {}).get("rows"))

    # Header F1 (multiset — duplicate headers must match in count).
    g_h_norm = [normalize(h) for h in g_headers]
    c_h_norm = [normalize(h) for h in c_headers]

    g_h_ctr = Counter(g_h_norm)
    c_h_ctr = Counter(c_h_norm)
    h_tp = sum((g_h_ctr & c_h_ctr).values())
    h_fp = sum((c_h_ctr - g_h_ctr).values())
    h_fn = sum((g_h_ctr - c_h_ctr).values())
    if not g_headers and not c_headers:
        header_f1 = 1.0  # two header-less tables agree (mirror the empty-rows case)
    else:
        header_f1 = _f1(h_tp, h_fp, h_fn)

    if g_headers or c_headers:
        denom = max(len(g_headers), len(c_headers))
        lcs_penalty = lcs_len(g_headers, c_headers) / denom if denom else 0.0
    else:
        lcs_penalty = 1.0
    header_score = header_f1 * lcs_penalty

    header_diffs = []
    for h in g_headers:
        if normalize(h) not in c_h_ctr:
            header_diffs.append({"gold": h, "candidate": None, "reason": "R-HEADER-DIFFERS"})
    for h in c_headers:
        if normalize(h) not in g_h_ctr:
            header_diffs.append({"gold": None, "candidate": h, "reason": "R-HEADER-DIFFERS"})
    if (
        not header_diffs
        and g_h_ctr == c_h_ctr
        and g_h_norm != c_h_norm
        and g_headers
        and c_headers
    ):
        header_diffs.append({
            "gold": list(g_headers),
            "candidate": list(c_headers),
            "reason": "R-HEADER-ORDER",
        })

    # Order-preserving row alignment (anchored + banded NW, min sim MIN_ROW_SIM).
    row_alignment = []
    cell_diffs = []
    missing_rows = []
    extra_rows = []
    column_swaps: list[dict] = []
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
        g_norm_rows = _normalized_rows(g_rows)
        c_norm_rows = _normalized_rows(c_rows)
        row_alignment = _align_rows(g_norm_rows, c_norm_rows)
        matched_g = {i for i, _, _ in row_alignment}
        matched_c = {j for _, j, _ in row_alignment}
        for i in range(n_g):
            if i not in matched_g:
                missing_rows.append(i)
                cell_fn += sum(1 for _ in _row_keyvals(g_rows[i]))
        for j in range(n_c):
            if j not in matched_c:
                extra_rows.append(j)
                cell_fp += sum(1 for _ in _row_keyvals(c_rows[j]))

        # Per-column bookkeeping for systematic wrong-column detection.
        col_mismatches: Counter = Counter()
        col_swap_targets: dict = defaultdict(Counter)

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
                        reason = _norm_residual_reason(gv, cv)
                        if reason == "R-CELL-DIFFERS":
                            target = _wrong_column_target(gk, gv, cv, g_kv, c_kv)
                            if target is not None:
                                reason = "R-CELL-WRONG-COLUMN"
                                if target:
                                    col_swap_targets[gk][target] += 1
                        col_mismatches[gk] += 1
                        cell_diffs.append({
                            "g_row": i, "c_row": j, "header": gk,
                            "gold": gv, "candidate": cv,
                            "reason": reason,
                        })
            for k, (ck, cv) in enumerate(c_kv):
                if not used[k]:
                    cell_fp += 1
                    cell_diffs.append({
                        "g_row": i, "c_row": j, "header": ck,
                        "gold": None, "candidate": cv, "reason": "R-CELL-DIFFERS",
                    })
        cell_f1 = _f1(cell_tp, cell_fp, cell_fn)

        # If most of a column's mismatches match a single other gold column,
        # flag the systematic swap (per-column summary).
        for hdr, targets in col_swap_targets.items():
            other, cnt = targets.most_common(1)[0]
            if cnt * 2 > col_mismatches[hdr]:
                column_swaps.append({
                    "column": hdr,
                    "matches_column": other,
                    "count": cnt,
                    "mismatches": col_mismatches[hdr],
                })

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
        "column_swaps": column_swaps,
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

def metadata_score(g_content, c_content) -> tuple[float, dict]:
    g_strings = list(g_content) if isinstance(g_content, list) else []
    c_strings = list(c_content) if isinstance(c_content, list) else []
    g_strings = [str(s) for s in g_strings]
    c_strings = [str(s) for s in c_strings]

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
    return f1, {"f1": f1, "diffs": diffs}


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #

# Memoized per (gold_node, candidate_node) identity — align_records and align()
# revisit the same node pairs. Cleared at the start of each score() run. Keying
# on id() is safe here (unlike the row lists) because the node dicts are held in
# gold_records/cand_records for the whole run, so their ids can't be recycled.
_content_sim_cache: dict[tuple[int, int], tuple[float, dict]] = {}


def clear_content_caches() -> None:
    """Reset the per-run memoization caches (called once per score())."""
    _content_sim_cache.clear()


def content_similarity(g_node: dict, c_node: dict) -> tuple[float, dict]:
    key = (id(g_node), id(c_node))
    hit = _content_sim_cache.get(key)
    if hit is not None:
        return hit
    result = _content_similarity(g_node, c_node)
    _content_sim_cache[key] = result
    return result


def _content_similarity(g_node: dict, c_node: dict) -> tuple[float, dict]:
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


def node_content_size(node: dict) -> int:
    """Content size for ContentScore weighting: cells for tables, pairs for kv, 1 otherwise."""
    t = node.get("type")
    content = node.get("content")
    if t == "table":
        rows = (content or {}).get("rows") if isinstance(content, dict) else None
        return max(1, sum(len(r) for r in (rows or []) if isinstance(r, list)))
    if t == "key_value":
        return max(1, len(content) if isinstance(content, list) else 0)
    return 1


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
