"""eval.md §3.4 — string normalization shared across alignment and scoring."""

from __future__ import annotations

import re
import unicodedata


_HYPHEN_LIKE = {
    "‐": "-",  # hyphen
    "‑": "-",  # non-breaking hyphen
    "‒": "-",  # figure dash
    "–": "-",  # en dash
    "—": "-",  # em dash
    "―": "-",  # horizontal bar
    "−": "-",  # minus sign
}

_QUOTE_LIKE = {
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
}

_NBSP_LIKE = {
    " ": " ",
    " ": " ",
    " ": " ",
    " ": " ",
    " ": " ",
    "​": "",
}


def _fold_chars(s: str) -> str:
    out = []
    for ch in s:
        if ch in _HYPHEN_LIKE:
            out.append(_HYPHEN_LIKE[ch])
        elif ch in _QUOTE_LIKE:
            out.append(_QUOTE_LIKE[ch])
        elif ch in _NBSP_LIKE:
            out.append(_NBSP_LIKE[ch])
        else:
            out.append(ch)
    return "".join(out)


def normalize(s) -> str:
    """Apply eval.md §3.4 normalization for comparison purposes."""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = _fold_chars(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.lower()
    # Strip surrounding formatting punctuation.
    s = s.strip(":.· ")
    s = s.strip()
    return s


_NUM_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")


def _maybe_number(s: str):
    s = s.replace(",", "").replace(" ", "")
    if _NUM_RE.match(s):
        try:
            return float(s)
        except ValueError:
            return None
    return None


def equal_strings(a, b) -> bool:
    """Return True if a and b are equivalent under eval.md §3.4 (incl. numeric tolerance)."""
    na = normalize(a)
    nb = normalize(b)
    if na == nb:
        return True
    fa = _maybe_number(na)
    fb = _maybe_number(nb)
    if fa is not None and fb is not None:
        return abs(fa - fb) <= 1e-6
    return False


def lcs_len(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence between two normalized string lists."""
    if not a or not b:
        return 0
    na = [normalize(x) for x in a]
    nb = [normalize(x) for x in b]
    m, n = len(na), len(nb)
    prev = [0] * (n + 1)
    cur = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if na[i - 1] == nb[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev, cur = cur, prev
        for k in range(n + 1):
            cur[k] = 0
    return prev[n]


def edit_distance_norm(a: str, b: str) -> float:
    """1 - levenshtein(a,b)/max(|a|,|b|), on normalized strings."""
    na = normalize(a)
    nb = normalize(b)
    if not na and not nb:
        return 1.0
    m, n = len(na), len(nb)
    if m == 0 or n == 0:
        return 0.0
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if na[i - 1] == nb[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    dist = prev[n]
    return 1.0 - dist / max(m, n)
