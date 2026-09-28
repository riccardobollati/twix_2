"""Experimental, deterministic discovery of recurring PDF headers and KV groups.

Run from the repository root:
    env/bin/python test/scan_structure.py id_05 new_04

No template, generated separator, model call, or document-specific vocabulary is
used. This finds structural *candidates*, not semantic schemas or table bounds.
See test/README.md for the rules, output interpretation, and limitations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from statistics import median
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
MAX_SCAN_PAGES = 500


@dataclass(frozen=True)
class Rules:
    min_pages: int = 2
    gap_factor: float = 2.4  # relative to ordinary inter-word spacing
    row_tolerance: float = 0.48  # relative to median word height
    header_stability: float = 0.65
    family_overlap: float = 0.72
    geometry_tolerance: float = 0.075  # relative to header/group width
    body_alignment: float = 0.5
    body_lookahead: int = 12
    header_gap: float = 2.4  # in word heights, measured between text boxes
    kv_gap: float = 5.0  # in word heights, for vertically stacked pairs


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.translate(str.maketrans({"–": "-", "—": "-", "’": "'"}))
    return " ".join(text.split()).strip(" :;.,")


def bbox(items: list[dict]) -> list[float]:
    return [min(w["x0"] for w in items), min(w["top"] for w in items),
            max(w["x1"] for w in items), max(w["bottom"] for w in items)]


def text_like(text: str) -> bool:
    chars = [c for c in text if c.isalnum()]
    return bool(chars) and sum(c.isalpha() for c in chars) / len(chars) >= 0.7


def phrase(words: list[dict]) -> dict:
    x0, top, x1, bottom = bbox(words)
    leaves = [{k: w[k] for k in ("text", "x0", "top", "x1", "bottom")}
              for item in words for w in item.get("words", [item])]
    return dict(text=" ".join(w["text"] for w in words), x0=x0, top=top,
                x1=x1, bottom=bottom, words=leaves)


def make_rows(words: list[dict], rules: Rules) -> list[dict]:
    """Sweep in y, then split each line at gaps larger than ordinary spaces."""
    if not words:
        return []
    height = median(w["bottom"] - w["top"] for w in words)
    lines: list[list[dict]] = []
    center = 0.0
    for word in sorted(words, key=lambda w: ((w["top"] + w["bottom"]) / 2, w["x0"])):
        cy = (word["top"] + word["bottom"]) / 2
        if not lines or cy - center > rules.row_tolerance * height:
            lines.append([word])
            center = cy
        else:
            lines[-1].append(word)
            center += (cy - center) / len(lines[-1])
    result = []
    for line in lines:
        line.sort(key=lambda w: w["x0"])
        h = median(w["bottom"] - w["top"] for w in line)
        char_width = median((w["x1"] - w["x0"]) / max(1, len(w["text"])) for w in line)
        gaps = [b["x0"] - a["x1"] for a, b in zip(line, line[1:])]
        ordinary = [g for g in gaps if 0 < g < 0.6 * h]
        space = median(sorted(ordinary)[:max(1, len(ordinary) // 2)]) if ordinary else char_width * 0.5
        threshold = max(0.6 * h, 1.35 * char_width, rules.gap_factor * space)
        pieces: list[list[dict]] = []
        for word in line:
            if (not pieces or word["x0"] - pieces[-1][-1]["x1"] > threshold
                    or pieces[-1][-1]["text"].endswith(":")):
                pieces.append([word])
            else:
                pieces[-1].append(word)
        chunks = [phrase(part) for part in pieces]
        result.append({**phrase(line), "chunks": chunks, "height": h,
                       "gap_threshold": threshold})
    return result


def split_regions(words: list[dict], width: float) -> list[list[dict]]:
    """Split only large, entirely empty vertical gutters (e.g. book spreads)."""
    if not words:
        return []
    height = median(w["bottom"] - w["top"] for w in words)
    intervals = sorted((w["x0"], w["x1"]) for w in words)
    end = intervals[0][1]
    empty_gaps = []
    for left, right in intervals[1:]:
        if left > end:
            empty_gaps.append((end, left))
        end = max(end, right)
    ordinary_gaps = sorted(right - left for left, right in empty_gaps if right - left > height)
    baseline = median(ordinary_gaps[:max(1, len(ordinary_gaps) // 2)]) if len(ordinary_gaps) >= 2 else height
    cuts = []
    for left, right in empty_gaps:
        if right - left > max(4 * height, 0.04 * width, 2.5 * baseline):
            cut = (left + right) / 2
            # Do not split off an isolated marginal note or a page number.
            lhs = [w for w in words if w["x1"] <= cut]
            rhs = [w for w in words if w["x0"] >= cut]
            if len(lhs) >= 6 and len(rhs) >= 6:
                if bbox(lhs)[2] - bbox(lhs)[0] > width * 0.18 and bbox(rhs)[2] - bbox(rhs)[0] > width * 0.18:
                    cuts.append(cut)
    groups = [[] for _ in range(len(cuts) + 1)]
    for word in words:
        groups[sum(word["x0"] >= cut for cut in cuts)].append(word)
    return [g for g in groups if g]


def load_pdf(path: Path, start: int = 1, end: int | None = None) -> tuple[list[dict], int]:
    import pdfplumber

    pages = []
    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        if start < 1 or start > total or (end is not None and (end < start or end > total)):
            raise ValueError(f"page range must be inside 1..{total}")
        requested_end = end or total
        selected_end = min(requested_end, start + MAX_SCAN_PAGES - 1)
        if selected_end < requested_end:
            print(f"  Limiting scan to {MAX_SCAN_PAGES} pages ({start}..{selected_end} of {total})", flush=True)
        for number in range(start, selected_end + 1):
            page = pdf.pages[number - 1]
            words = [{k: w[k] for k in ("text", "x0", "top", "x1", "bottom")}
                     for w in page.dedupe_chars().extract_words()]
            # Ignore invalid boxes; report pages with no usable text.
            words = [w for w in words if w["text"].strip() and w["x1"] > w["x0"] and w["bottom"] > w["top"]]
            pages.append(dict(page=number, width=page.width, height=page.height, words=words))
            page.close()
            if number == start or number % 5 == 0 or number == selected_end:
                print(f"  Read page {number}/{total}", flush=True)
    return pages, total


def prepare_pages(pages: list[dict], rules: Rules) -> tuple[list[dict], dict[str, set[int]]]:
    prepared = []
    support: dict[str, set[int]] = defaultdict(set)
    for page in pages:
        regions = []
        for region_words in split_regions(page["words"], page["width"]):
            rows = make_rows(region_words, rules)
            regions.append(rows)
            for row in rows:
                for chunk in row["chunks"]:
                    support[normalize(chunk["text"])].add(page["page"])
        prepared.append({k: v for k, v in page.items() if k != "words"} | {
            "regions": regions, "word_count": len(page["words"])})
    return prepared, support


def is_header_line(row: dict, support: dict, rules: Rules) -> bool:
    chunks = row["chunks"]
    if any(":" in c["text"] for c in chunks):
        return False
    # Repeated categorical values can dominate a data row's text. Dates,
    # amounts and identifiers are stronger evidence that it is a body row.
    def label_like(text: str) -> bool:
        # Compact letter+exponent units (m2, cm4) are common header fragments.
        return text_like(text) or bool(re.fullmatch(r"[^\W\d_]{1,3}\d", normalize(text)))

    numeric = sum(bool(re.search(r"\d", c["text"])) and not label_like(c["text"])
                  for c in chunks)
    if numeric / len(chunks) >= 0.2:
        return False
    stable = [c for c in chunks if len(support.get(normalize(c["text"]), ())) >= rules.min_pages]
    return ((text_like(row["text"]) or sum(label_like(c["text"]) for c in chunks) / len(chunks) >= 0.7)
            and len(stable) / len(chunks) >= rules.header_stability)


def aligned_body(header: list[dict], following: list[dict], rules: Rules) -> list[dict]:
    """Require data under multiple header anchors; allow left/centre/right alignment."""
    anchor_row = max(header, key=lambda r: len(r["chunks"]))
    anchors = anchor_row["chunks"]
    if len(anchors) < 2:
        return []
    x0, _, x1, bottom = bbox(header)
    height = median(r["height"] for r in header)
    result = []
    for row in following[:rules.body_lookahead]:
        if row["top"] - bottom > height * 4:
            break
        if sum(c["text"].endswith(":") for c in row["chunks"]) >= 2:
            break
        cells = [c for c in row["chunks"] if c["x1"] >= x0 and c["x0"] <= x1]
        possible = []
        for a, anchor in enumerate(anchors):
            ax = (anchor["x0"] + anchor["x1"]) / 2
            for b, cell in enumerate(cells):
                distance = min(abs(cell["x0"] - anchor["x0"]), abs(cell["x1"] - anchor["x1"]),
                               abs((cell["x0"] + cell["x1"]) / 2 - ax))
                if distance <= height * 1.5:
                    possible.append((distance, a, b))
        used_anchors, used_cells = set(), set()
        for _, a, b in sorted(possible):
            if a not in used_anchors and b not in used_cells:
                used_anchors.add(a)
                used_cells.add(b)
        hits = len(used_anchors)
        if len(cells) >= 2 and hits >= max(2, math.ceil(len(anchors) * rules.body_alignment)):
            result.append({"bbox": bbox([row]), "matched_columns": hits,
                           "column_count": len(anchors), "text": row["text"]})
        # A comment cell can begin above the other cells of the first data row.
        # Look past it, but only when it sits under the last header column.
        elif (not result and len(row["chunks"]) == 1 and len(cells) == 1
              and abs(cells[0]["x0"] - anchors[-1]["x0"]) <= height * 1.5):
            bottom = row["bottom"]
            continue
        # An unrelated region is a boundary, not evidence farther down.
        elif not result:
            break
        bottom = row["bottom"]
    return result


def table_candidates(page: dict, support: dict, rules: Rules, kv_groups: list[dict] | None = None) -> list[dict]:
    candidates = []
    for region_id, rows in enumerate(page["regions"]):
        def eligible(row: dict) -> bool:
            return (is_header_line(row, support, rules)
                    and not any(g["region"] == region_id and g["bbox"][1] <= row["top"] <= g["bbox"][3]
                                for g in (kv_groups or [])))

        def coherent(a: dict, b: dict) -> bool:
            return (max(a["height"], b["height"]) / min(a["height"], b["height"]) <= 1.5
                    and b["top"] - a["bottom"] <= rules.header_gap * min(a["height"], b["height"]))

        def header_line(row: dict, previous: list[dict]) -> bool:
            # A full header line has labels in multiple horizontal positions.
            # Allow a short, aligned fragment of one wrapped column label.
            informative = [c for c in row["chunks"] if len(normalize(c["text"])) >= 2]
            if len(informative) >= 2:
                return True
            if len(row["chunks"]) != 1 or not previous:
                return False
            fragment = row["chunks"][0]
            if len(fragment["text"]) > 35 or len(fragment["text"].split()) > 3:
                return False
            return any(abs(fragment["x0"] - c["x0"]) <= row["height"] * 1.5
                       for earlier in previous for c in earlier["chunks"])

        i = 0
        while i < len(rows):
            if not header_line(rows[i], []) or not eligible(rows[i]):
                i += 1
                continue
            start = i
            # Begin at spatial evidence on multiple columns. Pulling preceding
            # isolated text upward can absorb a wrapped value from an earlier table.
            stop = i + 1
            while stop < len(rows) and eligible(rows[stop]):
                if not coherent(rows[stop - 1], rows[stop]):
                    break
                if not header_line(rows[stop], rows[start:stop]):
                    break
                stop += 1
            header = rows[start:stop]
            body = aligned_body(header, rows[stop:], rules)
            if body:
                labels = []
                for line, row in enumerate(header, 1):
                    for chunk in row["chunks"]:
                        labels.append({**chunk, "line": line, "key": normalize(chunk["text"])})
                candidates.append(dict(kind="table", page=page["page"], region=region_id,
                                       bbox=bbox(header), labels=labels, body_evidence=body,
                                       reason="repeated spaced header band with aligned data below"))
            i = stop
    return candidates


def kv_pairs(row: dict, support: dict, rules: Rules) -> list[dict]:
    chunks = row["chunks"]
    colon_indices = [i for i, c in enumerate(chunks) if c["text"].endswith(":")]
    pairs = []
    if len(colon_indices) >= 2:
        for n, index in enumerate(colon_indices):
            chunk = chunks[index]
            key, _, inline = chunk["text"].partition(":")
            if not text_like(key) or len(key) > 60:
                continue
            stop = colon_indices[n + 1] if n + 1 < len(colon_indices) else len(chunks)
            value = " ".join([inline.strip()] + [c["text"] for c in chunks[index + 1:stop]]).strip()
            pairs.append({**chunk, "text": key.strip(), "key": normalize(key), "value": value,
                          "value_x": chunks[index + 1]["x0"] if stop > index + 1 else None,
                          "value_bbox": bbox(chunks[index + 1:stop]) if stop > index + 1 else None})
        return pairs
    # A single vertical key/value row must have a genuine spatial gap. Extra
    # fragments far to the right can be unrelated diagram labels.
    if len(chunks) < 2:
        return []
    key, value = chunks[:2]
    if ":" in key["text"].rstrip(":") or re.search(r"\d", key["text"]):
        return []
    if len(chunks) > 2 and chunks[2]["x0"] - value["x1"] <= 5 * row["height"]:
        # A colon inside a value ("Range   UB: 100 to 300") can split it into
        # several phrases. Preserve that value without treating it as a new key.
        if value["text"].endswith(":") and chunks[2]["x0"] - value["x1"] <= 1.5 * row["height"]:
            value = phrase(chunks[1:])
        else:
            return []
    if (text_like(key["text"]) and 2 <= len(normalize(key["text"])) <= 60
            and len(key["text"].split()) <= 6
            and len(support.get(normalize(key["text"]), ())) >= rules.min_pages):
        pairs.append({**key, "text": key["text"].rstrip(":"), "key": normalize(key["text"]),
                      "value": value["text"], "value_x": value["x0"], "value_bbox": bbox([value])})
    return pairs


def kv_candidates(page: dict, support: dict, tables: list[dict], rules: Rules) -> list[dict]:
    result = []
    for region_id, rows in enumerate(page["regions"]):
        pending: list[tuple[dict, list[dict]]] = []

        def flush() -> None:
            if len(pending) >= 2 or (pending and len(pending[0][1]) >= 2):
                labels = [{**p, "line": n} for n, (_, pairs) in enumerate(pending, 1) for p in pairs]
                boxes = [bbox([label]) for label in labels] + [label["value_bbox"] for label in labels if label["value_bbox"]]
                group_bbox = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
                result.append(dict(kind="kv", page=page["page"], region=region_id,
                                   bbox=group_bbox, labels=labels,
                                   reason="aligned label/value pairs; value variation checked across occurrences"))
            pending.clear()

        for row in rows:
            if any(t["region"] == region_id and t["bbox"][1] <= row["top"] <= t["bbox"][3] for t in tables):
                flush()
                continue
            pairs = kv_pairs(row, support, rules)
            if not pairs:
                # Small diagram fragments between two KV rows do not terminate a group.
                if pending and row["top"] - pending[-1][0]["bottom"] > rules.kv_gap * row["height"]:
                    flush()
                continue
            if pending:
                prev_row, prev_pairs = pending[-1]
                if (len(pairs) == len(prev_pairs) == 1
                        and prev_pairs[0]["value_x"] is not None
                        and prev_pairs[0]["value_bbox"] is not None
                        and pairs[0]["value_x"] is not None
                        and abs(pairs[0]["x0"] - prev_pairs[0]["value_x"]) <= row["height"]
                        and row["top"] - prev_row["bottom"] <= rules.kv_gap * row["height"]):
                    # A subsequent indented line belongs to the value column.
                    prev_pairs[0]["value"] += " " + row["text"]
                    previous = prev_pairs[0]["value_bbox"]
                    current = bbox([row])
                    prev_pairs[0]["value_bbox"] = [min(previous[0], current[0]), min(previous[1], current[1]),
                                                    max(previous[2], current[2]), max(previous[3], current[3])]
                    continue
                vertically_aligned = (len(pairs) == len(prev_pairs) == 1
                                      and pairs[0]["value_x"] is not None
                                      and prev_pairs[0]["value_x"] is not None
                                      and abs(pairs[0]["x0"] - prev_pairs[0]["x0"]) <= row["height"]
                                      and abs(pairs[0]["value_x"] - prev_pairs[0]["value_x"]) <= row["height"])
                if not vertically_aligned or row["top"] - prev_row["bottom"] > rules.kv_gap * row["height"]:
                    flush()
            pending.append((row, pairs))
            if len(pairs) >= 2:
                flush()
        flush()
    return result


def ordered_matches(left: list[dict], right: list[dict]) -> list[tuple[int, int]]:
    """Exact normalized phrase LCS; insertion/deletion allowed, reordering rejected."""
    n, m = len(left), len(right)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = 1 + dp[i + 1][j + 1] if left[i]["key"] == right[j]["key"] else max(dp[i + 1][j], dp[i][j + 1])
    pairs = []
    i = j = 0
    while i < n and j < m:
        if left[i]["key"] == right[j]["key"]:
            pairs.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def relative_positions(candidate: dict) -> list[tuple[float, float]]:
    labels = candidate["labels"]
    x0, top, x1, bottom = bbox(labels)
    h = median(c["bottom"] - c["top"] for c in labels)
    return [((c["x0"] - x0) / max(h, x1 - x0),
             (c["top"] - top) / max(h, bottom - top)) for c in labels]


def compatibility(a: dict, b: dict, rules: Rules) -> tuple[float, list[tuple[int, int]]]:
    if a["kind"] != b["kind"]:
        return 0.0, []
    matches = ordered_matches(a["labels"], b["labels"])
    score = len(matches) / max(len(a["labels"]), len(b["labels"]))
    if len(matches) < 2 or score < rules.family_overlap:
        return 0.0, []
    # Normalize the matched anchors, so an optional leading/trailing column or
    # extra caption does not move every shared label in the comparison space.
    pos_a = relative_positions({"labels": [a["labels"][i] for i, _ in matches]})
    pos_b = relative_positions({"labels": [b["labels"][j] for _, j in matches]})
    residuals = [max(abs(xa - xb), abs(ya - yb)) for (xa, ya), (xb, yb) in zip(pos_a, pos_b)]
    # Permit a few displaced fragments after insertion; most anchors must agree.
    if sum(d <= rules.geometry_tolerance for d in residuals) / len(residuals) < 0.8:
        return 0.0, []
    return score, matches


def cluster_candidates(candidates: list[dict], rules: Rules) -> tuple[list[dict], list[dict]]:
    """Index by labels and compare to frozen representatives, never all page pairs."""
    families: list[dict] = []
    index: dict[tuple[str, str], set[int]] = defaultdict(set)
    for candidate in sorted(candidates, key=lambda c: (c["page"], c["bbox"][1], c["bbox"][0], c["kind"])):
        shortlist: Counter = Counter()
        for key in {c["key"] for c in candidate["labels"]}:
            shortlist.update(index.get((candidate["kind"], key), ()))
        best, best_score, pairs = None, 0.0, []
        for family_id in sorted(shortlist):
            representative = families[family_id]["representative"]
            score, matched = compatibility(representative, candidate, rules)
            if score > best_score:
                best, best_score, pairs = family_id, score, matched
        if best is None:
            best = len(families)
            families.append(dict(kind=candidate["kind"], representative=candidate,
                                 fields=[dict(c) for c in candidate["labels"]], occurrences=[]))
            pairs = list(enumerate(range(len(candidate["labels"]))))
            best_score = 1.0
            for label in candidate["labels"]:
                index[(candidate["kind"], label["key"])].add(best)
        family = families[best]
        # Representative fields are always the leading entries. New optional
        # fragments do not alter the representative or permit chain merging.
        field_indices = {j: i for i, j in pairs}
        for j, label in enumerate(candidate["labels"]):
            if j not in field_indices:
                extras = [i for i in range(len(family["representative"]["labels"]), len(family["fields"]))
                          if family["fields"][i]["key"] == label["key"] and family["fields"][i]["line"] == label["line"]
                          and i not in field_indices.values()]
                field_indices[j] = extras[0] if extras else len(family["fields"])
                if not extras:
                    family["fields"].append(dict(label))
        family["occurrences"].append(candidate | {
            "match_score": round(best_score, 4), "field_indices": [field_indices[j] for j in range(len(candidate["labels"]))]})

    confirmed, unconfirmed = [], []
    counters: Counter = Counter()
    for family in families:
        pages = sorted({o["page"] for o in family["occurrences"]})
        values: dict[int, set[str]] = defaultdict(set)
        value_pages: dict[int, set[int]] = defaultdict(set)
        for occurrence in family["occurrences"]:
            for field, label in zip(occurrence["field_indices"], occurrence["labels"]):
                value = normalize(label.get("value", ""))
                if value:
                    values[field].add(value)
                    value_pages[field].add(occurrence["page"])
        family["pages"] = pages
        family["varying_value_fields"] = [i for i, v in sorted(values.items()) if len(v) > 1 and len(value_pages[i]) >= rules.min_pages]
        family["distinct_value_counts"] = {str(i): len(v) for i, v in sorted(values.items())}
        reasons = []
        if len(pages) < rules.min_pages:
            reasons.append("insufficient distinct pages")
        if family["kind"] == "kv" and not family["varying_value_fields"]:
            reasons.append("no observed cross-page value variation")
        if reasons:
            family["unconfirmed_reasons"] = reasons
            unconfirmed.append(family)
        else:
            counters[family["kind"]] += 1
            family["id"] = f"{family['kind']}_{counters[family['kind']]:03d}"
            confirmed.append(family)
    return confirmed, unconfirmed


def scan_pages(pages: list[dict], rules: Rules | None = None) -> dict:
    rules = rules or Rules()
    prepared, support = prepare_pages(pages, rules)
    # KV confirmation runs first: repeated values like EN10365 otherwise look
    # like headers. Confirmed KV bands cannot also become table headers.
    kvs = [c for page in prepared for c in kv_candidates(page, support, [], rules)]
    confirmed_kvs, _ = cluster_candidates(kvs, rules)
    kv_by_page: dict[int, list[dict]] = defaultdict(list)
    for group in confirmed_kvs:
        for occurrence in group["occurrences"]:
            kv_by_page[occurrence["page"]].append(occurrence)
    candidates = list(kvs)
    for page in prepared:
        tables = table_candidates(page, support, rules, kv_by_page[page["page"]])
        candidates.extend(tables)
    families, unconfirmed = cluster_candidates(candidates, rules)
    numbers = [p["page"] for p in pages]
    for family in families:
        counts = [[0] * len(numbers) for _ in family["fields"]]
        page_index = {p: i for i, p in enumerate(numbers)}
        for occurrence in family["occurrences"]:
            for field in occurrence["field_indices"]:
                counts[field][page_index[occurrence["page"]]] += 1
        family["heatmap_counts"] = counts
    return dict(version=3, rules=asdict(rules), pages=numbers,
                page_summary=[dict(page=p["page"], width=p["width"], height=p["height"],
                                   word_count=p["word_count"], regions=len(p["regions"])) for p in prepared],
                empty_text_pages=[p["page"] for p in prepared if not p["word_count"]],
                families=families, unconfirmed=unconfirmed,
                candidate_count=len(candidates))


def spatial_coverage(boxes: list[tuple[int, float, float, float, float]]) -> list[tuple]:
    """Exact rectangle coverage: (x0, top, x1, bottom, distinct occurrence count).

    Sweep at actual word-box edges. No histogram bins, coordinate rounding,
    kernel smoothing, or resampling. Overlapping words in one occurrence count once.
    Adjacent slabs with equal coverage are merged into taller rectangles.
    """
    events: dict[float, list[tuple]] = defaultdict(list)
    for identifier, (page, x0, top, x1, bottom) in enumerate(boxes):
        if x1 <= x0 or bottom <= top:
            continue
        events[top].append((identifier, (page, x0, x1)))
        events[bottom].append((identifier, None))
    active: dict[int, tuple] = {}
    result: list[tuple] = []
    previous: dict[tuple, int] = {}
    ys = sorted(events)
    for top, bottom in zip(ys, ys[1:]):
        for identifier, value in events[top]:
            if value is None:
                active.pop(identifier, None)
            else:
                active[identifier] = value
        x_events: dict[float, Counter] = defaultdict(Counter)
        for occurrence, x0, x1 in active.values():
            x_events[x0][occurrence] += 1
            x_events[x1][occurrence] -= 1
        counts: Counter = Counter()
        xs = sorted(x_events)
        spans: list[tuple] = []
        for x0, x1 in zip(xs, xs[1:]):
            for occurrence, delta in x_events[x0].items():
                counts[occurrence] += delta
                if counts[occurrence] == 0:
                    del counts[occurrence]
            level = len(counts)
            if not level:
                continue
            if spans and spans[-1][1] == x0 and spans[-1][2] == level:
                spans[-1] = (spans[-1][0], x1, level)
            else:
                spans.append((x0, x1, level))
        current = {}
        for x0, x1, level in spans:
            key = (x0, x1, level)
            if key in previous and result[previous[key]][3] == top:
                index = previous[key]
                result[index] = (x0, result[index][1], x1, bottom, level)
            else:
                index = len(result)
                result.append((x0, top, x1, bottom, level))
            current[key] = index
        previous = current
    return result


def spatial_word_boxes(family: dict) -> list[tuple]:
    """Place each occurrence's header/key words relative to its top-left word."""
    boxes = set()
    for occurrence_id, occurrence in enumerate(family["occurrences"]):
        words = []
        for label in occurrence["labels"]:
            if "words" not in label:
                raise ValueError("word coordinates are missing; rescan the PDF to upgrade this report")
            words.extend(label["words"])
        if not words:
            continue
        origin_x = min(word["x0"] for word in words)
        origin_y = min(word["top"] for word in words)
        for word in words:
            x0, top = word["x0"] - origin_x, word["top"] - origin_y
            x1, bottom = word["x1"] - origin_x, word["bottom"] - origin_y
            if x1 > x0 and bottom > top:
                boxes.add((occurrence_id, x0, top, x1, bottom))
    return sorted(boxes)


def plot_results(report: dict, output: Path, title: str) -> list[str]:
    """Overlay each group's occurrences at their word-relative coordinates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.ticker import MaxNLocator

    artifacts = []
    cmap = LinearSegmentedColormap.from_list(
        "word_recurrence", ["#ffffff", "#fee8c8", "#fdbb84", "#e34a33", "#7f0000"])

    def heatmap(family: dict) -> None:
        boxes = spatial_word_boxes(family)
        coverage = spatial_coverage(boxes)
        width = max((box[3] for box in boxes), default=1.0)
        height = max((box[4] for box in boxes), default=1.0)
        fig, ax = plt.subplots(figsize=(12, max(3.5, min(9, 12 * height / width + 1.8))),
                               layout="constrained")
        vertices = [[(x0, top), (x1, top), (x1, bottom), (x0, bottom)]
                    for x0, top, x1, bottom, _ in coverage]
        norm = Normalize(vmin=0, vmax=len(family["occurrences"]))
        collection = PolyCollection(vertices, array=[c[4] for c in coverage],
                                    cmap=cmap, norm=norm, edgecolors="none", antialiaseds=False)
        ax.add_collection(collection)
        ax.set(xlim=(0, width), ylim=(height, 0), aspect="equal",
               xlabel="x from group leftmost word (PDF points)",
               ylabel="y from group topmost word (PDF points)")
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_color("#94a3b8")
        bar = fig.colorbar(collection, ax=ax, fraction=0.032, pad=0.03)
        bar.set_label("Group occurrences with a header/key word here")
        bar.locator = MaxNLocator(nbins=6, integer=True)
        bar.update_ticks()
        if not coverage:
            ax.text(width / 2, height / 2, "No recurring words detected", ha="center", color="#64748b")
        terms = " / ".join(field["text"] for field in family["fields"][:3])
        ax.set_title(f"{title} | {family['id']} - {terms[:85]}\n"
                     f"{len(family['occurrences'])} occurrences across {len(family['pages'])} pages",
                     fontsize=12, pad=14)
        fig.text(0.5, 0.005, "Exact word boxes relative to each occurrence; whitespace stays empty.",
                 ha="center", fontsize=8, color="#475569")
        for extension in ("png", "svg"):
            name = f"{family['id']}.{extension}"
            fig.savefig(output / name, dpi=180)
            artifacts.append(name)
        plt.close(fig)

    for family in report["families"]:
        heatmap(family)
    return artifacts


def render_overlays(pdf_path: Path, report: dict, output: Path, numbers: list[int]) -> list[str]:
    import pdfplumber
    from PIL import ImageDraw

    artifacts = []
    with pdfplumber.open(pdf_path) as pdf:
        for number in numbers:
            if number not in report["pages"]:
                raise ValueError(f"overlay page {number} is outside the scanned range")
            page = pdf.pages[number - 1]
            rendered = page.to_image(resolution=110).original.convert("RGB")
            draw = ImageDraw.Draw(rendered)
            sx, sy = rendered.width / page.width, rendered.height / page.height
            for family in report["families"]:
                color = "#dc2626" if family["kind"] == "table" else "#059669"
                for occurrence in family["occurrences"]:
                    if occurrence["page"] != number:
                        continue
                    x0, y0, x1, y1 = occurrence["bbox"]
                    draw.rectangle((x0 * sx, y0 * sy, x1 * sx, y1 * sy), outline=color, width=2)
                    draw.text((x0 * sx, max(0, y0 * sy - 12)), family["id"], fill=color)
            name = f"page_{number:04d}_overlay.png"
            rendered.save(output / name)
            artifacts.append(name)
    return artifacts


def resolve_pdf(value: str) -> Path:
    direct = Path(value).expanduser()
    if direct.is_file():
        return direct.resolve()
    dataset = ROOT / "data" / value / f"{value}.pdf"
    if dataset.is_file():
        return dataset
    raise ValueError(f"PDF or dataset not found: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("documents", nargs="+", help="PDF paths or dataset names")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "test" / "scan_results")
    parser.add_argument("--result-name", help="Output folder name for a single PDF (useful for same-named PDFs)")
    parser.add_argument("--min-pages", type=int, default=2, help="Distinct pages needed to confirm repetition (>=2)")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int)
    parser.add_argument("--gap-factor", type=float, default=2.4)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--overlay-pages", default="", help="Comma-separated physical pages to annotate")
    args = parser.parse_args()
    if args.min_pages < 2 or args.gap_factor <= 0:
        parser.error("--min-pages must be >=2 and --gap-factor must be positive")
    if args.result_name and (len(args.documents) != 1 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.result_name)
                             or args.result_name in {".", ".."}):
        parser.error("--result-name requires one PDF and a safe folder name")
    try:
        paths = [resolve_pdf(value) for value in args.documents]
        if len({p.stem for p in paths}) != len(paths):
            raise ValueError("input basenames must be distinct to avoid overwriting outputs")
        overlay_pages = sorted({int(p) for p in args.overlay_pages.split(",") if p.strip()})
        for path in paths:
            print(f"Scanning {path.name}", flush=True)
            pages, total = load_pdf(path, args.start_page, args.end_page)
            rules = Rules(min_pages=args.min_pages, gap_factor=args.gap_factor)
            report = scan_pages(pages, rules)
            report.update(source_pdf=str(path), total_pdf_pages=total)
            output = args.out_dir.resolve() / (args.result_name or path.stem)
            output.mkdir(parents=True, exist_ok=True)
            artifacts = [] if args.no_plots else plot_results(report, output, path.stem)
            artifacts += render_overlays(path, report, output, overlay_pages) if overlay_pages else []
            # Only manage this script's own plot names inside this document's
            # output folder, so reruns cannot leave obsolete family plots behind.
            for old in output.iterdir():
                if old.name not in artifacts and re.fullmatch(r"(?:(?:overview|(?:table|kv)_\d+)(?:_size_\d+)?\.(?:png|svg)|page_\d+_overlay\.png)", old.name):
                    old.unlink()
            report["artifacts"] = artifacts
            (output / "scan.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            counts = Counter(f["kind"] for f in report["families"])
            print(f"  {len(pages)} pages: {counts['table']} table groups, {counts['kv']} KV groups; "
                  f"{len(report['unconfirmed'])} unconfirmed groups\n  Saved {output}", flush=True)
            if report["empty_text_pages"]:
                print(f"  No text layer on pages {report['empty_text_pages']}; image OCR is not performed.", flush=True)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
