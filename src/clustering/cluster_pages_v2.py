"""Cluster PDF pages deterministically using detected headers and table borders."""
from __future__ import annotations

from collections import defaultdict
from collections import Counter
import json
import math
from pathlib import Path
import re
from statistics import median

import numpy as np

from . import cluster_pages as base

METHODS = {
    "header_strict": ("complete", 0.80, 0.86),
    "balanced_complete": ("complete", 0.60, 0.83),
    "balanced_average": ("average", 0.60, 0.83),
    "structure_complete": ("complete", 0.35, 0.82),
    "structure_average": ("average", 0.35, 0.82),
}
TABLE_ALGORITHM_REVISION = 6
LOCAL_COLUMN_MIN_SIMILARITY = 0.80
LOCAL_MISMATCH_CAP = 0.79


def noisy_scan(report: dict, headers_by_page: dict[int, list[dict]], page_numbers: list[int]) -> bool:
    """Recognize body rows misclassified as many unrelated header families."""
    families = sum(family["kind"] == "table" for family in report["families"])
    if families == 0:
        return True
    label_counts = [sum(len(header["labels"]) for header in headers_by_page.get(page, []))
                    for page in page_numbers]
    return families >= 8 and median(label_counts) > 100


def word_tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[^\W\d_]+", value.casefold()) if len(token) >= 3}


def consensus_headers(pdf, numbers: list[int]) -> dict[int, list[dict]]:
    """Find a repeated aligned header band when scanner families are absent/noisy.

    Support is measured across pages, and the band must contain at least six
    distinct repeated alphabetic tokens spread over nearly half a page. This
    rejects vertically repeating key/value labels in catalog cards.
    """
    top_words = {}
    support: Counter[str] = Counter()
    for number in numbers:
        page = pdf.pages[number - 1]
        words = [word for word in page.extract_words()
                 if (word["top"] + word["bottom"]) / 2 < 0.32 * page.height]
        top_words[number] = words
        support.update({token for word in words for token in word_tokens(word["text"])})
    frequent = {token for token, count in support.items()
                if count >= max(3, math.ceil(0.60 * len(numbers)))}
    candidates = {}
    for number in numbers:
        page = pdf.pages[number - 1]
        tolerance = max(6.0, 0.008 * page.height)
        rows: dict[int, list[dict]] = defaultdict(list)
        for word in top_words[number]:
            if word_tokens(word["text"]) & frequent:
                center_y = (word["top"] + word["bottom"]) / 2
                rows[round(center_y / tolerance)].append(word)
        qualifying = []
        for row in rows.values():
            tokens = set().union(*(word_tokens(word["text"]) & frequent for word in row))
            centers = [(word["x0"] + word["x1"]) / 2 for word in row]
            span = max(centers) - min(centers)
            if len(row) >= 6 and len(tokens) >= 6 and span >= 0.45 * page.width:
                qualifying.append((len(tokens), span, -median((word["top"] + word["bottom"]) / 2 for word in row), row))
        if qualifying:
            best = max(qualifying, key=lambda item: item[:3])[3]
            anchor = median((word["top"] + word["bottom"]) / 2 for word in best)
            nearby = [word for word in top_words[number]
                      if word_tokens(word["text"]) & frequent
                      and abs((word["top"] + word["bottom"]) / 2 - anchor) <= max(18, 0.035 * page.height)]
            candidates[number] = nearby
    if len(candidates) < max(3, math.ceil(0.60 * len(numbers))):
        return {}
    result = {}
    for number, words in candidates.items():
        labels = [{"text": word["text"], "x0": word["x0"], "x1": word["x1"],
                   "top": word["top"], "bottom": word["bottom"],
                   "line": round((word["top"] + word["bottom"]) / 2 / 6)} for word in words]
        bbox = [min(word["x0"] for word in words), min(word["top"] for word in words),
                max(word["x1"] for word in words), max(word["bottom"] for word in words)]
        result[number] = [{"family": "consensus_table", "page": number, "bbox": bbox,
                           "labels": labels, "recovered": True}]
    return result


def recover_headers(page, current: list[dict], families: list[dict]) -> list[dict]:
    """Confirm known repeated headers that the candidate scanner missed."""
    found = list(current)
    for family in families:
        if family["kind"] != "table":
            continue
        prototype = family["representative"]
        left, top, right, bottom = prototype["bbox"]
        if any(abs(item["bbox"][1] - top) < 12 and
               max(0, min(item["bbox"][2], right) - max(item["bbox"][0], left)) > 0.5 * (right - left)
               for item in found):
            continue
        expected = set(re.findall(r"\w+", " ".join(label["text"] for label in prototype["labels"]).casefold()))
        if len(expected) < 2:
            continue
        page_left, page_top, page_right, page_bottom = page.bbox
        crop_box = (max(page_left, left - 12), max(page_top, top - 5),
                    min(page_right, right + 12), min(page_bottom, bottom + 5))
        if crop_box[0] >= crop_box[2] or crop_box[1] >= crop_box[3]:
            continue
        crop = page.crop(crop_box)
        observed = set(re.findall(r"\w+", (crop.extract_text() or "").casefold()))
        if len(expected & observed) >= max(2, int(np.ceil(0.75 * len(expected)))):
            found.append({"family": family["id"], "page": page.page_number,
                          "bbox": prototype["bbox"], "labels": prototype["labels"],
                          "recovered": True})
    return sorted(found, key=lambda item: (item["bbox"][1], item["bbox"][0]))


def relevant_lines(lines: list[dict], headers: list[dict], width: float, height: float) -> list[dict]:
    """Limit borders to header-bearing table bands when headers are known."""
    orthogonal = []
    for line in lines:
        if base.horizontal(line) or base.vertical(line):
            orthogonal.append({**line, "x0": min(line["x0"], line["x1"]),
                               "x1": max(line["x0"], line["x1"]),
                               "top": min(line["top"], line["bottom"]),
                               "bottom": max(line["top"], line["bottom"])})
    if not headers:
        return orthogonal
    header_span = max(item["bbox"][2] for item in headers) - min(item["bbox"][0] for item in headers)
    if header_span < 0.35 * width:
        # A one-column scanner hit cannot bound an entire table grid.
        return orthogonal
    selected = []
    for line in orthogonal:
        for header in headers:
            left, top, right, bottom = header["bbox"]
            margin = max(12, 0.03 * width)
            x_overlap = max(0, min(line["x1"], right + margin) - max(line["x0"], left - margin))
            x_inside = left - margin <= line["x0"] <= right + margin
            if not (x_overlap >= 0.25 * max(1, right - left) or x_inside):
                continue
            next_header = min((other["bbox"][1] for other in headers
                               if other["bbox"][1] > bottom + 5), default=height)
            if line["bottom"] >= top - 0.035 * height and line["top"] <= next_header:
                selected.append(line)
                break
    return selected


def has_structure(lines: list[dict], width: float, height: float) -> bool:
    """Require a small grid, avoiding isolated rules and page furniture."""
    horizontals = [line for line in lines if base.horizontal(line) and line["x1"] - line["x0"] >= 0.12 * width]
    verticals = [line for line in lines if base.vertical(line)]
    y_levels = {round(line["top"] / max(3, 0.005 * height)) for line in horizontals}
    x_groups = defaultdict(set)
    for line in verticals:
        x_groups[round(line["x0"] / max(3, 0.005 * width))].add(round(line["top"] / max(3, 0.005 * height)))
    repeated_columns = sum(len(ys) >= 3 for ys in x_groups.values())
    long_columns = sum(line["bottom"] - line["top"] >= 0.04 * height for line in verticals)
    wide_enough = [line for line in lines if base.horizontal(line) and line["x1"] - line["x0"] >= 0.04 * width]
    span_counts = defaultdict(set)
    for line in wide_enough:
        span_counts[(round(line["x0"] / 5), round(line["x1"] / 5))].add(round(line["top"] / 5))
    repeated_spans = sum(len(ys) >= 3 for ys in span_counts.values())
    return ((len(y_levels) >= 3 and repeated_columns >= 2)
            or (len(y_levels) >= 2 and long_columns >= 3)
            or (len(wide_enough) >= 12 and repeated_spans >= 2
                and len({round(line["top"] / 5) for line in wide_enough}) >= 5))


def structure_signature(lines: list[dict], width: float, height: float) -> np.ndarray:
    """Column positions and horizontal spans survive changes in row count."""
    vertical = np.zeros(32)
    horizontal = np.zeros(32)
    row_gaps = np.zeros(16)
    y_values = []
    for line in lines:
        if base.vertical(line):
            x = (line["x0"] + line["x1"]) / (2 * width)
            vertical[min(31, max(0, int(x * 32)))] += min(1.0, (line["bottom"] - line["top"]) / (0.2 * height))
        elif base.horizontal(line):
            x0, x1 = sorted((line["x0"] / width, line["x1"] / width))
            for index in range(max(0, int(x0 * 32)), min(32, int(np.ceil(x1 * 32)))):
                horizontal[index] += max(0, min(x1, (index + 1) / 32) - max(x0, index / 32))
            y_values.append(line["top"])
    levels = sorted({round(y / max(2, 0.003 * height)) for y in y_values})
    for first, second in zip(levels, levels[1:]):
        gap = (second - first) * max(2, 0.003 * height) / height
        if 0.005 <= gap <= 0.16:
            row_gaps[min(15, int(gap / 0.16 * 16))] += 1
    parts = []
    for vector, weight in ((vertical, 0.50), (horizontal, 0.35), (row_gaps, 0.15)):
        norm = np.linalg.norm(vector)
        parts.append(weight * vector / norm if norm else vector)
    return np.concatenate(parts)


def header_geometry(headers: list[dict], width: float, height: float) -> np.ndarray:
    positions = np.zeros(32)
    bands = np.zeros(8)
    for header in headers:
        for label in header["labels"]:
            x = (label["x0"] + label["x1"]) / (2 * width)
            positions[min(31, max(0, int(x * 32)))] += 1
        y = (header["bbox"][1] + header["bbox"][3]) / (2 * height)
        bands[min(7, max(0, int(y * 8)))] += 1
    parts = []
    for vector, weight in ((positions, 0.8), (bands, 0.2)):
        norm = np.linalg.norm(vector)
        parts.append(weight * vector / norm if norm else vector)
    return np.concatenate(parts)


def table_columns(headers: list[dict], lines: list[dict], width: float) -> list[dict]:
    """Normalized vertical-rule positions for each detected table separately."""
    result = []
    for header in headers:
        left, top, right, _ = header["bbox"]
        span = right - left
        if span <= 0:
            continue
        margin = 0.02 * span
        positions = sorted((line["x0"] + line["x1"]) / 2 for line in lines
                           if base.vertical(line) and left - margin <= line["x0"] <= right + margin
                           and line["bottom"] >= top)
        unique = []
        for x in positions:
            if not unique or x - unique[-1] > max(2.0, 0.003 * width):
                unique.append(x)
        result.append({"family": header.get("family"), "x_span": [left / width, right / width],
                       "positions": [(x - left) / span for x in unique]})
    return result


def column_similarity(left: list[float], right: list[float], tolerance: float = 0.016) -> float:
    """One-to-one rule matches; one added column leaves most boundaries intact."""
    if not left or not right:
        return 0.0
    used = set()
    matched = 0
    for x in left:
        candidates = [(abs(x - y), index) for index, y in enumerate(right)
                      if index not in used and abs(x - y) <= tolerance]
        if candidates:
            _, index = min(candidates)
            used.add(index)
            matched += 1
    return matched / max(len(left), len(right))


def column_counts_reliable(features: list[dict]) -> bool:
    """A fluctuating rule detector cannot support a hard local mismatch gate."""
    counts: dict[str, list[int]] = defaultdict(list)
    for feature in features:
        for region in feature["table_columns"]:
            if region.get("family") and len(region["positions"]) >= 4:
                counts[region["family"]].append(len(region["positions"]))
    for values in counts.values():
        if len(values) < 3:
            continue
        low, high = np.percentile(values, [10, 90])
        if high - low > max(3.0, 0.25 * median(values)):
            return False
    return True


def local_layout_mismatch(a: dict, b: dict) -> bool:
    """A changed table component must not be hidden by other, unchanged tables."""
    if not (a.get("has_red") and b.get("has_red")
            and a.get("column_layout_reliable", True) and b.get("column_layout_reliable", True)):
        return False
    left = a.get("table_columns", [])
    right = b.get("table_columns", [])
    for region in left:
        first = region["positions"]
        if len(first) < 4:
            continue
        l0, l1 = region["x_span"]
        same_family = [other for other in right if region.get("family") is not None
                       and other.get("family") == region["family"] and len(other["positions"]) >= 4]
        candidates = same_family[:]
        if not candidates:
            # Different scanner families may still occupy the same table slot.
            # The local column signature decides whether they are compatible.
            for other in right:
                second = other["positions"]
                if len(second) < 4:
                    continue
                r0, r1 = other["x_span"]
                overlap = max(0.0, min(l1, r1) - max(l0, r0)) / max(1e-9, min(l1 - l0, r1 - r0))
                if overlap >= 0.5:
                    candidates.append(other)
        if candidates and max(column_similarity(first, other["positions"]) for other in candidates) < LOCAL_COLUMN_MIN_SIMILARITY:
            return True
    return False


def cosine_nonempty(a: np.ndarray, b: np.ndarray) -> float:
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / norm) if norm else 0.0


def header_similarity(a: dict, b: dict) -> float:
    left, right = set(a["header_words"]), set(b["header_words"])
    if not left or not right:
        return 0.0
    common = len(left & right)
    text = 0.65 * common / min(len(left), len(right)) + 0.35 * common / len(left | right)
    return 0.88 * text + 0.12 * cosine_nonempty(a["header_geometry"], b["header_geometry"])


def pair_similarity(a: dict, b: dict, header_weight: float) -> tuple[float, float, float]:
    head = header_similarity(a, b)
    structural = cosine_nonempty(a["structure"], b["structure"])
    if a["has_red"] and b["has_red"]:
        if a["headers"] and b["headers"]:
            combined = header_weight * head + (1 - header_weight) * structural
        elif not a["headers"] and not b["headers"]:
            combined = structural
        else:
            combined = 0.90 * structural
    elif a["headers"] and b["headers"]:
        # The word-derived shape is evidence only for the side lacking borders.
        weight = min(0.20, 1 - header_weight)
        combined = (1 - weight) * head + weight * structural
        if a["has_red"] != b["has_red"]:
            combined *= 0.96
    else:
        combined = 0.0
    if local_layout_mismatch(a, b):
        combined = min(combined, LOCAL_MISMATCH_CAP)
    return max(0.0, min(1.0, combined)), head, structural


def components(features: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    n = len(features)
    header, structure = (np.eye(n) for _ in range(2))
    for i in range(n):
        for j in range(i + 1, n):
            header[i, j] = header[j, i] = header_similarity(features[i], features[j])
            structure[i, j] = structure[j, i] = cosine_nonempty(features[i]["structure"], features[j]["structure"])
    return header, structure


def scores(features: list[dict], header_weight: float,
           parts: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(features)
    header, structure = parts if parts is not None else components(features)
    has_header = np.array([bool(f["headers"]) for f in features])
    has_red = np.array([f["has_red"] for f in features])
    both_header = np.outer(has_header, has_header)
    both_red = np.outer(has_red, has_red)
    neither_header = np.outer(~has_header, ~has_header)
    combined = np.zeros_like(header)
    combined[both_red & both_header] = (header_weight * header + (1 - header_weight) * structure)[both_red & both_header]
    combined[both_red & neither_header] = structure[both_red & neither_header]
    combined[both_red & ~both_header & ~neither_header] = 0.90 * structure[both_red & ~both_header & ~neither_header]
    fallback = ~both_red & both_header
    word_weight = min(0.20, 1 - header_weight)
    combined[fallback] = ((1 - word_weight) * header + word_weight * structure)[fallback]
    mismatch = np.not_equal.outer(has_red, has_red)
    combined[fallback & mismatch] *= 0.96
    for i in range(n):
        for j in range(i + 1, n):
            if local_layout_mismatch(features[i], features[j]):
                combined[i, j] = combined[j, i] = min(combined[i, j], LOCAL_MISMATCH_CAP)
    np.fill_diagonal(combined, 1.0)
    return combined, header, structure


def metrics(labels: list[int], distance: np.ndarray, features: list[dict],
            header: np.ndarray, structure: np.ndarray, first_page: int = 1) -> dict:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    within = [(i, j) for indices in groups.values() for p, i in enumerate(indices) for j in indices[p + 1:]]
    silhouettes = []
    if len(groups) > 1:
        for i, label in enumerate(labels):
            own = [j for j in groups[label] if j != i]
            if not own:
                silhouettes.append(0.0)
                continue
            a = float(np.mean([distance[i, j] for j in own]))
            b = min(float(np.mean([distance[i, j] for j in group])) for key, group in groups.items() if key != label)
            silhouettes.append((b - a) / max(a, b) if max(a, b) else 0.0)
    representatives = {str(label): features[min(indices, key=lambda i: (sum(distance[i, j] for j in indices), features[i]["page"]))]["page"]
                       for label, indices in groups.items()}
    agreement = (sum(set(features[i]["families"]) == set(features[j]["families"]) for i, j in within) / len(within)) if within else None
    return {
        "cluster_count": len(groups),
        "clusters": {str(label): [features[i]["page"] for i in indices] for label, indices in groups.items()},
        "representatives": representatives,
        "mean_silhouette": round(float(np.mean(silhouettes)), 4) if silhouettes else None,
        "mean_within_similarity": round(float(np.mean([1 - distance[i, j] for i, j in within])), 4) if within else None,
        "mean_within_header_similarity": round(float(np.mean([header[i, j] for i, j in within])), 4) if within else None,
        "mean_within_structure_similarity": round(float(np.mean([structure[i, j] for i, j in within])), 4) if within else None,
        "header_family_agreement": round(agreement, 4) if agreement is not None else None,
        "first_five_cluster_coverage": len({labels[i] for i, feature in enumerate(features)
                                            if first_page <= feature["page"] < first_page + 5}) / len(groups),
    }


def process(pdf_path: Path, max_pages: int, output_root: Path, scan_root: Path | None,
            overlay_limit: int, result_name: str | None = None, *, start_page: int = 1,
            scan_report: dict | None = None) -> dict:
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        if not 1 <= start_page <= len(pdf.pages):
            raise ValueError(f"start page {start_page} is outside the PDF")
        numbers = list(range(start_page, min(len(pdf.pages), start_page + max_pages - 1) + 1))
        scan = scan_report if scan_report is not None else base.scan_report(pdf_path, numbers, scan_root, result_name)
        if scan.get("pages") != numbers:
            raise ValueError("scan pages do not match the requested source pages")
        headers_by_page = base.table_headers(scan)
        replace_noisy_headers = noisy_scan(scan, headers_by_page, numbers)
        family_count = sum(family["kind"] == "table" for family in scan["families"])
        typical_tables = median(len(headers_by_page.get(number, [])) for number in numbers)
        column_layout_reliable = (not replace_noisy_headers
                                  and family_count <= 2 * max(1, typical_tables))
        consensus_by_page = consensus_headers(pdf, numbers) if replace_noisy_headers else {}
        features = []
        for scan_index, number in enumerate(numbers, 1):
            page = pdf.pages[number - 1]
            headers = (consensus_by_page.get(number, []) if replace_noisy_headers
                       else recover_headers(page, headers_by_page.get(number, []), scan["families"]))
            raw_red = base.vector_lines(page, min_length=12)
            if not raw_red:
                image = page.to_image(resolution=110).original
                raw_red = base.raster_lines(image, page.width, page.height, min_length=12)
                image.close()
            red = relevant_lines(raw_red, headers, page.width, page.height)
            has_red = has_structure(red, page.width, page.height)
            # Some pages have vector page furniture but a rasterized table
            # body. A nonempty vector list must not suppress raster detection.
            if not has_red and len(raw_red) >= 20:
                image = page.to_image(resolution=110).original
                raster = relevant_lines(base.raster_lines(image, page.width, page.height, min_length=12),
                                        headers, page.width, page.height)
                image.close()
                if has_structure(raster, page.width, page.height):
                    red, has_red = raster, True
            if not has_red:
                red = []
            words = []
            inferred = []
            if headers and not has_red:
                words = [{k: word[k] for k in ("text", "x0", "top", "x1", "bottom")}
                         for word in page.extract_words()
                         if word["text"].strip() and word["x1"] > word["x0"] and word["bottom"] > word["top"]]
                regions = base.table_regions(headers, [], page.width, page.height)
                inferred = base.green_lines(words, regions, headers, page.width, page.height)
            blue = base.blue_lines(headers)
            features.append(dict(page=number, headers=headers, families=sorted({h["family"] for h in headers}),
                                 header_words=base.header_signature(headers),
                                 header_geometry=header_geometry(headers, page.width, page.height),
                                 structure=structure_signature(red if has_red else inferred, page.width, page.height),
                                 table_columns=table_columns(headers, red, page.width) if has_red else [],
                                 column_layout_reliable=column_layout_reliable,
                                 red_lines=red, green_lines=inferred, blue_lines=blue,
                                 has_red=has_red, eligible=bool(headers or has_red), word_count=len(words)))
            if scan_index == 1 or scan_index % 25 == 0 or scan_index == len(numbers):
                source_suffix = f" (source {number})" if start_page != 1 else ""
                print(f"  page {scan_index}/{len(numbers)}: "
                      f"{len(red)} red, {len(blue)} blue, {len(inferred)} inferred"
                      f"{source_suffix}", flush=True)
        column_layout_reliable = column_layout_reliable and column_counts_reliable(features)
        for feature in features:
            feature["column_layout_reliable"] = column_layout_reliable
        active = [f for f in features if f["eligible"]]
        outcomes = {}
        parts = components(active) if active else None
        for name, (linkage, weight, cutoff) in METHODS.items():
            if active:
                combined, head, struct = scores(active, weight, parts)
                labels, distance = base.cluster({"combined": combined}, {"combined": 1.0}, linkage, cutoff)
                outcome = metrics(labels, distance, active, head, struct, start_page)
            else:
                outcome = {"cluster_count": 0, "clusters": {}, "representatives": {},
                           "mean_silhouette": None, "mean_within_similarity": None,
                           "mean_within_header_similarity": None,
                           "mean_within_structure_similarity": None,
                           "header_family_agreement": None,
                           "first_five_cluster_coverage": None}
            outcomes[name] = {"linkage": linkage, "header_weight": weight, "minimum_similarity": cutoff,
                              **outcome}
        output = output_root / (result_name or pdf_path.stem)
        output.mkdir(parents=True, exist_ok=True)
        representatives = sorted({page for outcome in outcomes.values() for page in outcome["representatives"].values()})
        if len(representatives) > overlay_limit:
            raise ValueError(f"{len(representatives)} representatives exceed --overlay-limit {overlay_limit}")
        pictures = representatives
        for old in output.glob("page_*_features.png"):
            if old.name not in {f"page_{number:04d}_features.png" for number in pictures}:
                old.unlink()
        for number in pictures:
            base.overlay(pdf.pages[number - 1], features[number - start_page],
                         output / f"page_{number:04d}_features.png")
        result = {
            "version": 2, "algorithm_revision": TABLE_ALGORITHM_REVISION,
            "result_name": result_name or pdf_path.stem,
            "source_pdf": str(pdf_path), "source_total_pages": len(pdf.pages),
            "source_mtime_ns": pdf_path.stat().st_mtime_ns, "page_count": len(numbers),
            "source_start_page": start_page,
            "column_layout_reliable": column_layout_reliable,
            "skipped_pages": [f["page"] for f in features if not f["eligible"]],
            "feature_definitions": {"red": "usable PDF border lines near detected headers, summarized by column positions and row spacing; each table's local column rules also impose a mismatch gate",
                                    "blue": "header text overlap plus header-label geometry",
                                    "green": "word-center row/column alignment below headers, only when usable red borders are absent"},
            "pages": [{"page": f["page"], "red_line_count": len(f["red_lines"]),
                       "green_line_count": len(f["green_lines"]), "blue_line_count": len(f["blue_lines"]),
                       "word_count": f["word_count"], "header_families": f["families"],
                       "table_column_counts": [len(region["positions"]) for region in f["table_columns"]],
                       "has_red_structure": f["has_red"], "eligible": f["eligible"]} for f in features],
            "algorithms": outcomes, "overlays": [f"page_{number:04d}_features.png" for number in pictures],
        }
        (output / "clusters.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result


def write_comparison(reports: list[dict], output_root: Path) -> Path:
    lines = ["# Clustering viewing 2", "", "Deterministic page clusters from blue header labels/geometry and red PDF table borders. A repeated, horizontally aligned header band is recovered when the candidate scanner misses the header or confuses body rows with headers. Green word alignment is inferred only where usable red structure is absent and a header is present.", "",
             "All methods use the same features with different weights and hierarchical linkage. When the header-family scan and column-rule counts are stable, each detected bordered table has an independent normalized column-rule sequence; a strong mismatch in one table caps page similarity even if another table is unchanged. Documents with many competing header families or unstable rule counts use the page-wide score. Complete linkage favors representative clusters; average linkage shows the broader alternative. Missing header or border evidence is not treated as a perfect match.", "",
             "## Corpus summary", "",
             f"{len(reports)} PDFs; {sum(report['page_count'] for report in reports):,} scanned pages; {sum(len(report['skipped_pages']) for report in reports):,} pages without eligible table evidence.", "",
             "| Method | Total clusters | Singleton clusters | Mean first 5 coverage |", "| --- | ---: | ---: | ---: |"]
    for method in METHODS:
        outcomes = [report["algorithms"][method] for report in reports]
        coverage = [outcome["first_five_cluster_coverage"] for outcome in outcomes
                    if outcome["first_five_cluster_coverage"] is not None]
        lines.append(f"| {method} | {sum(outcome['cluster_count'] for outcome in outcomes)} | "
                     f"{sum(len(pages) == 1 for outcome in outcomes for pages in outcome['clusters'].values())} | "
                     f"{sum(coverage) / len(coverage):.3f} |" if coverage else
                     f"| {method} | 0 | 0 | — |")
    empty_documents = [report["result_name"] for report in reports
                       if len(report["skipped_pages"]) == report["page_count"]]
    lines += ["", "**Suggested starting view:** `balanced_complete` uses 60% header and 40% structure similarity with a complete-linkage cutoff of 0.83. It keeps optional columns close while limiting broad merges. The structure-weighted views show possible later merges; the strict view exposes rare header variants.", ""]
    if empty_documents:
        lines.append(f"Documents without eligible table evidence: {', '.join(empty_documents)}.")
        lines.append("")
    lines += ["", "## Document comparison", "",
              "| PDF | Method | Clusters | Representatives | Silhouette | Within header | Within structure | First 5 coverage |", "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for report in reports:
        name = report["result_name"]
        for method, outcome in report["algorithms"].items():
            show = lambda value: "—" if value is None else f"{value:.3f}"
            reps = ", ".join(map(str, outcome["representatives"].values()))
            lines.append(f"| {name} | {method} | {outcome['cluster_count']} | {reps} | {show(outcome['mean_silhouette'])} | {show(outcome['mean_within_header_similarity'])} | {show(outcome['mean_within_structure_similarity'])} | {show(outcome['first_five_cluster_coverage'])} |")
    lines += ["", "## Page assignments", ""]
    for report in reports:
        lines += [f"### {report['result_name']}", ""]
        if report["skipped_pages"]:
            lines.append(f"Pages without usable header or border evidence: {', '.join(map(str, report['skipped_pages']))}.")
        for method, outcome in report["algorithms"].items():
            groups = "; ".join(f"{key}: {','.join(map(str, pages))}" for key, pages in outcome["clusters"].items())
            lines.append(f"- **{method}** — {groups}")
        lines.append("")
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
