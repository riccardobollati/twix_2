"""Cluster pages independently by repeated key/value groups and their geometry."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
from statistics import median

import numpy as np

from . import cluster_pages as base

METHODS = {
    "kv_strict": ("complete", 0.88),
    "kv_balanced": ("complete", 0.77),
    "kv_broad": ("average", 0.72),
}


def normalize(value: str) -> str:
    return " ".join(value.casefold().strip(" :;.,").split())


def value_offset(label: dict) -> tuple[float, float] | None:
    box = label.get("value_bbox")
    if not box:
        return None
    h = max(1.0, label["bottom"] - label["top"])
    return ((box[0] - label["x1"]) / h,
            ((box[1] + box[3]) - label["top"] - label["bottom"]) / (2 * h))


def stable_keys(family: dict, page_count: int) -> set[str]:
    """Require stable labels and a changing value in at least one stable slot."""
    occurrences = family["occurrences"]
    pages = {o["page"] for o in occurrences}
    if len(pages) < max(3, min(10, page_count)):
        return set()
    support: dict[str, set[int]] = defaultdict(set)
    values: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for occurrence in occurrences:
        counts = Counter(normalize(label["key"]) for label in occurrence["labels"])
        # A column label followed by many data rows is a table, not a KV form.
        if len(occurrence["labels"]) > 6 and len(counts) * 2 < len(occurrence["labels"]):
            return set()
        for label in occurrence["labels"]:
            key = normalize(label["key"])
            if key and counts[key] == 1:
                support[key].add(occurrence["page"])
                value = normalize(label.get("value", ""))
                if value and value_offset(label) is not None:
                    values[key][occurrence["page"]].add(value)
    minimum = max(3, math.ceil(0.65 * len(pages)))
    core = {key for key, seen in support.items() if len(seen) >= minimum}
    varying = any(len({value for page_values in values[key].values() for value in page_values}) >= 2
                  and len(values[key]) >= 3 for key in core)
    return core if varying else set()


def table_header_boxes(scan: dict) -> dict[int, list[list[float]]]:
    boxes: dict[int, list[list[float]]] = defaultdict(list)
    for family in scan["families"]:
        if family["kind"] == "table":
            for occurrence in family["occurrences"]:
                boxes[occurrence["page"]].append(occurrence["bbox"])
    return boxes


def overlaps_header(label: dict, boxes: list[list[float]]) -> bool:
    x = (label["x0"] + label["x1"]) / 2
    y = (label["top"] + label["bottom"]) / 2
    return any(left <= x <= right and top <= y <= bottom
               for left, top, right, bottom in boxes)


def group_feature(family_id: str, occurrence: dict, core: set[str],
                  header_boxes: list[list[float]], page_height: float) -> dict | None:
    labels = [label for label in occurrence["labels"]
              if normalize(label["key"]) in core and not overlaps_header(label, header_boxes)]
    if not labels or len({normalize(label["key"]) for label in labels}) != len(labels):
        return None
    if len(labels) < 2 and len(core) > 1:
        return None
    if len(core) <= 2 and (occurrence["bbox"][1] < 0.045 * page_height
                           or occurrence["bbox"][3] > 0.955 * page_height):
        # Two short fields at a page edge are usually running metadata.
        return None
    h = median(max(1.0, label["bottom"] - label["top"]) for label in labels)
    x0 = min(label["x0"] for label in labels)
    y0 = min(label["top"] for label in labels)
    anchors = {}
    for label in labels:
        key = normalize(label["key"])
        offset = value_offset(label)
        anchors[key] = {
            "position": ((label["x0"] - x0) / h, (label["top"] - y0) / h),
            "value_offset": offset,
        }
    boxes = [label.get("value_bbox") for label in labels if label.get("value_bbox")]
    all_boxes = [[label["x0"], label["top"], label["x1"], label["bottom"]] for label in labels] + boxes
    return {"family": family_id, "page": occurrence["page"], "keys": sorted(anchors),
            "anchors": anchors, "labels": labels,
            "bbox": [min(box[0] for box in all_boxes), min(box[1] for box in all_boxes),
                     max(box[2] for box in all_boxes), max(box[3] for box in all_boxes)]}


def extract_groups(scan: dict, page_sizes: dict[int, tuple[float, float]]) -> tuple[dict[int, list[dict]], list[dict]]:
    by_page: dict[int, list[dict]] = defaultdict(list)
    accepted = []
    headers = table_header_boxes(scan)
    page_count = len(page_sizes)
    for family in scan["families"]:
        if family["kind"] != "kv":
            continue
        core = stable_keys(family, page_count)
        if not core:
            continue
        occurrences = [feature for occurrence in family["occurrences"]
                       if (feature := group_feature(family["id"], occurrence, core,
                                                    headers[occurrence["page"]],
                                                    page_sizes[occurrence["page"]][1])) is not None]
        if len({o["page"] for o in occurrences}) < max(3, min(10, page_count)):
            continue
        accepted.append({"id": family["id"], "core_keys": sorted(core),
                         "pages": sorted({o["page"] for o in occurrences})})
        for feature in occurrences:
            by_page[feature["page"]].append(feature)
    for groups in by_page.values():
        groups.sort(key=lambda group: (group["bbox"][1], group["bbox"][0], group["family"]))
    return by_page, accepted


def reject_tabular_rows(pdf, by_page: dict[int, list[dict]], families: list[dict]) -> tuple[dict[int, list[dict]], list[dict]]:
    """Reject scanned KV families whose sampled keys sit inside numeric data rows."""
    words_by_page = {}
    rejected = set()
    for family in families:
        samples = [(page, group) for page, groups in sorted(by_page.items())
                   for group in groups if group["family"] == family["id"]][:3]
        row_votes = []
        for page_number, group in samples:
            if page_number not in words_by_page:
                words_by_page[page_number] = pdf.pages[page_number - 1].extract_words()
            page = pdf.pages[page_number - 1]
            label = group["labels"][0]
            height = label["bottom"] - label["top"]
            center = (label["top"] + label["bottom"]) / 2
            row = [word for word in words_by_page[page_number]
                   if abs((word["top"] + word["bottom"]) / 2 - center) <= max(3, 0.55 * height)]
            numeric = sum(bool(re.search(r"\d", word["text"])) for word in row)
            span = (max(word["x1"] for word in row) - min(word["x0"] for word in row)) if row else 0
            row_votes.append(len(row) >= 8 and numeric >= 5 and numeric / len(row) >= 0.45
                             and span >= 0.40 * page.width)
        if row_votes and sum(row_votes) >= math.ceil(len(row_votes) / 2):
            rejected.add(family["id"])
    kept = {page: [group for group in groups if group["family"] not in rejected]
            for page, groups in by_page.items()}
    return {page: groups for page, groups in kept.items() if groups}, [f for f in families if f["id"] not in rejected]


def group_similarity(a: dict, b: dict) -> float:
    left, right = set(a["keys"]), set(b["keys"])
    shared = sorted(left & right)
    if not shared:
        return 0.0
    text = 0.65 * len(shared) / min(len(left), len(right)) + 0.35 * len(shared) / max(len(left), len(right))
    if len(shared) == 1 and max(len(left), len(right)) > 2:
        return 0.0
    residuals = []
    reference = shared[0]
    for key in shared:
        first, second = a["anchors"][key], b["anchors"][key]
        if len(shared) > 1:
            relative_a = [x - y for x, y in zip(first["position"], a["anchors"][reference]["position"])]
            relative_b = [x - y for x, y in zip(second["position"], b["anchors"][reference]["position"])]
            residuals.append(max(abs(x - y) for x, y in zip(relative_a, relative_b)))
        if first["value_offset"] is not None and second["value_offset"] is not None:
            residuals.append(max(abs(x - y) for x, y in zip(first["value_offset"], second["value_offset"])))
        elif (first["value_offset"] is None) != (second["value_offset"] is None):
            residuals.append(2.0)
    geometry = math.exp(-median(residuals) / 2.5) if residuals else 0.0
    return 0.65 * text + 0.35 * geometry


def page_similarity(a: list[dict], b: list[dict]) -> float:
    """Match KV groups one-to-one; an optional extra group incurs a mild cost."""
    if not a or not b:
        return 0.0
    candidates = sorted(((group_similarity(left, right), i, j)
                         for i, left in enumerate(a) for j, right in enumerate(b)), reverse=True)
    used_left, used_right = set(), set()
    total = 0.0
    for score, i, j in candidates:
        if score < 0.35:
            break
        if i not in used_left and j not in used_right:
            used_left.add(i)
            used_right.add(j)
            total += score
    return (0.65 * total / min(len(a), len(b)) + 0.35 * total / max(len(a), len(b)))


def cluster_pages(by_page: dict[int, list[dict]], page_count: int, first_page: int = 1) -> dict:
    active = sorted(by_page)
    n = len(active)
    similarity = np.eye(n)
    for i, page_a in enumerate(active):
        for j in range(i + 1, n):
            score = page_similarity(by_page[page_a], by_page[active[j]])
            similarity[i, j] = similarity[j, i] = score
    outcomes = {}
    for name, (linkage, threshold) in METHODS.items():
        if not active:
            outcomes[name] = {"linkage": linkage, "minimum_similarity": threshold,
                              "cluster_count": 0, "clusters": {}, "representatives": {},
                              "mean_within_similarity": None, "first_five_cluster_coverage": None}
            continue
        labels, distance = base.cluster({"kv": similarity}, {"kv": 1.0}, linkage, threshold)
        clusters: dict[int, list[int]] = defaultdict(list)
        for page, label in zip(active, labels):
            clusters[label].append(page)
        representatives = {}
        within = []
        index = {page: i for i, page in enumerate(active)}
        for label, pages in clusters.items():
            representatives[str(label)] = min(pages, key=lambda page: (
                sum(distance[index[page], index[other]] for other in pages), page))
            within.extend(1 - distance[index[a], index[b]]
                          for pos, a in enumerate(pages) for b in pages[pos + 1:])
        outcomes[name] = {"linkage": linkage, "minimum_similarity": threshold,
                          "cluster_count": len(clusters),
                          "clusters": {str(key): pages for key, pages in clusters.items()},
                          "representatives": representatives,
                          "mean_within_similarity": round(float(np.mean(within)), 4) if within else None,
                          "first_five_cluster_coverage": len({labels[i] for i, page in enumerate(active)
                                                              if first_page <= page < first_page + 5}) / len(clusters)}
    return outcomes


def overlay(page, groups: list[dict], path: Path, dpi: int = 110) -> None:
    from PIL import Image, ImageDraw

    original = page.to_image(resolution=dpi).original.convert("RGBA")
    rendered = Image.alpha_composite(original, Image.new("RGBA", original.size, (110, 110, 110, 65)))
    draw = ImageDraw.Draw(rendered)
    sx, sy = rendered.width / page.width, rendered.height / page.height
    for group in groups:
        for label in group["labels"]:
            key = (label["x0"] * sx, label["top"] * sy, label["x1"] * sx, label["bottom"] * sy)
            draw.rectangle(key, outline=(15, 80, 235, 255), width=3)
            if box := label.get("value_bbox"):
                draw.rectangle((box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy),
                               outline=(0, 155, 75, 255), width=3)
    rendered.convert("RGB").save(path)
    original.close()
    rendered.close()


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
        page_sizes = {number: (pdf.pages[number - 1].width, pdf.pages[number - 1].height)
                      for number in numbers}
        by_page, families = extract_groups(scan, page_sizes)
        by_page, families = reject_tabular_rows(pdf, by_page, families)
        outcomes = cluster_pages(by_page, len(numbers), start_page)
        output = output_root / (result_name or pdf_path.stem)
        output.mkdir(parents=True, exist_ok=True)
        representatives = sorted({page for outcome in outcomes.values() for page in outcome["representatives"].values()})
        if len(representatives) > overlay_limit:
            raise ValueError(f"{len(representatives)} KV representatives exceed --overlay-limit {overlay_limit}")
        wanted = {f"kv_page_{page:04d}.png" for page in representatives}
        for old in output.glob("kv_page_*.png"):
            if old.name not in wanted:
                old.unlink()
        for page in representatives:
            overlay(pdf.pages[page - 1], by_page[page], output / f"kv_page_{page:04d}.png")
        result = {
            "version": 1, "kind": "kv", "result_name": result_name or pdf_path.stem,
            "source_pdf": str(pdf_path), "source_total_pages": len(pdf.pages),
            "source_mtime_ns": pdf_path.stat().st_mtime_ns, "page_count": len(numbers),
            "source_start_page": start_page,
            "families": families, "skipped_pages": [page for page in numbers if page not in by_page],
            "pages": [{"page": page, "eligible": page in by_page,
                       "groups": [{"family": group["family"], "keys": group["keys"],
                                   "bbox": group["bbox"],
                                   "pairs": [{"key": label["key"], "value": label.get("value", "")}
                                             for label in group["labels"]]}
                                  for group in by_page.get(page, [])]}
                      for page in numbers],
            "algorithms": outcomes, "overlays": sorted(wanted),
        }
        (output / "kv_clusters.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
