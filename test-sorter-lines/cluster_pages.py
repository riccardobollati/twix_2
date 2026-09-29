"""Compare deterministic page clusters using table borders, text rows, and headers.

Examples:
    env/bin/python test-sorter-lines/cluster_pages.py id_05 new_04
    env/bin/python test-sorter-lines/cluster_pages.py data/id_05/id_05.pdf --max-pages 20

This is an exploratory, pre-template stage. It never calls a model. The source
PDF is authoritative; existing scan.json/lines.json files are reused only when
their source and page coverage match, otherwise their features are rebuilt in
memory. Results are written under test-sorter-lines/clustering_results/.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import math
from pathlib import Path
import re
from statistics import median

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from detect_lines import raster_lines, vector_lines


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GRID_X, GRID_Y = 20, 16


def load_scanner():
    spec = importlib.util.spec_from_file_location("scan_structure_for_clustering", ROOT / "test" / "scan_structure.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_pdf(value: str) -> Path:
    direct = Path(value).expanduser()
    if direct.is_file():
        return direct.resolve()
    dataset = ROOT / "data" / value / f"{value}.pdf"
    if dataset.is_file():
        return dataset.resolve()
    raise ValueError(f"PDF or dataset not found: {value}")


def scan_report(pdf_path: Path, numbers: list[int], cache_root: Path | None,
                result_name: str | None = None) -> dict:
    if cache_root is not None:
        cached = cache_root / (result_name or pdf_path.stem) / "scan.json"
        if cached.is_file() and cached.stat().st_mtime_ns >= pdf_path.stat().st_mtime_ns:
            report = json.loads(cached.read_text(encoding="utf-8"))
            if report.get("source_pdf") == str(pdf_path) and report.get("pages") == numbers:
                return report
    scanner = load_scanner()
    pages, _ = scanner.load_pdf(pdf_path, numbers[0], numbers[-1])
    return scanner.scan_pages(pages)


def table_headers(report: dict) -> dict[int, list[dict]]:
    by_page: dict[int, list[dict]] = defaultdict(list)
    for family in report["families"]:
        if family["kind"] != "table":
            continue
        for occurrence in family["occurrences"]:
            by_page[occurrence["page"]].append({"family": family["id"], **occurrence})
    for headers in by_page.values():
        headers.sort(key=lambda h: (h["bbox"][1], h["bbox"][0]))
    return by_page


def horizontal(line: dict) -> bool:
    return abs(line["bottom"] - line["top"]) <= 2 and abs(line["x1"] - line["x0"]) >= 5


def vertical(line: dict) -> bool:
    return abs(line["x1"] - line["x0"]) <= 2 and abs(line["bottom"] - line["top"]) >= 5


def table_regions(headers: list[dict], lines: list[dict], width: float, height: float) -> list[tuple[float, float, float, float]]:
    """Header band plus body until the next overlapping header or bottom rule.

    A red rule directly below the last populated body row closes a region;
    repeated adjacent tables are therefore kept separate when possible.
    """
    if not headers:
        return [(0.0, 0.0, width, height)]
    regions = []
    hlines = [line for line in lines if horizontal(line)]
    for header in headers:
        left, top, right, bottom = header["bbox"]
        margin = max(5, 0.015 * width)
        left, right = max(0, left - margin), min(width, right + margin)
        next_top = min((other["bbox"][1] for other in headers
                        if other["bbox"][1] > bottom + 2
                        and min(right, other["bbox"][2]) - max(left, other["bbox"][0]) > 0.4 * (right - left)),
                       default=height)
        # The first horizontal rule after a large row gap often bounds the
        # table. Keep the nearest header boundary as the stronger limit.
        last = min(height, next_top)
        candidates = sorted(line["top"] for line in hlines
                            if bottom + 8 < line["top"] <= last
                            and min(right, line["x1"]) - max(left, line["x0"]) > 0.55 * (right - left))
        if candidates:
            gap = max(28, 0.055 * height)
            boundary = candidates[0]
            for y in candidates[1:]:
                if y - boundary > gap:
                    break
                boundary = y
            last = min(last, boundary + 2)
        regions.append((left, top, right, max(bottom + 8, last)))
    return regions


def green_lines(words: list[dict], regions: list[tuple[float, float, float, float]],
                headers: list[dict], width: float, height: float) -> list[dict]:
    """Connect aligned *word centers* within candidate table bodies.

    Distinct y levels create horizontal lines, including multiple lines from a
    multiline cell. Repeated x centers across distinct rows create columns.
    """
    result = []
    forbidden = [(h["bbox"][1] - 1, h["bbox"][3] + 1, h["bbox"][0], h["bbox"][2]) for h in headers]
    for left, top, right, bottom in regions:
        centers = []
        for word in words:
            x = (word["x0"] + word["x1"]) / 2
            y = (word["top"] + word["bottom"]) / 2
            if not (left <= x <= right and top <= y <= bottom):
                continue
            if any(a <= y <= b and c - 2 <= x <= d + 2 for a, b, c, d in forbidden):
                continue
            centers.append((x, y))
        if not centers:
            continue
        row_tol = max(2.0, 0.003 * height)
        col_tol = max(2.5, 0.004 * width)
        rows = group_centers(centers, axis=1, tolerance=row_tol)
        for row in rows:
            xs = [point[0] for point in row]
            if len(row) >= 3 and max(xs) - min(xs) >= 0.16 * (right - left):
                y = median(point[1] for point in row)
                result.append(dict(x0=min(xs), top=y, x1=max(xs), bottom=y, source="word_row"))
        cols = group_centers(centers, axis=0, tolerance=col_tol)
        for col in cols:
            ys = sorted(point[1] for point in col)
            distinct = [ys[0]]
            for y in ys[1:]:
                if y - distinct[-1] > row_tol:
                    distinct.append(y)
            if len(distinct) >= 3 and distinct[-1] - distinct[0] >= 0.035 * height:
                x = median(point[0] for point in col)
                result.append(dict(x0=x, top=distinct[0], x1=x, bottom=distinct[-1], source="word_column"))
    return result


def group_centers(points: list[tuple[float, float]], axis: int, tolerance: float) -> list[list[tuple[float, float]]]:
    groups: list[list[tuple[float, float]]] = []
    for point in sorted(points, key=lambda p: (p[axis], p[1 - axis])):
        if not groups or point[axis] - median(p[axis] for p in groups[-1]) > tolerance:
            groups.append([point])
        else:
            groups[-1].append(point)
    return groups


def raster_signature(lines: list[dict], width: float, height: float) -> np.ndarray:
    """Length-weighted, coarse page geometry; horizontal/vertical kept apart."""
    grid = np.zeros((2, GRID_Y, GRID_X), dtype=np.float64)
    for line in lines:
        if horizontal(line):
            channel = 0
        elif vertical(line):
            channel = 1
        else:
            continue
        x0, x1 = sorted((line["x0"] / width, line["x1"] / width))
        y0, y1 = sorted((line["top"] / height, line["bottom"] / height))
        if channel == 0:
            row = min(GRID_Y - 1, max(0, int((y0 + y1) / 2 * GRID_Y)))
            for col in range(max(0, int(x0 * GRID_X)), min(GRID_X, math.ceil(x1 * GRID_X))):
                grid[channel, row, col] += max(0, min(x1, (col + 1) / GRID_X) - max(x0, col / GRID_X))
        else:
            col = min(GRID_X - 1, max(0, int((x0 + x1) / 2 * GRID_X)))
            for row in range(max(0, int(y0 * GRID_Y)), min(GRID_Y, math.ceil(y1 * GRID_Y))):
                grid[channel, row, col] += max(0, min(y1, (row + 1) / GRID_Y) - max(y0, row / GRID_Y))
    return grid.flatten()


def header_signature(headers: list[dict]) -> Counter[str]:
    result: Counter[str] = Counter()
    for header in headers:
        for label in header["labels"]:
            key = re.sub(r"\s+", " ", label["text"].casefold()).strip(" :;.,")
            if key:
                result[key] = 1  # body repetition and table count do not dominate
    return result


def blue_lines(headers: list[dict]) -> list[dict]:
    """Join centers of repeated header labels on each scanner row."""
    result = []
    for header in headers:
        rows: dict[int, list[dict]] = defaultdict(list)
        for label in header["labels"]:
            rows[int(label.get("line", 0))].append(label)
        for labels in rows.values():
            if len(labels) < 2:
                continue
            ordered = sorted(labels, key=lambda item: item["x0"])
            y = median((label["top"] + label["bottom"]) / 2 for label in ordered)
            centers = [(label["x0"] + label["x1"]) / 2 for label in ordered]
            for left, right in zip(centers, centers[1:]):
                result.append(dict(x0=left, top=y, x1=right, bottom=y, source="header_centers"))
    return result


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    product = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / product) if product else float(not np.any(a) and not np.any(b))


def jaccard(a: Counter[str], b: Counter[str]) -> float:
    keys = a.keys() | b.keys()
    return sum(min(a[k], b[k]) for k in keys) / sum(max(a[k], b[k]) for k in keys) if keys else 1.0


def similarity_matrices(features: list[dict]) -> dict[str, np.ndarray]:
    n = len(features)
    matrices = {name: np.eye(n) for name in ("red", "green", "blue")}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = features[i], features[j]
            values = (cosine(a["red"], b["red"]),
                      cosine(a["green"], b["green"]),
                      0.85 * jaccard(a["blue"], b["blue"])
                      + 0.15 * cosine(a["blue_geometry"], b["blue_geometry"]))
            for name, value in zip(matrices, values):
                matrices[name][i, j] = matrices[name][j, i] = value
    return matrices


CONFIGS = {
    "tight_complete": ("complete", {"red": 0.35, "green": 0.25, "blue": 0.40}, 0.82),
    "tight_average": ("average", {"red": 0.35, "green": 0.25, "blue": 0.40}, 0.82),
    "structure_complete": ("complete", {"red": 0.58, "green": 0.37, "blue": 0.05}, 0.80),
    "structure_average": ("average", {"red": 0.58, "green": 0.37, "blue": 0.05}, 0.80),
    "structure_single": ("single", {"red": 0.58, "green": 0.37, "blue": 0.05}, 0.80),
}


def cluster(similarities: dict[str, np.ndarray], weights: dict[str, float], method: str,
            threshold: float) -> tuple[list[int], np.ndarray]:
    combined = sum(weight * similarities[name] for name, weight in weights.items())
    distance = np.clip(1 - combined, 0, 1)
    np.fill_diagonal(distance, 0)
    if len(distance) == 1:
        return [1], distance
    tree = linkage(squareform(distance, checks=False), method=method, optimal_ordering=False)
    labels = list(map(int, fcluster(tree, t=1 - threshold + 1e-9, criterion="distance")))
    # Make IDs stable in page order, regardless of SciPy's tree numbering.
    rename = {label: index + 1 for index, label in enumerate(dict.fromkeys(labels))}
    return [rename[label] for label in labels], distance


def metrics(labels: list[int], distance: np.ndarray, similarities: dict[str, np.ndarray],
            features: list[dict]) -> dict:
    clusters: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        clusters[label].append(index)
    within = [(i, j) for i in range(len(labels)) for j in range(i + 1, len(labels)) if labels[i] == labels[j]]
    between = [(i, j) for i in range(len(labels)) for j in range(i + 1, len(labels)) if labels[i] != labels[j]]
    silhouettes = []
    if len(clusters) > 1:
        for i, label in enumerate(labels):
            own = [j for j in clusters[label] if j != i]
            if not own:
                silhouettes.append(0.0)
                continue
            a = float(np.mean([distance[i, j] for j in own]))
            b = min(float(np.mean([distance[i, j] for j in others]))
                    for other_label, others in clusters.items() if other_label != label)
            silhouettes.append((b - a) / max(a, b) if max(a, b) else 0.0)
    medoids = {}
    for label, indices in clusters.items():
        medoid = min(indices, key=lambda i: (sum(distance[i, j] for j in indices), features[i]["page"]))
        medoids[label] = features[medoid]["page"]
    family_sets = [set(f["families"]) for f in features]
    return {
        "cluster_count": len(clusters),
        "clusters": {str(label): [features[i]["page"] for i in indices] for label, indices in clusters.items()},
        "representatives": {str(k): v for k, v in medoids.items()},
        "mean_silhouette": round(float(np.mean(silhouettes)), 4) if silhouettes else None,
        "mean_within_similarity": round(float(np.mean([1 - distance[i, j] for i, j in within])), 4) if within else None,
        "max_between_similarity": round(max((1 - distance[i, j] for i, j in between), default=0), 4) if between else None,
        "mean_within_structure_similarity": round(float(np.mean([0.58 * similarities["red"][i, j]
                                                                    + 0.37 * similarities["green"][i, j]
                                                                    + 0.05 * similarities["blue"][i, j]
                                                                    for i, j in within])), 4) if within else None,
        "header_family_agreement": round(sum(family_sets[i] == family_sets[j] for i, j in within) / len(within), 4) if within else None,
        "first_five_cluster_coverage": len({labels[i] for i, feature in enumerate(features)
                                             if feature["page"] <= 5}) / len(clusters),
    }


def overlay(page, feature: dict, path: Path, dpi: int = 110) -> None:
    from PIL import Image, ImageDraw
    original = page.to_image(resolution=dpi).original.convert("RGBA")
    veil = Image.new("RGBA", original.size, (110, 110, 110, 75))
    rendered = Image.alpha_composite(original, veil)
    draw = ImageDraw.Draw(rendered)
    sx, sy = rendered.width / page.width, rendered.height / page.height
    for name, color, line_width in (("red_lines", (230, 35, 35, 255), 3),
                                    ("green_lines", (0, 155, 75, 255), 2),
                                    ("blue_lines", (15, 80, 235, 255), 2)):
        for line in feature[name]:
            draw.line((line["x0"] * sx, line["top"] * sy, line["x1"] * sx, line["bottom"] * sy),
                      fill=color, width=line_width)
    for header in feature["headers"]:
        for label in header["labels"]:
            x = (label["x0"] + label["x1"]) / 2
            y = (label["top"] + label["bottom"]) / 2
            draw.ellipse((x * sx - 3, y * sy - 3, x * sx + 3, y * sy + 3), fill=(15, 80, 235, 255))
        x0, top, x1, bottom = header["bbox"]
        draw.rectangle((x0 * sx, top * sy, x1 * sx, bottom * sy), outline=(15, 80, 235, 255), width=2)
    rendered.convert("RGB").save(path)
    original.close()
    rendered.close()


def process(pdf_path: Path, max_pages: int, output_root: Path, scan_root: Path | None,
            overlay_limit: int, result_name: str | None = None,
            representatives_only: bool = False) -> dict:
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        numbers = list(range(1, min(len(pdf.pages), max_pages) + 1))
        report = scan_report(pdf_path, numbers, scan_root, result_name)
        headers_by_page = table_headers(report)
        features = []
        for number in numbers:
            page = pdf.pages[number - 1]
            words = [{k: word[k] for k in ("text", "x0", "top", "x1", "bottom")}
                     for word in page.dedupe_chars().extract_words()
                     if word["text"].strip() and word["x1"] > word["x0"] and word["bottom"] > word["top"]]
            red = vector_lines(page, min_length=12)
            if not red:
                image = page.to_image(resolution=110).original
                red = raster_lines(image, page.width, page.height, min_length=12)
                image.close()
            headers = headers_by_page.get(number, [])
            regions = table_regions(headers, red, page.width, page.height)
            green = green_lines(words, regions, headers, page.width, page.height)
            blue = blue_lines(headers)
            features.append(dict(page=number, red=raster_signature(red, page.width, page.height),
                                 green=raster_signature(green, page.width, page.height),
                                 blue=header_signature(headers),
                                 blue_geometry=raster_signature(blue, page.width, page.height),
                                 families=sorted({h["family"] for h in headers}),
                                 headers=headers, red_lines=red, green_lines=green, blue_lines=blue,
                                 regions=regions, word_count=len(words),
                                 eligible=bool(words and (red or green or headers))))
            print(f"  page {number}/{len(numbers)}: {len(red)} red, {len(green)} green, {len(blue)} blue segments", flush=True)

        active = [feature for feature in features if feature["eligible"]]
        if not active:
            raise ValueError(f"no pages with table-structure evidence in {pdf_path}")
        similarities = similarity_matrices(active)
        outcomes = {}
        for name, (method, weights, threshold) in CONFIGS.items():
            labels, distance = cluster(similarities, weights, method, threshold)
            outcomes[name] = {"linkage": method, "weights": weights, "minimum_similarity": threshold,
                              **metrics(labels, distance, similarities, active)}
        output = output_root / (result_name or pdf_path.stem)
        output.mkdir(parents=True, exist_ok=True)
        representative_pages = {page for outcome in outcomes.values() for page in outcome["representatives"].values()}
        active_numbers = [feature["page"] for feature in active]
        changes = {active_numbers[i] for outcome in outcomes.values() for i in range(1, len(active_numbers))
                   if any(active_numbers[i] in group and active_numbers[i - 1] not in group
                          for group in outcome["clusters"].values())}
        pictures = sorted(representative_pages)
        if not representatives_only:
            pictures += [number for number in sorted(changes | set(active_numbers[:1]))
                         if number not in representative_pages]
        pictures = pictures[:overlay_limit]
        current_pictures = {f"page_{number:04d}_features.png" for number in pictures}
        for old in output.glob("page_*_features.png"):
            if old.name not in current_pictures and re.fullmatch(r"page_\d+_features\.png", old.name):
                old.unlink()
        for number in pictures:
            overlay(pdf.pages[number - 1], features[number - 1], output / f"page_{number:04d}_features.png")
        result = {
            "version": 1,
            "result_name": result_name or pdf_path.stem,
            "source_pdf": str(pdf_path), "source_total_pages": len(pdf.pages),
            "source_mtime_ns": pdf_path.stat().st_mtime_ns,
            "page_count": len(numbers),
            "skipped_pages": [feature["page"] for feature in features if not feature["eligible"]],
            "feature_definitions": {"red": "length-weighted PDF border segments in normalized 20x16 page grid, split by orientation",
                                    "green": "length-weighted aligned word-center rows and columns within header-seeded table regions",
                                    "blue": "85% repeated-header label Jaccard and 15% aligned header-center line geometry"},
            "pages": [{"page": f["page"], "red_line_count": len(f["red_lines"]),
                       "green_line_count": len(f["green_lines"]), "blue_line_count": len(f["blue_lines"]),
                       "word_count": f["word_count"],
                       "header_families": f["families"], "regions": f["regions"],
                       "eligible": f["eligible"]} for f in features],
            "algorithms": outcomes,
            "overlays": [f"page_{number:04d}_features.png" for number in pictures],
        }
        (output / "clusters.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result


def write_comparison(reports: list[dict], output_root: Path) -> Path:
    lines = ["# Deterministic page clustering comparison", "",
             "Red = PDF table lines. Green = aligned word-center rows and columns inside candidate table regions. Blue = lines through repeated header-label centers plus their text from the pre-template scan.", "",
             "Each run uses complete, average, or single hierarchical linkage with a fixed similarity cutoff. The same page features are used by all methods; only channel weights and linkage change. Cluster IDs follow page order.", "",
             "| PDF | Method | Clusters | Representative pages | Silhouette | Within similarity | Within structure similarity | Header-family agreement | First 5 coverage |", "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for report in reports:
        document = report.get("result_name", Path(report["source_pdf"]).stem)
        for name, outcome in report["algorithms"].items():
            show = lambda value: "—" if value is None else f"{value:.3f}"
            reps = ", ".join(str(p) for p in outcome["representatives"].values())
            lines.append(f"| {document} | {name} | {outcome['cluster_count']} | {reps} | "
                         f"{show(outcome['mean_silhouette'])} | {show(outcome['mean_within_similarity'])} | "
                         f"{show(outcome['mean_within_structure_similarity'])} | "
                         f"{show(outcome['header_family_agreement'])} | "
                         f"{show(outcome['first_five_cluster_coverage'])} |")
    lines += ["", "## Sampling implications", ""]
    for report in reports:
        document = report.get("result_name", Path(report["source_pdf"]).stem)
        tight = report["algorithms"]["tight_complete"]
        structural = report["algorithms"]["structure_single"]
        uncovered = tight["cluster_count"] - len({key for key, pages in tight["clusters"].items()
                                                  if any(page <= 5 for page in pages)})
        group_word = "group" if tight["cluster_count"] == 1 else "groups"
        broad_word = "group" if structural["cluster_count"] == 1 else "groups"
        coverage = ("all represented in pages 1–5" if uncovered == 0
                    else f"{uncovered} absent from pages 1–5")
        statement = (f"**{document}:** tight complete finds {tight['cluster_count']} page {group_word}, "
                     f"{coverage}. Representative pages: "
                     f"{', '.join(map(str, tight['representatives'].values()))}. "
                     f"Structure single finds {structural['cluster_count']} broader {broad_word}.")
        if report["skipped_pages"]:
            page_word = "Page" if len(report["skipped_pages"]) == 1 else "Pages"
            verb = "has" if len(report["skipped_pages"]) == 1 else "have"
            statement += f" {page_word} {', '.join(map(str, report['skipped_pages']))} {verb} no table evidence."
        lines.append(f"- {statement}")
    lines += ["", "Use the strict representatives to expose header variants when selecting template pages. Inspect groups that collapse under structure weighting for optional columns or changes in row count. Review sparse pages separately before adding them to template inference.", ""]
    lines += ["## Page assignments", ""]
    for report in reports:
        document = report.get("result_name", Path(report["source_pdf"]).stem)
        lines.append(f"### {document}")
        lines.append("")
        if report["skipped_pages"]:
            lines.append(f"Pages without table evidence, excluded from clustering: {', '.join(map(str, report['skipped_pages']))}.")
            lines.append("")
        for name, outcome in report["algorithms"].items():
            groups = "; ".join(f"{key}: {','.join(map(str, pages))}" for key, pages in outcome["clusters"].items())
            lines.append(f"- **{name}** — {groups}")
        lines += ["", f"Feature overlays: `{document}/page_####_features.png`; machine-readable data: `{document}/clusters.json`.", ""]
    lines += ["## Reading the metrics", "",
              "Silhouette uses each method's own distance; it is unavailable for a single cluster. Within structure similarity always uses the structure weighting, even for tight methods. Header-family agreement compares scanner IDs within clusters and is a diagnostic, not independent ground truth. First 5 coverage is the fraction of discovered clusters touched by pages 1–5.", "",
              "These are page-level groups, not extracted table instances. Border detection can include diagrams and page furniture; a missed header weakens the blue feature. Representative pages are medoids of the observed sample, so rare optional columns or later page formats still need explicit review.", ""]
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("documents", nargs="+", help="PDF paths or data/<name>/<name>.pdf dataset names")
    parser.add_argument("--max-pages", type=int, default=500, help="maximum pages per document (default: 500)")
    parser.add_argument("--output-root", type=Path, default=HERE / "clustering_results")
    parser.add_argument("--scan-root", type=Path, default=ROOT / "test" / "scan_results",
                        help="reuse matching pre-template scan reports here")
    parser.add_argument("--overlay-limit", type=int, default=12, help="maximum representative/transition PNGs per PDF")
    parser.add_argument("--representatives-only", action="store_true",
                        help="render cluster medoid overlays only")
    parser.add_argument("--result-name", help="output directory name for a single document")
    args = parser.parse_args()
    if not 1 <= args.max_pages <= 500 or args.overlay_limit < 0:
        parser.error("max-pages must be in 1..500 and overlay-limit nonnegative")
    if args.result_name and (len(args.documents) != 1 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.result_name)
                             or args.result_name in {".", ".."}):
        parser.error("--result-name requires one document and a safe folder name")
    paths = [resolve_pdf(value) for value in args.documents]
    if len({path.stem for path in paths}) != len(paths):
        parser.error("input PDF basenames must be distinct")
    reports = []
    for path in paths:
        print(f"Clustering {path.name}", flush=True)
        reports.append(process(path, args.max_pages, args.output_root, args.scan_root,
                               args.overlay_limit, args.result_name, args.representatives_only))
    print(f"Saved {write_comparison(reports, args.output_root)}", flush=True)


if __name__ == "__main__":
    main()
