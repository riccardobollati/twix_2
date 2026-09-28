"""Filter scanned table/KV groups by frequency, then review table headers with Laya.

Run from the repository root:
    env/bin/python test/purge_structure.py id_05

The crop PNG is for inspection. Laya's API accepts text, not images: the model
receives every OCR word intersecting the crop, with its original PDF word box.
This standalone experiment does not modify the TWIX extraction graph.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
import json
import math
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request

if __package__:
    from .scan_structure import ROOT, Rules, make_rows, plot_results
else:
    from scan_structure import ROOT, Rules, make_rows, plot_results


DEFAULT_URL = "http://127.0.0.1:8770/api/predict"
DEFAULT_MIN_OCCURRENCES = 10
QUESTION_ID = "legitimate_table_header"
QUESTION = (
    "Is the marked TARGET a genuine table column header? Answer yes only when "
    "its text labels identify multiple horizontally aligned columns and nearby "
    "words below form table data under those columns. Answer no for prose, data "
    "rows, key-value pairs, captions, page titles, or repeated page furniture. "
    "Use the original PDF word coordinates to judge the layout."
)
COORDINATE_PRECISION = 0  # whole PDF points keep the layout while fitting more words
MODEL_CONTEXT_LIMITS = {"english": 512, "multilingual": 1024, "typed-decisions": 1024}


def resolve_scan_folder(value: str) -> Path:
    direct = Path(value).expanduser()
    folder = direct if direct.is_dir() else ROOT / "test" / "scan_results" / value
    folder = folder.resolve()
    if not (folder / "scan.json").is_file():
        raise ValueError(f"scan.json not found in {folder}")
    return folder


def expanded_bbox(header: list[float], width: float, height: float,
                  margin_fraction: float = 0.10) -> list[float]:
    """Expand each side by a fraction of the physical PDF page size."""
    if width <= 0 or height <= 0 or margin_fraction < 0:
        raise ValueError("page dimensions must be positive and margin nonnegative")
    x0, top, x1, bottom = header
    return [max(0.0, x0 - margin_fraction * width),
            max(0.0, top - margin_fraction * height),
            min(width, x1 + margin_fraction * width),
            min(height, bottom + margin_fraction * height)]


def word_in_region(word: dict, region: list[float]) -> bool:
    return (word["x1"] > region[0] and word["x0"] < region[2]
            and word["bottom"] > region[1] and word["top"] < region[3])


def _ocr_word(row: dict) -> dict:
    return {"id": row.get("id"), "text": row["text"],
            **{key: float(row[key]) for key in ("x0", "top", "x1", "bottom")}}


def load_document_words(pdf_path: Path, page_numbers: set[int]) -> tuple[dict[int, list[dict]], dict[int, str]]:
    """Prefer saved full-document OCR; extract missing pages from the source PDF."""
    import pdfplumber

    if not page_numbers:
        return {}, {}
    selected: dict[int, list[dict]] = defaultdict(list)
    sources: dict[int, str] = {}
    csv_path = pdf_path.parent / "ocr_extraction_full.csv"
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"page", "text", "x0", "top", "x1", "bottom"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"OCR CSV has missing columns: {csv_path}")
            for row in reader:
                number = int(row["page"])
                if number in page_numbers and row["text"].strip():
                    selected[number].append(_ocr_word(row))
                    sources[number] = str(csv_path)
    missing = page_numbers - set(sources)
    if missing:
        with pdfplumber.open(pdf_path) as pdf:
            for number in sorted(missing):
                page = pdf.pages[number - 1]
                selected[number] = [_ocr_word(word) for word in page.dedupe_chars().extract_words()
                                    if word["text"].strip()]
                sources[number] = str(pdf_path)
                page.close()
    return selected, sources


def _coord(value: float) -> str:
    return f"{value:.{COORDINATE_PRECISION}f}"


def _box(values: list[float] | tuple[float, ...]) -> str:
    return ",".join(_coord(value) for value in values)


def build_state(occurrence: dict, words: list[dict], region: list[float],
                width: float, height: float) -> str:
    """Put the target first so it survives any model-side context truncation."""
    lines = ["PDF points, page origin top-left. Each Ytop-bottom applies to following x0-x1:word entries.",
             f"PAGE {occurrence['page']} SIZE {_coord(width)},{_coord(height)}",
             f"TARGET HEADER BOX {_box(occurrence['bbox'])}",
             "TARGET LABELS " + " | ".join(label["text"] for label in occurrence["labels"]),
             f"CONTEXT CROP BOX {_box(region)}",
             "CROP WORDS IN THIS REQUEST, NEAREST ROWS FIRST:"]
    rows = make_rows(words, Rules())
    target_top, target_bottom = occurrence["bbox"][1], occurrence["bbox"][3]
    rows.sort(key=lambda row: (0, 0) if row["bottom"] >= target_top and row["top"] <= target_bottom
              else (row["top"] - target_bottom, 0) if row["top"] > target_bottom
              else (target_top - row["bottom"], 1))
    for row in rows:
        parts = []
        current_y = None
        for word in row["words"]:
            y = (_coord(word["top"]), _coord(word["bottom"]))
            if y != current_y:
                parts.append(f"Y{y[0]}-{y[1]}")
                current_y = y
            parts.append(f"{_coord(word['x0'])}-{_coord(word['x1'])}:{word['text']}")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def ask_laya(url: str, payload: dict, timeout: float) -> tuple[float, dict]:
    request = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(500).decode("utf-8", errors="replace")
        raise RuntimeError(f"Laya HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Laya connection failed at {url}: {exc.reason}") from exc
    try:
        probability = float(result["answers"][QUESTION_ID]["noul"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Laya returned no valid {QUESTION_ID} noul answer") from exc
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise RuntimeError(f"Laya returned an invalid noul probability: {probability}")
    return probability, result


def score_occurrence(occurrence: dict, words: list[dict], region: list[float],
                     width: float, height: float, url: str, model: str,
                     timeout: float) -> tuple[float, list[dict], list[dict], bool]:
    """Send all crop words across bounded requests; split any saturated request."""
    target = [word for word in words if word_in_region(word, occurrence["bbox"])]
    other = [word for word in words if not word_in_region(word, occurrence["bbox"])]
    target_center = (occurrence["bbox"][1] + occurrence["bbox"][3]) / 2
    other.sort(key=lambda word: (abs((word["top"] + word["bottom"]) / 2 - target_center),
                                 word["top"], word["x0"]))
    capacity = max(1, 75 - len(target))
    groups = [other[index:index + capacity] for index in range(0, len(other), capacity)] or [[]]
    call_records = []
    requests = []
    leaves = []
    context_limit = MODEL_CONTEXT_LIMITS.get(model, 1024)

    def score_chunk(extra: list[dict]) -> None:
        chunk_words = target + extra
        state = build_state(occurrence, chunk_words, region, width, height)
        payload = {"state": state, "model": model,
                   "questions": {QUESTION_ID: {"type": "noul", "instructions": QUESTION}}}
        probability, response = ask_laya(url, payload, timeout)
        tokens = response.get("usage", {}).get("input_tokens")
        saturated = not isinstance(tokens, int) or tokens >= 0.95 * context_limit
        request = dict(payload=payload, context_word_count=len(chunk_words),
                       input_tokens=tokens, saturated=saturated)
        requests.append(request)
        call_records.append(dict(probability=probability, input_tokens=tokens,
                                 context_word_count=len(chunk_words), saturated=saturated,
                                 answer=response["answers"][QUESTION_ID]))
        if saturated and len(extra) > 1:
            midpoint = len(extra) // 2
            score_chunk(extra[:midpoint])
            score_chunk(extra[midpoint:])
        else:
            leaves.append((probability, response, saturated))

    for group in groups:
        score_chunk(group)
    usable = [item for item in leaves if not item[2]]
    probability = max(item[0] for item in (usable or leaves))
    incomplete = any(item[2] for item in leaves)
    return probability, call_records, requests, incomplete


def save_snapshot(pdf, occurrence: dict, region: list[float], path: Path,
                  resolution: int = 110) -> None:
    """Save the matching PDF crop with the target header outlined in red."""
    from PIL import ImageDraw

    page = pdf.pages[occurrence["page"] - 1]
    cropped = page.crop(tuple(region))
    image = cropped.to_image(resolution=resolution).original.convert("RGB")
    scale_x = image.width / (region[2] - region[0])
    scale_y = image.height / (region[3] - region[1])
    x0, top, x1, bottom = occurrence["bbox"]
    draw = ImageDraw.Draw(image)
    draw.rectangle(((x0 - region[0]) * scale_x, (top - region[1]) * scale_y,
                    (x1 - region[0]) * scale_x, (bottom - region[1]) * scale_y),
                   outline="#dc2626", width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    page.close()


def filtered_family(family: dict, kept: list[dict], all_pages: list[int]) -> dict:
    """Remove rejected occurrences and fields that only those occurrences had."""
    result = copy.deepcopy(family)
    used = sorted({index for occurrence in kept for index in occurrence["field_indices"]})
    remap = {old: new for new, old in enumerate(used)}
    result["fields"] = [family["fields"][index] for index in used]
    result["occurrences"] = copy.deepcopy(kept)
    for occurrence in result["occurrences"]:
        occurrence["field_indices"] = [remap[index] for index in occurrence["field_indices"]]
    result["representative"] = copy.deepcopy(result["occurrences"][0])
    result["pages"] = sorted({occurrence["page"] for occurrence in kept})
    page_index = {page: index for index, page in enumerate(all_pages)}
    counts = [[0] * len(all_pages) for _ in used]
    for occurrence in result["occurrences"]:
        for index in occurrence["field_indices"]:
            counts[index][page_index[occurrence["page"]]] += 1
    result["heatmap_counts"] = counts
    return result


def frequency_filter(scan: dict, minimum: int) -> tuple[dict, list[dict]]:
    """Select groups seen at least `minimum` times anywhere in the document."""
    if minimum < 1:
        raise ValueError("minimum group occurrences must be positive")
    decisions = []
    eligible = []
    for family in scan["families"]:
        count = len(family["occurrences"])
        keep = count >= minimum
        decisions.append(dict(family_id=family["id"], kind=family["kind"],
                              occurrence_count=count, distinct_pages=len(family["pages"]),
                              passed=keep))
        if keep:
            eligible.append(family)
    return {**scan, "families": eligible}, decisions


def purge_report(scan: dict, scores: dict[str, float], threshold: float,
                 inconclusive: set[str] | None = None) -> tuple[dict, list[dict]]:
    """Keep accepted occurrences only in families still recurring across pages."""
    inconclusive = inconclusive or set()
    minimum = scan["rules"]["min_pages"]
    purged = copy.deepcopy(scan)
    purged["families"] = []
    decisions = []
    for family in scan["families"]:
        if family["kind"] == "kv":
            purged["families"].append(copy.deepcopy(family))
            continue
        if family["kind"] != "table":
            continue
        accepted = []
        for index, occurrence in enumerate(family["occurrences"], 1):
            identifier = f"{family['id']}_p{occurrence['page']:04d}_o{index:04d}"
            probability = scores[identifier]
            keep = probability >= threshold or identifier in inconclusive
            decisions.append(dict(occurrence_id=identifier, family_id=family["id"],
                                  page=occurrence["page"], probability=probability,
                                  model_accepts=probability >= threshold,
                                  status=("inconclusive_truncated" if identifier in inconclusive
                                          else "accepted" if keep else "rejected"),
                                  purge_retains=keep))
            if keep:
                accepted.append(occurrence)
        if len({occurrence["page"] for occurrence in accepted}) >= minimum:
            purged["families"].append(filtered_family(family, accepted, scan["pages"]))
    retained = {family["id"] for family in purged["families"]}
    for decision in decisions:
        decision["included_in_purged_group"] = decision["purge_retains"] and decision["family_id"] in retained
    purged["unconfirmed"] = []
    purged["candidate_count"] = sum(len(family["occurrences"]) for family in purged["families"])
    return purged, decisions


def save_purged(purged: dict, output_dir: Path, title: str, image_names: list[str],
                requests: list[dict], use_laya: bool) -> None:
    artifacts = plot_results(purged, output_dir, title, prefix="purged_")
    for old in output_dir.iterdir():
        if old.is_file() and re.fullmatch(r"purged_(?:table|kv)_\d+\.(?:png|svg)", old.name) and old.name not in artifacts:
            old.unlink()
    snapshot_dir = output_dir / "purged_snapshots"
    if snapshot_dir.is_dir():
        for old in snapshot_dir.iterdir():
            if old.is_file() and old.name not in image_names and re.fullmatch(r"(?:table|kv)_\d+_p\d+_o\d+\.png", old.name):
                old.unlink()
        if not any(snapshot_dir.iterdir()):
            snapshot_dir.rmdir()
    purged["artifacts"] = artifacts + [f"purged_snapshots/{name}" for name in image_names]
    request_path = output_dir / "purged_requests.jsonl"
    if use_laya:
        request_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in requests), encoding="utf-8")
    else:
        request_path.unlink(missing_ok=True)
    (output_dir / "purged_scan.json").write_text(json.dumps(purged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def archive_laya_outputs(output_dir: Path) -> None:
    """Keep earlier model results when replacing them with a frequency-only run."""
    existing = [path for path in output_dir.iterdir()
                if (path.name in {"purged_scan.json", "purged_requests.jsonl", "purged_snapshots", "purge_run.log"}
                    or re.fullmatch(r"purged_(?:table|kv)_\d+\.png", path.name))]
    if not existing:
        return
    report_path = output_dir / "purged_scan.json"
    if report_path.is_file():
        prior = json.loads(report_path.read_text(encoding="utf-8"))
        if prior.get("purge", {}).get("laya_enabled") is False:
            return
    archive = output_dir / "previous_laya"
    suffix = 2
    while archive.exists():
        archive = output_dir / f"previous_laya_{suffix}"
        suffix += 1
    archive.mkdir()
    for path in existing:
        path.rename(archive / path.name)
    archived_report = archive / "purged_scan.json"
    if archived_report.is_file():
        report = json.loads(archived_report.read_text(encoding="utf-8"))
        report["purge"]["source_scan"] = "../../scan.json"
        archived_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(scan_folder: Path, url: str, model: str, threshold: float, margin: float,
        timeout: float, save_images: bool = True, pdf_override: Path | None = None,
        minimum_occurrences: int = DEFAULT_MIN_OCCURRENCES, use_laya: bool = True) -> dict:
    scan = json.loads((scan_folder / "scan.json").read_text(encoding="utf-8"))
    eligible_scan, frequency_decisions = frequency_filter(scan, minimum_occurrences)
    output_dir = scan_folder / "purged"
    if not use_laya:
        output_dir.mkdir(parents=True, exist_ok=True)
        archive_laya_outputs(output_dir)
        purged = copy.deepcopy(eligible_scan)
        purged["unconfirmed"] = []
        purged["candidate_count"] = sum(len(family["occurrences"]) for family in purged["families"])
        purged["purge"] = dict(source_scan="../scan.json", source_pdf=scan.get("source_pdf"),
                               laya_enabled=False, minimum_group_occurrences=minimum_occurrences,
                               frequency_decisions=frequency_decisions,
                               examined_occurrences=0,
                               retained_occurrences=sum(len(family["occurrences"])
                                                        for family in purged["families"] if family["kind"] == "table"),
                               request_count=0, decisions=[])
        save_purged(purged, output_dir, scan_folder.name, [], [], use_laya=False)
        return purged

    import pdfplumber

    pdf_path = (pdf_override or Path(scan["source_pdf"])).expanduser().resolve()
    if not pdf_path.is_file():
        raise ValueError(f"source PDF not found: {pdf_path}; pass --pdf")
    output_dir.mkdir(parents=True, exist_ok=True)
    dimensions = {page["page"]: (page["width"], page["height"]) for page in scan["page_summary"]}
    table_families = [family for family in eligible_scan["families"] if family["kind"] == "table"]
    occurrences = [(family, index, occurrence)
                   for family in table_families for index, occurrence in enumerate(family["occurrences"], 1)]
    pages = {occurrence["page"] for _, _, occurrence in occurrences}
    document_words, ocr_sources = load_document_words(pdf_path, pages)
    snapshot_dir = output_dir / "purged_snapshots"
    requests = []
    scores = {}
    response_details = {}
    image_names = []
    pdf = pdfplumber.open(pdf_path) if save_images and occurrences else None
    try:
        for number, (family, index, occurrence) in enumerate(occurrences, 1):
            identifier = f"{family['id']}_p{occurrence['page']:04d}_o{index:04d}"
            width, height = dimensions[occurrence["page"]]
            region = expanded_bbox(occurrence["bbox"], width, height, margin)
            words = [word for word in document_words[occurrence["page"]] if word_in_region(word, region)]
            words.sort(key=lambda word: (word["top"], word["x0"]))
            if pdf is not None:
                name = f"{identifier}.png"
                save_snapshot(pdf, occurrence, region, snapshot_dir / name)
                image_names.append(name)
            probability, calls, chunk_requests, incomplete = score_occurrence(
                occurrence, words, region, width, height, url, model, timeout)
            scores[identifier] = probability
            usage = {"input_tokens": sum(call["input_tokens"] or 0 for call in calls),
                     "output_tokens": 0}
            response_details[identifier] = dict(crop_bbox=region, context_word_count=len(words),
                                                ocr_source=ocr_sources[occurrence["page"]],
                                                snapshot=f"purged_snapshots/{identifier}.png" if pdf else None,
                                                chunk_probabilities=[call["probability"] for call in calls],
                                                model_calls=len(calls), usage=usage,
                                                possible_context_truncation=incomplete)
            requests.extend({"occurrence_id": identifier, "call_index": call_index, **request}
                            for call_index, request in enumerate(chunk_requests, 1))
            print(f"  [{number}/{len(occurrences)}] {identifier}: p(header)={probability:.3f} "
                  f"({len(calls)} Laya call(s){'; incomplete context' if incomplete else ''})", flush=True)
    finally:
        if pdf is not None:
            pdf.close()

    inconclusive = {identifier for identifier, detail in response_details.items()
                    if detail["possible_context_truncation"]}
    purged, decisions = purge_report(eligible_scan, scores, threshold, inconclusive)
    for decision in decisions:
        decision.update(response_details[decision["occurrence_id"]])
    purged["purge"] = dict(source_scan="../scan.json", source_pdf=str(pdf_path), laya_url=url,
                           laya_enabled=True, model=model, question=QUESTION, probability_threshold=threshold,
                           crop_margin_fraction=margin, coordinate_precision_pdf_points=1,
                           minimum_group_occurrences=minimum_occurrences,
                           frequency_decisions=frequency_decisions,
                           examined_occurrences=len(occurrences),
                           retained_occurrences=sum(len(family["occurrences"])
                                                    for family in purged["families"] if family["kind"] == "table"),
                           request_count=len(requests), decisions=decisions)
    save_purged(purged, output_dir, pdf_path.stem, image_names, requests, use_laya=True)
    return purged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("documents", nargs="*", help="scan-results folders or document names such as id_05")
    parser.add_argument("--all", action="store_true", help="process every folder containing scan.json in test/scan_results")
    parser.add_argument("--laya", action=argparse.BooleanOptionalAction, default=True,
                        help="run the Laya header check after frequency filtering (default: enabled)")
    parser.add_argument("--url", default=DEFAULT_URL, help="local Laya playground decision endpoint")
    parser.add_argument("--model", default="typed-decisions", help="Laya checkpoint (default: typed-decisions)")
    parser.add_argument("--threshold", type=float, default=0.3, help="minimum noul probability to keep an occurrence (default: 0.3)")
    parser.add_argument("--min-occurrences", type=int, default=DEFAULT_MIN_OCCURRENCES,
                        help="minimum occurrences of a table or KV group before the Laya stage (default: 10)")
    parser.add_argument("--margin", type=float, default=0.10, help="page fraction added around each header box")
    parser.add_argument("--timeout", type=float, default=60, help="seconds per Laya request")
    parser.add_argument("--no-images", action="store_true", help="skip audit crop PNGs")
    parser.add_argument("--pdf", type=Path, help="override source PDF (one document only)")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or not 0 <= args.margin <= 1 or args.timeout <= 0 or args.min_occurrences < 1:
        parser.error("threshold and margin must be in 0..1; timeout and min-occurrences must be positive")
    if args.all and args.documents:
        parser.error("choose document names or --all")
    scan_root = ROOT / "test" / "scan_results"
    if args.all:
        documents = ([str(folder) for folder in sorted(scan_root.iterdir())
                      if (folder / "scan.json").is_file()] if scan_root.is_dir() else [])
    else:
        documents = args.documents
    if not documents:
        parser.error("no scan result folders selected")
    if args.pdf and len(documents) != 1:
        parser.error("--pdf can only be used with one document")
    failures = 0
    for value in documents:
        try:
            folder = resolve_scan_folder(value)
            print(f"Purging {folder.name}", flush=True)
            result = run(folder, args.url, args.model, args.threshold, args.margin,
                         args.timeout, not args.no_images, args.pdf, args.min_occurrences, args.laya)
            tables = sum(f["kind"] == "table" for f in result["families"])
            kvs = sum(f["kind"] == "kv" for f in result["families"])
            summary = (f"Kept {result['purge']['retained_occurrences']}/{result['purge']['examined_occurrences']} "
                       "header occurrences" if args.laya else "Frequency filter complete (Laya skipped)")
            print(f"  {summary}: {tables} table group(s), {kvs} KV group(s); "
                  f"saved {folder / 'purged' / 'purged_scan.json'}", flush=True)
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            failures += 1
            print(f"  Failed {value}: {exc}", file=sys.stderr, flush=True)
    if failures:
        parser.exit(1, f"{failures} purge run(s) failed\n")


if __name__ == "__main__":
    main()
