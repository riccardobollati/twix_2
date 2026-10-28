"""Detect PDF ruling lines from vector geometry with a raster fallback."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re



def page_numbers(specification: str | None, total: int, limit: int | None = 30) -> list[int]:
    if specification is None:
        return list(range(1, min(total, limit) + 1)) if limit is not None else list(range(1, total + 1))
    selected = set()
    for token in specification.split(","):
        token = token.strip()
        if not re.fullmatch(r"\d+(?:-\d+)?", token):
            raise ValueError(f"invalid page selection: {token!r}")
        first, _, last = token.partition("-")
        start, stop = int(first), int(last or first)
        if start < 1 or stop < start or stop > total:
            raise ValueError(f"page selection {token!r} is outside 1-{total}")
        selected.update(range(start, stop + 1))
    if not selected:
        raise ValueError("select at least one page")
    return sorted(selected)


def _visible_color(color: object) -> bool:
    """Ignore white fill rectangles, which do not draw a visible rule."""
    if isinstance(color, (int, float)):
        return color < 0.95
    if isinstance(color, (tuple, list)) and len(color) >= 3:
        return not all(isinstance(value, (int, float)) and value >= 0.95 for value in color[:3])
    return True


def vector_lines(page, min_length: float, max_rule_thickness: float = 2.5) -> list[dict]:
    """Read PDF line objects, stroked rectangle edges, and thin filled rules."""
    result = []
    seen = set()

    def add(x0: float, top: float, x1: float, bottom: float, source: str) -> None:
        if not all(math.isfinite(value) for value in (x0, top, x1, bottom)):
            return
        if math.hypot(x1 - x0, bottom - top) < min_length:
            return
        if not all(-1 <= value <= limit + 1 for value, limit in
                   ((x0, page.width), (x1, page.width), (top, page.height), (bottom, page.height))):
            return
        key = tuple(round(value, 2) for value in (x0, top, x1, bottom))
        if key in seen:
            return
        seen.add(key)
        result.append(dict(x0=x0, top=top, x1=x1, bottom=bottom, source=source))

    for line in page.lines:
        add(line["x0"], page.height - line["y0"],
            line["x1"], page.height - line["y1"], "pdf_line")

    for rect in page.rects:
        x0, x1, top, bottom = rect["x0"], rect["x1"], rect["top"], rect["bottom"]
        width, height = x1 - x0, bottom - top
        if rect.get("stroke") and _visible_color(rect.get("stroking_color")):
            for segment in ((x0, top, x1, top), (x0, bottom, x1, bottom),
                            (x0, top, x0, bottom), (x1, top, x1, bottom)):
                add(*segment, "rect_edge")
        elif rect.get("fill") and _visible_color(rect.get("non_stroking_color")):
            if 0 < height <= max_rule_thickness and width >= min_length:
                add(x0, (top + bottom) / 2, x1, (top + bottom) / 2, "thin_rect")
            elif 0 < width <= max_rule_thickness and height >= min_length:
                add((x0 + x1) / 2, top, (x0 + x1) / 2, bottom, "thin_rect")
    return result


def raster_lines(image, page_width: float, page_height: float,
                 min_length: float) -> list[dict]:
    """Find long horizontal/vertical dark runs on image-only pages."""
    import numpy as np
    from scipy import ndimage

    grey = np.asarray(image.convert("L"))
    dark = grey < 175
    pixel_height, pixel_width = dark.shape
    # Short word strokes should not become lines in an image-only document.
    raster_minimum = max(min_length, 20.0)
    min_horizontal = max(2, round(raster_minimum * pixel_width / page_width))
    min_vertical = max(2, round(raster_minimum * pixel_height / page_height))
    result = []
    for orientation, structure in (("h", np.ones((1, min_horizontal), dtype=bool)),
                                   ("v", np.ones((min_vertical, 1), dtype=bool))):
        mask = ndimage.binary_opening(dark, structure=structure)
        labels, count = ndimage.label(mask)
        for box in ndimage.find_objects(labels, max_label=count):
            if box is None:
                continue
            ys, xs = box
            width, height = xs.stop - xs.start, ys.stop - ys.start
            if orientation == "h":
                if width < min_horizontal or height > max(6, round(pixel_height * 2.5 / page_height)):
                    continue
                middle = (ys.start + ys.stop) / 2 * page_height / pixel_height
                result.append(dict(x0=xs.start * page_width / pixel_width, top=middle,
                                   x1=xs.stop * page_width / pixel_width, bottom=middle,
                                   source="raster"))
            else:
                if height < min_vertical or width > max(6, round(pixel_width * 2.5 / page_width)):
                    continue
                middle = (xs.start + xs.stop) / 2 * page_width / pixel_width
                result.append(dict(x0=middle, top=ys.start * page_height / pixel_height,
                                   x1=middle, bottom=ys.stop * page_height / pixel_height,
                                   source="raster"))
    return result


def overlay_lines(image, lines: list[dict], page_width: float, page_height: float,
                  grey_alpha: int, line_width: int):
    """Composite a translucent grey veil, then draw solid red detections."""
    from PIL import Image, ImageDraw

    base = image.convert("RGBA")
    veil = Image.new("RGBA", base.size, (115, 115, 115, grey_alpha))
    result = Image.alpha_composite(base, veil)
    draw = ImageDraw.Draw(result)
    x_scale, y_scale = result.width / page_width, result.height / page_height
    for line in lines:
        draw.line((round(line["x0"] * x_scale), round(line["top"] * y_scale),
                   round(line["x1"] * x_scale), round(line["bottom"] * y_scale)),
                  fill=(235, 32, 32, 255), width=line_width)
    return result.convert("RGB")


def process_pdf(pdf_path: Path, output_root: Path, pages: str | None, dpi: int,
                min_length: float, grey_alpha: int, line_width: int,
                use_raster_fallback: bool, all_pages: bool = False,
                result_name: str | None = None) -> Path:
    import pdfplumber

    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"PDF not found: {pdf_path}")
    if result_name is not None and (not result_name or Path(result_name).name != result_name
                                    or result_name in {".", ".."}):
        raise ValueError("result name must be a single folder name")
    output = output_root / (result_name or pdf_path.stem) / "results"
    report_path = output / "lines.json"
    if report_path.is_file():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("source_pdf") != str(pdf_path):
            raise ValueError(f"output name collision at {output}: another PDF has the same stem")
    output.mkdir(parents=True, exist_ok=True)

    reports = []
    with pdfplumber.open(pdf_path) as pdf:
        selected = page_numbers(pages, len(pdf.pages), limit=None if all_pages else 30)
        for number in selected:
            page = pdf.pages[number - 1]
            original = page.to_image(resolution=dpi).original
            lines = vector_lines(page, min_length)
            method = "pdf_vectors"
            if not lines and use_raster_fallback:
                lines = raster_lines(original, page.width, page.height, min_length)
                method = "raster_fallback"
            overlay = overlay_lines(original, lines, page.width, page.height, grey_alpha, line_width)
            name = f"page_{number:04d}_lines.png"
            overlay.save(output / name)
            reports.append(dict(page=number, width=page.width, height=page.height,
                                method=method, line_count=len(lines), image=name, lines=lines))
            print(f"  page {number} ({len(reports)}/{len(selected)} selected): "
                  f"{len(lines)} lines -> {output / name}", flush=True)
            overlay.close()
            original.close()
            page.close()

    current = {report["image"] for report in reports}
    for old in output.glob("page_*_lines.png"):
        if old.name not in current and re.fullmatch(r"page_\d+_lines\.png", old.name):
            old.unlink()
    report_path.write_text(json.dumps(dict(source_pdf=str(pdf_path), dpi=dpi,
                                           min_line_length_pdf_points=min_length,
                                           grey_alpha=grey_alpha, line_width_pixels=line_width,
                                           raster_fallback=use_raster_fallback,
                                           pages=reports), ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return output
