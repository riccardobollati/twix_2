"""Measure and plot block word coverage for every PDF page."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, PercentFormatter, StrMethodFormatter
import pdfplumber


WordKey = tuple[str, str]


@dataclass(frozen=True)
class PageWordMeasure:
    """Word coverage measurement for one PDF page."""

    page: int
    inside_block_words: int
    total_words: int

    @property
    def outside_block_words(self) -> int:
        return self.total_words - self.inside_block_words

    @property
    def inside_block_ratio(self) -> float:
        if self.total_words == 0:
            return math.nan
        return self.inside_block_words / self.total_words


def _word_key(word: dict) -> WordKey:
    """Return a stable key so a word in overlapping blocks is counted once."""

    if word.get("id") is not None:
        return ("id", str(word["id"]))
    return (
        "content",
        json.dumps(word, sort_keys=True, ensure_ascii=False, default=str),
    )


def _load_inside_words(
    blocks_json_path: str | Path,
) -> dict[int, set[WordKey]]:
    path = Path(blocks_json_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Block-separation JSON not found: {path}")

    with path.open(encoding="utf-8") as stream:
        separation = json.load(stream)

    if not isinstance(separation, dict):
        raise ValueError("block-separation JSON root must be an object")
    pages = separation.get("pages")
    if not isinstance(pages, list):
        raise ValueError('block-separation JSON must contain a "pages" list')

    words_by_page: dict[int, set[WordKey]] = {}
    for entry in pages:
        if not isinstance(entry, dict) or not isinstance(entry.get("page"), int):
            raise ValueError("every page entry must contain an integer page number")
        page_number = entry["page"]
        if page_number < 1:
            raise ValueError("block page numbers must be positive")

        blocks = entry.get("blocks", [])
        if not isinstance(blocks, list):
            raise ValueError("every page entry must contain a blocks list")

        page_words = words_by_page.setdefault(page_number, set())
        for block in blocks:
            if not isinstance(block, dict):
                continue
            words = block.get("words", [])
            if not isinstance(words, list):
                continue
            page_words.update(
                _word_key(word) for word in words if isinstance(word, dict)
            )

    return words_by_page


def measure_pdf_pages(
    pdf_path: str | Path,
    blocks_json_path: str | Path,
) -> list[PageWordMeasure]:
    """Count words inside blocks and total words on every PDF page.

    The numerator is the number of unique words assigned to blocks in
    ``block_separation.json``. The denominator is the number of words extracted
    from the corresponding PDF page with ``pdfplumber.extract_words()``.
    """

    path = Path(pdf_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    inside_words_by_page = _load_inside_words(blocks_json_path)
    measurements: list[PageWordMeasure] = []

    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        extra_pages = sorted(
            page_number
            for page_number in inside_words_by_page
            if page_number > page_count
        )
        if extra_pages:
            raise ValueError(
                "block-separation JSON references pages outside the PDF: "
                f"{extra_pages}"
            )

        for page_number, page in enumerate(pdf.pages, start=1):
            total_words = len(page.extract_words() or [])
            inside_block_words = len(
                inside_words_by_page.get(page_number, set())
            )
            if inside_block_words > total_words:
                raise ValueError(
                    f"page {page_number} has {inside_block_words} unique block "
                    f"words but the PDF contains only {total_words}; ensure the "
                    "PDF and block-separation JSON were generated together"
                )

            measurements.append(
                PageWordMeasure(
                    page=page_number,
                    inside_block_words=inside_block_words,
                    total_words=total_words,
                )
            )

    return measurements


def plot_page_measures(
    measurements: Sequence[PageWordMeasure],
    *,
    title: str = "Words inside and outside blocks per PDF page",
) -> Figure:
    """Create the stacked word-count bars and coverage-ratio line."""

    if not measurements:
        raise ValueError("measurements cannot be empty")

    pages = [item.page for item in measurements]
    inside = [item.inside_block_words for item in measurements]
    outside = [item.outside_block_words for item in measurements]
    ratios = [item.inside_block_ratio for item in measurements]

    figure = Figure(figsize=(16, 6), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.bar(
        pages,
        inside,
        width=0.9,
        color="#2563EB",
        edgecolor="none",
        label="Words inside blocks",
    )
    axis.bar(
        pages,
        outside,
        bottom=inside,
        width=0.9,
        color="#CBD5E1",
        edgecolor="none",
        label="Words outside blocks",
    )
    axis.set(
        title=title,
        xlabel="Page",
        ylabel="Word count",
        xlim=(0.5, len(pages) + 0.5),
    )
    axis.xaxis.set_major_locator(MaxNLocator(nbins=20, integer=True))
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)

    ratio_axis = axis.twinx()
    ratio_line = ratio_axis.plot(
        pages,
        ratios,
        color="#DC2626",
        linewidth=1.8,
        label="Words inside / total words",
        zorder=5,
    )[0]
    ratio_axis.set_ylabel(
        "Words inside blocks / total page words",
        color="#DC2626",
    )
    ratio_axis.set_ylim(0, 1.05)
    ratio_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ratio_axis.tick_params(axis="y", colors="#DC2626")
    axis.legend(
        [*axis.containers, ratio_line],
        [
            "Words inside blocks",
            "Words outside blocks",
            "Words inside / total words",
        ],
        loc="upper left",
    )
    return figure


def analyze_pdf_words(
    pdf_path: str | Path,
    blocks_json_path: str | Path,
    *,
    output_path: str | Path | None = None,
) -> tuple[list[PageWordMeasure], Figure]:
    """Measure word coverage and return the measurements and chart."""

    measurements = measure_pdf_pages(pdf_path, blocks_json_path)
    figure = plot_page_measures(
        measurements,
        title=f"Words inside and outside blocks - {Path(pdf_path).name}",
    )

    if output_path is not None:
        destination = Path(output_path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180)

    return measurements, figure


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot block word coverage for each page of a PDF."
    )
    parser.add_argument("pdf", type=Path, help="PDF to analyze")
    parser.add_argument(
        "blocks_json",
        type=Path,
        help="block_separation.json containing page-local block words",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output chart path (default: <pdf-name>_word_coverage.png)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output or args.pdf.with_name(
        f"{args.pdf.stem}_word_coverage.png"
    )
    measurements, _figure = analyze_pdf_words(
        args.pdf,
        args.blocks_json,
        output_path=output,
    )
    print(f"Analyzed {len(measurements)} pages")
    print(f"Chart written to {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
