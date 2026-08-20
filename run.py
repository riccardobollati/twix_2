"""Entry point for PDF template inference and evaluation.

Usage:
    python run.py path/to/document.pdf
    python run.py path/to/document.pdf --out-dir outputs/example
    python run.py <benchmark-name> --eval

    A benchmark name remains shorthand for data/<name>/<name>.pdf. Direct PDF
    inputs default to outputs/<pdf-stem>/ unless --out-dir is supplied.

    Outputs written to the selected output directory:
        ocr_extraction_full.csv        – full word-level OCR dump (step 1)
        template.json                  – inferred template
        pipeline/grouped.json          – records extracted by the pipeline
        pipeline/ground_truth.json     – LLM ground truth (only with --eval)
        pipeline/eval.json / eval.txt  – evaluation results (only with --eval)

    The graph detects and skips cover/index front matter before creating the
    train set. With --eval, the remaining pages are split into train and test;
    the graph builds the pipeline on train, runs it on test, generates ground
    truth via DeepSeek, and scores the result.

Environment variables:
    LLM_PROVIDER       Provider name (default and only active provider: openrouter).
    OPENROUTER_API_KEY Required for direct calls and OpenCode's OpenRouter provider.
    OPENROUTER_MODEL   Direct-call model (default: deepseek/deepseek-v4-flash).
    OPENAI_API_KEY     Required with --eval for held-out ground truth.
    OPENAI_MODEL       Held-out ground-truth model (default: gpt-5.4).
    OPENCODE_MODEL     OpenCode provider/model used for code generation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    load_dotenv()

import pdfplumber

from src.agents.graph import build_graph
from src.models import get_ground_truth_llm, get_llm
from src.models.opencode import DEFAULT_OPENCODE_MODEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _group_into_rows(words: list[dict], y_center_tol: float = 0.2) -> list[dict]:
    """Group words that share the same y-axis into row-level entries.

    A word joins an existing row iff its vertical center is within
    y_center_tol × average_height of every word already in that row.
    This is stricter than a plain y-overlap so that words on adjacent
    but distinct lines are never merged.

    y_center_tol: fraction of average word height allowed between vertical
    centers (default 0.2 — within one-fifth of a word-height of each other).
    """
    if not words:
        return []

    rows: list[list[dict]] = []
    for word in words:
        wc = (word["top"] + word["bottom"]) / 2
        wh =  word["bottom"] - word["top"]
        merged = False
        for row in rows:
            if all(
                abs(wc - (w["top"] + w["bottom"]) / 2) <= y_center_tol * ((wh + w["bottom"] - w["top"]) / 2)
                for w in row
            ):
                row.append(word)
                merged = True
                break
        if not merged:
            rows.append([word])

    result = []
    for row in rows:
        by_x = sorted(row, key=lambda w: w["x0"])
        result.append({
            "id":     row[0]["id"],
            "page":   row[0]["page"],
            "x0":     min(w["x0"]     for w in row),
            "top":    min(w["top"]    for w in row),
            "x1":     max(w["x1"]     for w in row),
            "bottom": max(w["bottom"] for w in row),
            "text":   " ".join(w["text"] for w in by_x),
        })
    return result


def extract_ocr_full(pdf_path: str, output_path: str, group_rows: bool = False) -> None:
    """Extract every word from the PDF with its bounding box and save as CSV.

    When group_rows=True, words on the same horizontal line are merged into a
    single row using the TWIX y-overlap grouping algorithm.
    """
    fields = ["id", "page", "x0", "top", "x1", "bottom", "text"]
    with pdfplumber.open(pdf_path) as pdf, \
         open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        row_id = 0
        for page_num, page in enumerate(pdf.pages, start=1):
            page_words = [
                {
                    "id":     row_id + i,
                    "page":   page_num,
                    "x0":     w["x0"],
                    "top":    w["top"],
                    "x1":     w["x1"],
                    "bottom": w["bottom"],
                    "text":   w["text"],
                }
                for i, w in enumerate(page.extract_words())
            ]
            if group_rows:
                page_words = _group_into_rows(page_words)
            for entry in page_words:
                writer.writerow(entry)
            row_id += len(page_words)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_document(document: str, out_dir: str | None) -> tuple[Path, Path, str]:
    """Resolve a PDF input while preserving the legacy benchmark-name shorthand."""
    direct_path = Path(document).expanduser()
    legacy_dir = Path("data") / document
    legacy_path = legacy_dir / f"{document}.pdf"

    if direct_path.is_file():
        pdf_path = direct_path.resolve()
        doc_name = pdf_path.stem
        output_dir = (
            Path(out_dir).expanduser().resolve()
            if out_dir
            else (Path("outputs") / doc_name).resolve()
        )
    elif legacy_path.is_file():
        pdf_path = legacy_path.resolve()
        doc_name = document
        output_dir = (
            Path(out_dir).expanduser().resolve() if out_dir else legacy_dir.resolve()
        )
    else:
        raise FileNotFoundError(
            f"PDF not found: {document!r}. Pass a PDF path or a benchmark name "
            f"whose file exists at {legacy_path}."
        )

    return pdf_path, output_dir, doc_name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


def _api_key_preview() -> str:
    key = os.getenv("OPENROUTER_API_KEY", "")
    return key[:10] if key else "<not set>"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate extraction code and structured data from a PDF document."
    )
    parser.add_argument(
        "document",
        help="Input PDF path, or a benchmark name under data/ for backward compatibility.",
    )
    parser.add_argument(
        "--out-dir",
        metavar="DIR",
        help="Artifact directory (default: outputs/<pdf-stem> for direct PDF inputs).",
    )
    parser.add_argument(
        "--pages", type=_positive_int, default=3, metavar="N",
        help="Number of structured-data pages to use as the training set (default: 3).",
    )
    parser.add_argument(
        "--start-page", type=_positive_int, default=None, metavar="N",
        help="Physical page where records begin; skips automatic front-matter detection.",
    )
    parser.add_argument(
        "--eval", action="store_true",
        help=(
            "Split into train/test, run the pipeline on the test set, generate "
            "ground truth via DeepSeek, and score the result."
        ),
    )
    parser.add_argument(
        "--eval-pages", type=_positive_int, default=None, metavar="N",
        help="Number of pages to use as the test set (default: all remaining pages after train).",
    )
    parser.add_argument(
        "--holdout-pages", type=_nonnegative_int, default=None, metavar="N",
        help=(
            "Number of train pages hidden from script generation but included in "
            "the train eval, so overfitted scripts fail where the repair loop can "
            "fix them. Default: auto (2 when train has ≥4 pages, 1 at 3, else 0). "
            "Pass 0 to disable."
        ),
    )
    parser.add_argument(
        "--group-rows", action="store_true",
        help=(
            "Group OCR words into row-level entries: words on the same horizontal "
            "line are merged into one CSV row (text concatenated, bbox unioned). "
            "Forces OCR re-extraction even if the CSV already exists."
        ),
    )
    parser.add_argument(
        "--reuse-ocr", action="store_true",
        help="Reuse an existing OCR CSV. By default OCR is regenerated for reproducibility.",
    )
    parser.add_argument(
        "--train-threshold", type=_probability, default=0.0, metavar="FLOAT",
        help=(
            "If set to a value > 0, the pipeline checks train accuracy after the "
            "train eval phase. If accuracy is below this threshold it launches "
            "an OpenCode repair session to fix the pipeline scripts, then re-evaluates. "
            "0.0 (default) disables the repair loop entirely."
        ),
    )
    parser.add_argument(
        "--train-max-retries", type=_nonnegative_int, default=3, metavar="N",
        help="Maximum number of OpenCode repair attempts before proceeding regardless (default: 3).",
    )
    parser.add_argument(
        "--gt-test-window", type=_positive_int, default=None, metavar="N",
        help="Pages per LLM call when generating test ground truth (default: 20).",
    )
    parser.add_argument(
        "--no-routing", action="store_true",
        help=(
            "Skip block-difficulty classification and use OPENCODE_EASY_MODEL "
            "for every extraction block."
        ),
    )
    parser.add_argument(
        "--caveman", action="store_true",
        help="Prepend /caveman to every OpenCode prompt to reduce output tokens (~65%%).",
    )
    parser.add_argument(
        "--save-log", action="store_true",
        help="Save raw OpenCode session output to <doc>/debug/ (one file per session).",
    )
    args = parser.parse_args()

    print(f"using kwy: {_api_key_preview()}", file=sys.stderr)

    total_steps = 2

    # --- resolve paths ---
    try:
        pdf_path, output_dir, doc_name = resolve_document(args.document, args.out_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    ocr_csv_path = output_dir / "ocr_extraction_full.csv"

    # --- step 1: full OCR extraction ---
    if args.reuse_ocr and ocr_csv_path.is_file() and not args.group_rows:
        print(f"[1/{total_steps}] Reusing OCR at {ocr_csv_path}.", file=sys.stderr)
    else:
        mode = "row-level" if args.group_rows else "word-level"
        print(f"[1/{total_steps}] Extracting full OCR ({mode}) from {pdf_path} ...", file=sys.stderr)
        extract_ocr_full(str(pdf_path), str(ocr_csv_path), group_rows=args.group_rows)
        print(f"[1/{total_steps}] OCR saved to {ocr_csv_path}", file=sys.stderr)

    # --- step 2: run graph ---
    # The graph handles everything:
    #   detect_beginning → environment_creation → template_inference →
    #   record_separation → run_record_separation → block_separation →
    #   run_block_separation →
    #   block_samples → block_extraction_scripting → collect_extraction_scripts →
    #   grouping [→ ground_truth → eval]  (last two only when eval_mode=True)
    print(f"[2/{total_steps}] Running graph ...", file=sys.stderr)
    graph = build_graph(
        get_llm(),
        train_repair_max_retries=args.train_max_retries,
        ground_truth_llm=get_ground_truth_llm() if args.eval else None,
    )
    graph.invoke({
        "full_pdf_path":              str(pdf_path),
        "full_ocr_csv_path":          str(ocr_csv_path),
        "output_dir":                 str(output_dir),
        "source_sha256":              _sha256(pdf_path),
        "opencode_model":             DEFAULT_OPENCODE_MODEL,
        "doc_name":                   doc_name,
        "train_pages":                args.pages,
        "eval_mode":                  args.eval,
        "eval_pages":                 args.eval_pages,
        "train_accuracy_threshold":   args.train_threshold,
        "train_repair_max_retries":   args.train_max_retries,
        "no_routing":                 args.no_routing,
        **({
            "detected_start_page": args.start_page,
            "beginning_confidence": "manual",
            "beginning_reason": "Start page supplied with --start-page.",
        } if args.start_page is not None else {}),
        **({"gt_test_window": args.gt_test_window} if args.gt_test_window is not None else {}),
        **({"holdout_pages": args.holdout_pages} if args.holdout_pages is not None else {}),
        **({"use_caveman": True} if args.caveman else {}),
        **({"save_log": True} if args.save_log else {}),
    })
    print(f"[2/{total_steps}] Graph complete.", file=sys.stderr)

    pipeline_dir = os.path.join(output_dir, "pipeline")
    print(f"  template    → {os.path.join(output_dir, 'template.json')}", file=sys.stderr)
    print(f"  grouped     → {os.path.join(pipeline_dir, 'grouped.json')}", file=sys.stderr)
    if args.eval:
        print(f"  ground truth → {os.path.join(pipeline_dir, 'ground_truth.json')}", file=sys.stderr)

        eval_json_path = os.path.join(pipeline_dir, "eval.json")
        if os.path.isfile(eval_json_path):
            with open(eval_json_path, encoding="utf-8") as f:
                ev = json.load(f)
            print(f"  accuracy     → {ev['accuracy']:.3f}", file=sys.stderr)
            print(f"  exact_match  → {ev['exact_match']}", file=sys.stderr)
            print(f"  eval.json    → {eval_json_path}", file=sys.stderr)
            print(f"  eval.txt     → {os.path.join(pipeline_dir, 'eval.txt')}", file=sys.stderr)


if __name__ == "__main__":
    main()
