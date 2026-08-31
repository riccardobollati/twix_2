#!/usr/bin/env python3
"""Run the pipeline on a PDF, generate ground truth, and score.

Usage:
    python single_eval.py <pdf_path> <name> [--pages N] [--seed S]

    pdf_path    path to the PDF, e.g. data/id_60/agent_space/test/id_60.pdf
    name        label for this eval run — outputs go to base_folder/evals/eval_<name>/
    --pages N   sample N pages for ground truth (default: all)
    --seed S    random seed for page sampling

Expects (found by walking up from pdf_path):
    template.json
    pipeline/pipeline.py

Outputs in <base_folder>/evals/eval_<name>/:
    ocr.csv
    extracted.json
    grouped.json
    ground_truth.json
    ground_truth_grouped.json
    eval.json
    eval.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys

import pdfplumber
from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, os.path.dirname(__file__))

from src.eval.convert import load_json_dedup
from src.eval.render import render_to_file
from src.eval.score import score as eval_score
from src.ground_truth_gen.helpers import load_ocr_pages, parse_json_response, render_page_b64
from src.ground_truth_gen.prompt import build_prompt
from src.models import get_ground_truth_llm
from src.record_grouping import group_blocks_into_records


def _find_base_folder(start: str) -> str:
    """Walk up from start until a directory containing template.json is found."""
    current = os.path.abspath(start)
    while True:
        if os.path.isfile(os.path.join(current, "template.json")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise SystemExit(f"Error: could not find template.json above {start}")
        current = parent


def _extract_ocr(pdf_path: str, out_path: str) -> None:
    fields = ["id", "page", "x0", "top", "x1", "bottom", "text"]
    with pdfplumber.open(pdf_path) as pdf, \
         open(out_path, "w", newline="", encoding="utf-8", errors="replace") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        word_id = 0
        for page_num, page in enumerate(pdf.pages, start=1):
            for word in (page.extract_words() or []):
                writer.writerow({
                    "id": word_id, "page": page_num,
                    "x0": word["x0"], "top": word["top"],
                    "x1": word["x1"], "bottom": word["bottom"],
                    "text": word["text"],
                })
                word_id += 1


def _run_pipeline(pipeline_py: str, pdf_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(
        [sys.executable, os.path.abspath(pipeline_py),
         os.path.abspath(pdf_path),
         "--out-dir", os.path.abspath(out_dir)],
        check=True,
    )


def _generate_ground_truth(
    pdf_path: str,
    template: list,
    ocr_path: str | None,
    indices: list[int],
    llm,
    window: int = 20,
) -> list[dict]:
    system_prompt = build_prompt(template, include_images=True)
    window = max(1, window)
    wins   = [indices[i:i + window] for i in range(0, len(indices), window)]
    page_numbers = [index + 1 for index in indices]
    ocr_pages = (
        load_ocr_pages(ocr_path, page_numbers)
        if ocr_path and os.path.isfile(ocr_path)
        else {}
    )

    blocks: list[dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        print(f"  {len(indices)} page(s), {len(wins)} window(s) of up to {window}")
        for win_indices in wins:
            page_nums = [i + 1 for i in win_indices]
            pages_str = ", ".join(str(p) for p in page_nums)
            label = f"pages {pages_str}" if len(page_nums) > 1 else f"page {page_nums[0]}"
            print(f"  {label}...", end=" ", flush=True)

            if len(page_nums) == 1:
                user_text = f"Extract every physical block from page {page_nums[0]} of the document."
            else:
                user_text = (
                    f"Extract every physical block from pages {pages_str}. "
                    f"The images are in page order ({pages_str}); keep blocks page-local."
                )

            content: list[dict] = [{"type": "text", "text": user_text}]
            for idx in win_indices:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{render_page_b64(pdf.pages[idx])}"
                    },
                })
            if ocr_path and os.path.isfile(ocr_path):
                for page_num in page_nums:
                    ocr_text = ocr_pages[page_num]["text"]
                    content.append({
                        "type": "text",
                        "text": (
                            f"OCR words on page {page_num} (exact character "
                            f"sequences, reading order):\n{ocr_text}"
                        ),
                    })

            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=content),
            ])
            try:
                parsed     = parse_json_response(response.content)
                window_blocks = parsed.get("blocks", [])
            except (json.JSONDecodeError, AttributeError) as exc:
                print(f"WARN: {exc}")
                print(f"  [RAW RESPONSE]\n{response.content}\n  [END RAW]")
                window_blocks = []

            normalized = []
            for block in window_blocks if isinstance(window_blocks, list) else []:
                if not isinstance(block, dict):
                    continue
                try:
                    page = int(block.get("page"))
                except (TypeError, ValueError):
                    page = page_nums[0] if len(page_nums) == 1 else None
                if page in page_nums:
                    normalized.append({**block, "page": page})
            print(f"{len(normalized)} block(s)")
            if not normalized:
                print(f"  [RAW RESPONSE]\n{response.content}\n  [END RAW]")
            blocks.extend(normalized)

    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pipeline on a PDF, generate ground truth, and score."
    )
    parser.add_argument("pdf_path", help="Path to the PDF.")
    parser.add_argument("name", help="Eval label — outputs go to eval_<name>/ next to the PDF.")
    parser.add_argument(
        "--pages", type=int, default=None, metavar="N",
        help="Number of pages to sample for ground truth (default: all).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for page sampling.")
    parser.add_argument(
        "--gt-window", type=int, default=20, metavar="N",
        help="Pages per LLM call when generating ground truth (default: 20).",
    )
    parser.add_argument(
        "--reuse", action="store_true",
        help="Skip pipeline and ground-truth generation if extracted.json and ground_truth.json already exist.",
    )
    args = parser.parse_args()

    pdf_path      = os.path.abspath(args.pdf_path)
    base_folder   = _find_base_folder(os.path.dirname(pdf_path))
    eval_dir      = os.path.join(base_folder, "evals", f"eval_{args.name}")
    pipeline_py   = os.path.join(base_folder, "pipeline", "pipeline.py")
    template_path = os.path.join(base_folder, "template.json")
    ocr_path      = os.path.join(eval_dir, "ocr.csv")
    extracted_path = os.path.join(eval_dir, "extracted.json")
    grouped_path   = os.path.join(eval_dir, "grouped.json")
    gt_path       = os.path.join(eval_dir, "ground_truth.json")
    gt_grouped_path = os.path.join(eval_dir, "ground_truth_grouped.json")

    os.makedirs(eval_dir, exist_ok=True)

    for p, label in [
        (pdf_path,      "PDF"),
        (pipeline_py,   "pipeline/pipeline.py"),
        (template_path, "template.json"),
    ]:
        if not os.path.isfile(p):
            print(f"Error: {label} not found at {p}", file=sys.stderr)
            sys.exit(1)

    with open(template_path, encoding="utf-8") as f:
        template = json.load(f).get("template", [])

    can_reuse = args.reuse and os.path.isfile(extracted_path) and os.path.isfile(gt_path)

    if can_reuse:
        print(f"--reuse: loading existing extracted.json and ground_truth.json from {eval_dir}")
        candidate_blocks = load_json_dedup(extracted_path)
        gt_results = load_json_dedup(gt_path)
    else:
        if args.reuse:
            missing = [p for p in (extracted_path, gt_path) if not os.path.isfile(p)]
            print(f"--reuse: missing {missing}, recomputing.")

        # --- OCR ---
        print("Extracting OCR...")
        _extract_ocr(pdf_path, ocr_path)
        print(f"  OCR saved → {ocr_path}")

        # --- run pipeline ---
        print("Running pipeline...")
        _run_pipeline(pipeline_py, pdf_path, eval_dir)

        candidate_blocks = load_json_dedup(extracted_path)

        # --- ground truth ---
        print("Generating ground truth...")
        with pdfplumber.open(pdf_path) as pdf:
            total = len(pdf.pages)
        n_pages = min(args.pages, total) if args.pages is not None else total
        rng = random.Random(args.seed)
        indices = sorted(rng.sample(range(total), n_pages))
        print(f"Sampling pages: {[i + 1 for i in indices]} / {total}")

        gt_results = _generate_ground_truth(
            pdf_path,
            template,
            ocr_path,
            indices,
            get_ground_truth_llm(),
            window=args.gt_window,
        )

        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(gt_results, f, indent=2, ensure_ascii=False)
        gt_results = load_json_dedup(gt_path)
        print(f"Ground truth saved → {gt_path}")

    # Filter candidate to pages covered by ground truth. Use the union of all
    # window pages so multi-page GT windows don't decimate the candidate.
    gt_pages: set = set()
    for block in gt_results:
        if block.get("page") is not None:
            gt_pages.add(block["page"])
    if gt_pages:
        candidate_blocks = [
            block for block in candidate_blocks if block.get("page") in gt_pages
        ]

    candidate = group_blocks_into_records(candidate_blocks, template)
    grouped_ground_truth = group_blocks_into_records(gt_results, template)
    with open(grouped_path, "w", encoding="utf-8") as f:
        json.dump(candidate, f, indent=2, ensure_ascii=False)
    with open(gt_grouped_path, "w", encoding="utf-8") as f:
        json.dump(grouped_ground_truth, f, indent=2, ensure_ascii=False)

    # --- score ---
    eval_result = eval_score(
        grouped_ground_truth, candidate,
        reference="model_reference", threshold=0.9,
    )

    eval_json_path = os.path.join(eval_dir, "eval.json")
    eval_txt_path  = os.path.join(eval_dir, "eval.txt")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)
    render_to_file(eval_result, eval_txt_path)

    print(f"accuracy:    {eval_result['accuracy']:.3f}")
    print(f"exact_match: {eval_result['exact_match']}")
    print(f"eval.json  → {eval_json_path}")
    print(f"eval.txt   → {eval_txt_path}")


if __name__ == "__main__":
    main()
