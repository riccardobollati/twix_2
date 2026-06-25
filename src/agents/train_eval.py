from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import time
from typing import Any

import pdfplumber
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.eval.score import score as eval_score
from src.eval.render import render_to_file
from src.ground_truth_gen.prompt import build_prompt
from src.ground_truth_gen.ground_truth import _parse_json, _load_page_ocr
from src.agents.grouping import _maybe_merge_single_block

WINDOW = 5


def _render_page_b64(page) -> str:
    buf = io.BytesIO()
    page.to_image(resolution=150).original.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def train_eval_node(state: dict[str, Any], llm: BaseChatModel) -> dict[str, Any]:
    """Run the pipeline on the train PDF and evaluate against generated ground truth.

    Reads:
        state["agent_space"]  – used to derive document_dir/pipeline/
        state["pdf_path"]     – train PDF
        state["ocr_csv_path"] – train OCR CSV
        state["template"]     – inferred template

    Writes:
        document_dir/pipeline/train_eval/grouped.json
        document_dir/pipeline/train_eval/ground_truth.json
        document_dir/pipeline/train_eval/eval.json
        document_dir/pipeline/train_eval/eval.txt
    """
    document_dir   = os.path.dirname(state["agent_space"])
    pipeline_dir   = os.path.join(document_dir, "pipeline")
    train_eval_dir = os.path.join(pipeline_dir, "train_eval")
    os.makedirs(train_eval_dir, exist_ok=True)

    pipeline_py = os.path.join(pipeline_dir, "pipeline.py")
    pdf_path    = state["pdf_path"]
    ocr_path    = state["ocr_csv_path"]
    template    = state["template"]

    cost_entries: list[dict] = []

    # ── 1. Run pipeline on train PDF ──────────────────────────────────────────
    print("train_eval_node: running pipeline on train PDF...")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, pipeline_py, pdf_path, "--out-dir", train_eval_dir],
        capture_output=True,
        text=True,
    )
    duration = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(f"pipeline.py failed on train PDF:\n{result.stderr}")
    if result.stdout:
        print(result.stdout, end="")
    cost_entries.append({"phase": "train_grouping", "duration_s": round(duration, 3)})

    grouped_path = os.path.join(train_eval_dir, "grouped.json")
    _maybe_merge_single_block(grouped_path, template)

    # ── 2. Generate ground truth for train pages (only on first run) ─────────
    gt_path = os.path.join(train_eval_dir, "ground_truth.json")
    if os.path.isfile(gt_path):
        print("train_eval_node: reusing existing ground truth (repair retry)")
        with open(gt_path, encoding="utf-8") as f:
            records = json.load(f)
        gt_duration  = 0.0
        gt_input     = 0
        gt_output    = 0
    else:
        print(f"train_eval_node: generating train ground truth (window={WINDOW})...")
        system_prompt  = build_prompt(template)
        records: list[dict] = []
        record_counter = 0
        gt_input       = 0
        gt_output      = 0

        t0 = time.time()
        with pdfplumber.open(pdf_path) as pdf:
            total   = len(pdf.pages)
            windows = [list(range(i, min(i + WINDOW, total))) for i in range(0, total, WINDOW)]
            print(f"  {total} page(s), {len(windows)} window(s) of up to {WINDOW}")

            for win_indices in windows:
                page_nums = [i + 1 for i in win_indices]
                pages_str = ", ".join(str(p) for p in page_nums)

                user_text = (
                    f"Extract all records from page {page_nums[0]} of the document."
                    if len(page_nums) == 1
                    else (
                        f"Extract all records from pages {pages_str} of the document. "
                        f"The images are provided in page order ({pages_str})."
                    )
                )

                content: list = [{"type": "text", "text": user_text}]
                for idx in win_indices:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{_render_page_b64(pdf.pages[idx])}"},
                    })
                if ocr_path and os.path.isfile(ocr_path):
                    for page_num in page_nums:
                        ocr_text = _load_page_ocr(ocr_path, page_num)
                        content.append({
                            "type": "text",
                            "text": f"OCR words on page {page_num} (exact character sequences, reading order):\n{ocr_text}",
                        })

                print(f"  pages {pages_str}...", end=" ", flush=True)
                response = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=content),
                ])

                usage = response.usage_metadata or {}
                inp = usage.get("input_tokens") or 0
                out = usage.get("output_tokens") or 0
                if not inp and not out:
                    out = usage.get("total_tokens") or 0
                gt_input  += inp
                gt_output += out

                try:
                    parsed    = _parse_json(response.content)
                    page_recs = parsed.get("records", [])
                except (json.JSONDecodeError, AttributeError) as exc:
                    print(f"WARN: {exc}")
                    page_recs = []

                n_blocks = 0
                for rec in page_recs:
                    blocks = rec.get("data", [])
                    for block in blocks:
                        block["page"] = page_nums[0]
                    records.append({"record_id": str(record_counter), "data": blocks})
                    record_counter += 1
                    n_blocks += len(blocks)
                print(f"{len(page_recs)} record(s), {n_blocks} block(s)")

        gt_duration = time.time() - t0
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        _maybe_merge_single_block(gt_path, template)

    gt_cost: dict = {"phase": "train_ground_truth", "duration_s": round(gt_duration, 3)}
    if gt_input:
        gt_cost["input_tokens"] = gt_input
    if gt_output:
        gt_cost["output_tokens"] = gt_output
    cost_entries.append(gt_cost)

    # ── 3. Evaluate ───────────────────────────────────────────────────────────
    t0 = time.time()
    with open(gt_path, encoding="utf-8") as f:
        gold = json.load(f)
    with open(grouped_path, encoding="utf-8") as f:
        candidate = json.load(f)

    eval_result  = eval_score(gold, candidate, reference="human", threshold=0.9)
    eval_duration = time.time() - t0

    eval_json_path = os.path.join(train_eval_dir, "eval.json")
    eval_txt_path  = os.path.join(train_eval_dir, "eval.txt")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, ensure_ascii=False)
    render_to_file(eval_result, eval_txt_path)

    print(
        f"train_eval_node: accuracy={eval_result['accuracy']:.3f}  "
        f"exact_match={eval_result['exact_match']}"
    )
    cost_entries.append({"phase": "train_eval", "duration_s": round(eval_duration, 3)})

    # ── 4. Append to accuracy history (tracks each loop iteration) ───────────
    history_path = os.path.join(train_eval_dir, "eval_history.json")
    if os.path.isfile(history_path):
        with open(history_path, encoding="utf-8") as f:
            history: list[dict] = json.load(f)
    else:
        history = []
    history.append({
        "attempt":     state.get("train_repair_attempts", 0),
        "accuracy":    eval_result["accuracy"],
        "exact_match": eval_result["exact_match"],
    })
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    return {"cost": cost_entries}
