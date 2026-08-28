from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents._usage import apply_usage, split_tokens
from src.ground_truth_gen.helpers import format_words_layout
from src.models.catalog import resolve_easy_model, resolve_hard_model

_MAX_SAMPLES = 5
_MAX_WORKERS = 4  # parallel classification calls

_SYSTEM_PROMPT = """\
You are a classification assistant for a PDF data extraction pipeline.
You will receive sample OCR layouts for a single block type \
(a recurring section of a PDF document that appears once or more per page).

Your task is to decide whether extracting this block requires a powerful model (HARD) \
or whether a cheaper, smaller model is sufficient (EASY).

There are exactly two reasons to classify a block as HARD:

  1. VARIABLE STRUCTURE — the block contains a section or sub-group that repeats \
a different number of times across samples. For example, a line-item table with \
3 rows on one page and 7 on another, or an optional sub-section that appears in \
some samples but not others.

  2. MULTI-LINE VALUES — at least one field has a value that wraps across multiple \
lines in any sample (e.g., a long address, a free-text description).

If NEITHER of those is true — even if values differ between samples, even if there \
are minor positional shifts — classify the block as EASY.

Respond with ONLY a JSON object on a single line, e.g.:
{"difficulty": "easy", "reason": "flat fixed structure, all values single-line"}
or
{"difficulty": "hard", "reason": "line-item table row count varies between samples"}
"""


def _load_samples(sample_dir: str) -> list[str]:
    """Return layout-preserving OCR for up to ``_MAX_SAMPLES`` instances."""
    fnames = sorted(f for f in os.listdir(sample_dir) if f.endswith(".json"))[:_MAX_SAMPLES]
    layouts: list[str] = []
    for fname in fnames:
        json_path = os.path.join(sample_dir, fname)
        try:
            with open(json_path, encoding="utf-8") as f:
                sample = json.load(f)
            layout = format_words_layout(sample.get("words", []))
            if layout:
                layouts.append(layout)
        except (OSError, json.JSONDecodeError):
            pass
    return layouts


def block_difficulty_router_node(state: dict[str, Any], llm: BaseChatModel) -> dict[str, Any]:
    """Classify each block type and assign an OpenCode model accordingly.

    The per-block classification calls are independent and run in parallel.

    Reads:
        state["pdf_path"]   – used to locate train_dir/block_samples/

    Writes:
        block_opencode_models  – dict[template_id, model_string]
    """
    t0 = time.time()
    train_dir         = os.path.dirname(state["pdf_path"])
    block_samples_dir = os.path.join(train_dir, "block_samples")

    template_ids = sorted(
        d for d in os.listdir(block_samples_dir)
        if os.path.isdir(os.path.join(block_samples_dir, d))
    )
    model_easy = resolve_easy_model(state.get("easy_model"))
    model_hard = resolve_hard_model(state.get("hard_model"))

    if state.get("no_routing"):
        model_assignments = {template_id: model_easy for template_id in template_ids}
        print(
            "  block_difficulty_router: routing disabled; assigned "
            f"{model_easy} to all {len(template_ids)} block type(s)"
        )
        return {
            "block_opencode_models": model_assignments,
            "cost": [{
                "phase": "block_difficulty_router",
                "duration_s": round(time.time() - t0, 3),
                "routing_disabled": True,
            }],
        }

    def _classify(template_id: str) -> tuple[str, str, tuple[int, int, int]]:
        """Return (model, log message, (input, output, total) tokens) for one block type."""
        sample_dir        = os.path.join(block_samples_dir, template_id)
        layouts = _load_samples(sample_dir)

        if not layouts:
            return model_hard, f"{template_id} → no samples, defaulting to hard → {model_hard}", (0, 0, 0)

        samples = "\n\n".join(
            f"Sample {index} OCR layout:\n{layout}"
            for index, layout in enumerate(layouts, start=1)
        )
        content = (
            f"Block type '{template_id}' — {len(layouts)} sample OCR layout(s):\n\n"
            f"{samples}\n\nClassify this block type as EASY or HARD based on "
            "the criteria above."
        )

        difficulty = "hard"
        tokens = (0, 0, 0)
        try:
            response = llm.invoke([
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=content),
            ])
            tokens = split_tokens(response.usage_metadata or {})
            text = response.content.strip()
            fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if fenced:
                text = fenced.group(1)
            parsed     = json.loads(text)
            difficulty = parsed.get("difficulty", "hard").lower()
            reason     = parsed.get("reason", "")
            message    = f"{template_id} → {difficulty}  ({reason})"
        except Exception as exc:
            message = f"{template_id} classification error ({exc}), defaulting to hard"

        model = model_easy if difficulty == "easy" else model_hard
        return model, message, tokens

    model_assignments: dict[str, str] = {}
    input_tokens = output_tokens = total_tokens = 0

    if template_ids:
        with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(template_ids))) as executor:
            futures = [executor.submit(_classify, tid) for tid in template_ids]
            results = [f.result() for f in futures]  # futures are in template_id order

        for template_id, (model, message, (inp, out, tot)) in zip(template_ids, results):
            model_assignments[template_id] = model
            input_tokens  += inp
            output_tokens += out
            total_tokens  += tot
            print(f"  block_difficulty_router: {message}")

    duration = time.time() - t0
    cost_entry: dict = {"phase": "block_difficulty_router", "duration_s": round(duration, 3)}
    apply_usage(cost_entry, input_tokens, output_tokens, total_tokens)
    return {
        "block_opencode_models": model_assignments,
        "cost": [cost_entry],
    }
