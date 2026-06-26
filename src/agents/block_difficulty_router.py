from __future__ import annotations

import base64
import json
import os
import re
import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

MODEL_EASY = "gpt-5.4-mini"
MODEL_HARD = "gpt-5.4"

_MAX_SAMPLES = 5

_SYSTEM_PROMPT = """\
You are a classification assistant for a PDF data extraction pipeline.
You will receive sample images and structured word data for a single block type \
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


def _load_samples(sample_dir: str) -> tuple[list[str], list[dict]]:
    """Return (base64 PNG list, JSON word-data list) for up to _MAX_SAMPLES instances."""
    fnames = sorted(f for f in os.listdir(sample_dir) if f.endswith(".json"))[:_MAX_SAMPLES]
    images: list[str] = []
    word_data: list[dict] = []
    for fname in fnames:
        base = fname[:-5]
        json_path = os.path.join(sample_dir, fname)
        img_path  = os.path.join(sample_dir, f"{base}.png")
        try:
            with open(json_path, encoding="utf-8") as f:
                word_data.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            pass
        if os.path.isfile(img_path):
            with open(img_path, "rb") as f:
                images.append(base64.b64encode(f.read()).decode())
    return images, word_data


def block_difficulty_router_node(state: dict[str, Any], llm: BaseChatModel) -> dict[str, Any]:
    """Classify each block type as easy or hard and assign a Codex model accordingly.

    Reads:
        state["pdf_path"]   – used to locate train_dir/block_samples/

    Writes:
        block_codex_models  – dict[template_id, model_string]
    """
    t0 = time.time()
    train_dir         = os.path.dirname(state["pdf_path"])
    block_samples_dir = os.path.join(train_dir, "block_samples")

    template_ids = sorted(
        d for d in os.listdir(block_samples_dir)
        if os.path.isdir(os.path.join(block_samples_dir, d))
    )

    model_assignments: dict[str, str] = {}

    for template_id in template_ids:
        sample_dir    = os.path.join(block_samples_dir, template_id)
        images, jsons = _load_samples(sample_dir)

        if not images and not jsons:
            model_assignments[template_id] = MODEL_HARD
            print(f"  block_difficulty_router: {template_id} → no samples, defaulting to hard → {MODEL_HARD}")
            continue

        content: list = []
        content.append({
            "type": "text",
            "text": f"Block type '{template_id}' — {len(images)} sample image(s) follow:",
        })
        for img_b64 in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            })
        if jsons:
            content.append({
                "type": "text",
                "text": (
                    f"Corresponding word-level JSON data for the same "
                    f"{len(jsons)} sample(s):\n{json.dumps(jsons, indent=2)}"
                ),
            })
        content.append({
            "type": "text",
            "text": "Classify this block type as EASY or HARD based on the criteria above.",
        })

        difficulty = "hard"
        try:
            response = llm.invoke([
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=content),
            ])
            text = response.content.strip()
            fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if fenced:
                text = fenced.group(1)
            parsed     = json.loads(text)
            difficulty = parsed.get("difficulty", "hard").lower()
            reason     = parsed.get("reason", "")
            print(f"  block_difficulty_router: {template_id} → {difficulty}  ({reason})")
        except Exception as exc:
            print(f"  block_difficulty_router: {template_id} classification error ({exc}), defaulting to hard")

        model = MODEL_EASY if difficulty == "easy" else MODEL_HARD
        model_assignments[template_id] = model

    duration = time.time() - t0
    return {
        "block_codex_models": model_assignments,
        "cost": [{"phase": "block_difficulty_router", "duration_s": round(duration, 3)}],
    }
