from __future__ import annotations

import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents._usage import apply_usage, split_tokens

MODEL_EASY = os.getenv("CODEX_EASY_MODEL", "gpt-5.4-mini")
MODEL_HARD = os.getenv("CODEX_HARD_MODEL", os.getenv("CODEX_MODEL", "gpt-5.4"))

_MAX_SAMPLES = 5
_MAX_WORKERS = 4  # parallel classification calls

_SYSTEM_PROMPT = """\
You are a classification assistant for a PDF data extraction pipeline.
You will receive sample images and the word text for a single block type \
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


def _load_samples(sample_dir: str) -> tuple[list[str], list[str]]:
    """Return (base64 PNG list, word-text list) for up to _MAX_SAMPLES instances.

    The word-level coordinate JSON is deliberately reduced to the word texts —
    the images carry the layout signal, so shipping the full coordinate dump
    only inflates the input tokens.
    """
    fnames = sorted(f for f in os.listdir(sample_dir) if f.endswith(".json"))[:_MAX_SAMPLES]
    images: list[str] = []
    word_texts: list[str] = []
    for fname in fnames:
        base = fname[:-5]
        json_path = os.path.join(sample_dir, fname)
        img_path  = os.path.join(sample_dir, f"{base}.png")
        try:
            with open(json_path, encoding="utf-8") as f:
                sample = json.load(f)
            word_texts.append(" ".join(w.get("text", "") for w in sample.get("words", [])))
        except (OSError, json.JSONDecodeError):
            pass
        if os.path.isfile(img_path):
            with open(img_path, "rb") as f:
                images.append(base64.b64encode(f.read()).decode())
    return images, word_texts


def block_difficulty_router_node(state: dict[str, Any], llm: BaseChatModel) -> dict[str, Any]:
    """Classify each block type as easy or hard and assign a Codex model accordingly.

    The per-block classification calls are independent and run in parallel.

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

    def _classify(template_id: str) -> tuple[str, str, tuple[int, int, int]]:
        """Return (model, log message, (input, output, total) tokens) for one block type."""
        sample_dir        = os.path.join(block_samples_dir, template_id)
        images, word_texts = _load_samples(sample_dir)

        if not images and not word_texts:
            return MODEL_HARD, f"{template_id} → no samples, defaulting to hard → {MODEL_HARD}", (0, 0, 0)

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
        if word_texts:
            listing = "\n".join(f"  sample {i}: {t}" for i, t in enumerate(word_texts))
            content.append({
                "type": "text",
                "text": f"Word texts of the same {len(word_texts)} sample(s):\n{listing}",
            })
        content.append({
            "type": "text",
            "text": "Classify this block type as EASY or HARD based on the criteria above.",
        })

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

        model = MODEL_EASY if difficulty == "easy" else MODEL_HARD
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
        "block_codex_models": model_assignments,
        "cost": [cost_entry],
    }
