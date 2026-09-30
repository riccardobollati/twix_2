"""Shared OpenRouter/OpenCode model identifiers and normalization helpers."""

from __future__ import annotations

import os


DEFAULT_DIRECT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_EASY_MODEL = "openrouter/deepseek/deepseek-v4-flash"
DEFAULT_HARD_MODEL = "openrouter/deepseek/deepseek-v4-pro"
DEFAULT_REFERENCE_MODEL = "openai/gpt-5.4"

MODEL_OPTIONS = (
    ("DeepSeek V4.1 Flash", "deepseek/deepseek-v4.1-flash"),
    ("DeepSeek V4 Flash", "deepseek/deepseek-v4-flash"),
    ("DeepSeek V4 Pro", "deepseek/deepseek-v4-pro"),
    ("GLM 5.3 Flash", "z-ai/glm-5.3-flash"),
    ("Qwen3.8 Flash", "qwen/qwen3.8-flash"),
)

REFERENCE_MODEL_OPTIONS = (
    ("GPT-5.4", DEFAULT_REFERENCE_MODEL),
    ("DeepSeek V4.1 Flash", "deepseek/deepseek-v4.1-flash"),
    ("GLM 5.3 Flash", "z-ai/glm-5.3-flash"),
    ("Qwen3.8 Flash", "qwen/qwen3.8-flash"),
)
TEMPLATE_MODEL_OPTIONS = (("GPT-5.4", DEFAULT_REFERENCE_MODEL), *MODEL_OPTIONS)


def direct_model_id(model: str) -> str:
    """Return an OpenRouter API slug, accepting OpenCode's provider prefix."""
    normalized = model.strip()
    return normalized.removeprefix("openrouter/")


def opencode_model_id(model: str) -> str:
    """Return an OpenCode provider/model target for an OpenRouter model."""
    normalized = model.strip()
    if not normalized:
        return normalized
    return normalized if normalized.startswith("openrouter/") else f"openrouter/{normalized}"


def resolve_easy_model(override: str | None = None) -> str:
    return opencode_model_id(
        (override or "").strip()
        or os.getenv("OPENCODE_EASY_MODEL", "").strip()
        or os.getenv("OPENCODE_MODEL", "").strip()
        or DEFAULT_EASY_MODEL
    )


def resolve_hard_model(override: str | None = None) -> str:
    return opencode_model_id(
        (override or "").strip()
        or os.getenv("OPENCODE_HARD_MODEL", "").strip()
        or os.getenv("OPENCODE_MODEL", "").strip()
        or DEFAULT_HARD_MODEL
    )


def resolve_reference_model(override: str | None = None) -> str:
    return direct_model_id(
        (override or "").strip()
        or os.getenv("OPENROUTER_GROUND_TRUTH_MODEL", "").strip()
        or DEFAULT_REFERENCE_MODEL
    )


def resolve_template_model(override: str | None = None) -> str:
    return direct_model_id(
        (override or "").strip()
        or os.getenv("OPENROUTER_TEMPLATE_INFERENCE_MODEL", "").strip()
        or DEFAULT_REFERENCE_MODEL
    )
