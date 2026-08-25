from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI


DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_GROUND_TRUTH_MODEL = "openai/gpt-5.4"
DEFAULT_GROUND_TRUTH_MAX_TOKENS = 128_000
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_REASONING_EFFORTS = {"minimal", "low", "medium", "high", "xhigh"}


def _connection_kwargs() -> dict:
    """Shared OpenRouter transport settings for every direct model call."""
    headers: dict[str, str] = {}
    if app_url := os.getenv("OPENROUTER_APP_URL", "").strip():
        headers["HTTP-Referer"] = app_url
    if app_name := os.getenv("OPENROUTER_APP_NAME", "TWIX 2").strip():
        headers["X-OpenRouter-Title"] = app_name

    return {
        "api_key": os.environ["OPENROUTER_API_KEY"],
        "base_url": os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
        "default_headers": headers,
        "temperature": 0,
        "model_kwargs": {"response_format": {"type": "json_object"}},
    }


def get_model(
    *, reasoning_effort: str | None = None, model: str | None = None
) -> BaseChatModel:
    """Return the OpenRouter chat model used by non-code-generation phases.

    OpenRouter exposes an OpenAI-compatible Chat Completions API, so LangChain's
    OpenAI transport is used only as the protocol client. The default upstream
    model remains DeepSeek V4 Flash.
    """
    reasoning_effort = (
        reasoning_effort
        if reasoning_effort is not None
        else os.getenv("OPENROUTER_REASONING_EFFORT", "high")
    ).strip().lower()
    if reasoning_effort not in _REASONING_EFFORTS:
        allowed = ", ".join(sorted(_REASONING_EFFORTS))
        raise ValueError(f"OPENROUTER_REASONING_EFFORT must be one of: {allowed}")

    return ChatOpenAI(
        model=model or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL),
        **_connection_kwargs(),
        extra_body={
            "reasoning": {
                "effort": reasoning_effort,
                "exclude": True,
            }
        },
    )


def get_ground_truth_model() -> BaseChatModel:
    """Return OpenAI GPT-5.4 through OpenRouter for held-out ground truth."""
    raw_max_tokens = os.getenv(
        "OPENROUTER_GROUND_TRUTH_MAX_TOKENS",
        str(DEFAULT_GROUND_TRUTH_MAX_TOKENS),
    )
    try:
        max_tokens = int(raw_max_tokens)
    except ValueError as exc:
        raise ValueError(
            "OPENROUTER_GROUND_TRUTH_MAX_TOKENS must be a positive integer"
        ) from exc
    if max_tokens <= 0:
        raise ValueError(
            "OPENROUTER_GROUND_TRUTH_MAX_TOKENS must be a positive integer"
        )

    return ChatOpenAI(
        model=os.getenv(
            "OPENROUTER_GROUND_TRUTH_MODEL", DEFAULT_GROUND_TRUTH_MODEL
        ),
        max_tokens=max_tokens,
        **_connection_kwargs(),
    )
