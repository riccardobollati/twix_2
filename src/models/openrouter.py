from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI


DEFAULT_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_REASONING_EFFORTS = {"minimal", "low", "medium", "high", "xhigh"}


def get_model() -> BaseChatModel:
    """Return the OpenRouter chat model used by non-code-generation phases.

    OpenRouter exposes an OpenAI-compatible Chat Completions API, so LangChain's
    OpenAI transport is used only as the protocol client. The default upstream
    model remains DeepSeek V4 Flash.
    """
    reasoning_effort = os.getenv(
        "OPENROUTER_REASONING_EFFORT", "high"
    ).strip().lower()
    if reasoning_effort not in _REASONING_EFFORTS:
        allowed = ", ".join(sorted(_REASONING_EFFORTS))
        raise ValueError(f"OPENROUTER_REASONING_EFFORT must be one of: {allowed}")

    headers: dict[str, str] = {}
    if app_url := os.getenv("OPENROUTER_APP_URL", "").strip():
        headers["HTTP-Referer"] = app_url
    if app_name := os.getenv("OPENROUTER_APP_NAME", "TWIX 2").strip():
        headers["X-OpenRouter-Title"] = app_name

    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL),
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
        default_headers=headers,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
        extra_body={
            "reasoning": {
                "effort": reasoning_effort,
                "exclude": True,
            }
        },
    )
