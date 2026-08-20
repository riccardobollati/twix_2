from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI


DEFAULT_MODEL = "gpt-5.4"


def get_model() -> BaseChatModel:
    """Return the OpenAI vision model used for held-out ground truth."""
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0,
    )
