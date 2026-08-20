import os

from langchain_core.language_models.chat_models import BaseChatModel

try:
    from dotenv import load_dotenv
except ImportError:
    pass
else:
    load_dotenv()

_PROVIDERS: dict[str, str] = {
    "openrouter": "src.models.openrouter",
}


def get_llm() -> BaseChatModel:
    """Instantiate the LLM for the provider in LLM_PROVIDER (default: openrouter).

    To add a new provider:
        1. Create src/models/<provider>.py with a get_model() -> BaseChatModel function.
        2. Add an entry to _PROVIDERS above.
    """
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()

    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Available: {', '.join(_PROVIDERS)}"
        )

    module_path = _PROVIDERS[provider]
    import importlib
    module = importlib.import_module(module_path)
    return module.get_model()


def get_ground_truth_llm() -> BaseChatModel:
    """Return the OpenAI model reserved for held-out/test ground truth."""
    import importlib

    module = importlib.import_module("src.models.openai")
    return module.get_model()
