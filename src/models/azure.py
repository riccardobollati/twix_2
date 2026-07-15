import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import AzureChatOpenAI


DEFAULT_MODEL = "gpt-5.4"
DEFAULT_API_VERSION = "2024-12-01-preview"


def get_model() -> BaseChatModel:
    """Return an AzureChatOpenAI model bound to the configured deployment.

    Required environment variables:
        AZURE_ENDPOINT                 – Resource endpoint, e.g. https://<resource>.openai.azure.com/
        AZURE_OPENAI_API_KEY           – Azure OpenAI API key.
        AZURE_OPENAI_DEPLOYMENT_NAME   – Deployment name (default: gpt-5.4).
        AZURE_OPENAI_API_VERSION       – API version (default: 2024-12-01-preview).
    """
    return AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", DEFAULT_MODEL),
        openai_api_version=os.environ.get("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION),
        temperature=0,
    )
