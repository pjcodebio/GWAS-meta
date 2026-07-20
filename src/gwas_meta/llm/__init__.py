"""LLM provider abstraction for GWAS meta-analysis."""

from .anthropic_provider import AnthropicProvider
from .base import LLMProvider, SearchCriteria
from .openai_provider import OpenAIProvider

__all__ = [
    "AnthropicProvider",
    "LLMProvider",
    "OpenAIProvider",
    "SearchCriteria",
    "create_provider",
]


def create_provider(provider_name: str, **kwargs: object) -> LLMProvider:
    """Factory that instantiates an :class:`LLMProvider` by name.

    Parameters
    ----------
    provider_name:
        One of ``"anthropic"`` or ``"openai"``.
    **kwargs:
        Forwarded to the provider constructor (e.g. ``api_key``, ``model``).

    Returns
    -------
    LLMProvider
        A ready-to-use provider instance.

    Raises
    ------
    ValueError
        If *provider_name* is not recognised.
    """
    if provider_name == "anthropic":
        return AnthropicProvider(**kwargs)  # type: ignore[arg-type]
    elif provider_name == "openai":
        return OpenAIProvider(**kwargs)  # type: ignore[arg-type]
    else:
        raise ValueError(
            f"Unknown provider: {provider_name!r}. Choose 'anthropic' or 'openai'."
        )
