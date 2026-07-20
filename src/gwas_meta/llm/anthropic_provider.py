"""Anthropic (Claude) LLM provider."""

from __future__ import annotations

import json
import logging
import os

import anthropic

from .base import LLMProvider, SearchCriteria, _extract_json
from .prompts import (
    CRITERIA_SYSTEM_PROMPT,
    CRITERIA_USER_TEMPLATE,
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """LLM provider backed by the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 4096,
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "An Anthropic API key must be provided either as a parameter "
                "or via the ANTHROPIC_API_KEY environment variable."
            )
        self._client = anthropic.Anthropic(api_key=resolved_key)
        self._model = model
        self._max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def parse_research_question(
        self, question: str, prompt: str | None = None
    ) -> SearchCriteria:
        """Send the research question to Claude and parse the JSON response."""
        user_message = (
            prompt if prompt is not None
            else CRITERIA_USER_TEMPLATE.format(research_question=question)
        )

        logger.info("Requesting search criteria from Anthropic (%s)", self._model)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=CRITERIA_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        text = response.content[0].text
        logger.debug("Raw Anthropic response: %s", text)

        data = _extract_json(text)
        return SearchCriteria(
            trait_description=data.get("trait_description", ""),
            efo_terms=data.get("efo_terms", []),
            inclusion_criteria=data.get("inclusion_criteria", []),
            exclusion_criteria=data.get("exclusion_criteria", []),
            ancestry_preference=data.get("ancestry_preference"),
            min_sample_size=data.get("min_sample_size"),
        )

    def summarize_results(
        self,
        question: str,
        n_variants: int,
        n_significant: int,
        top_hits: list[dict],
    ) -> str:
        """Send meta-analysis results to Claude and return a plain-text summary."""
        user_message = SUMMARY_USER_TEMPLATE.format(
            research_question=question,
            n_variants=n_variants,
            n_significant=n_significant,
            top_hits_json=json.dumps(top_hits, indent=2),
        )

        logger.info("Requesting result summary from Anthropic (%s)", self._model)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=SUMMARY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        return response.content[0].text
