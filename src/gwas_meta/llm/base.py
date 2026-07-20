"""Abstract base class and shared utilities for LLM providers."""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SearchCriteria:
    """Structured search criteria extracted from a research question."""

    trait_description: str
    efo_terms: list[str] = field(default_factory=list)
    inclusion_criteria: list[str] = field(default_factory=list)
    exclusion_criteria: list[str] = field(default_factory=list)
    ancestry_preference: str | None = None
    min_sample_size: int | None = None


def _extract_json(text: str) -> dict:
    """Extract a JSON object from *text*, stripping markdown code fences if present.

    The LLM response may wrap JSON in ```json ... ``` or plain ``` ... ```.
    This helper handles both cases, as well as bare JSON.

    Raises
    ------
    json.JSONDecodeError
        If no valid JSON object can be extracted.
    """
    # Try stripping markdown fences first.
    pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            logger.debug("JSON inside code fence was not valid, falling back.")

    # Fall back to parsing the whole text.
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Last resort: find the first { ... } block.
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group(0))

    raise json.JSONDecodeError("No JSON object found in LLM response", text, 0)


class LLMProvider(ABC):
    """Abstract interface that every LLM backend must implement."""

    @abstractmethod
    def parse_research_question(
        self, question: str, prompt: str | None = None
    ) -> SearchCriteria:
        """Parse a free-text research question into :class:`SearchCriteria`.

        If *prompt* is given it is sent verbatim as the user message (the
        Step 1 UI lets the user edit it); otherwise the default template is
        applied to *question*.
        """
        ...

    @abstractmethod
    def summarize_results(
        self,
        question: str,
        n_variants: int,
        n_significant: int,
        top_hits: list[dict],
    ) -> str:
        """Return a human-readable summary of meta-analysis results."""
        ...
