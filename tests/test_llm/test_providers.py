"""Unit tests for LLM providers using mocked SDK clients."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from gwas_meta.llm import AnthropicProvider, OpenAIProvider, SearchCriteria, create_provider
from gwas_meta.llm.base import _extract_json


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SAMPLE_CRITERIA_JSON = {
    "trait_description": "Type 2 diabetes mellitus",
    "efo_terms": ["type 2 diabetes mellitus", "diabetes mellitus"],
    "inclusion_criteria": ["genome-wide significance threshold of 5e-8"],
    "exclusion_criteria": ["candidate gene studies only"],
    "ancestry_preference": "European",
    "min_sample_size": 1000,
}


def _make_anthropic_response(text: str) -> MagicMock:
    """Build a mock Anthropic Messages response."""
    block = MagicMock()
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


def _make_openai_response(text: str) -> MagicMock:
    """Build a mock OpenAI ChatCompletion response."""
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


# ---------------------------------------------------------------------------
# _extract_json tests
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_plain_json(self) -> None:
        text = json.dumps(SAMPLE_CRITERIA_JSON)
        assert _extract_json(text) == SAMPLE_CRITERIA_JSON

    def test_json_in_code_fence(self) -> None:
        text = f"```json\n{json.dumps(SAMPLE_CRITERIA_JSON)}\n```"
        assert _extract_json(text) == SAMPLE_CRITERIA_JSON

    def test_json_in_plain_fence(self) -> None:
        text = f"```\n{json.dumps(SAMPLE_CRITERIA_JSON)}\n```"
        assert _extract_json(text) == SAMPLE_CRITERIA_JSON

    def test_json_with_surrounding_text(self) -> None:
        text = f"Here is the JSON:\n{json.dumps(SAMPLE_CRITERIA_JSON)}\nDone."
        assert _extract_json(text) == SAMPLE_CRITERIA_JSON

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            _extract_json("This is not JSON at all")


# ---------------------------------------------------------------------------
# AnthropicProvider tests
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    @patch("gwas_meta.llm.anthropic_provider.anthropic.Anthropic")
    def test_parse_research_question(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_anthropic_response(
            json.dumps(SAMPLE_CRITERIA_JSON)
        )

        provider = AnthropicProvider(api_key="test-key")
        result = provider.parse_research_question("What variants are associated with T2D?")

        assert isinstance(result, SearchCriteria)
        assert result.trait_description == "Type 2 diabetes mellitus"
        assert result.efo_terms == ["type 2 diabetes mellitus", "diabetes mellitus"]
        assert result.ancestry_preference == "European"
        assert result.min_sample_size == 1000
        mock_client.messages.create.assert_called_once()

    @patch("gwas_meta.llm.anthropic_provider.anthropic.Anthropic")
    def test_parse_research_question_code_fence(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        fenced = f"```json\n{json.dumps(SAMPLE_CRITERIA_JSON)}\n```"
        mock_client.messages.create.return_value = _make_anthropic_response(fenced)

        provider = AnthropicProvider(api_key="test-key")
        result = provider.parse_research_question("T2D variants?")

        assert isinstance(result, SearchCriteria)
        assert result.trait_description == "Type 2 diabetes mellitus"

    @patch("gwas_meta.llm.anthropic_provider.anthropic.Anthropic")
    def test_summarize_results(self, mock_anthropic_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client
        mock_client.messages.create.return_value = _make_anthropic_response(
            "This is the summary."
        )

        provider = AnthropicProvider(api_key="test-key")
        summary = provider.summarize_results(
            question="T2D variants?",
            n_variants=50000,
            n_significant=12,
            top_hits=[{"rsid": "rs123", "p_value": 1e-10}],
        )

        # The provider must return the model's text verbatim -- exact equality
        # tests pass-through wiring, not merely that the mock echoed a keyword.
        assert summary == "This is the summary."
        mock_client.messages.create.assert_called_once()

    def test_missing_api_key_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                AnthropicProvider()


# ---------------------------------------------------------------------------
# OpenAIProvider tests
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    @patch("gwas_meta.llm.openai_provider.openai.OpenAI")
    def test_parse_research_question(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_response(
            json.dumps(SAMPLE_CRITERIA_JSON)
        )

        provider = OpenAIProvider(api_key="test-key")
        result = provider.parse_research_question("What variants are associated with T2D?")

        assert isinstance(result, SearchCriteria)
        assert result.trait_description == "Type 2 diabetes mellitus"
        assert result.efo_terms == ["type 2 diabetes mellitus", "diabetes mellitus"]
        assert result.ancestry_preference == "European"
        assert result.min_sample_size == 1000
        mock_client.chat.completions.create.assert_called_once()

    @patch("gwas_meta.llm.openai_provider.openai.OpenAI")
    def test_parse_research_question_code_fence(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        fenced = f"```json\n{json.dumps(SAMPLE_CRITERIA_JSON)}\n```"
        mock_client.chat.completions.create.return_value = _make_openai_response(fenced)

        provider = OpenAIProvider(api_key="test-key")
        result = provider.parse_research_question("T2D variants?")

        assert isinstance(result, SearchCriteria)
        assert result.trait_description == "Type 2 diabetes mellitus"

    @patch("gwas_meta.llm.openai_provider.openai.OpenAI")
    def test_summarize_results(self, mock_openai_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_openai_response(
            "This is the summary."
        )

        provider = OpenAIProvider(api_key="test-key")
        summary = provider.summarize_results(
            question="T2D variants?",
            n_variants=50000,
            n_significant=12,
            top_hits=[{"rsid": "rs123", "p_value": 1e-10}],
        )

        # The provider must return the model's text verbatim -- exact equality
        # tests pass-through wiring, not merely that the mock echoed a keyword.
        assert summary == "This is the summary."
        mock_client.chat.completions.create.assert_called_once()

    def test_missing_api_key_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                OpenAIProvider()


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestFactory:
    @patch("gwas_meta.llm.anthropic_provider.anthropic.Anthropic")
    def test_create_anthropic(self, mock_cls: MagicMock) -> None:
        provider = create_provider("anthropic", api_key="test-key")
        assert isinstance(provider, AnthropicProvider)

    @patch("gwas_meta.llm.openai_provider.openai.OpenAI")
    def test_create_openai(self, mock_cls: MagicMock) -> None:
        provider = create_provider("openai", api_key="test-key")
        assert isinstance(provider, OpenAIProvider)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("gemini")
