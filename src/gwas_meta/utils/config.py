"""Configuration loader."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "settings.yaml"


def load_config(config_path: Path | None = None) -> dict:
    """Load settings from YAML config file and environment variables."""
    load_dotenv()

    path = config_path or _DEFAULT_CONFIG_PATH
    if path.exists():
        with open(path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Override LLM keys from env
    config.setdefault("llm", {})
    if api_key := os.getenv("ANTHROPIC_API_KEY"):
        config["llm"].setdefault("anthropic", {})["api_key"] = api_key
    if api_key := os.getenv("OPENAI_API_KEY"):
        config["llm"].setdefault("openai", {})["api_key"] = api_key

    return config
