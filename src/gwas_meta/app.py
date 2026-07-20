"""GWAS Meta-Analysis Automation Tool -- Streamlit entry point."""

import logging
import shutil
from datetime import datetime
from pathlib import Path

import streamlit as st

from gwas_meta.utils.config import load_config
from gwas_meta.utils.provenance import ProvenanceLogger

LOG_DIR = Path("logs")


def _setup_logging() -> None:
    """Configure file logging for the current run.

    Each run creates a timestamped log file in ``logs/``.
    Files persist across sessions so users can review past runs.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = LOG_DIR / f"run_{timestamp}.log"

    root = logging.getLogger()
    # Avoid adding duplicate handlers on Streamlit reruns
    if any(
        isinstance(h, logging.FileHandler) and "logs/run_" in str(getattr(h, "baseFilename", ""))
        for h in root.handlers
    ):
        return

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    logging.getLogger("gwas_meta").info("Log file: %s", log_file)


def _clear_cache() -> None:
    """Remove leftover summary-statistics files from previous runs."""
    config = load_config()
    cache_dir = Path(
        config.get("data", {}).get("cache_dir", ".cache/summary_stats")
    )
    if cache_dir.is_dir():
        shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)


def init_session_state():
    """Initialize all session state keys with defaults."""
    # Set up persistent file logging (once per Streamlit session)
    if "_logging_configured" not in st.session_state:
        _setup_logging()
        st.session_state["_logging_configured"] = True

    # Clean up downloaded study files from any previous / interrupted run
    if "_cache_cleared" not in st.session_state:
        _clear_cache()
        st.session_state["_cache_cleared"] = True

    defaults = {
        "step": 1,
        "config": load_config(),
        # Step 1
        "research_question": "",
        # Prompt actually sent to the LLM in Step 2. Auto-generated from the
        # research question while `research_prompt_autogen` is True; once the
        # user edits it by hand, the flag flips and their text is preserved.
        "research_prompt": "",
        "research_prompt_autogen": True,
        # Step 2
        "search_criteria": None,
        "criteria_edited": False,
        # Step 3
        "traits_found": [],
        "studies_found": [],
        "selected_studies": [],
        "upload_mode": False,
        "uploaded_files": {},  # study_id -> Path (locally uploaded datasets)
        # Step 4
        "downloaded_files": {},
        "download_errors": {},
        # Step 5
        "meta_results": None,
        "aligned_data": None,
        # Step 6
        "summary_text": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    if "provenance" not in st.session_state:
        cfg = st.session_state.config
        catalog_url = cfg.get("gwas_catalog", {}).get("rest_base_url")
        st.session_state.provenance = ProvenanceLogger(gwas_catalog_url=catalog_url)


def render_sidebar():
    """Render sidebar with step indicator and LLM config."""
    with st.sidebar:
        st.title("GWAS Meta-Analysis")
        st.markdown("---")

        steps = [
            "Research Question",
            "Inclusion Criteria",
            "Study Selection",
            "Download Data",
            "Run Meta-Analysis",
            "Results & Summary",
        ]
        for i, name in enumerate(steps, 1):
            if i == st.session_state.step:
                st.markdown(f"**→ Step {i}: {name}**")
            elif i < st.session_state.step:
                st.markdown(f"~~Step {i}: {name}~~  ✓")
            else:
                st.markdown(f"Step {i}: {name}")

        st.markdown("---")
        st.subheader("LLM Configuration")
        provider = st.selectbox(
            "Provider",
            ["anthropic", "openai"],
            index=0 if st.session_state.config.get("llm", {}).get("provider") == "anthropic" else 1,
        )
        st.session_state.config.setdefault("llm", {})["provider"] = provider

        st.markdown("---")
        if st.session_state.step > 1:
            if st.button("← Back"):
                prev = st.session_state.step - 1
                # In upload-only mode, skip steps 2 and 4
                if st.session_state.upload_mode:
                    if prev == 4:
                        prev = 3
                    elif prev == 2:
                        prev = 1
                        st.session_state.upload_mode = False
                st.session_state.step = prev
                st.rerun()


def main():
    st.set_page_config(
        page_title="GWAS Meta-Analysis Tool",
        page_icon="🧬",
        layout="wide",
    )

    init_session_state()
    render_sidebar()

    step = st.session_state.step

    if step == 1:
        from gwas_meta.pages.step1_question import render
    elif step == 2:
        from gwas_meta.pages.step2_criteria import render
    elif step == 3:
        from gwas_meta.pages.step3_studies import render
    elif step == 4:
        from gwas_meta.pages.step4_download import render
    elif step == 5:
        from gwas_meta.pages.step5_meta import render
    elif step == 6:
        from gwas_meta.pages.step6_results import render
    else:
        st.error(f"Unknown step: {step}")
        return

    render()


if __name__ == "__main__":
    main()
