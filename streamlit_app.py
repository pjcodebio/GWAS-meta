"""Deployment entry point (Streamlit Community Cloud and other hosts).

The tool's real entry point lives at ``src/gwas_meta/app.py`` and imports the
``gwas_meta`` package, which is not pip-installed on the host, so we put
``src/`` on the import path and delegate to it.

Secrets. The Anthropic API key is read from the environment
(``ANTHROPIC_API_KEY``). Streamlit Community Cloud injects configured secrets
via ``st.secrets`` rather than as environment variables, so we bridge them into
the environment first — keeping the same ``os.getenv``-based lookup working on
Streamlit Cloud and env-based hosts alike, with no change in the app itself.

Access gate. If ``APP_PASSWORD`` is set in the host's secrets, the app is
locked behind a shared access code (the free Streamlit tier has no built-in
viewer restriction). When it is unset — e.g. local development — the gate is a
no-op. This also protects the shared, prepaid LLM budget: the app (and its API
calls) never load until the code is entered.

Process-pool safety. Pass-1 chunking uses a ProcessPoolExecutor; on hosts that
create workers via spawn/forkserver, each worker re-imports this module. The
executable body is therefore guarded by ``if __name__ == "__main__"`` so a
worker importing this file (as ``__mp_main__``) does NOT re-run the app — the
same guard app.py already uses. Without it, every worker re-executes the whole
Streamlit app in bare mode and chunking fails.
"""

import os
import sys
from pathlib import Path

import streamlit as st


def _secret(key: str):
    """Return a secret value, or None if secrets are unavailable/unset.

    Accessing ``st.secrets`` with no secrets file raises, so guard every read.
    """
    try:
        return st.secrets.get(key)
    except Exception:
        return None


def _check_access() -> bool:
    """Gate the app behind APP_PASSWORD when one is configured."""
    expected = _secret("APP_PASSWORD")
    if not expected:  # no password configured -> open (local dev / ungated)
        return True
    if st.session_state.get("_authenticated"):
        return True

    st.title("🧬 gwas-meta")
    st.caption(
        "This deployment is access-controlled. Enter the access code to "
        "continue. (It runs on a shared, prepaid API budget — thanks for "
        "being considerate.)"
    )
    code = st.text_input("Access code", type="password")
    if code:
        if code == str(expected):
            st.session_state["_authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect access code.")
    return False


def _run() -> None:
    # Bridge Streamlit Cloud secrets -> environment (no-op locally).
    for _key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        if not os.getenv(_key):
            _val = _secret(_key)
            if _val:
                os.environ[_key] = str(_val)

    if not _check_access():
        st.stop()

    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from gwas_meta.app import main

    main()


# Guard the executable body so ProcessPoolExecutor workers that re-import this
# module (spawn/forkserver create them as __mp_main__) do not re-run the app.
if __name__ == "__main__":
    _run()
