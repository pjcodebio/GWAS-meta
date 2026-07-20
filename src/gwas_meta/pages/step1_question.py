"""Step 1: User inputs their research question."""

import streamlit as st

from gwas_meta.llm.prompts import CRITERIA_SYSTEM_PROMPT, build_criteria_prompt


def render():
    st.header("Step 1: Research Question")

    with st.expander("Requirements & estimated costs", expanded=False):
        st.markdown(
            """
**Disk space** — Each GWAS study downloads a compressed summary statistics
file (300–700 MB). A typical run with 5–7 studies uses **1.5–5 GB** on disk.
Files are cached in `.cache/summary_stats/` and accumulate across runs.

**Memory (RAM)** — The pipeline uses chromosome-chunked processing:
only one chromosome's data is held in memory at a time, so **peak RAM
stays under ~4 GB** even for large runs. Works on machines with 8 GB RAM.

**LLM API cost** — The tool makes exactly **2 API calls** per run
(~1,200 input tokens + ~1,500 output tokens). Estimated cost:
**< $0.01** per run.

| Scenario | Studies | Disk | Peak RAM | API cost |
|---|---|---|---|---|
| Small | 2–3 | 0.6–2 GB | ~1 GB | < $0.01 |
| Typical | 5–7 | 1.5–5 GB | ~2 GB | < $0.01 |
| Large | 10–50+ | 3–35 GB | ~2–4 GB | < $0.01 |
"""
        )

    st.markdown(
        "Describe the phenotype or trait you want to meta-analyse. "
        "Be specific about the trait, ancestry, and any conditions of interest."
    )

    question = st.text_input(
        "Research question",
        value=st.session_state.research_question,
        placeholder="e.g. What genetic variants are associated with type 2 diabetes in European populations?",
    )
    st.session_state.research_question = question

    # --- Editable AI prompt ---
    # Show the exact user message that will be sent to the LLM in Step 2 and let
    # the user edit it. While `research_prompt_autogen` is True the prompt tracks
    # the question above; the first manual edit flips the flag (via on_change) so
    # the user's wording is preserved. "Reset to question" re-enables tracking.
    if st.session_state.research_prompt_autogen:
        st.session_state.research_prompt = build_criteria_prompt(question)

    st.markdown("#### Prompt sent to the AI")
    st.caption(
        "This is the exact message sent to the model in Step 2 (a fixed system "
        "instruction asking for structured JSON is also sent). Edit it to steer "
        "how your question is interpreted — or just leave it as-is."
    )

    def _on_prompt_edit() -> None:
        # A manual edit stops the prompt from being overwritten by the question.
        st.session_state.research_prompt_autogen = False

    st.text_area(
        "AI prompt",
        key="research_prompt",
        height=160,
        label_visibility="collapsed",
        on_change=_on_prompt_edit,
    )
    if not st.session_state.research_prompt_autogen:
        if st.button("↻ Reset prompt to match question"):
            st.session_state.research_prompt_autogen = True
            st.rerun()

    # The system instruction does the heavy lifting — it defines the persona,
    # the exact JSON schema, and the GWAS conventions the model follows. Show it
    # (read-only) so the user sees the full context sent alongside their prompt,
    # not just the thin user message above.
    with st.expander("System instruction (sent with every request, read-only)"):
        st.caption(
            "This fixed instruction — not the short prompt above — is what turns "
            "your question into structured criteria: it sets the expert-GWAS "
            "persona, the six output fields, and the convention to include EFO "
            "synonyms and standard inclusion/exclusion rules. The model's own "
            "genetics knowledge fills in the rest."
        )
        st.text_area(
            "System prompt",
            value=CRITERIA_SYSTEM_PROMPT,
            height=280,
            disabled=True,
            label_visibility="collapsed",
        )

    prompt_text = st.session_state.research_prompt

    st.markdown("---")
    col1, col2 = st.columns([4, 1])
    with col1:
        if st.button("Or upload datasets yourself →"):
            st.session_state.upload_mode = True
            st.session_state.step = 3
            st.rerun()
    with col2:
        can_advance = bool(question.strip()) and bool(prompt_text.strip())
        if st.button("Next →", type="primary", disabled=not can_advance):
            prov = st.session_state.get("provenance")
            if prov is not None and not prov.has_event("research_question"):
                prov.event("research_question", {"question": question.strip()})
            st.session_state.step = 2
            st.rerun()
