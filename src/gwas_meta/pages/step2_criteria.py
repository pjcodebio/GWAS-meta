"""Step 2: LLM generates inclusion/exclusion criteria; user reviews and edits."""

import dataclasses
from datetime import date

import streamlit as st

from gwas_meta.llm import SearchCriteria, create_provider


def _criteria_to_pdf_bytes(
    criteria: SearchCriteria, research_question: str
) -> bytes:
    """Render the current search criteria as a one-page PDF."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    usable_w = pdf.w - pdf.l_margin - pdf.r_margin

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "GWAS Meta-Analysis -- Search Criteria",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {date.today().isoformat()}",
             new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)

    def _section(title: str) -> None:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)

    # fpdf2's multi_cell defaults new_x to RIGHT, which leaves the cursor at
    # the right edge of the page so the next call overflows off-screen.
    # Always send the cursor back to the left margin on the next line.
    def _mcell(text: str) -> None:
        safe = (text or "").encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(
            usable_w, 5, safe,
            new_x="LMARGIN", new_y="NEXT",
        )

    def _bullets(items: list[str]) -> None:
        if not items:
            _mcell("(none)")
            pdf.ln(2)
            return
        for item in items:
            _mcell(f"- {item}")
        pdf.ln(2)

    if research_question:
        _section("Research Question")
        _mcell(research_question)
        pdf.ln(2)

    _section("Trait Description")
    _mcell(criteria.trait_description or "(none)")
    pdf.ln(2)

    _section("EFO Search Terms")
    _bullets(criteria.efo_terms)

    _section("Inclusion Criteria")
    _bullets(criteria.inclusion_criteria)

    _section("Exclusion Criteria")
    _bullets(criteria.exclusion_criteria)

    _section("Ancestry Preference")
    _mcell(criteria.ancestry_preference or "(any)")
    pdf.ln(2)

    _section("Minimum Sample Size")
    min_n = criteria.min_sample_size
    _mcell(f"{min_n:,}" if min_n else "(no minimum)")

    out = pdf.output(dest="S")
    return bytes(out) if not isinstance(out, bytes) else out


def _generate_criteria() -> tuple[SearchCriteria, str, str, str]:
    config = st.session_state.config
    llm_cfg = config.get("llm", {})
    provider_name = llm_cfg.get("provider", "anthropic")
    provider_cfg = llm_cfg.get(provider_name, {})
    provider = create_provider(provider_name, **provider_cfg)
    # The (possibly user-edited) Step 1 prompt is sent verbatim; fall back to the
    # default template if it was never populated (e.g. upload path).
    prompt = st.session_state.get("research_prompt") or None
    criteria = provider.parse_research_question(
        st.session_state.research_question, prompt=prompt
    )
    model = provider_cfg.get("model") or getattr(provider, "model", None) or "default"
    return criteria, provider_name, model, prompt or ""


def render():
    st.header("Step 2: Inclusion / Exclusion Criteria")
    st.markdown(
        "The LLM has analysed your research question and suggested the criteria below. "
        "Review and edit them before searching the GWAS Catalog."
    )

    # Generate criteria if not yet done
    if st.session_state.search_criteria is None:
        with st.spinner("Generating search criteria with LLM..."):
            try:
                prov = st.session_state.get("provenance")
                if prov is not None:
                    with prov.time_block("search_terms_proposed") as scratch:
                        criteria, provider_name, model, prompt = _generate_criteria()
                        scratch["provider"] = provider_name
                        scratch["model"] = model
                        scratch["prompt"] = prompt
                        scratch["criteria"] = dataclasses.asdict(criteria)
                else:
                    criteria, _, _, _ = _generate_criteria()
                st.session_state.search_criteria = criteria
            except Exception as e:
                st.error(f"LLM error: {e}")
                return

    criteria: SearchCriteria = st.session_state.search_criteria

    # Editable fields
    trait = st.text_input("Trait description", value=criteria.trait_description)
    efo_terms = st.text_area(
        "EFO search terms (one per line)",
        value="\n".join(criteria.efo_terms),
        height=100,
    )
    inclusion = st.text_area(
        "Inclusion criteria (one per line)",
        value="\n".join(criteria.inclusion_criteria),
        height=120,
    )
    exclusion = st.text_area(
        "Exclusion criteria (one per line)",
        value="\n".join(criteria.exclusion_criteria),
        height=120,
    )
    ancestry = st.text_input(
        "Ancestry preference (leave blank for any)",
        value=criteria.ancestry_preference or "",
    )
    min_n = st.number_input(
        "Minimum sample size (0 = no minimum)",
        value=criteria.min_sample_size or 0,
        min_value=0,
        step=100,
    )

    # Update criteria object from edits
    criteria.trait_description = trait
    criteria.efo_terms = [t.strip() for t in efo_terms.split("\n") if t.strip()]
    criteria.inclusion_criteria = [c.strip() for c in inclusion.split("\n") if c.strip()]
    criteria.exclusion_criteria = [c.strip() for c in exclusion.split("\n") if c.strip()]
    criteria.ancestry_preference = ancestry.strip() or None
    criteria.min_sample_size = min_n if min_n > 0 else None

    st.markdown("---")
    col_pdf, _spacer, col_next = st.columns([1, 3, 1])
    with col_pdf:
        st.download_button(
            "Download criteria (PDF)",
            data=_criteria_to_pdf_bytes(criteria, st.session_state.research_question),
            file_name=f"gwas_search_criteria_{date.today().isoformat()}.pdf",
            mime="application/pdf",
        )
    with col_next:
        if st.button("Search GWAS Catalog →", type="primary"):
            st.session_state.search_criteria = criteria
            prov = st.session_state.get("provenance")
            if prov is not None:
                prov.event("search_terms_final", {"criteria": dataclasses.asdict(criteria)})
            st.session_state.step = 3
            st.rerun()
