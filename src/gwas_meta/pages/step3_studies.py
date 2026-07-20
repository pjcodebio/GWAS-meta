"""Step 3: Search GWAS Catalog and let user select studies."""

import re
from pathlib import Path

import streamlit as st
import pandas as pd

from gwas_meta.gwas_client import GWASCatalogClient


def _parse_total_sample_size(sample_desc: str) -> int | None:
    """Extract total sample size by summing all numbers in the description.

    Example: "208 Japanese ancestry cases, 287 Japanese ancestry controls" → 495
    Returns None if no numbers are found.
    """
    numbers = re.findall(r"\d[\d,]*", sample_desc)
    if not numbers:
        return None
    return sum(int(n.replace(",", "")) for n in numbers)


def _search_catalog(criteria) -> tuple[list, list]:
    """Search GWAS Catalog for matching traits and studies."""
    config = st.session_state.config.get("gwas_catalog", {})
    client = GWASCatalogClient(
        base_url=config.get("rest_base_url", "https://www.ebi.ac.uk/gwas/rest/api"),
        timeout=config.get("request_timeout", 30),
        rate_limit_delay=config.get("rate_limit_delay", 0.5),
    )

    all_traits = []
    all_studies = []
    seen_study_ids = set()

    for term in criteria.efo_terms:
        # 1) Exact EFO trait match → studies by EFO id
        traits = client.search_traits(term)
        all_traits.extend(traits)
        for trait in traits:
            studies = client.search_by_efo(trait.efo_id)
            for s in studies:
                if s.study_id not in seen_study_ids:
                    seen_study_ids.add(s.study_id)
                    all_studies.append(s)

        # 2) Broad disease-trait name search (partial match)
        broad_studies = client.search_studies(term)
        for s in broad_studies:
            if s.study_id not in seen_study_ids:
                seen_study_ids.add(s.study_id)
                all_studies.append(s)

    # Filter by summary stats availability
    all_studies = [s for s in all_studies if s.has_summary_stats]

    # Filter by minimum sample size
    if criteria.min_sample_size:
        min_n = criteria.min_sample_size
        all_studies = [
            s for s in all_studies
            if (_parse_total_sample_size(s.initial_sample_size) or 0) >= min_n
        ]

    return all_traits, all_studies


def _upload_dir() -> Path:
    """Return (and create) the directory for user-uploaded datasets."""
    cache_dir = st.session_state.config.get("data", {}).get(
        "cache_dir", ".cache/summary_stats"
    )
    upload_dir = Path(cache_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def _render_upload_section():
    """Render file upload widget and local folder path input."""
    st.subheader("Upload datasets")
    st.caption(
        "Upload harmonized summary statistics files (.tsv, .tsv.gz, .h.tsv.gz). "
        "Files must have columns: hm_chrom, hm_pos, hm_effect_allele, hm_other_allele, "
        "and hm_beta or hm_odds_ratio."
    )

    tab_folder, tab_upload = st.tabs(["Local folder", "Browser upload"])

    # --- Tab 1: Local folder path ---
    with tab_folder:
        folder_path = st.text_input(
            "Path to folder containing summary statistics files",
            placeholder="/Users/you/Desktop/files",
            key="local_folder_path",
        )
        if folder_path:
            folder = Path(folder_path)
            if not folder.is_dir():
                st.error(f"Not a valid directory: {folder_path}")
            else:
                found = sorted(
                    f for f in folder.iterdir()
                    if f.is_file() and (
                        f.name.endswith(".h.tsv.gz")
                        or f.name.endswith(".tsv.gz")
                        or f.name.endswith(".tsv")
                    )
                )
                if not found:
                    st.warning("No .tsv / .tsv.gz / .h.tsv.gz files found in that folder.")
                else:
                    st.success(f"Found **{len(found)}** summary statistics files.")
                    with st.expander("Show files"):
                        for f in found:
                            st.text(f"  {f.name}")

                    if st.button("Load folder", type="primary", key="load_folder_btn"):
                        uploaded_map = st.session_state.uploaded_files.copy()
                        newly = []
                        for f in found:
                            study_id = _derive_study_id(f.name)
                            uploaded_map[study_id] = f  # point directly, no copy
                            newly.append({"study_id": study_id, "filename": f.name,
                                          "source": "local_folder"})
                        st.session_state.uploaded_files = uploaded_map
                        prov = st.session_state.get("provenance")
                        if prov is not None and newly:
                            prov.event("manual_upload", {"files": newly})
                        st.rerun()

    # --- Tab 2: Browser upload (original) ---
    with tab_upload:
        uploaded = st.file_uploader(
            "Drag and drop or click to upload",
            type=["tsv", "gz"],
            accept_multiple_files=True,
            key="dataset_uploader",
        )

        if not uploaded:
            return

        upload_dir = _upload_dir()
        pending_files = {}
        for f in uploaded:
            pending_files[f.name] = f

        st.markdown(f"**{len(pending_files)}** file(s) ready to submit:")
        cols = st.columns([6, 1])
        with cols[0]:
            for name in pending_files:
                st.text(f"  {name}")

        if st.button("Submit uploads", type="primary"):
            uploaded_map = st.session_state.uploaded_files.copy()
            newly = []
            for name, f in pending_files.items():
                study_id = _derive_study_id(name)
                dest = upload_dir / name
                dest.write_bytes(f.getvalue())
                uploaded_map[study_id] = dest
                newly.append({"study_id": study_id, "filename": name,
                              "source": "browser_upload"})

            st.session_state.uploaded_files = uploaded_map
            prov = st.session_state.get("provenance")
            if prov is not None and newly:
                prov.event("manual_upload", {"files": newly})
            st.rerun()


def _derive_study_id(filename: str) -> str:
    """Derive a study ID from an uploaded filename.

    Strips common extensions to get a clean identifier.
    """
    name = filename
    for ext in (".h.tsv.gz", ".tsv.gz", ".tsv", ".gz"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return name


def render():
    st.header("Step 3: Study Selection")

    upload_mode = st.session_state.upload_mode
    uploaded_files = st.session_state.uploaded_files

    # ---------- Normal mode: search catalog first ----------
    if not upload_mode:
        criteria = st.session_state.search_criteria
        if criteria is None:
            st.warning("No search criteria defined. Go back to Step 2.")
            return

        # Search if not done yet
        if not st.session_state.studies_found:
            with st.spinner("Searching GWAS Catalog..."):
                try:
                    prov = st.session_state.get("provenance")
                    if prov is not None:
                        with prov.time_block("catalog_search_results") as scratch:
                            traits, studies = _search_catalog(criteria)
                            scratch["n_traits"] = len(traits)
                            scratch["n_studies"] = len(studies)
                            scratch["studies"] = [
                                {
                                    "study_id": s.study_id,
                                    "title": s.title,
                                    "traits": ", ".join(t.trait_name for t in s.traits),
                                    "sample_size": s.initial_sample_size,
                                    "ancestry": getattr(s, "ancestry", None) or getattr(s, "initial_sample_size", None),
                                    "journal": s.journal,
                                    "date": s.pub_date,
                                    "pubmed_id": getattr(s, "pubmed_id", None),
                                }
                                for s in studies
                            ]
                    else:
                        traits, studies = _search_catalog(criteria)
                    st.session_state.traits_found = traits
                    st.session_state.studies_found = studies
                except Exception as e:
                    st.error(f"Search error: {e}")
                    return

    # ---------- Upload section (both modes) ----------
    _render_upload_section()

    st.markdown("---")

    # ---------- Build combined study table ----------
    studies = st.session_state.studies_found if not upload_mode else []

    # Build rows for catalog studies
    rows = []
    for s in studies:
        catalog_url = f"https://www.ebi.ac.uk/gwas/studies/{s.study_id}"
        pubmed_url = (
            f"https://pubmed.ncbi.nlm.nih.gov/{s.pubmed_id}/"
            if s.pubmed_id else ""
        )
        rows.append({
            "Select": True,
            "Study ID": s.study_id,
            "Title": s.title[:80] + ("..." if len(s.title) > 80 else ""),
            "Journal": s.journal,
            "Date": s.pub_date,
            "Sample Size": s.initial_sample_size,
            "Traits": ", ".join(t.trait_name for t in s.traits),
            "GWAS Catalog": catalog_url,
            "PubMed": pubmed_url,
        })

    # Add rows for uploaded files
    for study_id, path in uploaded_files.items():
        rows.append({
            "Select": True,
            "Study ID": study_id,
            "Title": Path(path).name,
            "Journal": "-",
            "Date": "-",
            "Sample Size": "-",
            "Traits": "-",
            "GWAS Catalog": "",
            "PubMed": "",
        })

    if not rows:
        if upload_mode:
            st.info("Upload your summary statistics files above to get started.")
        else:
            st.warning(
                "No studies with summary statistics found. "
                "Try broadening your search terms in Step 2."
            )
        return

    st.markdown(f"**{len(rows)}** studies available.")

    # Select / Deselect all toggle
    select_all = st.checkbox("Select all studies", value=True)

    df = pd.DataFrame(rows)
    df["Select"] = select_all

    edited_df = st.data_editor(
        df,
        column_config={
            "Select": st.column_config.CheckboxColumn("Select", default=True),
            "GWAS Catalog": st.column_config.LinkColumn("GWAS Catalog", display_text="View"),
            "PubMed": st.column_config.LinkColumn("PubMed", display_text="View"),
        },
        disabled=["Study ID", "Title", "Journal", "Date", "Sample Size", "Traits", "GWAS Catalog", "PubMed"],
        hide_index=True,
        width="stretch",
    )

    selected_ids = set(edited_df.loc[edited_df["Select"], "Study ID"])
    all_ids = list(edited_df["Study ID"])

    def _log_selection() -> None:
        prov = st.session_state.get("provenance")
        if prov is None:
            return
        prov.event("studies_selected", {
            "included": sorted(selected_ids),
            "excluded": sorted(sid for sid in all_ids if sid not in selected_ids),
        })
    selected_catalog = [s for s in studies if s.study_id in selected_ids]
    selected_uploaded = {
        sid: path for sid, path in uploaded_files.items() if sid in selected_ids
    }
    total_selected = len(selected_catalog) + len(selected_uploaded)

    st.markdown(f"**{total_selected}** studies selected.")

    st.markdown("---")
    col1, col2 = st.columns([4, 1])

    # Decide next step label and target
    has_catalog = len(selected_catalog) > 0
    has_uploaded = len(selected_uploaded) > 0

    with col2:
        if has_catalog and not has_uploaded:
            # Only catalog studies → need to download
            btn_label = "Download Data →"
            if st.button(btn_label, type="primary", disabled=total_selected < 2):
                _log_selection()
                st.session_state.selected_studies = selected_catalog
                st.session_state.step = 4
                st.rerun()
        elif has_uploaded and not has_catalog:
            # Only uploaded files → skip download, go to meta-analysis
            btn_label = "Run Meta-Analysis →"
            if st.button(btn_label, type="primary", disabled=total_selected < 2):
                _log_selection()
                st.session_state.selected_studies = []
                st.session_state.downloaded_files = dict(selected_uploaded)
                st.session_state.step = 5
                st.rerun()
        else:
            # Mix of both → download catalog, merge uploaded
            btn_label = "Download & Continue →"
            if st.button(btn_label, type="primary", disabled=total_selected < 2):
                _log_selection()
                st.session_state.selected_studies = selected_catalog
                # Pre-load uploaded files into downloaded_files so step 4
                # only downloads catalog studies, then step 5 sees all.
                st.session_state.uploaded_files = selected_uploaded
                st.session_state.step = 4
                st.rerun()

    if total_selected < 2:
        st.info("Select at least 2 studies to proceed.")
