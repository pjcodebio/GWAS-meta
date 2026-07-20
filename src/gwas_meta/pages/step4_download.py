"""Step 4: Download harmonized summary statistics from FTP."""

import time

import pandas as pd
import streamlit as st

from gwas_meta.data import compute_lambda_gc_from_file
from gwas_meta.gwas_client import GWASFTPClient

_LAMBDA_GC_WARN_THRESHOLD = 1.1


def _fmt_bytes(n: int) -> str:
    """Format a byte count as a human-readable string."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.0f} KB"
    if n < 1024 ** 3:
        return f"{n / 1024 ** 2:.1f} MB"
    return f"{n / 1024 ** 3:.2f} GB"


def render():
    st.header("Step 4: Download Summary Statistics")

    studies = st.session_state.selected_studies
    if not studies:
        st.warning("No studies selected. Go back to Step 3.")
        return

    # Prune any cached downloads that are no longer part of the current
    # selection (user may have gone back to step 3 and deselected studies).
    selected_ids = {s.study_id for s in studies}
    uploaded = st.session_state.get("uploaded_files", {})
    keep_ids = selected_ids | set(uploaded.keys())
    st.session_state.downloaded_files = {
        sid: p for sid, p in st.session_state.downloaded_files.items()
        if sid in keep_ids
    }
    st.session_state.download_errors = {
        sid: e for sid, e in st.session_state.download_errors.items()
        if sid in selected_ids
    }

    # Download only studies that aren't already cached. This lets the
    # user go back to step 3, tick more studies, and return here to
    # download just the new ones (and retry any prior failures).
    pending = [s for s in studies if s.study_id not in st.session_state.downloaded_files]

    if not pending:
        st.success(
            f"Downloaded {len(st.session_state.downloaded_files)} / {len(studies)} studies."
        )
        if st.session_state.download_errors:
            st.warning(
                f"Failed for: {', '.join(st.session_state.download_errors.keys())}"
            )
            for sid, err in st.session_state.download_errors.items():
                st.text(f"  {sid}: {err}")
    else:
        if st.session_state.downloaded_files:
            st.info(
                f"{len(st.session_state.downloaded_files)} already cached -- "
                f"downloading {len(pending)} new study(ies)."
            )
        config = st.session_state.config.get("gwas_catalog", {})
        cache_dir = st.session_state.config.get("data", {}).get("cache_dir", ".cache/summary_stats")
        client = GWASFTPClient(
            host=config.get("ftp_host", "ftp.ebi.ac.uk"),
            base_path=config.get("ftp_base_path", "/pub/databases/gwas/summary_statistics"),
            cache_dir=cache_dir,
        )

        progress = st.progress(0, text="Connecting to FTP server...")
        status_text = st.empty()

        downloaded = dict(st.session_state.downloaded_files)
        errors: dict = {}
        total = len(pending)
        _dl_t0 = time.perf_counter()

        # Download sequentially so we can safely update Streamlit progress
        # from the main thread (Streamlit widgets aren't thread-safe).
        for i, study in enumerate(pending):
            sid = study.study_id
            status_text.text(f"Downloading {sid} ({i + 1}/{total})...")

            # Track bytes for smooth per-file progress
            file_bytes = [0]
            file_size = [0]

            def _on_bytes(n: int) -> None:
                file_bytes[0] += n
                # Compute combined progress: completed studies + fraction of current
                if file_size[0] > 0:
                    file_frac = min(file_bytes[0] / file_size[0], 1.0)
                else:
                    file_frac = 0.0
                overall = (i + file_frac) / total
                progress.progress(
                    min(overall, 0.99),
                    text=(
                        f"Downloading {sid} ({i + 1}/{total})... "
                        f"{_fmt_bytes(file_bytes[0])}"
                        + (f" / {_fmt_bytes(file_size[0])}" if file_size[0] > 0 else "")
                    ),
                )

            # Try to get file size for progress display
            try:
                ftp = client._connect()
                try:
                    rpath = study.ftp_path or client._resolve_ftp_path(sid, ftp)
                    size = client._get_file_size(ftp, rpath)
                    if size:
                        file_size[0] = size
                finally:
                    try:
                        ftp.quit()
                    except Exception:
                        pass
            except Exception:
                rpath = study.ftp_path

            try:
                path = client.download_harmonized(sid, rpath, bytes_callback=_on_bytes)
                downloaded[sid] = path
            except Exception as e:
                msg = str(e) or repr(e)
                errors[sid] = msg

            progress.progress((i + 1) / total, text=f"Downloaded {sid} ({i + 1}/{total})")

        progress.progress(1.0, text="Done!")
        status_text.empty()
        st.session_state.downloaded_files = downloaded
        st.session_state.download_errors = errors

        prov = st.session_state.get("provenance")
        if prov is not None:
            event_name = (
                "download_complete" if not prov.has_event("download_complete")
                else "download_incremental"
            )
            prov.event(
                event_name,
                {
                    "n_requested": total,
                    "n_downloaded_this_batch": sum(
                        1 for s in pending if s.study_id in downloaded
                    ),
                    "n_errors_this_batch": len(errors),
                    "downloaded_study_ids": sorted(downloaded.keys()),
                    "failed_study_ids": sorted(errors.keys()),
                },
                compute_seconds=time.perf_counter() - _dl_t0,
            )

        # Merge any user-uploaded files into the downloaded set
        uploaded = st.session_state.get("uploaded_files", {})
        if uploaded:
            downloaded.update(uploaded)

        if downloaded:
            st.success(f"Successfully downloaded {len(downloaded)} files.")
        if errors:
            st.warning(f"Failed for {len(errors)} studies.")
            for sid, err in errors.items():
                st.text(f"  {sid}: {err}")

    # --- Per-study QC: genomic inflation factor (lambda GC) ---
    downloaded = st.session_state.downloaded_files or {}
    if downloaded:
        _render_qc_table(downloaded)

    # Need at least 2 included studies after QC exclusions
    excluded = st.session_state.get("qc_excluded", set())
    n_ok = sum(1 for sid in downloaded if sid not in excluded)

    st.markdown("---")
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("Run Meta-Analysis →", type="primary", disabled=n_ok < 2):
            prov = st.session_state.get("provenance")
            if prov is not None:
                prov.event(
                    "qc_exclusions",
                    {"excluded": sorted(excluded),
                     "remaining": sorted(sid for sid in downloaded if sid not in excluded)},
                )
            st.session_state.step = 5
            st.rerun()

    if n_ok < 2:
        st.info("Need at least 2 included studies to proceed.")


def _render_qc_table(downloaded: dict) -> None:
    """Compute lambda GC per study (cached) and render exclusion UI."""
    st.markdown("### Per-study quality control")
    st.caption(
        "Genomic inflation factor (λGC). Values close to 1.0 indicate a "
        "clean study. Values above 1.1 may signal population stratification "
        "or other confounding — consider excluding such studies before "
        "running the meta-analysis."
    )

    qc_cache: dict = st.session_state.setdefault("qc_lambda_gc", {})
    excluded: set = st.session_state.setdefault("qc_excluded", set())

    # Compute lambda GC for any newly downloaded study
    pending = [sid for sid in downloaded if sid not in qc_cache]
    if pending:
        progress = st.progress(0, text="Computing λGC...")
        _qc_t0 = time.perf_counter()
        for i, sid in enumerate(pending):
            progress.progress(
                i / len(pending),
                text=f"Computing λGC for {sid} ({i + 1}/{len(pending)})...",
            )
            qc_cache[sid] = compute_lambda_gc_from_file(downloaded[sid])
        progress.empty()
        prov = st.session_state.get("provenance")
        if prov is not None:
            prov.event(
                "qc_lambda_gc",
                {"lambda_gc": {sid: float(qc_cache[sid]) for sid in downloaded
                               if sid in qc_cache and not pd.isna(qc_cache[sid])},
                 "studies_evaluated": list(downloaded.keys())},
                compute_seconds=time.perf_counter() - _qc_t0,
            )

    # Drop excluded entries that are no longer in downloaded (stale state)
    excluded &= set(downloaded.keys())

    # Build display table
    rows = []
    for sid, path in downloaded.items():
        lam = qc_cache.get(sid, float("nan"))
        if pd.isna(lam):
            lam_str = "n/a"
            flag = ""
        elif lam > _LAMBDA_GC_WARN_THRESHOLD:
            lam_str = f"{lam:.3f}"
            flag = "⚠ elevated"
        else:
            lam_str = f"{lam:.3f}"
            flag = "✓"
        rows.append({
            "Study": sid,
            "λGC": lam_str,
            "Status": flag,
            "Include": sid not in excluded,
        })

    df = pd.DataFrame(rows)
    edited = st.data_editor(
        df,
        column_config={
            "Include": st.column_config.CheckboxColumn(
                "Include", help="Uncheck to exclude this study from meta-analysis."
            ),
            "λGC": st.column_config.TextColumn("λGC", help="Genomic inflation factor."),
            "Status": st.column_config.TextColumn("Status"),
            "Study": st.column_config.TextColumn("Study"),
        },
        disabled=["Study", "λGC", "Status"],
        hide_index=True,
        use_container_width=True,
        key="qc_editor",
    )

    # Sync exclusion set with checkbox state
    new_excluded = {row["Study"] for _, row in edited.iterrows() if not row["Include"]}
    if new_excluded != excluded:
        st.session_state.qc_excluded = new_excluded
        st.rerun()
