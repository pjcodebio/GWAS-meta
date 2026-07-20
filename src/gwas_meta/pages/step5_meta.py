"""Step 5: Run meta-analysis on downloaded summary statistics."""

import csv
import shutil
import time
from pathlib import Path

import streamlit as st

from gwas_meta.data import (
    DEFAULT_VALID_HM_CODES,
    align_studies,
    check_trait_compatibility,
    chunk_studies_to_disk,
    filter_by_heterogeneity,
    find_shared_cohorts,
    load_chromosome_chunks,
)
from gwas_meta.meta_analysis import (
    MetaAnalysisResult,
    leave_one_out_max_p,
    run_meta_analysis_batch,
)

_RESULTS_CSV = Path("results/meta_results.csv")

_CSV_HEADER_BASE = [
    "variant_id", "rsid", "beta_fixed", "se_fixed", "z_fixed", "p_fixed",
    "beta_random", "se_random", "z_random", "p_random",
    "q_stat", "i_squared", "tau_squared", "n_studies", "study_ids",
    # Direction-of-effect summary
    "n_pos", "n_neg", "n_zero",
    # Leave-one-out; blank for non-hits or k < 3
    "loo_max_p", "loo_worst_dropped",
]


def _init_results_csv(study_ids: list[str]) -> Path:
    """Create results CSV with header only (overwrites any previous run).

    Per-study beta columns (``beta_<study_id>``) are appended after the
    base columns so the file works as a self-contained results report.
    """
    _RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    header = _CSV_HEADER_BASE + [f"beta_{sid}" for sid in study_ids]
    with open(_RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
    return _RESULTS_CSV


def _append_results_csv(
    results: list[MetaAnalysisResult], study_ids: list[str]
) -> None:
    """Append a batch of results to the CSV (no header).

    Per-study beta values are written in the order of ``study_ids``;
    missing values become blank cells.
    """
    with open(_RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        for r in results:
            per_study = r.per_study_betas or {}
            per_study_cells = [
                ("" if (b := per_study.get(sid)) is None
                 else f"{b:.6g}")
                for sid in study_ids
            ]
            import math
            loo_p_cell = ("" if r.loo_max_p is None or math.isnan(r.loo_max_p)
                          else f"{r.loo_max_p:.6g}")
            writer.writerow([
                r.variant_id, r.rsid,
                r.beta_fixed, r.se_fixed, r.z_fixed, r.p_fixed,
                r.beta_random, r.se_random, r.z_random, r.p_random,
                r.q_stat, r.i_squared, r.tau_squared, r.n_studies,
                ";".join(r.study_ids),
                r.n_pos, r.n_neg, r.n_zero,
                loo_p_cell, r.loo_worst_dropped,
                *per_study_cells,
            ])


def render():
    st.header("Step 5: Meta-Analysis")

    all_downloaded = st.session_state.downloaded_files or {}
    excluded = st.session_state.get("qc_excluded", set())
    downloaded = {sid: path for sid, path in all_downloaded.items() if sid not in excluded}
    if excluded:
        st.info(
            f"Excluded {len(excluded)} stud{'y' if len(excluded) == 1 else 'ies'} "
            f"based on Step 4 QC: {', '.join(sorted(excluded))}"
        )
    if not downloaded or len(downloaded) < 2:
        st.warning("Not enough data. Go back to Step 4.")
        return

    # --- k = 2 guardrail ---
    # With only 2 studies, Cochran's Q has 1 degree of freedom and the
    # DerSimonian-Laird τ² estimator is very unstable — random-effects
    # β, SE, and p_random should not be interpreted at face value.
    # Fixed-effects (IVW) and per-study β remain informative.
    if len(downloaded) == 2:
        st.warning(
            "**Only 2 studies included (k = 2).** The random-effects "
            "estimator (DerSimonian–Laird τ²) has 1 degree of freedom "
            "and is very unstable at k = 2; `beta_random`, `se_random`, "
            "`p_random`, and `tau_squared` should not be interpreted at "
            "face value. Fixed-effects (IVW) and heterogeneity direction "
            "(via `n_pos` / `n_neg` in the CSV) remain informative. For "
            "small-k random-effects inference, Hartung–Knapp–Sidik–"
            "Jonkman is the standard correction; it is not implemented "
            "in this tool."
        )
        prov = st.session_state.get("provenance")
        if prov is not None and not prov.has_event("k2_warning"):
            prov.event("k2_warning", {"n_studies": 2})

    # --- Sample-overlap guardrail ---
    # IVW assumes independent studies. Scan each study's initial-sample
    # description for named cohorts (UK Biobank, FinnGen, ...) and warn
    # when two studies share one. This is a heuristic — a real correction
    # requires bivariate LDSC or MetaSubtract.
    selected_studies = st.session_state.get("selected_studies", []) or []
    included_meta = [s for s in selected_studies if s.study_id in downloaded]

    # --- Trait-mismatch guardrail ---
    # Studies must share at least one EFO trait, otherwise their effect
    # sizes are not comparable. Compares EFO IDs exactly — hierarchical
    # relationships (parent/child terms) are not consulted.
    if len(included_meta) >= 2:
        trait_finding = check_trait_compatibility(included_meta)
        if trait_finding.is_mismatch:
            lines = []
            for sid, pairs in trait_finding.per_study.items():
                if pairs:
                    trait_txt = ", ".join(
                        f"{name} ({efo})" for efo, name in pairs
                    )
                else:
                    trait_txt = "*no EFO trait recorded*"
                lines.append(f"- **{sid}**: {trait_txt}")
            st.warning(
                "**Trait mismatch detected.** The selected studies do not "
                "share any EFO trait. Meta-analysing effect sizes across "
                "different traits produces uninterpretable output.\n\n"
                + "\n".join(lines)
                + "\n\nIf this is intentional (e.g. related sub-phenotypes), "
                "you can proceed; otherwise, return to Step 3 to reselect."
            )
            prov = st.session_state.get("provenance")
            if prov is not None and not prov.has_event("trait_mismatch_warning"):
                prov.event("trait_mismatch_warning", {
                    "per_study_traits": {
                        sid: [{"efo_id": efo, "trait_name": name}
                              for efo, name in pairs]
                        for sid, pairs in trait_finding.per_study.items()
                    },
                })

    if len(included_meta) >= 2:
        # Shared-cohort finder — Option B semantics: always emit a status,
        # never a green tick. Absence of evidence is not evidence of absence.
        report = find_shared_cohorts(included_meta)
        if report.any_found:
            lines: list[str] = []
            for f in report.shared_pubmed_ids:
                lines.append(
                    f"- **Shared PubMed ID {f.pubmed_id}** (same publication): "
                    f"{', '.join(f.study_ids)}"
                )
            for o in report.shared_cohort_keywords:
                lines.append(
                    f"- **{o.cohort}** named in: {', '.join(o.study_ids)}"
                )
            st.warning(
                "**Evidence of shared cohorts found.** IVW meta-analysis "
                "assumes independent studies; shared participants inflate "
                "the meta Z-statistic.\n\n"
                + "\n".join(lines)
                + "\n\nThis is a metadata-based finding (PubMed ID + "
                "well-known-biobank keywords). For a rigorous correction, "
                "consider **MetaSubtract** or **bivariate LDSC**."
            )
        else:
            st.info(
                "**No shared cohort found in metadata.** The tool scanned "
                "the PubMed IDs and sample-size descriptions of the included "
                "studies and did not detect a shared publication or a "
                "well-known biobank named in more than one study. This is "
                "*not* proof of independence — a dataset whose underlying "
                "cohort is not named in the sample description, or whose "
                "sub-cohorts appear under different PubMed IDs, will slip "
                "past both checks. Independence must be verified from the "
                "source publications before results are interpreted."
            )
        prov = st.session_state.get("provenance")
        if prov is not None and not prov.has_event("sample_overlap_check"):
            prov.event("sample_overlap_check", report.to_dict())

    # Already computed?
    if st.session_state.meta_results is not None:
        results = st.session_state.meta_results
        st.success(f"Meta-analysis complete: {len(results)} variants analysed.")
    else:
        config = st.session_state.config.get("data", {})
        valid_codes = config.get("valid_hm_codes", DEFAULT_VALID_HM_CODES)
        min_count = config.get("min_study_count", 2)
        cache_dir = Path(config.get("cache_dir", ".cache/summary_stats"))
        chunks_dir = cache_dir / "chunked"
        sig_threshold = st.session_state.config.get("meta_analysis", {}).get("significance_threshold", 5e-8)
        q_threshold = 1e-6

        prov = st.session_state.get("provenance")
        if prov is not None and not prov.has_event("meta_settings"):
            prov.event("meta_settings", {
                "min_study_count": min_count,
                "valid_hm_codes": valid_codes,
                "q_threshold": q_threshold,
                "sig_threshold": sig_threshold,
                "n_studies_input": len(downloaded),
                "study_ids": sorted(downloaded.keys()),
            })

        # --- Pass 1: Chunk each study by chromosome to disk ---
        # Only one study in RAM at a time → peak ≈ size of largest study.
        status1 = st.status(
            "Pass 1: Chunking studies by chromosome...", expanded=True,
        )
        progress1 = st.progress(0)
        load_errors: list[str] = []

        def _on_study_chunked(study_id: str, i: int, total: int) -> None:
            status1.write(f"Chunked {study_id} ({i}/{total})")
            progress1.progress(i / total)

        try:
            _ck_t0 = time.perf_counter()
            chroms, build_verdicts = chunk_studies_to_disk(
                downloaded, chunks_dir,
                valid_hm_codes=valid_codes,
                progress_callback=_on_study_chunked,
            )
            if prov is not None:
                prov.event(
                    "chunking_complete",
                    {"n_studies": len(downloaded), "n_chromosomes": len(chroms),
                     "chromosomes": list(chroms),
                     "build_verdicts": {
                         sid: {
                             "verdict": v.verdict,
                             "n_matches": v.n_matches,
                             "grch37_hits": v.grch37_hits,
                             "grch38_hits": v.grch38_hits,
                         }
                         for sid, v in build_verdicts.items()
                     }},
                    compute_seconds=time.perf_counter() - _ck_t0,
                )
        except Exception as e:
            st.error(f"Failed during chunking: {e}")
            return

        # --- Genome-build warnings from sentinel-SNP probe ---
        # Flag studies that look like GRCh37 rather than GRCh38, or that
        # have inconsistent sentinel positions. Catalog harmonisation
        # should always produce GRCh38, so a mismatch typically means a
        # user-uploaded file in the wrong build.
        flagged = {sid: v for sid, v in build_verdicts.items()
                   if hasattr(v, "is_warning") and v.is_warning}
        if flagged:
            lines = []
            for sid, v in sorted(flagged.items()):
                if v.verdict == "grch37":
                    lines.append(
                        f"- **{sid}**: appears to be **GRCh37** "
                        f"(sentinel hits: {', '.join(v.grch37_hits)})"
                    )
                elif v.verdict == "mixed":
                    lines.append(
                        f"- **{sid}**: mixed sentinel positions "
                        f"(GRCh37 hits: {', '.join(v.grch37_hits)}, "
                        f"GRCh38 hits: {', '.join(v.grch38_hits)})"
                    )
            st.warning(
                "**Genome-build mismatch detected.** The tool expects "
                "GRCh38-harmonised summary statistics. The following "
                "studies look wrong:\n\n"
                + "\n".join(lines)
                + "\n\nMeta-analysis with GRCh38 studies will likely "
                "yield zero overlapping variants. Reconvert the affected "
                "files to GRCh38 (e.g. with `liftOver`) or drop them."
            )

        progress1.empty()
        status1.update(
            label=f"Pass 1 complete: {len(downloaded)} studies chunked "
                  f"across {len(chroms)} chromosomes",
            state="complete",
        )

        # --- Pass 2: Per-chromosome align → meta-analyse → save ---
        # Peak RAM ≈ largest chromosome × n_studies (typically 1–2 GB).
        # Fix the per-study column order up-front so the CSV header is stable
        # even when a particular study is missing from a given chromosome.
        included_study_ids = sorted(downloaded.keys())
        st.session_state.included_study_ids = included_study_ids
        _init_results_csv(included_study_ids)
        all_results: list[MetaAnalysisResult] = []

        status2 = st.status("Pass 2: Processing chromosomes...", expanded=True)
        progress2 = st.progress(0)

        _am_t0 = time.perf_counter()
        per_chrom_rows: list[dict] = []
        n_skipped = 0
        total_aligned = 0

        for i, chrom in enumerate(chroms):
            status2.update(label=f"Processing chromosome {chrom}...")

            studies = load_chromosome_chunks(chunks_dir, chrom)
            if len(studies) < min_count:
                status2.write(f"chr{chrom}: skipped (fewer than {min_count} studies)")
                per_chrom_rows.append({"chrom": chrom, "n_studies": len(studies),
                                       "aligned_variants": 0, "meta_variants": 0,
                                       "skipped": True})
                n_skipped += 1
                progress2.progress((i + 1) / len(chroms))
                del studies
                continue

            n_studies_chrom = len(studies)
            # Build variant_id → rsid lookup from the chunked studies before
            # alignment frees them. First non-empty rsid across studies wins.
            rsid_map: dict[str, str] = {}
            for _st in studies:
                if "rsid" not in _st.variants.columns:
                    continue
                _vids = _st.variants["variant_id"].values
                _rs = _st.variants["rsid"].values
                for _vid, _r in zip(_vids, _rs):
                    if _r and _vid not in rsid_map:
                        rsid_map[_vid] = _r
            aligned = align_studies(studies, min_study_count=min_count)
            del studies

            if aligned.empty:
                status2.write(f"chr{chrom}: no overlapping variants")
                per_chrom_rows.append({"chrom": chrom, "n_studies": n_studies_chrom,
                                       "aligned_variants": 0, "meta_variants": 0,
                                       "skipped": False})
                progress2.progress((i + 1) / len(chroms))
                del aligned
                continue

            n_vars = aligned["variant_id"].nunique()
            total_aligned += n_vars
            status2.write(f"chr{chrom}: {n_vars} variants aligned — running meta-analysis...")

            chrom_results: list[MetaAnalysisResult] = run_meta_analysis_batch(aligned)

            # Pivot per-variant betas AND SEs to wide form. Betas feed the
            # per-study display columns and the direction summary
            # (n_pos / n_neg / n_zero); SEs are needed for the leave-one-out
            # sensitivity check.
            betas_wide = aligned.pivot(
                index="variant_id", columns="study_id", values="beta"
            )
            ses_wide = aligned.pivot(
                index="variant_id", columns="study_id", values="se"
            )
            beta_lookup = betas_wide.to_dict("index")
            se_lookup = ses_wide.to_dict("index")
            del aligned, betas_wide, ses_wide
            for r in chrom_results:
                raw_b = beta_lookup.get(r.variant_id, {})
                raw_s = se_lookup.get(r.variant_id, {})
                r.per_study_betas = {
                    sid: float(b)
                    for sid, b in raw_b.items()
                    if b is not None and not (isinstance(b, float) and b != b)
                }
                r.per_study_ses = {
                    sid: float(s)
                    for sid, s in raw_s.items()
                    if s is not None and not (isinstance(s, float) and s != s)
                }
                r.rsid = rsid_map.get(r.variant_id, "")
                # Direction-of-effect summary
                r.n_pos = sum(1 for b in r.per_study_betas.values() if b > 0)
                r.n_neg = sum(1 for b in r.per_study_betas.values() if b < 0)
                r.n_zero = sum(1 for b in r.per_study_betas.values() if b == 0)
                # Leave-one-out sensitivity — only for hits with k >= 3.
                # For k = 2, dropping either study leaves k = 1 and no
                # meta-analysis is possible.
                if r.p_fixed < sig_threshold and r.n_studies >= 3:
                    common_sids = sorted(
                        set(r.per_study_betas) & set(r.per_study_ses)
                    )
                    r.loo_max_p, r.loo_worst_dropped = leave_one_out_max_p(
                        [r.per_study_betas[s] for s in common_sids],
                        [r.per_study_ses[s] for s in common_sids],
                        common_sids,
                    )
            del beta_lookup, se_lookup, rsid_map

            _append_results_csv(chrom_results, included_study_ids)
            all_results.extend(chrom_results)
            per_chrom_rows.append({"chrom": chrom, "n_studies": n_studies_chrom,
                                   "aligned_variants": n_vars,
                                   "meta_variants": len(chrom_results),
                                   "skipped": False})
            del chrom_results

            status2.write(f"chr{chrom}: done — {n_vars} variants ✓")
            progress2.progress((i + 1) / len(chroms))

        if prov is not None:
            prov.event(
                "alignment_meta_complete",
                {
                    "n_chroms_processed": len(chroms) - n_skipped,
                    "n_chroms_skipped": n_skipped,
                    "total_aligned_variants": total_aligned,
                    "total_meta_variants": len(all_results),
                    "per_chrom": per_chrom_rows,
                },
                compute_seconds=time.perf_counter() - _am_t0,
            )

        progress2.empty()
        status2.update(label="All chromosomes processed", state="complete")

        if not all_results:
            # Assemble an actionable error listing the likely causes,
            # with the build-verdict summary prepended when relevant.
            build_lines = []
            for sid, v in sorted(build_verdicts.items()):
                if hasattr(v, "verdict") and v.verdict != "grch38":
                    build_lines.append(f"- {sid}: build verdict = **{v.verdict}** "
                                       f"({v.n_matches} sentinel matches)")
            build_block = (
                "\n\n**Sentinel-SNP build check (not all GRCh38):**\n"
                + "\n".join(build_lines)
            ) if build_lines else ""
            st.error(
                "**No overlapping variants found across studies.** "
                "Likely causes, in order of probability:\n\n"
                "1. **Mixed genome builds** — e.g. one study is GRCh37 "
                "and another is GRCh38. Positions won't match. "
                "Verify each source file and re-lift to GRCh38 as needed.\n"
                "2. **Different variant call sets** — studies from very "
                "different genotyping arrays or imputation panels can "
                "share few common variants after QC.\n"
                "3. **QC filters too strict** — MAF threshold, `hm_code` "
                "filter, or plausibility bounds may have removed too "
                "many variants. Loosen in `config/settings.yaml` and "
                "retry.\n"
                "4. **min_study_count** setting too high for the number "
                "of studies selected."
                + build_block
            )
            return

        # Post-meta QC: heterogeneity filter (default threshold 1e-6)
        q_threshold = 1e-6
        if q_threshold > 0:
            import pandas as _pd

            results_df = _pd.DataFrame([
                {
                    "variant_id": r.variant_id,
                    "rsid": r.rsid,
                    "beta_fixed": r.beta_fixed, "se_fixed": r.se_fixed,
                    "z_fixed": r.z_fixed, "p_fixed": r.p_fixed,
                    "beta_random": r.beta_random, "se_random": r.se_random,
                    "z_random": r.z_random, "p_random": r.p_random,
                    "q_stat": r.q_stat, "i_squared": r.i_squared,
                    "tau_squared": r.tau_squared, "n_studies": r.n_studies,
                    "study_ids": r.study_ids,
                    "per_study_betas": r.per_study_betas,
                    "per_study_ses": r.per_study_ses,
                    "n_pos": r.n_pos, "n_neg": r.n_neg, "n_zero": r.n_zero,
                    "loo_max_p": r.loo_max_p,
                    "loo_worst_dropped": r.loo_worst_dropped,
                }
                for r in all_results
            ])
            n_before = len(results_df)
            results_df = filter_by_heterogeneity(results_df, q_pvalue_threshold=q_threshold)
            n_het_removed = n_before - len(results_df)

            if n_het_removed > 0:
                st.info(
                    f"Heterogeneity filter: removed {n_het_removed:,} variants "
                    f"(Cochran's Q p < {q_threshold:.0e})"
                )
                # Rebuild result list from filtered DataFrame
                all_results = [
                    MetaAnalysisResult(
                        variant_id=row.variant_id,
                        beta_fixed=row.beta_fixed, se_fixed=row.se_fixed,
                        z_fixed=row.z_fixed, p_fixed=row.p_fixed,
                        beta_random=row.beta_random, se_random=row.se_random,
                        z_random=row.z_random, p_random=row.p_random,
                        q_stat=row.q_stat, i_squared=row.i_squared,
                        tau_squared=row.tau_squared, n_studies=int(row.n_studies),
                        study_ids=row.study_ids if isinstance(row.study_ids, list) else [],
                        per_study_betas=row.per_study_betas if isinstance(row.per_study_betas, dict) else {},
                        per_study_ses=row.per_study_ses if isinstance(row.per_study_ses, dict) else {},
                        rsid=row.rsid if isinstance(row.rsid, str) else "",
                        n_pos=int(row.n_pos), n_neg=int(row.n_neg),
                        n_zero=int(row.n_zero),
                        loo_max_p=float(row.loo_max_p),
                        loo_worst_dropped=(row.loo_worst_dropped
                                           if isinstance(row.loo_worst_dropped, str)
                                           else ""),
                    )
                    for row in results_df.itertuples(index=False)
                ]
            del results_df

        st.session_state.meta_results = all_results
        st.session_state.aligned_data = None
        if prov is not None:
            n_sig = sum(1 for r in all_results if r.p_fixed < sig_threshold)
            n_sug = sum(1 for r in all_results if r.p_fixed < 1e-5)
            prov.event("meta_results_summary", {
                "n_total": len(all_results),
                "n_significant": n_sig,
                "n_suggestive": n_sug,
                "n_heterogeneity_removed": locals().get("n_het_removed", 0),
            })
        st.success(f"Meta-analysis complete: {len(all_results)} variants analysed.")
        st.info(f"Results saved to `{_RESULTS_CSV}` ({len(all_results):,} rows)")

        # Clean up all cached data (downloads + chunks)
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir, ignore_errors=True)
            cache_dir.mkdir(parents=True, exist_ok=True)

    # Show quick summary
    results = st.session_state.meta_results
    sig_threshold = st.session_state.config.get("meta_analysis", {}).get(
        "significance_threshold", 5e-8
    )
    n_sig = sum(1 for r in results if r.p_fixed < sig_threshold)
    st.metric("Genome-wide significant hits", n_sig)
    st.metric("Total variants analysed", len(results))

    st.markdown("---")
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("View Results →", type="primary"):
            st.session_state.step = 6
            st.rerun()
