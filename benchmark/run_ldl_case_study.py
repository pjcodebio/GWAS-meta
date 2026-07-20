#!/usr/bin/env python3
"""LDL cholesterol case study (§3.3) — end-to-end reproduction.

Downloads the two input studies via GWAS Catalog FTP:
  * GCST90019512 -- UK Biobank biomarker LDL-C, 341,875 European
  * GCST90475420 -- MVP lipids LDL-C, 404,745 European (Verma et al. 2024)

Runs the tool's shipped disk-backed pipeline (chunk_studies_to_disk ->
per-chromosome load_chromosome_chunks -> align_studies ->
run_meta_analysis_batch) — i.e. the wrapper is the tool minus the UI,
sharing both the QC code path and config/settings.yaml — then scores
recovery of 30 expected LDL loci (15 canonical + 15
GLGC-2013 new) against the meta output. Recovery criterion: at least
one variant with p_fixed < 5e-8 within +/- 500 kb of the expected
locus position (GRCh38).

Outputs:
  meta_results.csv                  -- full per-variant engine output
  ldl_recovery.csv                  -- per-locus recovery table
  ldl_summary.json                  -- headline pass/fail + counts
  provenance.json                   -- run header (compare to
                                       benchmark/ldl_case_study/provenance_reference.json)

Usage (from repo root):
  python benchmark/run_ldl_case_study.py --out-dir benchmark/ldl_case_study/out
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gwas_meta.data.harmonize import (
    DEFAULT_VALID_HM_CODES,
    align_studies,
    chunk_studies_to_disk,
    load_chromosome_chunks,
)
from gwas_meta.data.qc import compute_lambda_gc_from_file, filter_by_heterogeneity
from gwas_meta.data.sample_overlap import find_shared_cohorts
from gwas_meta.gwas_client.ftp_client import GWASFTPClient
from gwas_meta.gwas_client.rest_client import GWASCatalogClient
from gwas_meta.meta_analysis import run_meta_analysis_batch
from gwas_meta.utils.config import load_config

GCSTS = ["GCST90019512", "GCST90475420"]
STUDY_LABELS = {"GCST90019512": "UKB", "GCST90475420": "MVP"}
GWS = 5e-8
EXPECTED_LOCI_CSV = Path(__file__).parent / "ldl_case_study" / "expected_loci.csv"


def fetch_inputs(cache_dir: Path) -> tuple[dict[str, Path], list]:
    rest = GWASCatalogClient()
    ftp = GWASFTPClient(cache_dir=str(cache_dir))
    paths: dict[str, Path] = {}
    studies: list = []
    for gcst in GCSTS:
        print(f"  {gcst} ({STUDY_LABELS[gcst]}):")
        study = None
        try:
            study = rest.get_study(gcst)
        except Exception as e:
            print(f"    REST get_study failed ({e}); FTP will auto-resolve")
        p = ftp.download_harmonized(
            gcst, ftp_path=(study.ftp_path if study else None)
        )
        paths[gcst] = p
        if study is not None:
            studies.append(study)
        print(f"    -> {p} ({p.stat().st_size / 1e6:.0f} MB)")
    return paths, studies


def score_recovery(
    results_df: pd.DataFrame, loci_df: pd.DataFrame, gws_threshold: float = GWS
) -> pd.DataFrame:
    parts = results_df["variant_id"].str.extract(
        r"^chr(?P<chr>[^:]+):(?P<pos>\d+):"
    )
    rdf = results_df.assign(
        chr=parts["chr"].astype(str),
        pos=parts["pos"].astype(int),
    )
    rows = []
    for _, locus in loci_df.iterrows():
        chr_str = str(locus["chr"])
        pos = int(locus["expected_pos_hg38"])
        w = int(locus["window_kb"]) * 1000
        window = rdf[
            (rdf["chr"] == chr_str) & (rdf["pos"] >= pos - w) & (rdf["pos"] <= pos + w)
        ]
        sig = window[window["p_fixed"] < gws_threshold]
        recovered = len(sig) > 0
        # Deterministic lead: lowest p_fixed, then p_random, then variant_id.
        # At the double-precision floor (p_fixed == 0) many variants tie, so a
        # fixed tie-break is needed for run-to-run stability; the specific lead
        # at a floored locus is therefore a convention (see REPRODUCE.md).
        lead = (
            sig.sort_values(["p_fixed", "p_random", "variant_id"]).iloc[0]
            if recovered else None
        )
        rows.append({
            "locus": locus["locus"],
            "tier": locus["tier"],
            "recovered": recovered,
            "n_sig_in_window": int(len(sig)),
            "lead_variant": lead["variant_id"] if recovered else "",
            "lead_p_fixed": float(lead["p_fixed"]) if recovered else np.nan,
            "lead_beta_fixed": float(lead["beta_fixed"]) if recovered else np.nan,
            "lead_p_random": float(lead["p_random"]) if recovered else np.nan,
            "lead_i_squared": float(lead["i_squared"]) if recovered else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", default=".cache/summary_stats")
    ap.add_argument(
        "--q-filter",
        type=float,
        default=0.0,
        help="Cochran's Q p-value threshold for heterogeneity filtering "
             "(matches the Streamlit UI's Step 5 filter). Variants with "
             "Q p-value below this threshold are removed. Default 0 "
             "(disabled) — pass 1e-6 to reproduce the manuscript's "
             "post-Q GWS count.",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)

    # Resolve QC / meta constants from config/settings.yaml — the same source
    # the Streamlit tool reads — so the wrapper and the UI share one source of
    # truth (recorded under config_resolved in ldl_summary.json).
    config = load_config()
    data_cfg = config.get("data", {})
    meta_cfg = config.get("meta_analysis", {})
    valid_hm_codes = data_cfg.get("valid_hm_codes", DEFAULT_VALID_HM_CODES)
    min_study_count = int(data_cfg.get("min_study_count", 2))
    gws_threshold = float(meta_cfg.get("significance_threshold", GWS))
    print(
        f"Config (settings.yaml): valid_hm_codes={valid_hm_codes} "
        f"min_study_count={min_study_count} gws_threshold={gws_threshold:g}"
    )

    t_wall = time.time()

    print("=" * 64)
    print("Step 1 -- Fetch inputs from GWAS Catalog FTP (~1-2 GB total)")
    print("=" * 64)
    inputs, study_meta = fetch_inputs(cache)

    print("\n" + "=" * 64)
    print("Step 1b -- Shared-cohort check (metadata heuristic)")
    print("=" * 64)
    overlap_report = find_shared_cohorts(study_meta)
    if overlap_report.any_found:
        print("  [WARN sample_overlap] Evidence of shared cohorts found:")
        for f in overlap_report.shared_pubmed_ids:
            print(f"    - Shared PubMed ID {f.pubmed_id}: {', '.join(f.study_ids)}")
        for o in overlap_report.shared_cohort_keywords:
            print(f"    - Named biobank '{o.cohort}' in: {', '.join(o.study_ids)}")
        print("  Verify independence from source publications before "
              "interpreting the meta-analysis.")
    else:
        print("  No shared cohort found in metadata (PubMed IDs + biobank "
              "keywords).")
        print("  NOTE: absence of evidence is not evidence of absence — "
              "independence still requires reading the source publications.")

    print("\n" + "=" * 64)
    print("Step 2 -- Per-study λGC (genomic inflation)")
    print("=" * 64)
    lambda_gc: dict[str, float] = {}
    for gcst, path in inputs.items():
        lam = float(compute_lambda_gc_from_file(path))
        lambda_gc[gcst] = lam
        print(f"  {gcst} ({STUDY_LABELS[gcst]}): λGC = {lam:.3f}")

    # Two-pass disk-backed pipeline — identical to the Streamlit tool's Step 5
    # (pages/step5_meta.py) and to the RP wrapper. Pass 1 chunks each study to
    # disk by chromosome; Pass 2 aligns + meta-analyses one chromosome at a
    # time. This is the shipped code path, and it bounds peak RAM to roughly
    # one chromosome x n_studies rather than the whole genome.
    print("\n" + "=" * 64)
    print("Step 3 -- Pass 1: chunk studies to disk by chromosome (tool path)")
    print("=" * 64)
    t0 = time.time()
    chunks_dir = out_dir / "chunks"
    chroms, build_verdicts = chunk_studies_to_disk(
        inputs, chunks_dir, valid_hm_codes=valid_hm_codes,
    )
    t_chunk = time.time() - t0
    print(f"  Chunked {len(inputs)} studies across {len(chroms)} chromosomes "
          f"in {t_chunk:.1f}s")
    for sid, v in sorted(build_verdicts.items()):
        verdict = getattr(v, "verdict", "unknown")
        note = " [WARN: not GRCh38]" if verdict not in ("grch38", "unknown") else ""
        print(f"    {sid}: genome-build verdict = {verdict}{note}")

    print("\n" + "=" * 64)
    print("Step 4 -- Pass 2: per-chromosome align -> meta (tool path)")
    print("=" * 64)
    t0 = time.time()
    frames: list[pd.DataFrame] = []
    # Per-study betas for the genome-wide-significant variants only. The pooled
    # meta output does not carry per-study effects, but the cross-cohort
    # direction concordance and the per-study beta columns of Table 3.3.1 need
    # them. The GWS set is small (tens of thousands of rows), so accumulating
    # its aligned rows across chromosomes stays cheap.
    gws_beta_frames: list[pd.DataFrame] = []
    n_aligned = 0
    for chrom in chroms:
        studies = load_chromosome_chunks(chunks_dir, chrom)
        if len(studies) < min_study_count:
            del studies
            continue
        aligned = align_studies(studies, min_study_count=min_study_count)
        del studies
        if aligned.empty:
            del aligned
            continue
        n_aligned += aligned["variant_id"].nunique()
        res = run_meta_analysis_batch(aligned, as_dataframe=True)
        frames.append(res)
        gws_ids_chrom = set(res.loc[res["p_fixed"] < gws_threshold, "variant_id"])
        if gws_ids_chrom:
            gws_beta_frames.append(
                aligned.loc[
                    aligned["variant_id"].isin(gws_ids_chrom),
                    ["variant_id", "study_id", "beta"],
                ]
            )
        del aligned
        gc.collect()
    t_align_meta = time.time() - t0

    results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    del frames
    shutil.rmtree(chunks_dir, ignore_errors=True)
    print(f"  Aligned: {n_aligned:,}  |  Meta rows: {len(results):,}  "
          f"in {t_align_meta:.1f}s")

    gws_pre_q = int((results["p_fixed"] < gws_threshold).sum())

    # Heterogeneity narrative + per-locus recovery (Table 3.3.1 and the §3.3
    # heterogeneity discussion) are computed on the RAW (pre-Q) meta. The
    # Cochran-Q filter below is a post-hoc clean-up that defines only the
    # reported top-results counts (Table 3.3.2); it does not change recovery.
    preq_gws_mask = results["p_fixed"] < gws_threshold
    median_i2_gws = (
        float(results.loc[preq_gws_mask, "i_squared"].median())
        if int(preq_gws_mask.sum()) else float("nan")
    )
    # Per-study betas for the raw GWS set, pivoted wide (variant_id x study_id).
    gws_long = (
        pd.concat(gws_beta_frames, ignore_index=True) if gws_beta_frames
        else pd.DataFrame(columns=["variant_id", "study_id", "beta"])
    )
    del gws_beta_frames
    wide = gws_long.pivot_table(
        index="variant_id", columns="study_id", values="beta", aggfunc="first"
    ).dropna()
    del gws_long
    # Cross-cohort direction concordance over the raw GWS set: fraction of GWS
    # variants where the two cohorts' per-study betas agree in sign.
    direction_concordance_gws = (
        float((np.sign(wide.iloc[:, 0]) == np.sign(wide.iloc[:, 1])).mean())
        if len(wide) else float("nan")
    )
    gc.collect()

    # Cochran-Q filter → the reported GWS / suggestive counts (Table 3.3.2).
    if args.q_filter > 0:
        print("\n" + "=" * 64)
        print(f"Step 5b -- Cochran's Q heterogeneity filter (p < {args.q_filter:.0e})")
        print("=" * 64)
        results_q = filter_by_heterogeneity(results, q_pvalue_threshold=args.q_filter)
        n_het_removed = int(len(results)) - int(len(results_q))
        print(f"  Removed {n_het_removed:,} variants for extreme heterogeneity")
    else:
        results_q = results
        n_het_removed = 0
    gws_total = int((results_q["p_fixed"] < GWS).sum())
    suggestive = int((results_q["p_fixed"] < 1e-5).sum())

    # Raw (pre-Q) meta written to disk so it stays consistent with the
    # recovery table scored below.
    meta_out = out_dir / "meta_results.csv"
    results.to_csv(meta_out, index=False)
    print(f"  Wrote {meta_out}")

    print("\n" + "=" * 64)
    print("Step 6 -- Score recovery of expected loci (raw / pre-Q meta)")
    print("=" * 64)
    loci = pd.read_csv(EXPECTED_LOCI_CSV)
    recovery = score_recovery(results, loci, gws_threshold)
    # Attach harmonized per-study betas at each lead variant (beta UKB / beta
    # MVP), reusing the aligned per-study effects already pivoted in `wide`.
    # Every lead is genome-wide-significant, hence present in `wide`; combined
    # with the existing lead_beta_fixed (beta Meta) this makes Table 3.3.1
    # fully reproducible from ldl_recovery.csv alone.
    lead_betas = wide.reindex(recovery["lead_variant"].replace("", np.nan))
    for study_id in wide.columns:
        recovery[f"beta_{study_id}"] = lead_betas[study_id].values
    b0, b1 = wide.columns[0], wide.columns[1]
    recovery["sign"] = np.where(
        recovery[f"beta_{b0}"].notna() & recovery[f"beta_{b1}"].notna(),
        np.where(
            np.sign(recovery[f"beta_{b0}"]) == np.sign(recovery[f"beta_{b1}"]),
            "SAME", "OPP",
        ),
        "",
    )
    del wide
    gc.collect()
    recovery.to_csv(out_dir / "ldl_recovery.csv", index=False)

    n_canonical = int((recovery["tier"] == "canonical").sum())
    n_glgc = int((recovery["tier"] == "glgc_new").sum())
    n_canonical_rec = int(
        ((recovery["tier"] == "canonical") & recovery["recovered"]).sum()
    )
    n_glgc_rec = int(
        ((recovery["tier"] == "glgc_new") & recovery["recovered"]).sum()
    )

    summary = {
        "gcst_inputs": GCSTS,
        "config_resolved": {
            "source": "config/settings.yaml (utils.config.load_config)",
            "valid_hm_codes": list(valid_hm_codes),
            "min_study_count": min_study_count,
            "significance_threshold": gws_threshold,
        },
        "aligned_variants": n_aligned,
        "meta_rows": int(len(results)),
        "gws_variants": gws_total,
        "gws_variants_pre_q": gws_pre_q,
        "suggestive_variants": suggestive,
        "q_filter_threshold": args.q_filter,
        "het_variants_removed": n_het_removed,
        "direction_concordance_gws": direction_concordance_gws,
        "median_i2_gws_percent": median_i2_gws,
        "lambda_gc": lambda_gc,
        "sample_overlap_check": overlap_report.to_dict(),
        "recovery": {
            "canonical": {"recovered": n_canonical_rec, "expected": n_canonical},
            "glgc_new":  {"recovered": n_glgc_rec,      "expected": n_glgc},
            "total":     {"recovered": n_canonical_rec + n_glgc_rec,
                          "expected":  n_canonical + n_glgc},
        },
        "timings_seconds": {
            "chunk": t_chunk, "align_meta": t_align_meta,
            "total_wall": time.time() - t_wall,
        },
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    (out_dir / "ldl_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 64)
    print("HEADLINE RESULTS")
    print("=" * 64)
    print(f"  Canonical loci recovered: {n_canonical_rec} / {n_canonical}")
    print(f"  GLGC-2013 new recovered:  {n_glgc_rec} / {n_glgc}")
    print(f"  Total recovered:          {n_canonical_rec + n_glgc_rec} / "
          f"{n_canonical + n_glgc}")
    if args.q_filter > 0:
        print(f"  GWS variants (pre-Q):     {gws_pre_q:,}")
        print(f"  Removed for extreme het:  {n_het_removed:,}")
        print(f"  GWS variants (post-Q):    {gws_total:,}")
    else:
        print(f"  GWS variants (p < {gws_threshold:g}): {gws_total:,}")
    print(f"  Suggestive (p < 1e-5):    {suggestive:,}")
    print(f"  Aligned variants:         {n_aligned:,}")
    print(f"  Dir. concordance (GWS):   {direction_concordance_gws * 100:.1f} %")
    print(f"  Median I² (GWS set):      {median_i2_gws:.1f} %")
    print(f"  λGC:                      "
          + ", ".join(f"{STUDY_LABELS[g]}={lambda_gc[g]:.3f}" for g in GCSTS))
    print()
    missing = recovery[~recovery["recovered"]]
    if len(missing):
        print(f"  Not recovered ({len(missing)}):")
        for _, r in missing.iterrows():
            print(f"    - {r['locus']} ({r['tier']}, chr{loci.iloc[r.name]['chr']}"
                  f":{loci.iloc[r.name]['expected_pos_hg38']})")
    print(f"\n  Wrote: {out_dir/'meta_results.csv'}")
    print(f"  Wrote: {out_dir/'ldl_recovery.csv'}")
    print(f"  Wrote: {out_dir/'ldl_summary.json'}")
    return 0 if (n_canonical_rec == n_canonical and n_glgc_rec == n_glgc) else 1


if __name__ == "__main__":
    sys.exit(main())
