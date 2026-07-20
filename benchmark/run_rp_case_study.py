#!/usr/bin/env python3
"""Retinitis pigmentosa specificity case study (§3.5) — end-to-end reproduction.

Tests the converse of §3.3: the tool should recover the single common-variant
RP signal that genuinely exists (EYS on chr6) and stay silent everywhere else.

Inputs — both stages of Nishiguchi et al. 2021 (Commun Biol 4:140),
Japanese ancestry:
  * GCST90011892 -- 432 cases / 603 controls
  * GCST90011893 -- 208 cases / 287 controls
Combined 640 cases / 890 controls (matches manuscript).

Success criteria (from manuscript §3.5):
  * Exactly 1 GWS locus at chr6:EYS
  * Lead variant at chr6:64,990,459 (GRCh38); published OR 3.95, P 1.18e-13
  * All GWS variants (p_fixed < 5e-8) on chr6 only

Outputs:
  meta_results.csv          -- full per-variant engine output
  rp_recovery.csv           -- lead-locus table (EYS + GWS-per-chromosome)
  rp_summary.json           -- headline pass/fail

Usage (from repo root):
  python benchmark/run_rp_case_study.py --out-dir benchmark/rp_case_study/out
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
from gwas_meta.data.qc import compute_lambda_gc_from_file
from gwas_meta.data.sample_overlap import find_shared_cohorts
from gwas_meta.gwas_client.ftp_client import GWASFTPClient
from gwas_meta.gwas_client.rest_client import GWASCatalogClient
from gwas_meta.meta_analysis import run_meta_analysis_batch
from gwas_meta.utils.config import load_config

GCSTS = ["GCST90011892", "GCST90011893"]
STUDY_LABELS = {
    "GCST90011892": "Nishiguchi 2021 stage 1 (432/603)",
    "GCST90011893": "Nishiguchi 2021 stage 2 (208/287)",
}
GWS = 5e-8

# Expected lead: EYS on chr6 (GRCh38 position from manuscript §3.5)
EYS_CHR = "6"
EYS_POS = 64_990_459
EYS_WINDOW_KB = 500


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
    results_df: pd.DataFrame, gws_threshold: float = GWS
) -> tuple[pd.DataFrame, dict]:
    parts = results_df["variant_id"].str.extract(
        r"^chr(?P<chr>[^:]+):(?P<pos>\d+):"
    )
    rdf = results_df.assign(
        chr=parts["chr"].astype(str),
        pos=parts["pos"].astype(int),
    )
    gws = rdf[rdf["p_fixed"] < gws_threshold]

    # EYS window on chr6
    w = EYS_WINDOW_KB * 1000
    eys_win = gws[
        (gws["chr"] == EYS_CHR)
        & (gws["pos"] >= EYS_POS - w)
        & (gws["pos"] <= EYS_POS + w)
    ]
    eys_recovered = len(eys_win) > 0
    lead = eys_win.loc[eys_win["p_fixed"].idxmin()] if eys_recovered else None

    # GWS count per chromosome
    gws_by_chr = (
        gws.groupby("chr").size().rename("n_gws").reset_index().sort_values(
            "n_gws", ascending=False
        )
    )

    off_target = int(gws[gws["chr"] != EYS_CHR].shape[0])

    # Direction split and heterogeneity across the GWS set (§3.5.3).
    n_risk = int((gws["beta_fixed"] > 0).sum())
    n_protective = int((gws["beta_fixed"] < 0).sum())
    median_i2 = float(gws["i_squared"].median()) if len(gws) else float("nan")

    rows = [{
        "locus": "EYS",
        "chr": EYS_CHR,
        "expected_pos_hg38": EYS_POS,
        "recovered": eys_recovered,
        "n_sig_in_window": int(len(eys_win)),
        "lead_variant": lead["variant_id"] if eys_recovered else "",
        "lead_p_fixed": float(lead["p_fixed"]) if eys_recovered else np.nan,
        "lead_beta_fixed": float(lead["beta_fixed"]) if eys_recovered else np.nan,
        "lead_i_squared": float(lead["i_squared"]) if eys_recovered else np.nan,
    }]
    recovery_df = pd.DataFrame(rows)

    stats = {
        "gws_total": int(len(gws)),
        "gws_by_chromosome": gws_by_chr.to_dict("records"),
        "gws_off_chr6": off_target,
        "risk_increasing": n_risk,
        "protective": n_protective,
        "median_i2_gws_percent": median_i2,
        "eys_recovered": eys_recovered,
        "eys_lead_variant": lead["variant_id"] if eys_recovered else "",
        "eys_lead_p_fixed": float(lead["p_fixed"]) if eys_recovered else None,
        "eys_lead_beta_fixed": float(lead["beta_fixed"]) if eys_recovered else None,
        "eys_lead_or": float(np.exp(lead["beta_fixed"])) if eys_recovered else None,
        "eys_lead_i_squared": float(lead["i_squared"]) if eys_recovered else None,
    }
    return recovery_df, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", default=".cache/summary_stats")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)

    # Resolve QC / meta constants from config/settings.yaml -- the same source
    # the Streamlit tool reads (utils.config.load_config) -- so the wrapper and
    # the UI share one source of truth. If settings.yaml drifts, the wrapper's
    # headline numbers drift too and stop matching REPRODUCE.md's expected
    # values, surfacing the mismatch instead of masking it.
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
    print("Step 1 -- Fetch inputs from GWAS Catalog FTP")
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
    print("Step 1c -- Per-study λGC (genomic inflation)")
    print("=" * 64)
    lambda_gc: dict[str, float] = {}
    for gcst, path in inputs.items():
        lam = float(compute_lambda_gc_from_file(path))
        lambda_gc[gcst] = lam
        print(f"  {gcst}: λGC = {lam:.3f}")

    # The two passes below mirror the Streamlit tool's Step 5 exactly
    # (pages/step5_meta.py): the wrapper is "the tool minus the UI". Pass 1
    # chunks each study to disk one at a time (peak RAM ~ one study); Pass 2
    # processes one chromosome at a time (peak RAM ~ largest chromosome x
    # n_studies). This is the same disk-backed path the manuscript's Manhattan
    # plots were produced with, so the wrapper exercises the shipped code — not
    # a parallel whole-file implementation.
    print("\n" + "=" * 64)
    print("Step 2 -- Pass 1: chunk studies to disk by chromosome (tool path)")
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
    print("Step 3 -- Pass 2: per-chromosome align -> meta (tool path)")
    print("=" * 64)
    t0 = time.time()
    frames: list[pd.DataFrame] = []
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
        frames.append(run_meta_analysis_batch(aligned, as_dataframe=True))
        del aligned
        gc.collect()
    t_align_meta = time.time() - t0

    results = (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )
    del frames
    gc.collect()
    print(f"  Aligned: {n_aligned:,}  |  Meta rows: {len(results):,}  "
          f"in {t_align_meta:.1f}s")

    # The on-disk chunks are a scratch artifact of the two-pass path; remove
    # them so the out-dir holds only results (chunk_studies_to_disk also wipes
    # this dir on the next run).
    shutil.rmtree(chunks_dir, ignore_errors=True)

    meta_out = out_dir / "meta_results.csv"
    results.to_csv(meta_out, index=False)
    print(f"  Wrote {meta_out}")

    print("\n" + "=" * 64)
    print("Step 5 -- Score EYS recovery + specificity")
    print("=" * 64)
    recovery, stats = score_recovery(results, gws_threshold=gws_threshold)
    recovery.to_csv(out_dir / "rp_recovery.csv", index=False)

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
        "gws_variants": stats["gws_total"],
        "gws_by_chromosome": stats["gws_by_chromosome"],
        "gws_off_chr6": stats["gws_off_chr6"],
        "risk_increasing": stats["risk_increasing"],
        "protective": stats["protective"],
        "median_i2_gws_percent": stats["median_i2_gws_percent"],
        "lambda_gc": lambda_gc,
        "sample_overlap_check": overlap_report.to_dict(),
        "eys": {
            "recovered": stats["eys_recovered"],
            "lead_variant": stats["eys_lead_variant"],
            "lead_p_fixed": stats["eys_lead_p_fixed"],
            "lead_beta_fixed": stats["eys_lead_beta_fixed"],
            "lead_or": stats["eys_lead_or"],
            "lead_i_squared": stats["eys_lead_i_squared"],
        },
        "timings_seconds": {
            "chunk": t_chunk, "align_meta": t_align_meta,
            "total_wall": time.time() - t_wall,
        },
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
    (out_dir / "rp_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 64)
    print("HEADLINE RESULTS")
    print("=" * 64)
    print(f"  Aligned variants:              {n_aligned:,}")
    print(f"  GWS variants (p < {gws_threshold:g}):      {stats['gws_total']:,}")
    print(f"  GWS variants off chr6:         {stats['gws_off_chr6']:,}")
    print(f"  Risk / protective (GWS):       {stats['risk_increasing']} / {stats['protective']}")
    print(f"  Median I² (GWS set):           {stats['median_i2_gws_percent']:.1f}")
    print(f"  λGC (stage 1 / stage 2):       "
          + " / ".join(f"{lambda_gc[g]:.3f}" for g in GCSTS))
    print(f"  EYS locus recovered:           {stats['eys_recovered']}")
    if stats["eys_recovered"]:
        print(f"  EYS lead variant:              {stats['eys_lead_variant']}")
        print(f"  EYS lead p_fixed:              {stats['eys_lead_p_fixed']:.3e}")
        print(f"  EYS lead OR (exp(beta_fixed)): {stats['eys_lead_or']:.3f}")
    print()
    print("  GWS variants per chromosome:")
    for row in stats["gws_by_chromosome"]:
        print(f"    chr{row['chr']:>3}: {row['n_gws']:,}")
    print(f"\n  Wrote: {out_dir/'meta_results.csv'}")
    print(f"  Wrote: {out_dir/'rp_recovery.csv'}")
    print(f"  Wrote: {out_dir/'rp_summary.json'}")

    ok = stats["eys_recovered"] and stats["gws_off_chr6"] == 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
