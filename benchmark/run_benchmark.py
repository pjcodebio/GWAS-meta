#!/usr/bin/env python3
"""Full-genome gout benchmark: reproduce Table 3.2.2 (a, b, c).

Runs the shipped tool on FinnGen R12 GOUT_STRICT + pan-UKBB gout
harmonized inputs, joins the engine output against the published
reference meta-analysis, and emits:

- Table 3.2.2a metrics (all matched + reference-significant subset):
    Pearson r on beta, SE, -log10(p); direction concordance;
    mean / max |Delta beta|; GWS-variant counts (engine / ref / shared).
- Table 3.2.2b variant accounting: raw input rows, after-QC counts
  from the loader, aligned overlap.
- (with --full-accounting) the per-chromosome four-way partition of the
  reference-only variants: in-raw-removed-by-QC / single-cohort /
  in-neither-raw / in-both-post-QC-missed, written to
  variant_accounting_full.csv.
- Table 3.2.2c canonical loci: SLC2A9 rs6449137, ABCG2 rs2231142,
  GCKR rs1260326, SLC22A11/12 rs2164495.
- Timings (chunk / align+meta / write) and total wall clock.

Uses the tool's shipped disk-backed pipeline (chunk_studies_to_disk ->
per-chromosome load_chromosome_chunks -> align_studies ->
run_meta_analysis_batch), with QC / meta constants resolved from
config/settings.yaml — i.e. the benchmark is the tool minus the UI,
matching the RP / LDL wrappers.

Usage (from repo root):
    python benchmark/run_benchmark.py \\
        --inputs-dir "/Volumes/.../benchmark_gout_full_genome_data" \\
        --out-dir    benchmark/gout_output
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import resource
import shutil
import subprocess
import sys
import time


def _peak_rss_gb() -> float:
    """Process-lifetime peak resident set size in GB.

    ``ru_maxrss`` is monotonic (max over the process so far). Units differ by
    platform: macOS reports bytes, Linux reports kilobytes.
    """
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 ** 3) if sys.platform == "darwin" else r / (1024 ** 2)
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataclasses import dataclass

from gwas_meta.data.harmonize import (
    DEFAULT_VALID_HM_CODES,
    align_studies,
    chunk_studies_to_disk,
    load_chromosome_chunks,
)
from gwas_meta.data.sample_overlap import find_shared_cohorts
from gwas_meta.meta_analysis import run_meta_analysis_batch
from gwas_meta.utils.config import load_config


@dataclass
class _StaticStudyStub:
    """Minimal stand-in for a Catalog study record.

    The gout benchmark inputs are not fetched from the Catalog (they are
    pre-processed slices of the FinnGen R12 combined meta output), so no
    Catalog metadata exists to feed the shared-cohort finder. We assemble
    a static stub carrying the two cohorts' known biobank names and paper
    references so the same heuristic runs here for reporting consistency.
    """

    study_id: str
    initial_sample_size: str
    pubmed_id: str | None

GWS = 5e-8
CANONICAL_RSIDS = ["rs6449137", "rs2231142", "rs1260326", "rs2164495"]
CANONICAL_LABELS = {
    "rs6449137": "SLC2A9",
    "rs2231142": "ABCG2 Q141K",
    "rs1260326": "GCKR",
    "rs2164495": "SLC22A11/12",
}


def count_gz_rows(path: Path) -> int:
    out = subprocess.check_output(
        f"gunzip -c '{path}' | wc -l", shell=True
    ).decode().strip()
    return int(out.split()[0]) - 1


def load_expected(path: Path) -> pd.DataFrame:
    usecols = [
        "chromosome", "base_pair_location",
        "effect_allele", "other_allele",
        "beta", "standard_error", "p_value", "rsid",
    ]
    df = pd.read_csv(
        path, sep="\t", comment="#", usecols=usecols,
        dtype={"effect_allele": str, "other_allele": str, "rsid": str},
    )
    ea = df["effect_allele"].str.strip().str.upper()
    oa = df["other_allele"].str.strip().str.upper()
    snp_mask = (ea.str.len() == 1) & (oa.str.len() == 1)
    df = df[snp_mask].copy()
    ea = ea[snp_mask]
    oa = oa[snp_mask]
    first = np.where(ea <= oa, ea, oa)
    second = np.where(ea <= oa, oa, ea)
    df["variant_id"] = (
        "chr" + df["chromosome"].astype(str)
        + ":" + df["base_pair_location"].astype(str)
        + ":" + first + ":" + second
    )
    df = df.drop_duplicates(subset="variant_id")
    return df


def concordance(sub: pd.DataFrame, label: str) -> dict:
    if len(sub) == 0:
        return {"label": label, "n": 0}
    beta_e = sub["beta_e"].astype(float)
    beta_r = sub["beta_r"].astype(float)
    se_e = sub["se_e"].astype(float)
    se_r = sub["se_r"].astype(float)
    p_e = sub["p_e"].astype(float).clip(lower=1e-300)
    p_r = sub["p_r"].astype(float).clip(lower=1e-300)
    dbeta = (beta_e - beta_r).abs()
    return {
        "label": label,
        "n": int(len(sub)),
        "pearson_r_beta": float(beta_e.corr(beta_r)),
        "pearson_r_se": float(se_e.corr(se_r)),
        "pearson_r_neglog10p": float((-np.log10(p_e)).corr(-np.log10(p_r))),
        "direction_concordance": float(
            (np.sign(beta_e) == np.sign(beta_r)).mean()
        ),
        "mean_abs_dbeta": float(dbeta.mean()),
        "max_abs_dbeta": float(dbeta.max()),
    }


def _chrom_sort_key(lab: str):
    c = lab[3:] if lab.startswith("chr") else lab
    return (0, int(c)) if c.isdigit() else (1, c)


def _spill_from_series(vid_series: pd.Series, tmpdir: Path, tag: str) -> set:
    """Write per-chromosome key files from an in-memory variant_id Series
    (deduplicated). The chromosome label is parsed from the key prefix so it
    matches every other source exactly. Returns the set of labels written."""
    vids = pd.unique(vid_series.astype(str))
    labs = pd.Series(vids).str.split(":", n=1).str[0]
    frame = pd.DataFrame({"vid": vids, "lab": labs.values})
    written = set()
    for lab, g in frame.groupby("lab", sort=False):
        (tmpdir / f"{tag}_{lab}.txt").write_text("\n".join(g["vid"].tolist()))
        written.add(lab)
    return written


def _spill_from_file(path: Path, tmpdir: Path, tag: str,
                     chunksize: int = 2_000_000) -> set:
    """Stream a harmonized TSV in chunks, build canonical variant_ids
    (chr{chrom}:{pos}:{sorted alleles} — the same key space as the engine),
    and append them to per-chromosome key files.

    Used for the raw cohort inputs *and* the published reference. Unlike the
    engine loader, this keeps every representable variant (SNPs, indels, and
    both alleles of multi-allelic sites) so that the reference/raw key universe
    is complete. Variants the engine drops as indels or multi-allelic therefore
    surface in the "removed by QC" category rather than vanishing from the
    accounting. Memory is bounded by chunksize: no genome-wide key set is ever
    held in RAM."""
    usecols = ["chromosome", "base_pair_location", "effect_allele", "other_allele"]
    handles: dict = {}
    written = set()
    reader = pd.read_csv(
        path, sep="\t", comment="#", usecols=usecols,
        dtype={"effect_allele": str, "other_allele": str, "chromosome": str},
        chunksize=chunksize,
    )
    for chunk in reader:
        ea = chunk["effect_allele"].astype(str).str.strip().str.upper()
        oa = chunk["other_allele"].astype(str).str.strip().str.upper()
        valid = (
            (ea.str.len() >= 1) & (oa.str.len() >= 1)
            & ~ea.isin(["", "NAN", "NA", "."]) & ~oa.isin(["", "NAN", "NA", "."])
        )
        if not valid.any():
            continue
        c = chunk["chromosome"][valid].astype(str)
        pos = chunk["base_pair_location"][valid].astype(str)
        ea_v = ea[valid].values
        oa_v = oa[valid].values
        first = np.where(ea_v <= oa_v, ea_v, oa_v)
        second = np.where(ea_v <= oa_v, oa_v, ea_v)
        vid = "chr" + c.values + ":" + pos.values + ":" + first + ":" + second
        lab = "chr" + c.values
        frame = pd.DataFrame({"vid": vid, "lab": lab})
        for lb, g in frame.groupby("lab", sort=False):
            h = handles.get(lb)
            if h is None:
                h = (tmpdir / f"{tag}_{lb}.txt").open("w")
                handles[lb] = h
                written.add(lb)
            h.write("\n".join(g["vid"].tolist()))
            h.write("\n")
    for h in handles.values():
        h.close()
    return written


def _load_keyset(tmpdir: Path, tag: str, lab: str) -> set:
    p = tmpdir / f"{tag}_{lab}.txt"
    if not p.is_file():
        return set()
    return {x for x in p.read_text().split("\n") if x}


def full_accounting(fg_raw: Path, ukbb_raw: Path, ref_path: Path,
                    qc_fg_keys: pd.Series, qc_ukbb_keys: pd.Series,
                    engine_keys: pd.Series,
                    outdir: Path) -> dict:
    """Per-chromosome reconciliation of the reference-only variants into the
    four mutually-exclusive categories of the §3.2.2 gap table.

    For each reference-unique canonical key R that is NOT in the engine output:
      - missed      R passed QC in BOTH cohorts yet is absent from the engine
                    output. Must be 0 — a non-zero value means the engine
                    silently dropped a variant it should have kept.
      - single      R passed QC in exactly one cohort, so it was dropped by the
                    min_study_count = 2 rule.
      - removed_qc  R appears in a raw input but survived QC in neither cohort
                    (removed by MAF / |beta| / SE / indel / multiallelic).
      - neither     R is in neither raw input (reference superset / external).

    Every key set is spilled to a temp directory and re-read one chromosome at a
    time, so peak memory is proportional to the largest single chromosome rather
    than the whole genome.
    """
    import shutil
    import tempfile

    tmpdir = Path(tempfile.mkdtemp(prefix="gout_acct_"))
    try:
        print("  spilling engine / post-QC keys by chromosome...")
        labs = set()
        labs |= _spill_from_series(qc_fg_keys, tmpdir, "qcfg")
        labs |= _spill_from_series(qc_ukbb_keys, tmpdir, "qcuk")
        labs |= _spill_from_series(engine_keys, tmpdir, "eng")
        print("  streaming reference for full (indel-inclusive) key universe...")
        labs |= _spill_from_file(ref_path, tmpdir, "ref")
        print("  streaming raw FinnGen for pre-QC keys...")
        labs |= _spill_from_file(fg_raw, tmpdir, "rawfg")
        print("  streaming raw pan-UKBB for pre-QC keys...")
        labs |= _spill_from_file(ukbb_raw, tmpdir, "rawuk")

        rows = []
        tot = {k: 0 for k in
               ("ref_only", "engine_only", "missed", "single", "removed_qc", "neither")}
        for lab in sorted(labs, key=_chrom_sort_key):
            rawfg = _load_keyset(tmpdir, "rawfg", lab)
            rawuk = _load_keyset(tmpdir, "rawuk", lab)
            qcfg = _load_keyset(tmpdir, "qcfg", lab)
            qcuk = _load_keyset(tmpdir, "qcuk", lab)
            eng = _load_keyset(tmpdir, "eng", lab)
            ref = _load_keyset(tmpdir, "ref", lab)

            ref_only = ref - eng
            engine_only = eng - ref
            both_qc = qcfg & qcuk
            missed = ref_only & both_qc
            rem = ref_only - missed
            single = rem & (qcfg | qcuk)
            rem2 = rem - single
            removed_qc = rem2 & (rawfg | rawuk)
            neither = rem2 - removed_qc

            rows.append({
                "chrom": lab,
                "ref_unique": len(ref),
                "engine": len(eng),
                "ref_only": len(ref_only),
                "engine_only": len(engine_only),
                "in_both_postqc_missed": len(missed),
                "single_cohort": len(single),
                "removed_by_qc": len(removed_qc),
                "in_neither_raw": len(neither),
            })
            tot["ref_only"] += len(ref_only)
            tot["engine_only"] += len(engine_only)
            tot["missed"] += len(missed)
            tot["single"] += len(single)
            tot["removed_qc"] += len(removed_qc)
            tot["neither"] += len(neither)
            del rawfg, rawuk, qcfg, qcuk, eng, ref
            gc.collect()

        per_chrom = pd.DataFrame(rows)
        per_chrom.to_csv(outdir / "variant_accounting_full.csv", index=False)

        gap = tot["missed"] + tot["single"] + tot["removed_qc"] + tot["neither"]
        totals = {
            "reference_only_gap": tot["ref_only"],
            "engine_only": tot["engine_only"],
            "in_raw_removed_by_qc": tot["removed_qc"],
            "single_cohort": tot["single"],
            "in_neither_raw": tot["neither"],
            "in_both_postqc_missed": tot["missed"],
            "categorized_total": gap,
            "reconciles": bool(gap == tot["ref_only"]),
            "pct_removed_by_qc": (100.0 * tot["removed_qc"] / gap) if gap else 0.0,
            "pct_single_cohort": (100.0 * tot["single"] / gap) if gap else 0.0,
        }
        return {"totals": totals, "per_chrom": per_chrom}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--full-accounting", action="store_true",
        help="Also compute the per-chromosome four-way variant accounting that "
             "reproduces the §3.2.2 gap table (in-raw-removed-by-QC / single-cohort "
             "/ in-neither-raw / in-both-post-QC-missed). Streams the raw inputs a "
             "second time and works one chromosome at a time; adds a few minutes.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(levelname)s: %(message)s",
    )

    inputs = Path(args.inputs_dir)
    fg_path = inputs / "input_finngen.h.tsv.gz"
    ukbb_path = inputs / "input_ukbb.h.tsv.gz"
    expected_path = inputs / "expected_meta.h.tsv.gz"
    for p in (fg_path, ukbb_path, expected_path):
        if not p.is_file():
            print(f"ERROR: missing input: {p}", file=sys.stderr)
            return 2

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolve QC / meta constants from config/settings.yaml — the same source
    # the Streamlit tool reads (utils.config.load_config) — so the benchmark and
    # the UI share one source of truth and the gout QC can't silently drift from
    # the tool's (e.g. a valid_hm_codes edit).
    config = load_config()
    data_cfg = config.get("data", {})
    valid_hm_codes = data_cfg.get("valid_hm_codes", DEFAULT_VALID_HM_CODES)
    min_study_count = int(data_cfg.get("min_study_count", 2))
    print(
        f"Config (settings.yaml): valid_hm_codes={valid_hm_codes} "
        f"min_study_count={min_study_count}"
    )
    print()

    t_wall_start = time.time()

    print(f"FinnGen input:  {fg_path}")
    print(f"UKBB input:     {ukbb_path}")
    print(f"Expected meta:  {expected_path}")
    print()

    print("Counting raw rows in input files (may take ~30 s)...")
    t0 = time.time()
    n_fg_raw = count_gz_rows(fg_path)
    n_ukbb_raw = count_gz_rows(ukbb_path)
    n_ref_raw = count_gz_rows(expected_path)
    print(f"  Raw rows -- FinnGen: {n_fg_raw:,}   UKBB: {n_ukbb_raw:,}   Expected: {n_ref_raw:,}")
    print(f"  (raw counting took {time.time() - t0:.1f}s)")
    print()

    inputs_map = {"FinnGen": fg_path, "UKBB": ukbb_path}

    # Two-pass disk-backed pipeline — identical to the Streamlit tool's Step 5
    # (pages/step5_meta.py) and to the RP / LDL wrappers. Pass 1 chunks each
    # study to disk by chromosome (peak RAM ~ one study); Pass 2 aligns +
    # meta-analyses one chromosome at a time (peak RAM ~ largest chromosome x
    # n_studies). This exercises the shipped loader + QC, not a parallel
    # whole-file loader.
    print("Pass 1: chunking studies to disk by chromosome (tool path)...")
    t0 = time.time()
    chunks_dir = outdir / "chunks"
    chroms, build_verdicts = chunk_studies_to_disk(
        inputs_map, chunks_dir, valid_hm_codes=valid_hm_codes,
    )
    t_chunk = time.time() - t0
    print(f"  Chunked {len(inputs_map)} studies across {len(chroms)} chromosomes "
          f"in {t_chunk:.1f}s")
    for sid, v in sorted(build_verdicts.items()):
        verdict = getattr(v, "verdict", "unknown")
        note = " [WARN: not GRCh38]" if verdict not in ("grch38", "unknown") else ""
        print(f"    {sid}: genome-build verdict = {verdict}{note}")
    print()

    print("Pass 2: per-chromosome align -> meta (tool path)...")
    t0 = time.time()
    frames: list[pd.DataFrame] = []
    n_aligned = 0
    # Per-study post-QC variant counts (Table 3.2.2b) and — when
    # --full-accounting is set — the full per-study post-QC key sets for the
    # four-way accounting. The on-disk chunks are already QC'd, so summing per
    # study across chromosomes recovers exactly what load_harmonized_file's
    # whole-file post-QC count/key set gave in the single-pass version.
    qc_counts: dict[str, int] = {sid: 0 for sid in inputs_map}
    qc_key_frames: dict[str, list[pd.Series]] | None = (
        {sid: [] for sid in inputs_map} if args.full_accounting else None
    )
    for chrom in chroms:
        studies = load_chromosome_chunks(chunks_dir, chrom)
        for s in studies:
            qc_counts[s.study_id] = qc_counts.get(s.study_id, 0) + len(s.variants)
            if qc_key_frames is not None:
                qc_key_frames[s.study_id].append(s.variants["variant_id"])
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
    # Peak RSS through the two-pass pipeline (chunk + align + meta), captured
    # before the validation-only reference load. This is the memory a normal
    # (non-benchmark) run needs; the commodity-laptop claim rests on it.
    peak_pipeline_gb = _peak_rss_gb()

    results_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    del frames
    # The on-disk chunks are a scratch artifact of the two-pass path; remove them
    # so the out-dir holds only results.
    shutil.rmtree(chunks_dir, ignore_errors=True)

    n_fg_qc = qc_counts["FinnGen"]
    n_ukbb_qc = qc_counts["UKBB"]
    print(f"  FinnGen after QC: {n_fg_qc:,}")
    print(f"  UKBB after QC:    {n_ukbb_qc:,}")
    print(f"  Overlapping variants: {n_aligned:,}")
    print(f"  Engine output rows:   {len(results_df):,}")
    print(f"  Chunk / align+meta time: {t_chunk:.1f}s / {t_align_meta:.1f}s")
    print()

    qc_fg_keys = qc_ukbb_keys = None
    if args.full_accounting:
        qc_fg_keys = (
            pd.concat(qc_key_frames["FinnGen"], ignore_index=True)
            if qc_key_frames["FinnGen"] else pd.Series(dtype=str)
        )
        qc_ukbb_keys = (
            pd.concat(qc_key_frames["UKBB"], ignore_index=True)
            if qc_key_frames["UKBB"] else pd.Series(dtype=str)
        )
    del qc_key_frames
    gc.collect()

    print("Loading reference meta-analysis and joining...")
    t0 = time.time()
    expected_df = load_expected(expected_path)
    engine_slim = results_df[["variant_id", "beta_fixed", "se_fixed", "p_fixed"]].rename(
        columns={"beta_fixed": "beta_e", "se_fixed": "se_e", "p_fixed": "p_e"}
    )
    ref_slim = expected_df[["variant_id", "beta", "standard_error", "p_value", "rsid"]].rename(
        columns={"beta": "beta_r", "standard_error": "se_r", "p_value": "p_r"}
    )
    merged = engine_slim.merge(ref_slim, on="variant_id", how="inner")
    t_write = time.time() - t0
    print(f"  Matched engine cap reference: {len(merged):,}")
    print(f"  Reference (post-SNP-filter):  {len(expected_df):,}")
    print(f"  Load+join time: {t_write:.1f}s")
    del expected_df, engine_slim, ref_slim
    gc.collect()
    print()

    all_stats = concordance(merged, "all_matched")
    ref_sig_mask = merged["p_r"].astype(float) < GWS
    ref_sig_stats = concordance(merged[ref_sig_mask], "reference_significant")

    gws_engine = int((merged["p_e"].astype(float) < GWS).sum())
    gws_ref = int((merged["p_r"].astype(float) < GWS).sum())
    gws_shared = int(
        ((merged["p_e"].astype(float) < GWS) & (merged["p_r"].astype(float) < GWS)).sum()
    )

    canonical_hits = merged[merged["rsid"].isin(CANONICAL_RSIDS)][
        ["rsid", "variant_id", "beta_e", "se_e", "p_e", "beta_r", "se_r", "p_r"]
    ].copy()
    canonical_hits["locus"] = canonical_hits["rsid"].map(CANONICAL_LABELS)
    canonical_hits = canonical_hits[[
        "locus", "rsid", "variant_id",
        "beta_e", "se_e", "p_e",
        "beta_r", "se_r", "p_r",
    ]]

    accounting = pd.DataFrame([
        {"study": "FinnGen",         "raw_rows": n_fg_raw,   "after_qc": n_fg_qc},
        {"study": "UKBB",            "raw_rows": n_ukbb_raw, "after_qc": n_ukbb_qc},
        {"study": "Reference",       "raw_rows": n_ref_raw,  "after_qc": None},
        {"study": "Aligned_overlap", "raw_rows": None,       "after_qc": n_aligned},
    ])
    accounting.to_csv(outdir / "variant_accounting.csv", index=False)
    canonical_hits.to_csv(outdir / "canonical_loci.csv", index=False)

    full_acct = None
    if args.full_accounting:
        del merged
        gc.collect()
        print()
        print("Computing full per-chromosome variant accounting "
              "(streams raw inputs; a few minutes)...")
        t_acct0 = time.time()
        full_acct = full_accounting(
            fg_path, ukbb_path, expected_path, qc_fg_keys, qc_ukbb_keys,
            results_df["variant_id"], outdir,
        )
        del qc_fg_keys, qc_ukbb_keys
        gc.collect()
        print(f"  Accounting time: {time.time() - t_acct0:.1f}s")

    # Shared-cohort finder — metadata-based, run for reporting consistency
    # with the other case-study wrappers. The two gout inputs are pre-
    # processed slices of the FinnGen R12 combined meta output; the check
    # runs against the known cohort descriptions and PubMed IDs.
    overlap_report = find_shared_cohorts([
        _StaticStudyStub(
            study_id="FinnGen_R12_GOUT_STRICT",
            initial_sample_size="FinnGen R12, GOUT_STRICT phenotype",
            pubmed_id="36653562",  # FinnGen R12 flagship
        ),
        _StaticStudyStub(
            study_id="pan-UKBB_gout",
            initial_sample_size="pan-UK Biobank, gout phenotype",
            pubmed_id=None,
        ),
    ])
    print()
    print("-" * 64)
    print("Shared-cohort check (metadata heuristic)")
    print("-" * 64)
    if overlap_report.any_found:
        print("  [WARN sample_overlap] Evidence of shared cohorts found:")
        for f in overlap_report.shared_pubmed_ids:
            print(f"    - Shared PubMed ID {f.pubmed_id}: {', '.join(f.study_ids)}")
        for o in overlap_report.shared_cohort_keywords:
            print(f"    - Named biobank '{o.cohort}' in: {', '.join(o.study_ids)}")
    else:
        print("  No shared cohort found in metadata.")
        print("  NOTE: absence of evidence is not evidence of absence — "
              "verify independence from source publications.")

    summary = {
        "table_3_2_2a": {
            "all_matched": all_stats,
            "reference_significant": ref_sig_stats,
            "gws_variants": {
                "engine": gws_engine, "reference": gws_ref, "shared": gws_shared,
            },
        },
        "table_3_2_2b": {
            "raw": {"finngen": n_fg_raw, "ukbb": n_ukbb_raw, "reference": n_ref_raw},
            "after_qc": {"finngen": n_fg_qc, "ukbb": n_ukbb_qc},
            "aligned_overlap": n_aligned,
        },
        "table_3_2_2_full_accounting": (full_acct["totals"] if full_acct else None),
        "sample_overlap_check": overlap_report.to_dict(),
        "timings_seconds": {
            "chunk": t_chunk, "align_meta": t_align_meta,
            "load_and_join_reference": t_write,
            "total_wall": time.time() - t_wall_start,
        },
        "memory_gb": {
            "pipeline_peak": peak_pipeline_gb,
            "total_peak": _peak_rss_gb(),
            "note": "ru_maxrss (RUSAGE_SELF); pipeline_peak excludes the "
                    "validation-only reference load",
        },
    }
    with (outdir / "benchmark_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("=" * 64)
    print("TABLE 3.2.2a -- CONCORDANCE (engine vs published reference)")
    print("=" * 64)
    for label, s in (("All matched", all_stats), ("Reference-significant", ref_sig_stats)):
        print(f"\n{label}   (n = {s['n']:,})")
        print(f"  Pearson r (beta):        {s.get('pearson_r_beta', float('nan')):.6f}")
        print(f"  Pearson r (SE):          {s.get('pearson_r_se', float('nan')):.6f}")
        print(f"  Pearson r (-log10 p):    {s.get('pearson_r_neglog10p', float('nan')):.6f}")
        print(f"  Direction concordance:   {s.get('direction_concordance', float('nan')) * 100:.4f} %")
        print(f"  Mean |dBeta|:            {s.get('mean_abs_dbeta', float('nan')):.3e}")
        print(f"  Max  |dBeta|:            {s.get('max_abs_dbeta', float('nan')):.3e}")
    print(f"\nGWS variants (p < 5e-8): engine {gws_engine:,} / ref {gws_ref:,} / shared {gws_shared:,}")

    print("\n" + "=" * 64)
    print("TABLE 3.2.2b -- VARIANT ACCOUNTING")
    print("=" * 64)
    print(accounting.to_string(index=False))

    if full_acct is not None:
        T = full_acct["totals"]
        print("\n" + "=" * 64)
        print("TABLE 3.2.2 -- FULL VARIANT ACCOUNTING (per-chromosome)")
        print("=" * 64)
        print(f"  Reference-only gap (unique keys):   {T['reference_only_gap']:,}")
        print(f"  Engine-only (must be 0):            {T['engine_only']:,}")
        print("  ---- partition of the gap ----")
        print(f"  In raw input, removed by QC:        {T['in_raw_removed_by_qc']:,}  ({T['pct_removed_by_qc']:.2f} %)")
        print(f"  Single-cohort (min_study_count={min_study_count}):  {T['single_cohort']:,}  ({T['pct_single_cohort']:.2f} %)")
        print(f"  In neither raw input:               {T['in_neither_raw']:,}")
        print(f"  In both post-QC, missed (bug if >0): {T['in_both_postqc_missed']:,}")
        print(f"  Categorized total reconciles to gap: {T['reconciles']}")

    print("\n" + "=" * 64)
    print("TABLE 3.2.2c -- CANONICAL LOCI")
    print("=" * 64)
    if len(canonical_hits) == 0:
        print("(no canonical rsIDs matched -- check reference rsid column)")
    else:
        for _, r in canonical_hits.iterrows():
            print(f"\n  {r['locus']:14s} {r['rsid']}  ({r['variant_id']})")
            print(f"    engine     beta = {r['beta_e']:+.4f}   se = {r['se_e']:.4f}   p = {r['p_e']:.2e}")
            print(f"    reference  beta = {float(r['beta_r']):+.4f}   se = {float(r['se_r']):.4f}   p = {float(r['p_r']):.2e}")

    t = summary["timings_seconds"]
    print("\n" + "=" * 64)
    print("TIMINGS")
    print("=" * 64)
    print(f"  chunk      : {t['chunk']:.1f}s")
    print(f"  align+meta : {t['align_meta']:.1f}s")
    print(f"  write      : {t['load_and_join_reference']:.1f}s")
    print(f"  total      : {t['total_wall']:.1f}s")

    print(f"\nWrote: {outdir / 'benchmark_summary.json'}")
    print(f"Wrote: {outdir / 'variant_accounting.csv'}")
    print(f"Wrote: {outdir / 'canonical_loci.csv'}")
    if full_acct is not None:
        print(f"Wrote: {outdir / 'variant_accounting_full.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
