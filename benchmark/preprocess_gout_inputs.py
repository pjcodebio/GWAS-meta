#!/usr/bin/env python3
"""Reproduce the three harmonised gout benchmark inputs from a single
primary public source.

Downloads `finngen_R12_GOUT_STRICT_meta_out.tsv.gz` (~1.77 GB) from the
FinnGen R12 public bucket and splits it column-wise into the three
files that `benchmark/run_benchmark.py` expects:

  input_finngen.h.tsv.gz   <- FINNGEN_beta / FINNGEN_sebeta / ...
  input_ukbb.h.tsv.gz      <- UKBB_beta    / UKBB_sebeta    / ...
  expected_meta.h.tsv.gz   <- all_inv_var_meta_beta / _sebeta / ...

All positions are GRCh38 forward-strand (FinnGen R12 release convention).
The transform is a pure column rename + selection -- no numeric
re-encoding is needed because the meta_out file already stores beta/SE/p
in 3-sig-fig scientific, which is the shape the appendix files have.

Usage (from repo root):
    python benchmark/preprocess_gout_inputs.py \\
        --out-dir <inputs-dir> \\
        [--raw-cache <dir>] \\
        [--verify <path-to-appendix-input_finngen.h.tsv.gz>]

    # then feed <inputs-dir> to the benchmark:
    python benchmark/run_benchmark.py --inputs-dir <inputs-dir> \\
        --out-dir benchmark/gout_output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

META_OUT_URL = (
    "https://storage.googleapis.com/finngen-public-data-r12/"
    "meta_analysis/ukbb/summary_stats/finngen_R12_GOUT_STRICT_meta_out.tsv.gz"
)
META_OUT_BYTES = 1_772_144_746

CANONICAL_RSIDS = {"rs6449137", "rs2231142", "rs1260326", "rs2164495"}

OUT_COLS = [
    "chromosome", "base_pair_location",
    "effect_allele", "other_allele",
    "beta", "standard_error",
    "effect_allele_frequency", "p_value",
    "rsid", "variant_id",
]


def curl_download(url: str, dest: Path, expected_bytes: int | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and (expected_bytes is None or dest.stat().st_size == expected_bytes):
        print(f"  cached: {dest} ({dest.stat().st_size:,} bytes)")
        return
    print(f"  downloading {url}")
    subprocess.run(["curl", "-#", "-fL", "-o", str(dest), url], check=True)
    print(f"  wrote {dest} ({dest.stat().st_size:,} bytes)")


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def split_meta_out(raw_path: Path, out_dir: Path) -> dict:
    """Stream the meta_out file and write the three harmonised outputs.

    For each row (chr, pos, ref, alt) we emit up to three rows into the
    three output files, one per cohort/meta view. NA in a cohort's beta
    means that cohort did not measure that variant, and no row is written
    to that file (matching the appendix behaviour).
    """
    print(f"  streaming {raw_path}")
    t0 = time.time()
    counts = {"finngen": 0, "ukbb": 0, "meta": 0}
    header_written = {"finngen": False, "ukbb": False, "meta": False}
    paths = {
        "finngen": out_dir / "input_finngen.h.tsv.gz",
        "ukbb":    out_dir / "input_ukbb.h.tsv.gz",
        "meta":    out_dir / "expected_meta.h.tsv.gz",
    }
    # Remove any stale outputs so appends are clean
    for p in paths.values():
        if p.exists():
            p.unlink()

    with pd.read_csv(
        raw_path, sep="\t", compression="gzip",
        chunksize=500_000, dtype=str, na_filter=False,
    ) as reader:
        for chunk in reader:
            # Common columns
            chrom = chunk["#CHR"]
            pos = chunk["POS"]
            ref = chunk["REF"]
            alt = chunk["ALT"]
            rsid = chunk["rsid"]
            variant_id = chrom + "_" + pos + "_" + ref + "_" + alt

            # FinnGen slice
            m_fg = chunk["FINNGEN_beta"] != "NA"
            fg = pd.DataFrame({
                "chromosome": chrom[m_fg],
                "base_pair_location": pos[m_fg],
                "effect_allele": alt[m_fg],
                "other_allele": ref[m_fg],
                "beta": chunk["FINNGEN_beta"][m_fg],
                "standard_error": chunk["FINNGEN_sebeta"][m_fg],
                "effect_allele_frequency": chunk["FINNGEN_af_alt"][m_fg],
                "p_value": chunk["FINNGEN_pval"][m_fg],
                "rsid": rsid[m_fg],
                "variant_id": variant_id[m_fg],
            })[OUT_COLS]

            # UKBB slice
            m_uk = chunk["UKBB_beta"] != "NA"
            uk = pd.DataFrame({
                "chromosome": chrom[m_uk],
                "base_pair_location": pos[m_uk],
                "effect_allele": alt[m_uk],
                "other_allele": ref[m_uk],
                "beta": chunk["UKBB_beta"][m_uk],
                "standard_error": chunk["UKBB_sebeta"][m_uk],
                "effect_allele_frequency": chunk["UKBB_af_alt"][m_uk],
                "p_value": chunk["UKBB_pval"][m_uk],
                "rsid": rsid[m_uk],
                "variant_id": variant_id[m_uk],
            })[OUT_COLS]

            # Meta slice (single-cohort rows carry through as the meta)
            m_meta = chunk["all_inv_var_meta_beta"] != "NA"
            meta = pd.DataFrame({
                "chromosome": chrom[m_meta],
                "base_pair_location": pos[m_meta],
                "effect_allele": alt[m_meta],
                "other_allele": ref[m_meta],
                "beta": chunk["all_inv_var_meta_beta"][m_meta],
                "standard_error": chunk["all_inv_var_meta_sebeta"][m_meta],
                "effect_allele_frequency": chunk["FINNGEN_af_alt"][m_meta].where(
                    chunk["FINNGEN_af_alt"][m_meta] != "NA",
                    chunk["UKBB_af_alt"][m_meta],
                ),
                "p_value": chunk["all_inv_var_meta_p"][m_meta],
                "rsid": rsid[m_meta],
                "variant_id": variant_id[m_meta],
            })[OUT_COLS]

            for name, df in (("finngen", fg), ("ukbb", uk), ("meta", meta)):
                df.to_csv(
                    paths[name], sep="\t", index=False,
                    header=not header_written[name],
                    mode="a", compression="gzip",
                )
                header_written[name] = True
                counts[name] += len(df)

            if sum(counts.values()) % 6_000_000 < 500_000:
                print(f"    ... rows so far: {counts}   ({time.time() - t0:.0f}s)")

    print(f"  finished in {time.time() - t0:.0f}s")
    print(f"  rows -- FinnGen: {counts['finngen']:,}   UKBB: {counts['ukbb']:,}   Meta: {counts['meta']:,}")
    return {name: {"path": str(paths[name]), "rows": n, "bytes": paths[name].stat().st_size}
            for name, n in counts.items()}


def spot_check(out_dir: Path, verify_path: Path | None) -> None:
    print("\nCanonical-loci spot check (input_finngen.h.tsv.gz)")
    print("-" * 64)
    out_fg = out_dir / "input_finngen.h.tsv.gz"
    cmd = f"gunzip -c '{out_fg}' | awk 'NR==1 || $9 ~ /^(rs6449137|rs2231142|rs1260326|rs2164495)$/'"
    print(subprocess.check_output(cmd, shell=True).decode())
    if verify_path and verify_path.is_file():
        print(f"...vs appendix {verify_path.name}...")
        cmd2 = f"gunzip -c '{verify_path}' | awk 'NR==1 || $9 ~ /^(rs6449137|rs2231142|rs1260326|rs2164495)$/'"
        print(subprocess.check_output(cmd2, shell=True).decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--raw-cache", default=None,
                    help="Directory for the 1.77 GB primary source cache. "
                         "Defaults to .cache/gout/ (alongside the other "
                         "case-study caches).")
    ap.add_argument("--verify", default=None,
                    help="Path to appendix input_finngen.h.tsv.gz for spot-check diff")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.raw_cache:
        raw_dir = Path(args.raw_cache)
    else:
        raw_dir = Path(__file__).resolve().parent.parent / ".cache" / "gout"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("Primary source: FinnGen R12 + pan-UKBB meta_out")
    print(f"  URL:   {META_OUT_URL}")
    print(f"  Size:  {META_OUT_BYTES:,} bytes (~1.77 GB)")
    print("=" * 64)
    raw = raw_dir / "finngen_R12_GOUT_STRICT_meta_out.tsv.gz"
    curl_download(META_OUT_URL, raw, expected_bytes=META_OUT_BYTES)
    md5_raw = md5(raw)
    print(f"  MD5:   {md5_raw}")

    print("\nSplitting into three appendix-shape files...")
    outputs = split_meta_out(raw, out_dir)

    spot_check(out_dir, Path(args.verify) if args.verify else None)

    prov = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "primary_source": {
            "url": META_OUT_URL,
            "md5": md5_raw,
            "bytes": raw.stat().st_size,
        },
        "outputs": outputs,
        "transform": (
            "Row-by-row column projection from the FinnGen R12 + pan-UKBB "
            "meta_out file into three .h.tsv.gz files. Values are copied "
            "verbatim (already 3-sig-fig scientific in meta_out). NA rows "
            "in a given cohort's beta are omitted from that cohort's output."
        ),
        "genome_build": "GRCh38 (FinnGen R12 release convention)",
        "strand": "forward (FinnGen R12 release convention)",
    }
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"\nWrote {out_dir / 'provenance.json'}")

    print("\nDone. Feed the outputs to the benchmark:")
    print(f"  python benchmark/run_benchmark.py \\")
    print(f"      --inputs-dir {out_dir} \\")
    print(f"      --out-dir benchmark/gout_output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
