#!/usr/bin/env python3
"""Eyeball-check LDL locus recovery from a meta_results.csv.

Reads the frozen expected-loci list (expected_loci.csv, same folder) and
streams a meta_results.csv, reporting for every locus how many variants
reach genome-wide significance (p_fixed < 5e-8) within its +/-500 kb
window, plus the lead (smallest-p) variant. No pandas, no re-analysis —
just a line-by-line scan you can run against any run's output.

Usage:
    python verify_recovery.py /path/to/meta_results.csv

Example (frozen appendix run on an external drive):
    python benchmark/ldl_case_study/verify_recovery.py \
        "/Volumes/Disque dur données/02_Academic/Master_Thesis_Large_Files/2026_06_14_appendix/LDL_case_study/meta_results.csv"
"""
import csv
import sys
from pathlib import Path

GWS = 5e-8
HERE = Path(__file__).parent
EXPECTED = HERE / "expected_loci.csv"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    meta = Path(sys.argv[1])
    if not meta.exists():
        print(f"ERROR: meta file not found: {meta}")
        return 2

    loci = []
    with open(EXPECTED) as f:
        for r in csv.DictReader(f):
            loci.append({
                "locus": r["locus"], "tier": r["tier"], "chr": str(r["chr"]),
                "pos": int(r["expected_pos_hg38"]),
                "w": int(r["window_kb"]) * 1000,
            })
    by_chr: dict[str, list[int]] = {}
    for i, l in enumerate(loci):
        by_chr.setdefault(l["chr"], []).append(i)
    res = {i: {"n": 0, "p": None, "lead": None} for i in range(len(loci))}

    print(f"Scanning {meta.name} against {len(loci)} expected loci ...")
    with open(meta) as f:
        f.readline()  # header
        for line in f:
            colon = line.find(":")
            chrom = line[3:colon]  # variant_id starts 'chrN:'
            if chrom not in by_chr:
                continue
            first_comma = line.find(",")
            pos = int(line[colon + 1:line.find(":", colon + 1)])
            p = None
            for i in by_chr[chrom]:
                l = loci[i]
                if l["pos"] - l["w"] <= pos <= l["pos"] + l["w"]:
                    if p is None:
                        p = float(line.split(",", 5)[4])
                    if p < GWS:
                        r = res[i]
                        r["n"] += 1
                        if r["p"] is None or p < r["p"]:
                            r["p"] = p
                            r["lead"] = line[:first_comma]
                    break

    print(f"\n{'locus':24s} {'tier':10s} {'REC':4s} {'lead variant':22s} "
          f"{'min p_fixed':>12s} {'n<5e-8':>7s}")
    print("-" * 84)
    n_rec = {"canonical": 0, "glgc_new": 0}
    n_tot = {"canonical": 0, "glgc_new": 0}
    for i, l in enumerate(loci):
        r = res[i]
        rec = r["n"] > 0
        n_tot[l["tier"]] += 1
        n_rec[l["tier"]] += int(rec)
        pstr = "" if r["p"] is None else f"{r['p']:.2e}"
        print(f"{l['locus']:24s} {l['tier']:10s} {'YES' if rec else 'NO ':4s} "
              f"{(r['lead'] or ''):22s} {pstr:>12s} {r['n']:>7d}")
    print("-" * 84)
    print(f"Canonical (Supp. Table 3): {n_rec['canonical']}/{n_tot['canonical']} "
          f"recovered")
    print(f"GLGC-2013 new (Table 2):   {n_rec['glgc_new']}/{n_tot['glgc_new']} "
          f"recovered")
    total_rec = n_rec['canonical'] + n_rec['glgc_new']
    total = n_tot['canonical'] + n_tot['glgc_new']
    print(f"TOTAL:                     {total_rec}/{total} recovered")
    return 0 if total_rec == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
