#!/usr/bin/env python3
"""Tier 1 adapter: run the shipped gwas-meta engine on the synthetic per-study CSV.

Calls the SAME batch entry point the interactive pipeline uses
(`run_meta_analysis_batch` from `gwas_meta.meta_analysis`, invoked by
`gwas_meta.pages.step5_meta.render`), so we validate the SHIPPED PRODUCTION
CODE PATH -- the vectorized engine that generates the real results -- not the
single-variant reference or a reimplementation. (The single-variant
`run_meta_analysis` is exercised as a reference in the unit tests, and a
consistency test certifies the batch path agrees with it.)

Engine field renames applied here to match metafor_validate.R's default COLS map:
    beta_fixed  -> beta_fe         q_stat        -> Q
    se_fixed    -> se_fe           tau_squared   -> tau2
    z_fixed     -> z_fe            i_squared/100 -> I2  (engine reports %, R script expects fraction)
    p_fixed     -> p_fe            n_studies     -> k
    beta_random -> beta_re
    se_random   -> se_re
    p_random    -> p_re

Usage:
    python run_engine_on_synthetic.py \
        --per_study synthetic_per_study_long.csv \
        --out       engine_output_synthetic.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gwas_meta.meta_analysis import run_meta_analysis_batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_study", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.per_study)
    needed = {"variant_id", "study_id", "beta", "standard_error"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"per_study CSV missing columns: {missing}")

    # The batch engine consumes the aligned long-format schema
    # [variant_id, study_id, beta, se]; the synthetic deposit names the SE
    # column `standard_error`, so rename to match.
    aligned = df[["variant_id", "study_id", "beta", "standard_error"]].rename(
        columns={"standard_error": "se"}
    )

    res = run_meta_analysis_batch(aligned, as_dataframe=True)

    out_df = pd.DataFrame({
        "variant_id": res["variant_id"].to_numpy(),
        "beta_fe": res["beta_fixed"].to_numpy(),
        "se_fe":   res["se_fixed"].to_numpy(),
        "z_fe":    res["z_fixed"].to_numpy(),
        "p_fe":    res["p_fixed"].to_numpy(),
        "beta_re": res["beta_random"].to_numpy(),
        "se_re":   res["se_random"].to_numpy(),
        "p_re":    res["p_random"].to_numpy(),
        "Q":       res["q_stat"].to_numpy(),
        "tau2":    res["tau_squared"].to_numpy(),
        "I2":      res["i_squared"].to_numpy() / 100.0,
        "k":       res["n_studies"].to_numpy(),
    })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}: {len(out_df)} variants")
    print(f"k distribution: {out_df['k'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
