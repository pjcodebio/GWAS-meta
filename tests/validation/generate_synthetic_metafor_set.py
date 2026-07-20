#!/usr/bin/env python3
"""
Synthetic per-study GWAS dataset generator for metafor engine validation (Tier 1).

Produces a long-format table of per-study (beta, standard_error) values designed to
exercise the full numerical space of the meta-analysis engine:
  - study counts k = 2..8
  - heterogeneity from I^2 = 0 (homogeneous) to high
  - p-value range from ~1 down to extreme (underflow-territory) values
  - degenerate / adversarial cases that break naive implementations

Output columns match the engine's StudySummaryStats schema so the same code path is used:
    variant_id, chromosome, position, effect_allele, other_allele,
    beta, standard_error, p_value, effect_allele_frequency, hm_code, study_id

Two files are written into the appendix:
    synthetic/synthetic_per_study_long.csv   <- feed to BOTH engines (one row per variant x study)
    synthetic/synthetic_manifest.csv         <- per-variant design metadata (k, target I2, case type)

Run:
    python generate_synthetic_metafor_set.py
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

# --------------------------------------------------------------------------
# Output location (matches RERUN_INSTRUCTIONS appendix structure)
# --------------------------------------------------------------------------
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic")
os.makedirs(OUTDIR, exist_ok=True)

RNG = np.random.default_rng(20260606)   # fixed seed -> reproducible deposit

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def p_from_beta_se(beta: float, se: float) -> float:
    """Two-sided p from a z = beta/se, computed in log space for tiny p."""
    z = abs(beta / se)
    # survival function in log space avoids underflow to 0.0
    logp = np.log(2.0) + norm.logsf(z)
    p = np.exp(logp)
    return p  # may legitimately underflow to 0.0 for very large z -> kept on purpose

def make_variant(vid, k, betas, ses, chrom="22", pos=None, eaf=0.25):
    """Build k per-study rows for one synthetic variant."""
    if pos is None:
        pos = int(RNG.integers(1, 50_000_000))
    rows = []
    for s in range(k):
        b, se = float(betas[s]), float(ses[s])
        rows.append(dict(
            variant_id=vid, chromosome=chrom, position=pos,
            effect_allele="A", other_allele="G",
            beta=b, standard_error=se, p_value=p_from_beta_se(b, se),
            effect_allele_frequency=eaf, hm_code=10,
            study_id=f"SYN_S{s+1}",
        ))
    return rows

# --------------------------------------------------------------------------
# 1. Grid: k = 2..8 x heterogeneity level x effect/precision regimes
# --------------------------------------------------------------------------
rows = []
manifest = []
vidx = 0

K_VALUES = [2, 3, 4, 5, 6, 8]
HET_LEVELS = {
    "I2_zero":      0.0,    # identical true effect -> Q small, I2 ~ 0
    "I2_low":       0.02,   # small between-study variance
    "I2_moderate":  0.08,
    "I2_high":      0.30,   # large between-study variance -> FE and RE diverge
}
EFFECT_REGIMES = {
    "null":     0.00,   # true effect ~ 0
    "small":    0.03,
    "moderate": 0.10,
    "large":    0.25,   # will produce small p at typical SE
}

REPS = 40  # variants per (k, het, regime) cell -> statistical mass

for k in K_VALUES:
    for het_name, tau2 in HET_LEVELS.items():
        for reg_name, mu in EFFECT_REGIMES.items():
            for _ in range(REPS):
                vidx += 1
                vid = f"syn_{vidx:06d}"
                # per-study sampling SEs: realistic spread, occasionally one imprecise study
                ses = RNG.uniform(0.015, 0.06, size=k)
                if RNG.random() < 0.15:
                    ses[RNG.integers(k)] *= RNG.uniform(3, 8)  # one low-precision study
                # true per-study effects: mu + between-study (tau) + within-study handled by SE
                tau = np.sqrt(tau2)
                true_effects = mu + RNG.normal(0, tau, size=k)
                betas = true_effects + RNG.normal(0, ses)  # observed = true + sampling noise
                rows.extend(make_variant(vid, k, betas, ses))
                manifest.append(dict(variant_id=vid, k=k, het=het_name,
                                     target_tau2=tau2, regime=reg_name, mu=mu,
                                     case="grid"))

# --------------------------------------------------------------------------
# 2. Degenerate / adversarial cases (where naive code breaks)
# --------------------------------------------------------------------------
def add_case(vid, k, betas, ses, case, chrom="22"):
    global rows, manifest
    rows.extend(make_variant(vid, k, np.array(betas), np.array(ses), chrom=chrom))
    manifest.append(dict(variant_id=vid, k=k, het="NA", target_tau2=np.nan,
                         regime="NA", mu=np.nan, case=case))

# 2a. k=2 identical betas -> Q=0, tau2=0, I2=0 (must not NaN / div0)
add_case("edge_identical_k2", 2, [0.10, 0.10], [0.03, 0.03], "k2_identical_Q0")

# 2b. k=2 opposite signs equal magnitude -> high Q, FE near 0
add_case("edge_opposite_k2", 2, [0.20, -0.20], [0.03, 0.03], "k2_opposite_signs")

# 2c. one study with huge SE -> must be down-weighted to ~irrelevant
add_case("edge_dominated_k3", 3, [0.10, 0.11, 0.50], [0.02, 0.02, 1.50], "one_study_dominated")

# 2d. extreme effect -> p underflow territory (z very large)
add_case("edge_extreme_p_k3", 3, [0.60, 0.62, 0.58], [0.02, 0.02, 0.02], "extreme_small_p")

# 2e. near-zero effects, large SE -> p ~ 1
add_case("edge_null_k4", 4, [0.001, -0.002, 0.0, 0.001], [0.05, 0.05, 0.05, 0.05], "near_null_p1")

# 2f. wide SE heterogeneity, homogeneous effect
add_case("edge_wide_se_k5", 5, [0.10]*5, [0.01, 0.03, 0.06, 0.10, 0.20], "wide_se_spread")

# 2g. large k, high heterogeneity
add_case("edge_bigk_het_k8", 8,
         [0.05, 0.30, -0.10, 0.20, 0.00, 0.40, -0.05, 0.15],
         [0.03]*8, "bigk_high_het")

# NOTE on k=1: a single-study "meta-analysis" is intentionally NOT emitted here,
# because metafor rma() requires k>=2. If your engine accepts k=1 (passthrough),
# test that separately as a unit test, not in the metafor cross-check.

# --------------------------------------------------------------------------
# Write
# --------------------------------------------------------------------------
df = pd.DataFrame(rows)[
    ["variant_id","chromosome","position","effect_allele","other_allele",
     "beta","standard_error","p_value","effect_allele_frequency","hm_code","study_id"]
]
man = pd.DataFrame(manifest)

long_path = os.path.join(OUTDIR, "synthetic_per_study_long.csv")
man_path  = os.path.join(OUTDIR, "synthetic_manifest.csv")
df.to_csv(long_path, index=False)
man.to_csv(man_path, index=False)

# console summary
n_var = df["variant_id"].nunique()
print(f"Wrote {long_path}")
print(f"Wrote {man_path}")
print(f"Variants: {n_var}  |  per-study rows: {len(df)}")
print("k distribution (variants):")
print(man.groupby('k')['variant_id'].nunique().to_string())
print(f"Seed: 20260606  (change in source for a fresh deposit)")
