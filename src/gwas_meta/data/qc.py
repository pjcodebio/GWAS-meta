"""Quality-control checks for GWAS summary statistics.

Per-study QC (run after loading, before meta-analysis):
    - Genomic inflation factor (lambda GC)
    - P-value vs Z-score consistency check

Post-meta QC:
    - Heterogeneity-based filtering (Cochran's Q p-value threshold)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-study QC
# ---------------------------------------------------------------------------

@dataclass
class StudyQCReport:
    """QC summary for a single study."""

    study_id: str
    n_variants_before: int
    n_variants_after: int
    lambda_gc: float
    n_pz_inconsistent: int


def compute_lambda_gc(p_values: np.ndarray) -> float:
    """Compute genomic inflation factor (lambda GC).

    lambda_gc = median(chi2_observed) / median(chi2_expected)

    where chi2_observed = qchisq(1 - p, df=1) and
    median(chi2_expected) = qchisq(0.5, df=1) ≈ 0.4549364.

    Parameters
    ----------
    p_values:
        Array of p-values.  NaN and zero values are dropped.

    Returns
    -------
    float
        Genomic inflation factor.  Values close to 1.0 indicate no
        systematic inflation.  Values > 1.0 suggest confounding or
        population stratification.
    """
    from scipy.stats import chi2

    p = np.asarray(p_values, dtype=np.float64)
    p = p[np.isfinite(p) & (p > 0) & (p <= 1)]

    if len(p) == 0:
        return np.nan

    chi2_obs = chi2.ppf(1.0 - p, df=1)
    median_obs = np.median(chi2_obs)
    median_expected = chi2.ppf(0.5, df=1)  # ≈ 0.4549364

    if median_expected == 0:
        return np.nan

    return float(median_obs / median_expected)


def compute_lambda_gc_from_file(path) -> float:
    """Compute genomic inflation factor directly from a harmonized .h.tsv.gz.

    Reads only the p-value column to keep memory low.  Tries the
    standard GWAS Catalog harmonized column names (``p_value`` first,
    then ``hm_p_value``).

    Parameters
    ----------
    path:
        Filesystem path to a harmonized summary-stats file
        (``.h.tsv.gz``).

    Returns
    -------
    float
        Lambda GC, or NaN if no usable p-value column is found.
    """
    from pathlib import Path as _Path

    p = _Path(path)
    # Probe header to pick a p-value column
    try:
        header = pd.read_csv(p, sep="\t", nrows=0, compression="gzip")
    except Exception as exc:
        logger.warning("Could not read header from %s: %s", p, exc)
        return float("nan")

    pcol = None
    for candidate in ("p_value", "hm_p_value", "pvalue", "P"):
        if candidate in header.columns:
            pcol = candidate
            break
    if pcol is None:
        logger.warning("No p-value column in %s (columns: %s)", p, list(header.columns))
        return float("nan")

    try:
        df = pd.read_csv(
            p, sep="\t", usecols=[pcol], compression="gzip",
            dtype={pcol: "float64"}, na_values=["NA", "NaN", ""],
        )
    except Exception as exc:
        logger.warning("Could not read p-values from %s: %s", p, exc)
        return float("nan")

    return compute_lambda_gc(df[pcol].values)


def check_pz_consistency(
    betas: np.ndarray,
    ses: np.ndarray,
    p_values: np.ndarray,
    max_log10_diff: float = 2.0,
) -> np.ndarray:
    """Check consistency between reported p-values and p derived from beta/SE.

    For each variant, computes ``p_derived = 2 * Phi(-|beta/SE|)`` and
    compares it with the reported p-value on the -log10 scale.  Variants
    where ``|(-log10 p_reported) - (-log10 p_derived)| > max_log10_diff``
    are flagged as inconsistent.

    Parameters
    ----------
    betas, ses, p_values:
        Arrays of the same length.
    max_log10_diff:
        Maximum tolerable difference on the -log10(p) scale.
        Default ``2.0`` (i.e., 100-fold difference is tolerated).

    Returns
    -------
    np.ndarray
        Boolean mask — ``True`` for *consistent* (i.e. OK) variants.
    """
    from scipy.stats import norm

    betas = np.asarray(betas, dtype=np.float64)
    ses = np.asarray(ses, dtype=np.float64)
    p_values = np.asarray(p_values, dtype=np.float64)

    z = np.where(ses > 0, np.abs(betas / ses), np.nan)
    p_derived = 2.0 * norm.sf(z)

    # Clamp away from zero for log
    p_rep = np.clip(p_values, 1e-300, 1.0)
    p_der = np.clip(p_derived, 1e-300, 1.0)

    log_diff = np.abs(-np.log10(p_rep) - (-np.log10(p_der)))

    # Consistent = small difference; also keep NaN rows (can't check)
    consistent = np.isnan(log_diff) | (log_diff <= max_log10_diff)
    return consistent


def run_study_qc(
    variants: pd.DataFrame,
    study_id: str,
    max_pz_log10_diff: float = 2.0,
) -> tuple[pd.DataFrame, StudyQCReport]:
    """Run per-study QC checks and return filtered DataFrame + report.

    Steps:
        1. Compute genomic inflation factor (lambda GC) — informational.
        2. Remove variants failing the p-value vs Z-score consistency check.

    Parameters
    ----------
    variants:
        DataFrame with columns ``beta``, ``standard_error``, ``p_value``.
    study_id:
        Study identifier (for logging).
    max_pz_log10_diff:
        Tolerance for the p-value consistency check (on -log10 scale).
        Set to ``0.0`` to disable.

    Returns
    -------
    (filtered_df, report)
    """
    n_before = len(variants)

    # 1. Genomic control
    p_vals = variants["p_value"].values if "p_value" in variants.columns else np.array([])
    lambda_gc = compute_lambda_gc(p_vals)
    logger.info(
        "Study %s: lambda_gc = %.4f (%d variants)",
        study_id, lambda_gc, n_before,
    )
    if lambda_gc > 1.1:
        logger.warning(
            "Study %s has elevated genomic inflation (lambda_gc=%.4f). "
            "This may indicate population stratification or other confounding.",
            study_id, lambda_gc,
        )

    # 2. P-value vs Z-score consistency
    n_pz_removed = 0
    if (
        max_pz_log10_diff > 0
        and "p_value" in variants.columns
        and "beta" in variants.columns
        and "standard_error" in variants.columns
    ):
        consistent = check_pz_consistency(
            variants["beta"].values,
            variants["standard_error"].values,
            variants["p_value"].values,
            max_log10_diff=max_pz_log10_diff,
        )
        n_pz_removed = int((~consistent).sum())
        if n_pz_removed > 0:
            variants = variants[consistent].copy()
            logger.info(
                "Study %s: removed %d variants failing p-value/Z-score "
                "consistency check (max_log10_diff=%g)",
                study_id, n_pz_removed, max_pz_log10_diff,
            )

    report = StudyQCReport(
        study_id=study_id,
        n_variants_before=n_before,
        n_variants_after=len(variants),
        lambda_gc=lambda_gc,
        n_pz_inconsistent=n_pz_removed,
    )

    return variants, report


# ---------------------------------------------------------------------------
# Post-meta QC
# ---------------------------------------------------------------------------

def filter_by_heterogeneity(
    results_df: pd.DataFrame,
    q_pvalue_threshold: float = 1e-6,
) -> pd.DataFrame:
    """Filter meta-analysis results by Cochran's Q p-value.

    Removes variants where the heterogeneity across studies is so extreme
    that the meta-analytic estimate is unreliable.

    Parameters
    ----------
    results_df:
        DataFrame from :func:`run_meta_analysis_batch` with at least
        ``q_stat`` and ``n_studies`` columns.
    q_pvalue_threshold:
        Variants with Cochran's Q p-value **below** this threshold are
        removed (i.e. extreme heterogeneity).  Default ``1e-6``.
        Set to ``0.0`` to disable.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    if q_pvalue_threshold <= 0:
        return results_df

    from scipy.stats import chi2

    q = results_df["q_stat"].values
    k = results_df["n_studies"].values
    df_q = np.maximum(k - 1, 1)

    q_pval = chi2.sf(q, df=df_q)

    keep = q_pval >= q_pvalue_threshold
    n_removed = int((~keep).sum())
    if n_removed > 0:
        logger.info(
            "Heterogeneity filter: removed %d / %d variants "
            "(Cochran's Q p < %g)",
            n_removed, len(results_df), q_pvalue_threshold,
        )

    return results_df[keep].reset_index(drop=True)
