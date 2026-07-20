"""Leave-one-out sensitivity analysis for fixed-effects (IVW) meta-hits."""

import math

from scipy.stats import norm


def leave_one_out_max_p(
    betas: "list[float]",
    ses: "list[float]",
    study_ids: "list[str]",
) -> "tuple[float, str]":
    """Leave-one-out sensitivity for a fixed-effects (IVW) meta-hit.

    For each study, recompute IVW with the remaining k-1 studies and
    return the maximum resulting p-value (worst-case) plus the ID of
    the dropped study that produced it. Requires ``k >= 3``; for fewer
    studies returns ``(nan, "")`` since dropping one leaves k <= 1.

    Uses the same inverse-variance weighting as
    :func:`~gwas_meta.meta_analysis.fixed_effects.ivw_fixed_effects`, but
    subtracts each study's contribution from the precomputed totals so the
    whole scan is O(k) rather than O(k^2).

    Parameters
    ----------
    betas :
        Per-study effect-size estimates for a single variant.
    ses :
        Per-study standard errors (same order as ``betas``).
    study_ids :
        Per-study identifiers (same order as ``betas``).

    Returns
    -------
    tuple[float, str]
        ``(loo_max_p, loo_worst_dropped)`` -- the worst-case fixed-effects
        p-value after dropping any single study, and the study whose removal
        produced it.
    """
    b = [float(x) for x in betas]
    s = [float(x) for x in ses]
    k = len(b)
    if k < 3:
        return float("nan"), ""

    w = [1.0 / (si * si) for si in s]
    total_w = sum(w)
    total_wb = sum(w_i * b_i for w_i, b_i in zip(w, b))

    worst_p = -1.0
    worst_sid = ""
    for j in range(k):
        w_rem = total_w - w[j]
        wb_rem = total_wb - w[j] * b[j]
        if w_rem <= 0:
            continue
        beta_loo = wb_rem / w_rem
        se_loo = 1.0 / math.sqrt(w_rem)
        z = beta_loo / se_loo if se_loo > 0 else 0.0
        p = 2.0 * norm.sf(abs(z))
        if p > worst_p:
            worst_p = p
            worst_sid = study_ids[j]
    return (worst_p if worst_p >= 0 else float("nan"), worst_sid)
