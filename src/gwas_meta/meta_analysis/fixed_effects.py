"""Inverse-variance weighted (IVW) fixed-effects meta-analysis."""

import numpy as np
from scipy.stats import norm

from gwas_meta.meta_analysis.models import MetaAnalysisInput


def ivw_fixed_effects(inp: MetaAnalysisInput) -> dict:
    """Run an IVW fixed-effects meta-analysis for a single variant.

    Weights:  w_i = 1 / se_i^2
    Beta:     sum(w_i * beta_i) / sum(w_i)
    SE:       1 / sqrt(sum(w_i))
    Z:        beta / se
    P:        two-sided p-value from the standard normal distribution

    Parameters
    ----------
    inp : MetaAnalysisInput
        Input data containing betas, standard errors, and study identifiers.

    Returns
    -------
    dict
        Keys: beta, se, z, p.
    """
    weights = 1.0 / (inp.standard_errors ** 2)
    beta = float(np.sum(weights * inp.betas) / np.sum(weights))
    se = float(1.0 / np.sqrt(np.sum(weights)))
    z = beta / se if se > 0.0 else 0.0
    p = float(2.0 * norm.sf(abs(z)))

    return {"beta": beta, "se": se, "z": z, "p": p}
