"""DerSimonian-Laird random-effects meta-analysis."""

import numpy as np
from scipy.stats import norm

from gwas_meta.meta_analysis.heterogeneity import tau_squared_dl
from gwas_meta.meta_analysis.models import MetaAnalysisInput


def dl_random_effects(inp: MetaAnalysisInput) -> dict:
    """Run a DerSimonian-Laird random-effects meta-analysis.

    Weights:  w_i* = 1 / (se_i^2 + tau^2)
    Beta*:    sum(w_i* * beta_i) / sum(w_i*)
    SE*:      1 / sqrt(sum(w_i*))
    Z:        beta* / se*
    P:        two-sided p-value from the standard normal distribution

    Parameters
    ----------
    inp : MetaAnalysisInput
        Input data containing betas, standard errors, and study identifiers.

    Returns
    -------
    dict
        Keys: beta, se, z, p, tau_squared.
    """
    tau2 = tau_squared_dl(inp.betas, inp.standard_errors)
    weights_star = 1.0 / (inp.standard_errors ** 2 + tau2)
    beta = float(np.sum(weights_star * inp.betas) / np.sum(weights_star))
    se = float(1.0 / np.sqrt(np.sum(weights_star)))
    z = beta / se if se > 0.0 else 0.0
    p = float(2.0 * norm.sf(abs(z)))

    return {"beta": beta, "se": se, "z": z, "p": p, "tau_squared": tau2}
