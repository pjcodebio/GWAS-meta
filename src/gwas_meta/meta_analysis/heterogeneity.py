"""Heterogeneity statistics for meta-analysis."""

import numpy as np


def cochrans_q(betas: np.ndarray, standard_errors: np.ndarray) -> float:
    """Compute Cochran's Q statistic for heterogeneity.

    Q = sum(w_i * (beta_i - beta_fixed)^2)
    where w_i = 1 / se_i^2 and beta_fixed = sum(w_i * beta_i) / sum(w_i).

    Parameters
    ----------
    betas : np.ndarray
        Effect-size estimates from each study.
    standard_errors : np.ndarray
        Standard errors corresponding to each beta.

    Returns
    -------
    float
        Cochran's Q statistic.
    """
    weights = 1.0 / (standard_errors ** 2)
    beta_fixed = np.sum(weights * betas) / np.sum(weights)
    q: float = float(np.sum(weights * (betas - beta_fixed) ** 2))
    return q


def i_squared(q: float, k: int) -> float:
    """Compute the I-squared heterogeneity index.

    I^2 = max(0, (Q - (k - 1)) / Q) * 100

    Parameters
    ----------
    q : float
        Cochran's Q statistic.
    k : int
        Number of studies.

    Returns
    -------
    float
        I-squared value as a percentage (0-100).
    """
    if k <= 1 or q == 0.0:
        return 0.0
    return max(0.0, (q - (k - 1)) / q * 100.0)


def tau_squared_dl(betas: np.ndarray, standard_errors: np.ndarray) -> float:
    """Estimate between-study variance (tau^2) using DerSimonian-Laird.

    tau^2 = max(0, (Q - (k - 1)) / (sum(w) - sum(w^2) / sum(w)))

    Parameters
    ----------
    betas : np.ndarray
        Effect-size estimates from each study.
    standard_errors : np.ndarray
        Standard errors corresponding to each beta.

    Returns
    -------
    float
        DerSimonian-Laird estimate of tau-squared.
    """
    k = len(betas)
    if k <= 1:
        return 0.0

    weights = 1.0 / (standard_errors ** 2)
    q = cochrans_q(betas, standard_errors)
    c = float(np.sum(weights) - np.sum(weights ** 2) / np.sum(weights))

    if c == 0.0:
        return 0.0

    tau2 = (q - (k - 1)) / c
    return max(0.0, tau2)
