"""Data models for the meta-analysis engine."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MetaAnalysisInput:
    """Input data for a single-variant meta-analysis.

    Parameters
    ----------
    variant_id : str
        Identifier for the genetic variant (e.g. rsID or chr:pos:ref:alt).
    betas : np.ndarray
        Effect-size estimates from each study.
    standard_errors : np.ndarray
        Standard errors corresponding to each beta.
    study_ids : list[str]
        Identifiers for the contributing studies.
    """

    variant_id: str
    betas: np.ndarray
    standard_errors: np.ndarray
    study_ids: list[str]

    def __post_init__(self) -> None:
        self.betas = np.asarray(self.betas, dtype=np.float64)
        self.standard_errors = np.asarray(self.standard_errors, dtype=np.float64)
        if len(self.betas) != len(self.standard_errors):
            raise ValueError(
                "betas and standard_errors must have the same length "
                f"({len(self.betas)} != {len(self.standard_errors)})"
            )
        if len(self.betas) != len(self.study_ids):
            raise ValueError(
                "betas and study_ids must have the same length "
                f"({len(self.betas)} != {len(self.study_ids)})"
            )
        if len(self.betas) == 0:
            raise ValueError("At least one study is required")
        if np.any(self.standard_errors <= 0):
            raise ValueError("All standard errors must be positive")


@dataclass
class MetaAnalysisResult:
    """Result of a single-variant meta-analysis.

    Contains fixed-effects (IVW) and random-effects (DerSimonian-Laird)
    estimates, together with heterogeneity statistics.
    """

    variant_id: str
    beta_fixed: float
    se_fixed: float
    z_fixed: float
    p_fixed: float
    beta_random: float
    se_random: float
    z_random: float
    p_random: float
    q_stat: float
    i_squared: float
    tau_squared: float
    n_studies: int
    study_ids: list[str]
    per_study_betas: dict[str, float] = field(default_factory=dict)
    per_study_ses: dict[str, float] = field(default_factory=dict)
    rsid: str = ""
    # Direction-of-effect summary: counts of contributing studies
    # with beta > 0, beta < 0, beta == 0.
    n_pos: int = 0
    n_neg: int = 0
    n_zero: int = 0
    # Leave-one-out sensitivity, populated only for hits with
    # n_studies >= 3; NaN otherwise. `loo_max_p` is the worst-case
    # fixed-effects p-value after dropping any single study.
    loo_max_p: float = float("nan")
    loo_worst_dropped: str = ""
