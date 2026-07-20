"""Data models for GWAS summary statistics."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StudySummaryStats:
    """Summary statistics for one GWAS study.

    The *variants* DataFrame uses the following standardised columns:

        variant_id, rsid, chromosome, position, effect_allele, other_allele,
        beta, standard_error, p_value, effect_allele_frequency, hm_code
    """

    study_id: str
    variants: pd.DataFrame = field(repr=False)

    # Canonical column order -------------------------------------------------
    EXPECTED_COLUMNS: list[str] = field(
        default_factory=lambda: [
            "variant_id",
            "rsid",
            "chromosome",
            "position",
            "effect_allele",
            "other_allele",
            "beta",
            "standard_error",
            "p_value",
            "effect_allele_frequency",
            "hm_code",
        ],
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        missing = set(self.EXPECTED_COLUMNS) - set(self.variants.columns)
        if missing:
            raise ValueError(
                f"StudySummaryStats for '{self.study_id}' is missing "
                f"columns: {sorted(missing)}"
            )

    @property
    def n_variants(self) -> int:
        return len(self.variants)

    def __len__(self) -> int:
        return self.n_variants
