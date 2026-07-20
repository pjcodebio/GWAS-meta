"""Sentinel-SNP genome-build check.

GWAS Catalog harmonized files are contractually GRCh38, but the tool
also accepts user-uploaded harmonized-format files with no such
guarantee. A mixed-build set produces zero overlapping variants at
Pass 2 with no hint that build mismatch is the cause. This module
provides a lightweight sanity check: probe each study for a small set
of well-known "sentinel" variants whose GRCh37 and GRCh38 positions
are known, and warn when the observed positions consistently match
GRCh37 rather than GRCh38.

The check is a heuristic. The verdict is deliberately conservative:
a build is only asserted when the sentinel evidence is unanimous.
- If none of the sentinel variants appear in the study (rsID missing
  or filtered out earlier), the verdict is ``"unknown"`` — no warning.
- If matches are found and *all* are GRCh38-consistent (no GRCh37
  hits), the verdict is ``"grch38"`` — the expected case, no warning.
- If ≥ 2 matches are found and *all* are GRCh37-consistent (no GRCh38
  hits), the verdict is ``"grch37"`` — a warning is emitted. A single
  lone GRCh37 hit is too weak to warn and stays ``"unknown"``.
- Any study with both GRCh37 and GRCh38 hits is reported as ``"mixed"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Sentinel variants: high-profile SNPs whose positions differ between
# GRCh37 and GRCh38. All rsIDs are widely known clinical / GWAS-defining
# variants, and chromosome-level positions are stable across dbSNP
# releases. When adding entries, use dbSNP as the source of truth.
SENTINEL_SNPS: list[dict] = [
    # APOE region (chr19)
    {"rsid": "rs7412",    "chromosome": "19", "grch37": 45412079, "grch38": 44908822},
    {"rsid": "rs429358",  "chromosome": "19", "grch37": 45411941, "grch38": 44908684},
    # HBB (chr11) — sickle cell
    {"rsid": "rs334",     "chromosome": "11", "grch37":  5248232, "grch38":  5227002},
    # MTHFR C677T (chr1)
    {"rsid": "rs1801133", "chromosome":  "1", "grch37": 11856378, "grch38": 11796321},
    # Factor V Leiden (chr1)
    {"rsid": "rs6025",    "chromosome":  "1", "grch37": 169519049, "grch38": 169549811},
    # HFE C282Y (chr6) — hereditary hemochromatosis
    {"rsid": "rs1800562", "chromosome":  "6", "grch37": 26093141,  "grch38": 26092913},
]


@dataclass
class BuildVerdict:
    """Result of the sentinel-SNP genome-build check for one study.

    Parameters
    ----------
    verdict:
        One of ``"grch38"``, ``"grch37"``, ``"mixed"``, or ``"unknown"``.
    n_matches:
        Number of sentinel rsIDs found in the study.
    grch37_hits:
        Sentinel rsIDs whose position matched the GRCh37 coordinate.
    grch38_hits:
        Sentinel rsIDs whose position matched the GRCh38 coordinate.
    mismatch_hits:
        Sentinel rsIDs found in the study but at neither GRCh37 nor
        GRCh38 positions (indicates a data-integrity issue rather than
        a build issue — reported for completeness).
    """

    verdict: str
    n_matches: int
    grch37_hits: list[str] = field(default_factory=list)
    grch38_hits: list[str] = field(default_factory=list)
    mismatch_hits: list[str] = field(default_factory=list)

    @property
    def is_warning(self) -> bool:
        """True when the verdict warrants a user-facing warning."""
        return self.verdict in ("grch37", "mixed")


def check_genome_build(df) -> BuildVerdict:
    """Probe a DataFrame for sentinel SNPs and return a build verdict.

    Parameters
    ----------
    df:
        DataFrame with ``rsid``, ``chromosome``, and ``position`` columns
        (any subset of the sentinel rsIDs may be present).

    Returns
    -------
    BuildVerdict
        Verdict ``"unknown"`` when no sentinel is found; otherwise a
        summary of GRCh37 vs GRCh38 matches.
    """
    if df is None or len(df) == 0 or "rsid" not in df.columns:
        return BuildVerdict(verdict="unknown", n_matches=0)

    grch37_hits: list[str] = []
    grch38_hits: list[str] = []
    mismatch_hits: list[str] = []

    for sent in SENTINEL_SNPS:
        rs = sent["rsid"]
        row = df[df["rsid"] == rs]
        if row.empty:
            continue
        # Multiple rows for one rsID → take the first
        r = row.iloc[0]
        try:
            pos = int(r["position"])
            chrom = str(r["chromosome"])
        except (KeyError, ValueError, TypeError):
            continue
        if chrom != sent["chromosome"]:
            mismatch_hits.append(rs)
            continue
        if pos == sent["grch38"]:
            grch38_hits.append(rs)
        elif pos == sent["grch37"]:
            grch37_hits.append(rs)
        else:
            mismatch_hits.append(rs)

    n_matches = len(grch37_hits) + len(grch38_hits)

    if n_matches == 0:
        verdict = "unknown"
    elif len(grch38_hits) > 0 and len(grch37_hits) == 0:
        verdict = "grch38"
    elif len(grch37_hits) >= 2 and len(grch38_hits) == 0:
        verdict = "grch37"
    elif len(grch37_hits) > 0 and len(grch38_hits) > 0:
        verdict = "mixed"
    else:
        # Only 1 grch37 hit and 0 grch38 hits → too weak to warn
        verdict = "unknown"

    return BuildVerdict(
        verdict=verdict,
        n_matches=n_matches,
        grch37_hits=grch37_hits,
        grch38_hits=grch38_hits,
        mismatch_hits=mismatch_hits,
    )
