"""Heuristic finder for shared cohorts across studies.

IVW meta-analysis assumes independent studies. When two studies share
participants (e.g. UK Biobank contributing to both a consortium GWAS and
pan-UKBB) the Z-statistic is inflated with no warning from the analysis
itself. Rigorous correction requires bivariate LDSC or MetaSubtract.

This module is a **shared-cohort finder**, not an independence check.
It reports evidence of overlap when it exists (shared PubMed ID, or a
well-known biobank named in more than one study's sample description),
and is silent when no such evidence exists. The absence of a finding
must not be read as evidence of independence: any dataset whose
underlying cohort is not named in the sample description, or whose two
sub-cohorts appear as distinct Catalog entries under different PubMed
IDs, will slip past both checks. Definitive independence still requires
reading the source publications and tracing the cohorts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Canonical cohort name -> list of case-insensitive regex patterns that
# indicate the cohort. Patterns are matched with re.search on the raw
# free-text description. Word boundaries prevent, e.g., "UKB" matching
# inside "UKBiobank" spelled with hyphens.
KNOWN_COHORTS: dict[str, list[str]] = {
    "UK Biobank": [
        r"\bUK[\s\-]?Biobank\b",
        r"\bUKBB\b",
        r"\bUKB\b",
        r"\bpan[\s\-]?UKBB?\b",
    ],
    "FinnGen": [
        r"\bFinnGen\b",
    ],
    "23andMe": [
        r"\b23andMe\b",
    ],
    "deCODE": [
        r"\bdeCODE\b",
    ],
    "BioBank Japan": [
        r"\bBio[\s\-]?Bank Japan\b",
        r"\bBBJ\b",
    ],
    "Million Veteran Program": [
        r"\bMillion Veterans? Program\b",
        r"\bMVP\b",
    ],
    "HUNT": [
        r"\bHUNT\b",
    ],
    "Estonian Biobank": [
        r"\bEstonian Biobank\b",
        r"\bEstBB\b",
    ],
    "Generation Scotland": [
        r"\bGeneration Scotland\b",
    ],
    "China Kadoorie Biobank": [
        r"\bChina Kadoorie\b",
        r"\bCKB\b",
    ],
    "Trøndelag Health Study": [
        r"\bTrøndelag Health Study\b",
        r"\bHUNT[\s\-]?Study\b",
    ],
    "Framingham Heart Study": [
        r"\bFramingham\b",
    ],
    "ARIC": [
        r"\bARIC\b",
        r"\bAtherosclerosis Risk in Communities\b",
    ],
    "CHARGE": [
        r"\bCHARGE\b",
    ],
}


@dataclass
class OverlapFinding:
    """One shared-cohort finding across a set of studies.

    Parameters
    ----------
    cohort:
        Canonical cohort name (e.g. ``"UK Biobank"``).
    study_ids:
        Study accessions that all mention this cohort.
    """

    cohort: str
    study_ids: list[str]


def detect_cohorts(text: str) -> set[str]:
    """Return the set of canonical cohort names mentioned in *text*."""
    if not text:
        return set()
    found: set[str] = set()
    for canonical, patterns in KNOWN_COHORTS.items():
        for pat in patterns:
            if re.search(pat, text, flags=re.IGNORECASE):
                found.add(canonical)
                break
    return found


def find_sample_overlap(
    studies: "list",
) -> list[OverlapFinding]:
    """Find cohorts named in more than one study's sample description.

    Parameters
    ----------
    studies:
        Iterable of objects with ``study_id`` and ``initial_sample_size``
        attributes (i.e. :class:`gwas_meta.gwas_client.models.GWASStudy`).

    Returns
    -------
    list[OverlapFinding]
        One entry per cohort that appears in >= 2 studies.
    """
    per_study: dict[str, set[str]] = {}
    for st in studies:
        text = getattr(st, "initial_sample_size", "") or ""
        per_study[st.study_id] = detect_cohorts(text)

    cohort_to_studies: dict[str, list[str]] = {}
    for sid, cohorts in per_study.items():
        for c in cohorts:
            cohort_to_studies.setdefault(c, []).append(sid)

    return [
        OverlapFinding(cohort=c, study_ids=sorted(sids))
        for c, sids in sorted(cohort_to_studies.items())
        if len(sids) >= 2
    ]


@dataclass
class PubmedFinding:
    """One shared-PubMed-ID finding across a set of studies.

    Studies sharing a PubMed ID are typically staged sub-cohorts of one
    publication (discovery + replication) and should not be meta-analysed
    as if independent.
    """

    pubmed_id: str
    study_ids: list[str]


def find_shared_pubmed_ids(studies: "list") -> list[PubmedFinding]:
    """Group studies by shared PubMed ID.

    Returns entries where two or more studies carry the same non-empty
    ``pubmed_id`` attribute.
    """
    by_pmid: dict[str, list[str]] = {}
    for st in studies:
        pmid = getattr(st, "pubmed_id", None)
        if not pmid:
            continue
        by_pmid.setdefault(str(pmid), []).append(st.study_id)

    return [
        PubmedFinding(pubmed_id=pmid, study_ids=sorted(sids))
        for pmid, sids in sorted(by_pmid.items())
        if len(sids) >= 2
    ]


@dataclass
class SharedCohortReport:
    """Aggregated shared-cohort-finder output.

    Attributes
    ----------
    shared_pubmed_ids:
        PubMed-ID matches (studies from the same publication).
    shared_cohort_keywords:
        Well-known-cohort keyword matches on the sample-size descriptions.
    n_studies:
        Number of studies scanned, for provenance.

    Notes
    -----
    ``any_found`` being ``False`` does **not** imply the inputs are
    independent — only that neither heuristic surfaced evidence of
    overlap. Definitive independence must be established by reading the
    source publications.
    """

    shared_pubmed_ids: list[PubmedFinding]
    shared_cohort_keywords: list[OverlapFinding]
    n_studies: int

    @property
    def any_found(self) -> bool:
        return bool(self.shared_pubmed_ids or self.shared_cohort_keywords)

    def to_dict(self) -> dict:
        return {
            "n_studies_scanned": self.n_studies,
            "shared_pubmed_ids": [
                {"pubmed_id": f.pubmed_id, "study_ids": f.study_ids}
                for f in self.shared_pubmed_ids
            ],
            "shared_cohort_keywords": [
                {"cohort": f.cohort, "study_ids": f.study_ids}
                for f in self.shared_cohort_keywords
            ],
            "any_found": self.any_found,
            "interpretation": (
                "Evidence of shared cohorts was found; verify from source "
                "publications and consider MetaSubtract / bivariate LDSC."
                if self.any_found
                else "No evidence of shared cohorts in metadata. This does "
                     "not rule out overlap — independence must be verified "
                     "from source publications."
            ),
        }


def find_shared_cohorts(studies: "list") -> SharedCohortReport:
    """Run both heuristics (PubMed ID + biobank keyword) and aggregate.

    Parameters
    ----------
    studies:
        Iterable of objects with at least ``study_id`` and
        ``initial_sample_size`` attributes; ``pubmed_id`` is used when
        available.
    """
    lst = list(studies)
    return SharedCohortReport(
        shared_pubmed_ids=find_shared_pubmed_ids(lst),
        shared_cohort_keywords=find_sample_overlap(lst),
        n_studies=len(lst),
    )
