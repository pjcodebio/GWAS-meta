"""Tests for the shared-cohort finder (biobank-keyword + PubMed-ID heuristics)."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from gwas_meta.data.sample_overlap import (
    KNOWN_COHORTS,
    detect_cohorts,
    find_sample_overlap,
    find_shared_cohorts,
    find_shared_pubmed_ids,
)


@dataclass
class FakeStudy:
    study_id: str
    initial_sample_size: str = ""
    pubmed_id: str | None = None


def test_detect_cohorts_uk_biobank_variants():
    for txt in ["UK Biobank cohort", "UKBB participants", "pan-UKBB study",
                "343,621 UK-Biobank Europeans", "ukb"]:
        assert "UK Biobank" in detect_cohorts(txt), txt


def test_detect_cohorts_word_boundary_negative():
    assert "UK Biobank" not in detect_cohorts("SUKBAT gene expression cohort")
    assert "UK Biobank" not in detect_cohorts("STRUCKB")


def test_detect_cohorts_multi_biobank():
    txt = "Two-cohort meta of UK Biobank and FinnGen R12 (MVP not included)"
    found = detect_cohorts(txt)
    assert found >= {"UK Biobank", "FinnGen", "Million Veteran Program"}


def test_detect_cohorts_empty():
    assert detect_cohorts("") == set()
    assert detect_cohorts(None) == set()  # type: ignore[arg-type]


def test_find_sample_overlap_flags_shared_ukb():
    studies = [
        FakeStudy("A", "343,621 Europeans, UK Biobank"),
        FakeStudy("B", "pan-UKBB gout"),
        FakeStudy("C", "FinnGen R12"),
    ]
    findings = find_sample_overlap(studies)
    assert len(findings) == 1
    assert findings[0].cohort == "UK Biobank"
    assert findings[0].study_ids == ["A", "B"]


def test_find_sample_overlap_silent_when_disjoint():
    studies = [
        FakeStudy("A", "UK Biobank"),
        FakeStudy("B", "FinnGen R12"),
    ]
    assert find_sample_overlap(studies) == []


def test_find_shared_pubmed_ids_flags_same_pmid():
    studies = [
        FakeStudy("GCST90011892", pubmed_id="33514863"),
        FakeStudy("GCST90011893", pubmed_id="33514863"),
        FakeStudy("GCST_other",  pubmed_id="99999999"),
    ]
    findings = find_shared_pubmed_ids(studies)
    assert len(findings) == 1
    assert findings[0].pubmed_id == "33514863"
    assert findings[0].study_ids == ["GCST90011892", "GCST90011893"]


def test_find_shared_pubmed_ids_ignores_missing_pmid():
    studies = [
        FakeStudy("A", pubmed_id=None),
        FakeStudy("B", pubmed_id=None),
    ]
    assert find_shared_pubmed_ids(studies) == []


def test_find_shared_pubmed_ids_coerces_to_str():
    studies = [
        FakeStudy("A", pubmed_id=12345),  # int
        FakeStudy("B", pubmed_id="12345"),  # str
    ]
    findings = find_shared_pubmed_ids(studies)
    assert len(findings) == 1
    assert findings[0].pubmed_id == "12345"


# --- Aggregated report semantics (Option B) --------------------------------

def test_shared_cohort_report_flags_ldl_case_silent():
    """LDL: UKB vs MVP — different papers, different biobanks. No hit."""
    studies = [
        FakeStudy("GCST90019512", "343,621 Europeans, UK Biobank", "31217584"),
        FakeStudy("GCST90475420", "404,745 Europeans, Million Veteran Program", "38900908"),
    ]
    rep = find_shared_cohorts(studies)
    assert rep.any_found is False
    assert rep.shared_pubmed_ids == []
    assert rep.shared_cohort_keywords == []
    # Silence must never be read as evidence of independence.
    assert "does not rule out overlap" in rep.to_dict()["interpretation"]


def test_shared_cohort_report_flags_rp_case_pmid():
    """RP: Nishiguchi's two stages share PMID — must flag red."""
    studies = [
        FakeStudy("GCST90011892",
                  "432 cases, 603 controls, Japanese ancestry",
                  pubmed_id="33514863"),
        FakeStudy("GCST90011893",
                  "208 cases, 287 controls, Japanese ancestry",
                  pubmed_id="33514863"),
    ]
    rep = find_shared_cohorts(studies)
    assert rep.any_found is True
    assert len(rep.shared_pubmed_ids) == 1
    assert rep.shared_pubmed_ids[0].pubmed_id == "33514863"
    assert rep.shared_cohort_keywords == []


def test_shared_cohort_report_flags_gout_case_silent():
    """Gout: FinnGen R12 vs pan-UKBB — different biobanks, different papers."""
    studies = [
        FakeStudy("FinnGen_R12_GOUT", "FinnGen R12, 500,000 Finns", "38653804"),
        FakeStudy("pan-UKBB_gout",    "pan-UK Biobank, 420,531 Europeans", None),
    ]
    rep = find_shared_cohorts(studies)
    assert rep.any_found is False


def test_shared_cohort_report_to_dict_shape():
    rep = find_shared_cohorts([
        FakeStudy("A", "UK Biobank", "111"),
        FakeStudy("B", "UK Biobank", "111"),
    ])
    d = rep.to_dict()
    assert set(d.keys()) == {
        "n_studies_scanned", "shared_pubmed_ids", "shared_cohort_keywords",
        "any_found", "interpretation",
    }
    assert d["n_studies_scanned"] == 2
    assert d["any_found"] is True


def test_known_cohorts_contains_expected_keys():
    for expected in ["UK Biobank", "FinnGen", "Million Veteran Program",
                     "BioBank Japan", "23andMe", "deCODE"]:
        assert expected in KNOWN_COHORTS
