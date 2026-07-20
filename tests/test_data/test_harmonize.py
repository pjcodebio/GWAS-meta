"""Tests for the data-loading and harmonisation layer."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gwas_meta.data.harmonize import (
    align_studies,
    align_studies_chunked,
    load_harmonized_file,
)
from gwas_meta.data.models import StudySummaryStats


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_FILE = FIXTURES_DIR / "sample_harmonized.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tsv(tmp_path: Path, content: str, name: str = "test.tsv") -> Path:
    """Write *content* to a TSV file inside *tmp_path* and return its path."""
    p = tmp_path / name
    p.write_text(textwrap.dedent(content))
    return p


_EXPECTED_COLS = StudySummaryStats(
    study_id="_tmp",
    variants=pd.DataFrame(
        columns=[
            "variant_id", "rsid", "chromosome", "position", "effect_allele",
            "other_allele", "beta", "standard_error", "p_value",
            "effect_allele_frequency", "hm_code",
        ]
    ),
).EXPECTED_COLUMNS


def _make_study_simple(study_id: str, rows: list[dict]) -> StudySummaryStats:
    """Build a StudySummaryStats ensuring all expected columns exist."""
    df = pd.DataFrame(rows)
    for col in _EXPECTED_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return StudySummaryStats(study_id=study_id, variants=df)


# ---------------------------------------------------------------------------
# Tests: load_harmonized_file
# ---------------------------------------------------------------------------

class TestLoadHarmonizedFile:
    """Tests using the sample fixture and inline TSVs."""

    def test_load_sample_fixture(self) -> None:
        stats = load_harmonized_file(SAMPLE_FILE, study_id="GCST000001")
        # The fixture has 10 rows; one has hm_code=5 (filtered) and one has
        # missing beta (empty field).  So we expect 8 rows.
        assert stats.study_id == "GCST000001"
        assert len(stats.variants) == 8

    def test_hm_code_filtering(self, tmp_path: Path) -> None:
        content = """\
        hm_rsid\thm_chrom\thm_pos\thm_other_allele\thm_effect_allele\thm_beta\thm_effect_allele_frequency\thm_code\tstandard_error\tp_value
        rs1\t1\t100\tA\tG\t0.05\t0.3\t10\t0.01\t5.7e-7
        rs2\t1\t200\tA\tC\t0.06\t0.4\t5\t0.02\t2.7e-3
        rs3\t1\t300\tT\tG\t0.07\t0.5\t12\t0.03\t1.96e-2
        """
        p = _write_tsv(tmp_path, content)
        stats = load_harmonized_file(p, "study1")
        # hm_code=5 should be filtered out -> 2 rows remain
        assert len(stats.variants) == 2
        codes = stats.variants["hm_code"].tolist()
        assert 5 not in codes

    def test_custom_valid_hm_codes(self, tmp_path: Path) -> None:
        content = """\
        hm_rsid\thm_chrom\thm_pos\thm_other_allele\thm_effect_allele\thm_beta\thm_effect_allele_frequency\thm_code\tstandard_error\tp_value
        rs1\t1\t100\tA\tG\t0.05\t0.3\t10\t0.01\t5.7e-7
        rs2\t1\t200\tA\tC\t0.06\t0.4\t5\t0.02\t2.7e-3
        """
        p = _write_tsv(tmp_path, content)
        stats = load_harmonized_file(p, "study1", valid_hm_codes=[5])
        assert len(stats.variants) == 1

    def test_missing_beta_se_dropped(self, tmp_path: Path) -> None:
        content = """\
        hm_rsid\thm_chrom\thm_pos\thm_other_allele\thm_effect_allele\thm_beta\thm_effect_allele_frequency\thm_code\tstandard_error\tp_value
        rs1\t1\t100\tA\tG\t0.05\t0.3\t10\t0.01\t5.7e-7
        rs2\t1\t200\tA\tC\t\t0.4\t10\t0.02\t2.7e-3
        rs3\t1\t300\tT\tG\t0.07\t0.5\t10\t\t1.96e-2
        """
        p = _write_tsv(tmp_path, content)
        stats = load_harmonized_file(p, "study1")
        # rs1: has both beta and SE -> kept
        # rs2: missing beta, no OR -> dropped
        # rs3: has beta, missing SE but has p-value -> SE derived -> kept
        assert len(stats.variants) == 2

    def test_variant_id_format(self, tmp_path: Path) -> None:
        content = """\
        hm_rsid\thm_chrom\thm_pos\thm_other_allele\thm_effect_allele\thm_beta\thm_effect_allele_frequency\thm_code\tstandard_error\tp_value
        rs1\t2\t12345\tA\tG\t0.05\t0.3\t10\t0.01\t5.7e-7
        """
        p = _write_tsv(tmp_path, content)
        stats = load_harmonized_file(p, "study1")
        vid = stats.variants.iloc[0]["variant_id"]
        # Alleles sorted: A, G  ->  chr2:12345:A:G
        assert vid == "chr2:12345:A:G"

    def test_variant_id_sorted_alleles(self, tmp_path: Path) -> None:
        """Effect/other order should not matter -- alleles are sorted."""
        content = """\
        hm_rsid\thm_chrom\thm_pos\thm_other_allele\thm_effect_allele\thm_beta\thm_effect_allele_frequency\thm_code\tstandard_error\tp_value
        rs1\t5\t999\tG\tA\t0.05\t0.3\t10\t0.01\t5.7e-7
        """
        p = _write_tsv(tmp_path, content)
        stats = load_harmonized_file(p, "study1")
        vid = stats.variants.iloc[0]["variant_id"]
        assert vid == "chr5:999:A:G"


# ---------------------------------------------------------------------------
# Tests: align_studies
# ---------------------------------------------------------------------------

class TestAlignStudies:
    def _base_rows(self) -> list[dict]:
        """Shared variants for two studies (same allele orientation)."""
        return [
            {
                "variant_id": "chr1:100:A:G",
                "chromosome": "1",
                "position": 100,
                "effect_allele": "G",
                "other_allele": "A",
                "beta": 0.10,
                "standard_error": 0.02,
                "p_value": 1e-4,
                "effect_allele_frequency": 0.3,
                "hm_code": 10,
            },
            {
                "variant_id": "chr2:200:C:T",
                "chromosome": "2",
                "position": 200,
                "effect_allele": "T",
                "other_allele": "C",
                "beta": 0.20,
                "standard_error": 0.03,
                "p_value": 1e-5,
                "effect_allele_frequency": 0.4,
                "hm_code": 10,
            },
        ]

    def test_basic_alignment_same_alleles(self) -> None:
        rows = self._base_rows()
        s1 = _make_study_simple("s1", rows)
        s2 = _make_study_simple("s2", rows)
        result = align_studies([s1, s2], min_study_count=2)
        assert result["variant_id"].nunique() == 2
        for vid, grp in result.groupby("variant_id"):
            assert len(grp) == 2
            # Same alleles, same beta
            betas = grp["beta"].tolist()
            assert betas[0] == pytest.approx(betas[1])

    def test_allele_flip(self) -> None:
        """When alleles are swapped in study 2, beta should be negated."""
        rows1 = [
            {
                "variant_id": "chr1:100:A:G",
                "chromosome": "1",
                "position": 100,
                "effect_allele": "G",
                "other_allele": "A",
                "beta": 0.10,
                "standard_error": 0.02,
                "p_value": 1e-4,
                "effect_allele_frequency": 0.3,
                "hm_code": 10,
            },
        ]
        rows2 = [
            {
                "variant_id": "chr1:100:A:G",
                "chromosome": "1",
                "position": 100,
                "effect_allele": "A",   # swapped
                "other_allele": "G",    # swapped
                "beta": 0.05,
                "standard_error": 0.01,
                "p_value": 1e-3,
                "effect_allele_frequency": 0.7,
                "hm_code": 10,
            },
        ]
        s1 = _make_study_simple("s1", rows1)
        s2 = _make_study_simple("s2", rows2)
        result = align_studies([s1, s2])
        vid_rows = result[result["variant_id"] == "chr1:100:A:G"]
        s1_row = vid_rows[vid_rows["study_id"] == "s1"].iloc[0]
        s2_row = vid_rows[vid_rows["study_id"] == "s2"].iloc[0]
        # Study 1 beta unchanged
        assert s1_row["beta"] == pytest.approx(0.10)
        assert s1_row["se"] == pytest.approx(0.02)
        # Study 2 beta should be flipped: -0.05
        assert s2_row["beta"] == pytest.approx(-0.05)
        # SE is a magnitude -- orientation-independent -- so it must NOT flip
        # and must be preserved unchanged through the alignment.
        assert s2_row["se"] == pytest.approx(0.01)

    def test_min_study_count_filtering(self) -> None:
        """Variants in fewer than min_study_count studies are excluded."""
        shared_row = {
            "variant_id": "chr1:100:A:G",
            "chromosome": "1",
            "position": 100,
            "effect_allele": "G",
            "other_allele": "A",
            "beta": 0.10,
            "standard_error": 0.02,
            "p_value": 1e-4,
            "effect_allele_frequency": 0.3,
            "hm_code": 10,
        }
        unique_row = {
            "variant_id": "chr3:300:A:T",
            "chromosome": "3",
            "position": 300,
            "effect_allele": "T",
            "other_allele": "A",
            "beta": 0.50,
            "standard_error": 0.05,
            "p_value": 1e-10,
            "effect_allele_frequency": 0.2,
            "hm_code": 10,
        }
        s1 = _make_study_simple("s1", [shared_row, unique_row])
        s2 = _make_study_simple("s2", [shared_row])
        result = align_studies([s1, s2], min_study_count=2)
        vids = set(result["variant_id"])
        assert "chr1:100:A:G" in vids
        assert "chr3:300:A:T" not in vids

    def test_requires_at_least_two_studies(self) -> None:
        s1 = _make_study_simple("s1", self._base_rows())
        with pytest.raises(ValueError, match="at least 2"):
            align_studies([s1])

    def test_ambiguous_snps_still_included(self) -> None:
        """A/T SNPs are flagged but not removed."""
        at_row = {
            "variant_id": "chr5:500:A:T",
            "chromosome": "5",
            "position": 500,
            "effect_allele": "A",
            "other_allele": "T",
            "beta": 0.03,
            "standard_error": 0.01,
            "p_value": 0.01,
            "effect_allele_frequency": 0.5,
            "hm_code": 10,
        }
        s1 = _make_study_simple("s1", [at_row])
        s2 = _make_study_simple("s2", [at_row])
        result = align_studies([s1, s2])
        assert "chr5:500:A:T" in set(result["variant_id"])

    def test_palindromic_snp_naive_flip_behaviour(self) -> None:
        """Behaviour-pinning test for palindromic (strand-ambiguous) SNPs.

        Palindromic variants (A/T and C/G) are their own complement, so their
        strand cannot be inferred from alleles alone: a study reporting A/T on
        the reverse strand is indistinguishable from a genuine effect/other
        allele swap on the forward strand. ``align_studies`` does NOT attempt
        strand resolution -- it treats reversed alleles as ``match_swap`` and
        naively negates beta (harmonize.py match_swap branch).

        This test pins that CURRENT behaviour; it is not an assertion of
        biological correctness. It is safe for GWAS Catalog data because those
        files are already normalised to the forward strand upstream, so the
        A/T reversal below only arises from a genuine allele swap. The residual
        risk is confined to user-uploaded, non-strand-normalised files -- a
        known, documented limitation. If strand-resolution logic is ever added,
        this test will flag that the behaviour changed, forcing a conscious
        decision rather than a silent one.
        """
        rows1 = [
            {
                "variant_id": "chr7:700:A:T",
                "chromosome": "7",
                "position": 700,
                "effect_allele": "A",
                "other_allele": "T",
                "beta": 0.08,
                "standard_error": 0.02,
                "p_value": 1e-4,
                "effect_allele_frequency": 0.5,
                "hm_code": 10,
            },
        ]
        rows2 = [
            {
                "variant_id": "chr7:700:A:T",
                "chromosome": "7",
                "position": 700,
                "effect_allele": "T",   # reversed vs reference -- ambiguous strand
                "other_allele": "A",    # reversed vs reference -- ambiguous strand
                "beta": 0.06,
                "standard_error": 0.02,
                "p_value": 1e-3,
                "effect_allele_frequency": 0.5,
                "hm_code": 10,
            },
        ]
        s1 = _make_study_simple("s1", rows1)
        s2 = _make_study_simple("s2", rows2)
        result = align_studies([s1, s2])
        vid_rows = result[result["variant_id"] == "chr7:700:A:T"]
        # Variant is retained (not dropped) for both studies.
        assert set(vid_rows["study_id"]) == {"s1", "s2"}
        s1_row = vid_rows[vid_rows["study_id"] == "s1"].iloc[0]
        s2_row = vid_rows[vid_rows["study_id"] == "s2"].iloc[0]
        # Reference study is untouched.
        assert s1_row["beta"] == pytest.approx(0.08)
        # PINNED: reversed alleles are treated as a swap and beta is naively
        # negated -- no palindrome-aware strand handling is performed.
        assert s2_row["beta"] == pytest.approx(-0.06)
        # SE is orientation-independent and preserved unchanged.
        assert s2_row["se"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# Tests: align_studies_chunked
# ---------------------------------------------------------------------------

class TestAlignStudiesChunked:
    def _multi_chrom_rows(self) -> list[dict]:
        """Variants spanning chr1, chr2, and chr3."""
        return [
            {
                "variant_id": "chr1:100:A:G",
                "chromosome": "1",
                "position": 100,
                "effect_allele": "G",
                "other_allele": "A",
                "beta": 0.10,
                "standard_error": 0.02,
                "p_value": 1e-4,
                "effect_allele_frequency": 0.3,
                "hm_code": 10,
            },
            {
                "variant_id": "chr2:200:C:T",
                "chromosome": "2",
                "position": 200,
                "effect_allele": "T",
                "other_allele": "C",
                "beta": 0.20,
                "standard_error": 0.03,
                "p_value": 1e-5,
                "effect_allele_frequency": 0.4,
                "hm_code": 10,
            },
            {
                "variant_id": "chr3:300:A:T",
                "chromosome": "3",
                "position": 300,
                "effect_allele": "T",
                "other_allele": "A",
                "beta": 0.05,
                "standard_error": 0.01,
                "p_value": 0.01,
                "effect_allele_frequency": 0.5,
                "hm_code": 10,
            },
        ]

    def test_matches_unchunked_output(self) -> None:
        """Chunked alignment must produce identical results to unchunked."""
        rows = self._multi_chrom_rows()
        s1 = _make_study_simple("s1", rows)
        s2 = _make_study_simple("s2", rows)

        unchunked = align_studies([s1, s2], min_study_count=2)
        chunked = align_studies_chunked([s1, s2], min_study_count=2)

        # Same set of variants
        assert set(chunked["variant_id"]) == set(unchunked["variant_id"])
        # Same number of rows
        assert len(chunked) == len(unchunked)

        # Same beta/se values per (variant_id, study_id)
        for df in [unchunked, chunked]:
            df.sort_values(["variant_id", "study_id"], inplace=True)
            df.reset_index(drop=True, inplace=True)
        pd.testing.assert_frame_equal(chunked, unchunked)

    def test_allele_flip_across_chromosomes(self) -> None:
        """Beta flips work correctly in chunked mode."""
        row_chr1 = {
            "variant_id": "chr1:100:A:G",
            "chromosome": "1",
            "position": 100,
            "effect_allele": "G",
            "other_allele": "A",
            "beta": 0.10,
            "standard_error": 0.02,
            "p_value": 1e-4,
            "effect_allele_frequency": 0.3,
            "hm_code": 10,
        }
        row_chr1_swapped = {
            **row_chr1,
            "effect_allele": "A",
            "other_allele": "G",
            "beta": 0.05,
        }
        s1 = _make_study_simple("s1", [row_chr1])
        s2 = _make_study_simple("s2", [row_chr1_swapped])

        result = align_studies_chunked([s1, s2])
        s2_row = result[result["study_id"] == "s2"].iloc[0]
        assert s2_row["beta"] == pytest.approx(-0.05)

    def test_progress_callback_called(self) -> None:
        """Progress callback is invoked once per chromosome."""
        rows = self._multi_chrom_rows()
        s1 = _make_study_simple("s1", rows)
        s2 = _make_study_simple("s2", rows)

        calls: list[tuple[str, int, int]] = []
        def _cb(chrom: str, i: int, total: int) -> None:
            calls.append((chrom, i, total))

        align_studies_chunked([s1, s2], progress_callback=_cb)
        assert len(calls) == 3  # chr1, chr2, chr3
        assert calls[-1][1] == calls[-1][2]  # last call: i == total

    def test_requires_at_least_two_studies(self) -> None:
        rows = self._multi_chrom_rows()
        s1 = _make_study_simple("s1", rows)
        with pytest.raises(ValueError, match="at least 2"):
            align_studies_chunked([s1])

    def test_chromosome_only_in_one_study_skipped(self) -> None:
        """If a chromosome exists in only one study, it's skipped."""
        shared = {
            "variant_id": "chr1:100:A:G",
            "chromosome": "1",
            "position": 100,
            "effect_allele": "G",
            "other_allele": "A",
            "beta": 0.10,
            "standard_error": 0.02,
            "p_value": 1e-4,
            "effect_allele_frequency": 0.3,
            "hm_code": 10,
        }
        unique = {
            "variant_id": "chr22:500:C:G",
            "chromosome": "22",
            "position": 500,
            "effect_allele": "G",
            "other_allele": "C",
            "beta": 0.30,
            "standard_error": 0.04,
            "p_value": 1e-6,
            "effect_allele_frequency": 0.2,
            "hm_code": 10,
        }
        s1 = _make_study_simple("s1", [shared, unique])
        s2 = _make_study_simple("s2", [shared])

        result = align_studies_chunked([s1, s2])
        assert set(result["variant_id"]) == {"chr1:100:A:G"}
