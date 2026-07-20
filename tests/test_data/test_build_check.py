"""Tests for the sentinel-SNP genome-build check (data/build_check.py).

The check is a user-facing safety net: it warns when an uploaded study
looks like the wrong genome build (GRCh37 instead of the expected
GRCh38), a failure that otherwise surfaces only as a silent zero-overlap
at meta-analysis time. These tests pin the verdict logic and the sentinel
coordinates against regressions.
"""

import pandas as pd

from gwas_meta.data.build_check import (
    SENTINEL_SNPS,
    BuildVerdict,
    check_genome_build,
)

# Sentinels indexed by rsID for convenient row construction.
_BY_RS = {s["rsid"]: s for s in SENTINEL_SNPS}


def _row(rsid: str, build: str) -> dict:
    """Build one study row placing ``rsid`` at its GRCh37 or GRCh38 position."""
    s = _BY_RS[rsid]
    return {"rsid": rsid, "chromosome": s["chromosome"], "position": s[build]}


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["rsid", "chromosome", "position"])


class TestVerdicts:
    def test_all_grch38_is_expected_no_warning(self) -> None:
        df = _df([_row(s["rsid"], "grch38") for s in SENTINEL_SNPS])
        v = check_genome_build(df)
        assert v.verdict == "grch38"
        assert v.n_matches == len(SENTINEL_SNPS)
        assert set(v.grch38_hits) == {s["rsid"] for s in SENTINEL_SNPS}
        assert v.grch37_hits == []
        assert v.is_warning is False

    def test_two_grch37_hits_warns(self) -> None:
        df = _df([_row("rs7412", "grch37"), _row("rs429358", "grch37")])
        v = check_genome_build(df)
        assert v.verdict == "grch37"
        assert v.n_matches == 2
        assert set(v.grch37_hits) == {"rs7412", "rs429358"}
        assert v.is_warning is True

    def test_single_grch37_hit_too_weak_to_warn(self) -> None:
        # One lone GRCh37 hit stays "unknown" (see docstring / line 128 >= 2).
        df = _df([_row("rs334", "grch37")])
        v = check_genome_build(df)
        assert v.verdict == "unknown"
        assert v.grch37_hits == ["rs334"]
        assert v.is_warning is False

    def test_mixed_builds_reported_as_mixed(self) -> None:
        # Any co-occurrence of GRCh37 and GRCh38 hits -> "mixed", regardless
        # of counts (conservative: never assert a build on conflicting evidence).
        df = _df([
            _row("rs7412", "grch37"),
            _row("rs429358", "grch37"),
            _row("rs334", "grch38"),
        ])
        v = check_genome_build(df)
        assert v.verdict == "mixed"
        assert set(v.grch37_hits) == {"rs7412", "rs429358"}
        assert v.grch38_hits == ["rs334"]
        assert v.is_warning is True

    def test_no_sentinels_present_is_unknown(self) -> None:
        df = _df([{"rsid": "rs_not_a_sentinel", "chromosome": "1", "position": 12345}])
        v = check_genome_build(df)
        assert v.verdict == "unknown"
        assert v.n_matches == 0


class TestEdgeCases:
    def test_empty_dataframe_is_unknown(self) -> None:
        v = check_genome_build(_df([]))
        assert v.verdict == "unknown"
        assert v.n_matches == 0

    def test_none_input_is_unknown(self) -> None:
        v = check_genome_build(None)
        assert v.verdict == "unknown"

    def test_missing_rsid_column_is_unknown(self) -> None:
        df = pd.DataFrame({"chromosome": ["19"], "position": [44908822]})
        v = check_genome_build(df)
        assert v.verdict == "unknown"

    def test_duplicate_rsid_takes_first_row(self) -> None:
        # Two rows for one rsID: the first is a valid GRCh38 position, the
        # second is garbage. iloc[0] must take the first -> counted as GRCh38.
        s = _BY_RS["rs7412"]
        df = _df([
            {"rsid": "rs7412", "chromosome": s["chromosome"], "position": s["grch38"]},
            {"rsid": "rs7412", "chromosome": s["chromosome"], "position": 999},
        ])
        v = check_genome_build(df)
        assert "rs7412" in v.grch38_hits

    def test_position_at_neither_build_is_mismatch(self) -> None:
        s = _BY_RS["rs334"]
        df = _df([{"rsid": "rs334", "chromosome": s["chromosome"], "position": 999}])
        v = check_genome_build(df)
        assert v.mismatch_hits == ["rs334"]
        assert v.n_matches == 0
        assert v.verdict == "unknown"

    def test_wrong_chromosome_is_mismatch(self) -> None:
        # Correct GRCh38 position but wrong chromosome label -> mismatch,
        # not a build hit.
        s = _BY_RS["rs334"]
        df = _df([{"rsid": "rs334", "chromosome": "99", "position": s["grch38"]}])
        v = check_genome_build(df)
        assert v.mismatch_hits == ["rs334"]
        assert v.verdict == "unknown"


class TestSentinelCoordinates:
    def test_sentinel_positions_are_frozen(self) -> None:
        # Pin the shipped sentinel coordinates so an accidental edit is caught.
        # Positions are dbSNP GRCh37/GRCh38 for these well-known variants.
        expected = {
            "rs7412": ("19", 45412079, 44908822),
            "rs429358": ("19", 45411941, 44908684),
            "rs334": ("11", 5248232, 5227002),
            "rs1801133": ("1", 11856378, 11796321),
            "rs6025": ("1", 169519049, 169549811),
            "rs1800562": ("6", 26093141, 26092913),
        }
        got = {s["rsid"]: (s["chromosome"], s["grch37"], s["grch38"])
               for s in SENTINEL_SNPS}
        assert got == expected

    def test_grch37_and_grch38_positions_differ(self) -> None:
        # The whole check relies on the two builds disagreeing per sentinel.
        for s in SENTINEL_SNPS:
            assert s["grch37"] != s["grch38"], f"{s['rsid']} has equal build positions"


def test_returns_build_verdict_instance() -> None:
    v = check_genome_build(_df([_row("rs7412", "grch38")]))
    assert isinstance(v, BuildVerdict)
