"""Tests for the meta-analysis engine using hand-calculated values."""

import numpy as np
import pytest
from scipy.stats import norm

import pandas as pd

from gwas_meta.meta_analysis import (
    MetaAnalysisInput,
    MetaAnalysisResult,
    cochrans_q,
    dl_random_effects,
    i_squared,
    ivw_fixed_effects,
    leave_one_out_max_p,
    run_meta_analysis,
    run_meta_analysis_batch,
    tau_squared_dl,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def three_study_input() -> MetaAnalysisInput:
    """Three-study example used throughout the tests."""
    return MetaAnalysisInput(
        variant_id="rs12345",
        betas=np.array([0.15, 0.20, 0.10]),
        standard_errors=np.array([0.05, 0.04, 0.06]),
        study_ids=["study_a", "study_b", "study_c"],
    )


@pytest.fixture()
def single_study_input() -> MetaAnalysisInput:
    """Single-study edge case."""
    return MetaAnalysisInput(
        variant_id="rs99999",
        betas=np.array([0.25]),
        standard_errors=np.array([0.03]),
        study_ids=["only_study"],
    )


# ---------------------------------------------------------------------------
# Hand-calculated reference values for the three-study example
# betas  = [0.15, 0.20, 0.10]
# SEs    = [0.05, 0.04, 0.06]
# w      = [400, 625, 277.777...]
# sum(w) = 1302.777...
# beta_fixed = (60 + 125 + 27.777...) / 1302.777... = 212.777... / 1302.777...
# ---------------------------------------------------------------------------

def _hand_calc_fixed() -> dict:
    """Return hand-calculated fixed-effects values."""
    betas = np.array([0.15, 0.20, 0.10])
    ses = np.array([0.05, 0.04, 0.06])
    w = 1.0 / ses ** 2  # [400, 625, 277.7778]
    sw = np.sum(w)

    beta = np.sum(w * betas) / sw
    se = 1.0 / np.sqrt(sw)
    z = beta / se
    p = 2.0 * norm.sf(abs(z))
    return {"beta": beta, "se": se, "z": z, "p": p, "weights": w, "sum_w": sw}


def _hand_calc_heterogeneity() -> dict:
    """Return hand-calculated heterogeneity values."""
    betas = np.array([0.15, 0.20, 0.10])
    ses = np.array([0.05, 0.04, 0.06])
    w = 1.0 / ses ** 2
    sw = np.sum(w)
    beta_fixed = np.sum(w * betas) / sw

    q = float(np.sum(w * (betas - beta_fixed) ** 2))
    k = 3
    i2 = max(0.0, (q - (k - 1)) / q * 100.0)
    c = sw - np.sum(w ** 2) / sw
    tau2 = max(0.0, (q - (k - 1)) / c)
    return {"q": q, "i2": i2, "tau2": tau2, "c": c}


# ---------------------------------------------------------------------------
# Heterogeneity tests
# ---------------------------------------------------------------------------

class TestCochransQ:
    def test_three_studies(self, three_study_input: MetaAnalysisInput) -> None:
        expected = _hand_calc_heterogeneity()["q"]
        result = cochrans_q(three_study_input.betas, three_study_input.standard_errors)
        assert result == pytest.approx(expected, rel=1e-6)

    def test_single_study(self, single_study_input: MetaAnalysisInput) -> None:
        # With one study Q is always 0
        q = cochrans_q(single_study_input.betas, single_study_input.standard_errors)
        assert q == pytest.approx(0.0, abs=1e-12)


class TestISquared:
    def test_three_studies(self) -> None:
        het = _hand_calc_heterogeneity()
        result = i_squared(het["q"], 3)
        assert result == pytest.approx(het["i2"], rel=1e-6)

    def test_single_study(self) -> None:
        assert i_squared(0.0, 1) == 0.0

    def test_no_heterogeneity(self) -> None:
        # Q exactly equal to k-1 means I^2 = 0
        assert i_squared(2.0, 3) == 0.0

    def test_floor_at_zero(self) -> None:
        # Q < k-1 should still give 0 (not negative)
        assert i_squared(1.0, 3) == 0.0


class TestTauSquaredDL:
    def test_three_studies(self, three_study_input: MetaAnalysisInput) -> None:
        expected = _hand_calc_heterogeneity()["tau2"]
        result = tau_squared_dl(
            three_study_input.betas, three_study_input.standard_errors
        )
        assert result == pytest.approx(expected, rel=1e-6)

    def test_single_study(self, single_study_input: MetaAnalysisInput) -> None:
        tau2 = tau_squared_dl(
            single_study_input.betas, single_study_input.standard_errors
        )
        assert tau2 == 0.0


# ---------------------------------------------------------------------------
# Fixed-effects tests
# ---------------------------------------------------------------------------

class TestFixedEffects:
    def test_beta_and_se(self, three_study_input: MetaAnalysisInput) -> None:
        ref = _hand_calc_fixed()
        result = ivw_fixed_effects(three_study_input)
        assert result["beta"] == pytest.approx(ref["beta"], rel=1e-6)
        assert result["se"] == pytest.approx(ref["se"], rel=1e-6)

    def test_z_and_p(self, three_study_input: MetaAnalysisInput) -> None:
        ref = _hand_calc_fixed()
        result = ivw_fixed_effects(three_study_input)
        assert result["z"] == pytest.approx(ref["z"], rel=1e-6)
        assert result["p"] == pytest.approx(ref["p"], rel=1e-6)

    def test_p_value_range(self, three_study_input: MetaAnalysisInput) -> None:
        result = ivw_fixed_effects(three_study_input)
        assert 0.0 < result["p"] < 1.0

    def test_single_study_matches_input(
        self, single_study_input: MetaAnalysisInput
    ) -> None:
        result = ivw_fixed_effects(single_study_input)
        assert result["beta"] == pytest.approx(0.25, rel=1e-6)
        assert result["se"] == pytest.approx(0.03, rel=1e-6)


# ---------------------------------------------------------------------------
# Random-effects tests
# ---------------------------------------------------------------------------

class TestRandomEffects:
    def test_three_studies(self, three_study_input: MetaAnalysisInput) -> None:
        het = _hand_calc_heterogeneity()
        tau2 = het["tau2"]

        betas = np.array([0.15, 0.20, 0.10])
        ses = np.array([0.05, 0.04, 0.06])
        w_star = 1.0 / (ses ** 2 + tau2)
        expected_beta = float(np.sum(w_star * betas) / np.sum(w_star))
        expected_se = float(1.0 / np.sqrt(np.sum(w_star)))

        result = dl_random_effects(three_study_input)
        assert result["beta"] == pytest.approx(expected_beta, rel=1e-6)
        assert result["se"] == pytest.approx(expected_se, rel=1e-6)
        assert result["tau_squared"] == pytest.approx(tau2, rel=1e-6)

    def test_p_value_range(self, three_study_input: MetaAnalysisInput) -> None:
        result = dl_random_effects(three_study_input)
        assert 0.0 < result["p"] < 1.0

    def test_single_study_equals_fixed(
        self, single_study_input: MetaAnalysisInput
    ) -> None:
        fixed = ivw_fixed_effects(single_study_input)
        random = dl_random_effects(single_study_input)
        # With one study tau^2 = 0, so random == fixed
        assert random["beta"] == pytest.approx(fixed["beta"], rel=1e-6)
        assert random["se"] == pytest.approx(fixed["se"], rel=1e-6)


# ---------------------------------------------------------------------------
# Integration: run_meta_analysis
# ---------------------------------------------------------------------------

class TestRunMetaAnalysis:
    def test_returns_result_object(self, three_study_input: MetaAnalysisInput) -> None:
        result = run_meta_analysis(three_study_input)
        assert isinstance(result, MetaAnalysisResult)

    def test_variant_id_preserved(self, three_study_input: MetaAnalysisInput) -> None:
        result = run_meta_analysis(three_study_input)
        assert result.variant_id == "rs12345"
        assert result.n_studies == 3
        assert result.study_ids == ["study_a", "study_b", "study_c"]

    def test_fixed_random_consistency(
        self, three_study_input: MetaAnalysisInput
    ) -> None:
        result = run_meta_analysis(three_study_input)
        ref_fixed = _hand_calc_fixed()
        ref_het = _hand_calc_heterogeneity()

        assert result.beta_fixed == pytest.approx(ref_fixed["beta"], rel=1e-6)
        assert result.se_fixed == pytest.approx(ref_fixed["se"], rel=1e-6)
        assert result.q_stat == pytest.approx(ref_het["q"], rel=1e-6)
        assert result.i_squared == pytest.approx(ref_het["i2"], rel=1e-6)
        assert result.tau_squared == pytest.approx(ref_het["tau2"], rel=1e-6)

    def test_p_values_in_range(self, three_study_input: MetaAnalysisInput) -> None:
        result = run_meta_analysis(three_study_input)
        assert 0.0 < result.p_fixed < 1.0
        assert 0.0 < result.p_random < 1.0

    def test_single_study(self, single_study_input: MetaAnalysisInput) -> None:
        result = run_meta_analysis(single_study_input)
        assert result.n_studies == 1
        assert result.i_squared == 0.0
        assert result.tau_squared == 0.0
        assert result.beta_fixed == pytest.approx(result.beta_random, rel=1e-6)
        assert result.se_fixed == pytest.approx(result.se_random, rel=1e-6)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            MetaAnalysisInput(
                variant_id="rs1",
                betas=np.array([0.1, 0.2]),
                standard_errors=np.array([0.05]),
                study_ids=["a", "b"],
            )

    def test_zero_se_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            MetaAnalysisInput(
                variant_id="rs1",
                betas=np.array([0.1]),
                standard_errors=np.array([0.0]),
                study_ids=["a"],
            )

    def test_negative_se_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            MetaAnalysisInput(
                variant_id="rs1",
                betas=np.array([0.1]),
                standard_errors=np.array([-0.05]),
                study_ids=["a"],
            )

    def test_empty_input_rejected(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            MetaAnalysisInput(
                variant_id="rs1",
                betas=np.array([]),
                standard_errors=np.array([]),
                study_ids=[],
            )


# ---------------------------------------------------------------------------
# Consistency: vectorized batch path == single-variant reference path
#
# The interactive pipeline (step5_meta.render) runs `run_meta_analysis_batch`,
# while `run_meta_analysis` is the hand-validated (and metafor-cross-checked)
# reference. These tests certify the fast genome-scale path agrees with the
# reference on identical inputs, so the reference's validation transfers to it.
# ---------------------------------------------------------------------------

def _build_long_df(variants: dict) -> pd.DataFrame:
    """Build an aligned long-format frame [variant_id, study_id, beta, se].

    ``variants`` maps variant_id -> (betas, ses); study ids are synthesized
    positionally as ``s0, s1, ...``.
    """
    rows = []
    for vid, (betas, ses) in variants.items():
        for j, (b, s) in enumerate(zip(betas, ses)):
            rows.append({"variant_id": vid, "study_id": f"s{j}",
                         "beta": float(b), "se": float(s)})
    return pd.DataFrame(rows)


_BATCH_FIELDS = [
    ("beta_fixed", "beta_fixed"), ("se_fixed", "se_fixed"),
    ("z_fixed", "z_fixed"), ("p_fixed", "p_fixed"),
    ("beta_random", "beta_random"), ("se_random", "se_random"),
    ("z_random", "z_random"), ("p_random", "p_random"),
    ("q_stat", "q_stat"), ("i_squared", "i_squared"),
    ("tau_squared", "tau_squared"),
]


class TestVectorizedConsistency:
    def _assert_matches_reference(self, variants: dict) -> None:
        aligned = _build_long_df(variants)
        batch = run_meta_analysis_batch(aligned, as_dataframe=True)
        batch = batch.set_index("variant_id")

        assert set(batch.index) == set(variants)
        for vid, (betas, ses) in variants.items():
            ref = run_meta_analysis(MetaAnalysisInput(
                variant_id=vid,
                betas=np.array(betas, dtype=np.float64),
                standard_errors=np.array(ses, dtype=np.float64),
                study_ids=[f"s{j}" for j in range(len(betas))],
            ))
            brow = batch.loc[vid]
            for bcol, rattr in _BATCH_FIELDS:
                assert brow[bcol] == pytest.approx(
                    getattr(ref, rattr), rel=1e-9, abs=1e-12
                ), f"{vid}.{bcol} batch != reference"
            assert int(brow["n_studies"]) == ref.n_studies

    def test_uniform_k2(self) -> None:
        # All variants have k=2 -> exercises the fast uniform-reshape branch.
        rng = np.random.default_rng(42)
        variants = {}
        for i in range(25):
            ses = rng.uniform(0.02, 0.08, size=2)
            betas = rng.normal(0.1, 0.05, size=2)
            variants[f"v{i}"] = (betas, ses)
        self._assert_matches_reference(variants)

    def test_mixed_k(self) -> None:
        # Variants with differing k -> exercises the group-by-study-count branch.
        rng = np.random.default_rng(7)
        variants = {}
        for i in range(40):
            k = int(rng.integers(2, 7))  # k in 2..6
            ses = rng.uniform(0.015, 0.06, size=k)
            betas = rng.normal(0.08, 0.06, size=k)
            variants[f"v{i}"] = (betas, ses)
        self._assert_matches_reference(variants)

    def test_high_heterogeneity_case(self) -> None:
        # Opposite-sign, high-Q variant where FE and RE diverge most.
        self._assert_matches_reference({
            "het": (np.array([0.30, -0.25, 0.05]), np.array([0.02, 0.02, 0.02])),
        })

    def test_dataframe_and_object_paths_agree(self) -> None:
        # as_dataframe=True vs the MetaAnalysisResult-object return path.
        aligned = _build_long_df({
            "a": (np.array([0.1, 0.2, 0.15]), np.array([0.03, 0.04, 0.05])),
            "b": (np.array([0.05, 0.06]), np.array([0.02, 0.03])),
        })
        as_df = run_meta_analysis_batch(aligned, as_dataframe=True).set_index("variant_id")
        as_obj = {r.variant_id: r for r in run_meta_analysis_batch(aligned)}
        for vid in as_obj:
            assert as_df.loc[vid, "p_fixed"] == pytest.approx(as_obj[vid].p_fixed, rel=1e-12)
            assert as_df.loc[vid, "tau_squared"] == pytest.approx(
                as_obj[vid].tau_squared, rel=1e-12
            )


# ---------------------------------------------------------------------------
# Leave-one-out sensitivity (leave_one_out_max_p)
# ---------------------------------------------------------------------------

class TestLeaveOneOut:
    def test_requires_three_studies(self) -> None:
        p, sid = leave_one_out_max_p([0.1, 0.2], [0.03, 0.03], ["a", "b"])
        assert np.isnan(p)
        assert sid == ""

    def test_identifies_most_influential_study(self) -> None:
        # A and B carry the signal (beta=0.2); C is null (beta=0.0).
        # Dropping C -> [0.2, 0.2] (most significant, smallest p).
        # Dropping A or B -> [0.2, 0.0] (least significant, largest p).
        # So the worst-case p comes from dropping A (first of the tie).
        betas = [0.2, 0.2, 0.0]
        ses = [0.03, 0.03, 0.03]
        p_max, worst = leave_one_out_max_p(betas, ses, ["A", "B", "C"])

        # Independent oracle for the [0.2, 0.0] leave-one-out result.
        w = np.array([1 / 0.03**2, 1 / 0.03**2])
        beta_lo = np.sum(w * np.array([0.2, 0.0])) / np.sum(w)
        se_lo = 1.0 / np.sqrt(np.sum(w))
        expected_p = float(2.0 * norm.sf(abs(beta_lo / se_lo)))

        assert worst == "A"
        assert p_max == pytest.approx(expected_p, rel=1e-9)

    def test_robust_hit_stays_significant(self) -> None:
        # Three concordant strong studies: even the worst-case drop is tiny.
        p_max, worst = leave_one_out_max_p(
            [0.5, 0.52, 0.48], [0.02, 0.02, 0.02], ["A", "B", "C"]
        )
        assert p_max < 1e-8
        assert worst in {"A", "B", "C"}
