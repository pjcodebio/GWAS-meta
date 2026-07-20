"""Loading, harmonisation and allele-alignment of GWAS summary statistics."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pa_csv

from .models import StudySummaryStats

logger = logging.getLogger(__name__)

# GWAS Catalog harmonised-file columns  ->  our internal names
# Old format (hm_* prefix) -- pre-2023 GWAS Catalog files
_COLUMN_MAP_OLD: dict[str, str] = {
    "hm_rsid": "rsid",
    "hm_chrom": "chromosome",
    "hm_pos": "position",
    "hm_effect_allele": "effect_allele",
    "hm_other_allele": "other_allele",
    "hm_beta": "beta",
    "hm_odds_ratio": "odds_ratio",
    "standard_error": "standard_error",
    "p_value": "p_value",
    "hm_effect_allele_frequency": "effect_allele_frequency",
    "hm_code": "hm_code",
}

# New GWAS-SSF format (no hm_ prefix) -- 2023+ GWAS Catalog files
_COLUMN_MAP_NEW: dict[str, str] = {
    "rsid": "rsid",
    "chromosome": "chromosome",
    "base_pair_location": "position",
    "effect_allele": "effect_allele",
    "other_allele": "other_allele",
    "beta": "beta",
    "odds_ratio": "odds_ratio",
    "standard_error": "standard_error",
    "p_value": "p_value",
    "effect_allele_frequency": "effect_allele_frequency",
    "hm_code": "hm_code",
}

_REQUIRED_COLUMNS_OLD = {"hm_chrom", "hm_pos", "hm_effect_allele", "hm_other_allele"}
_REQUIRED_COLUMNS_NEW = {"chromosome", "base_pair_location", "effect_allele", "other_allele"}

# hm_code values that represent successful harmonisation of non-palindromic
# variants per the GWAS Catalog / EBI harmoniser spec:
#   10 = forward strand, alleles correct
#   11 = forward strand, flipped alleles
#   12 = reverse strand, alleles correct
#   13 = reverse strand, flipped alleles
# Palindromic codes (1-9) and failure codes (14-18) are excluded so no
# strand-ambiguous variant reaches downstream meta-analysis.
# See https://ebispot.github.io/gwas-sumstats-harmoniser-documentation/Reference-guide/Hm_code
DEFAULT_VALID_HM_CODES: list[int] = [10, 11, 12, 13]


# ---------------------------------------------------------------------------
# Shared per-study QC
# ---------------------------------------------------------------------------

def apply_variant_qc(
    df: pd.DataFrame,
    study_id: str,
    *,
    valid_hm_codes: list[int],
    maf_threshold: float = 0.01,
    max_abs_beta: float = 10.0,
    max_se: float = 10.0,
) -> pd.DataFrame:
    """Apply the per-study variant QC pipeline — the single source of truth.

    Operates on a DataFrame whose columns have already been renamed to the
    internal names (``chromosome``, ``position``, ``effect_allele``,
    ``other_allele``, and any of ``beta``, ``odds_ratio``, ``standard_error``,
    ``p_value``, ``effect_allele_frequency``, ``hm_code``, ``rsid``).

    Both loaders call this so their QC can never drift:
    :func:`load_harmonized_file` (whole-file path) and
    :func:`_chunk_single_study` (disk-backed path).

    Steps, in order. Every step is a row-wise filter or transform except the
    multi-allelic group filter (which runs last), so the final variant set is
    independent of the ordering:

      1. hm_code filter (keep only ``valid_hm_codes``)
      2. derive beta from log(OR) where beta is missing
      3. derive SE from beta and p-value where SE is missing
      4. drop rows with missing/invalid beta or SE (require SE > 0)
      5. beta/SE plausibility bounds
      6. p-value vs Z-score consistency (skips derived-SE rows)
      7. coerce genomic-coordinate / allele types
      8. MAF filter
      9. drop indels (SNPs only)
     10. drop multi-allelic positions
     11. build the canonical ``variant_id``

    Returns the cleaned DataFrame carrying the standardised
    :class:`~gwas_meta.data.models.StudySummaryStats` columns.
    """
    # rsid is optional in source files; keep an empty column when missing so
    # downstream code can rely on it always being present.
    if "rsid" not in df.columns:
        df["rsid"] = ""
    else:
        df["rsid"] = df["rsid"].fillna("").astype(str)

    # Filter by hm_code -----------------------------------------------------
    if "hm_code" in df.columns:
        before = len(df)
        df["hm_code"] = pd.to_numeric(df["hm_code"], errors="coerce")
        df = df[df["hm_code"].isin(valid_hm_codes)]
        logger.debug("hm_code filter: %d -> %d rows", before, len(df))
    else:
        logger.warning(
            "No hm_code column found for %s; skipping hm_code filter", study_id
        )
        df["hm_code"] = None

    # Derive beta and SE when only odds ratios are available ----------------
    if "beta" in df.columns:
        df["beta"] = pd.to_numeric(df["beta"], errors="coerce")
    else:
        df["beta"] = np.nan

    if "odds_ratio" in df.columns:
        df["odds_ratio"] = pd.to_numeric(df["odds_ratio"], errors="coerce")
        # Fill missing betas from log(OR)
        missing_beta = df["beta"].isna() & df["odds_ratio"].notna()
        if missing_beta.any():
            df.loc[missing_beta, "beta"] = np.log(df.loc[missing_beta, "odds_ratio"])
            logger.info(
                "Derived beta from log(OR) for %d variants in %s",
                missing_beta.sum(), study_id,
            )

    if "standard_error" in df.columns:
        df["standard_error"] = pd.to_numeric(df["standard_error"], errors="coerce")
    else:
        df["standard_error"] = np.nan

    if "p_value" in df.columns:
        df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    else:
        df["p_value"] = np.nan

    # Derive SE from beta and p-value when SE is missing:  SE = |beta| / |Z|
    # Track which rows had SE derived so the p-vs-Z consistency check below
    # can skip them (the check is tautological on derived rows: Z = beta/SE
    # would just reproduce the p-value it was derived from).
    missing_se = df["standard_error"].isna() & df["beta"].notna() & df["p_value"].notna()
    df["_se_derived"] = False
    if missing_se.any():
        from scipy.stats import norm as _norm
        p_vals = df.loc[missing_se, "p_value"].values
        betas = df.loc[missing_se, "beta"].values
        # Clamp p-values away from 0 and 1 to avoid inf
        p_clamped = np.clip(p_vals, 1e-300, 1.0 - 1e-10)
        z_vals = np.abs(_norm.ppf(p_clamped / 2.0))
        se_derived = np.where(z_vals > 0, np.abs(betas) / z_vals, np.nan)
        df.loc[missing_se, "standard_error"] = se_derived
        df.loc[missing_se, "_se_derived"] = True
        logger.info(
            "Derived SE from beta/p-value for %d variants in %s",
            missing_se.sum(), study_id,
        )

    # Drop rows still missing beta or SE ------------------------------------
    before = len(df)
    df.dropna(subset=["beta", "standard_error"], inplace=True)
    # Drop rows with SE <= 0 (invalid)
    df = df[df["standard_error"] > 0]
    logger.debug("Dropped %d rows with missing/invalid beta/se", before - len(df))

    # Beta/SE plausibility bounds -------------------------------------------
    if max_abs_beta > 0 or max_se > 0:
        before = len(df)
        plausible = pd.Series(True, index=df.index)
        if max_abs_beta > 0:
            plausible &= df["beta"].abs() <= max_abs_beta
        if max_se > 0:
            plausible &= df["standard_error"] <= max_se
        df = df[plausible]
        n_implausible = before - len(df)
        if n_implausible > 0:
            logger.info(
                "Excluded %d variants with implausible beta/SE in %s "
                "(|beta| > %g or SE > %g)",
                n_implausible, study_id, max_abs_beta, max_se,
            )

    # P-value vs Z-score consistency check ------------------------------------
    # Skip rows whose SE was derived from (beta, p): the check would be
    # tautological because Z = beta / (|beta|/|Z(p)|) reproduces p exactly.
    # Only rows with an independently reported SE carry information here.
    from .qc import check_pz_consistency as _pz_check
    _has_all = (
        df["beta"].notna() & df["standard_error"].notna()
        & (df["standard_error"] > 0) & df["p_value"].notna()
        & ~df["_se_derived"]
    )
    n_derived_skipped = int(df["_se_derived"].sum())
    if _has_all.any():
        _consistent = np.ones(len(df), dtype=bool)
        _idx = _has_all.values
        _consistent[_idx] = _pz_check(
            df.loc[_has_all, "beta"].values,
            df.loc[_has_all, "standard_error"].values,
            df.loc[_has_all, "p_value"].values,
        )
        n_pz_bad = int((~_consistent).sum())
        if n_pz_bad > 0:
            df = df[_consistent]
            logger.info(
                "Removed %d p-value/Z-score inconsistent variants from %s "
                "(check ran on %d rows with reported SE; %d rows with derived "
                "SE were skipped)",
                n_pz_bad, study_id, int(_has_all.sum()), n_derived_skipped,
            )
        elif n_derived_skipped > 0:
            logger.info(
                "p-vs-Z check: no inconsistent variants in %s (%d rows with "
                "derived SE skipped as check would be tautological)",
                study_id, n_derived_skipped,
            )

    df.drop(columns=["_se_derived"], inplace=True)

    # Drop odds_ratio column -- no longer needed
    df.drop(columns=["odds_ratio"], errors="ignore", inplace=True)

    # Drop rows with missing essentials (genomic coordinates + alleles) ------
    df.dropna(
        subset=["chromosome", "position", "effect_allele", "other_allele"],
        inplace=True,
    )

    # Coerce types -----------------------------------------------------------
    # Chromosome may be read as float (e.g. 1.0) when NaN rows exist --
    # convert via int to avoid "1.0" strings.  Non-numeric chroms (X, Y, MT)
    # are kept as-is.
    chrom_numeric = pd.to_numeric(df["chromosome"], errors="coerce")
    int_mask = chrom_numeric.notna()
    df["chromosome"] = df["chromosome"].astype(str)
    df.loc[int_mask, "chromosome"] = chrom_numeric[int_mask].astype(int).astype(str)

    df["position"] = df["position"].astype(int)
    df["effect_allele"] = df["effect_allele"].astype(str).str.strip().str.upper()
    df["other_allele"] = df["other_allele"].astype(str).str.strip().str.upper()

    if "effect_allele_frequency" in df.columns:
        df["effect_allele_frequency"] = pd.to_numeric(
            df["effect_allele_frequency"], errors="coerce"
        )
    else:
        df["effect_allele_frequency"] = np.nan

    # MAF filter ------------------------------------------------------------
    if maf_threshold > 0:
        eaf = df["effect_allele_frequency"]
        maf = np.minimum(eaf, 1.0 - eaf)
        has_freq = maf.notna()
        before = len(df)
        # Keep variants with MAF >= threshold OR missing frequency (don't
        # discard variants simply because frequency is unreported).
        df = df[~has_freq | (maf >= maf_threshold)]
        n_low_maf = before - len(df)
        if n_low_maf > 0:
            logger.info(
                "Excluded %d low-MAF variants (MAF < %g) from %s",
                n_low_maf, maf_threshold, study_id,
            )

    # Exclude indels: keep only single-base substitutions
    before = len(df)
    snp_mask = (df["effect_allele"].str.len() == 1) & (df["other_allele"].str.len() == 1)
    df = df[snp_mask]
    n_indels = before - len(df)
    if n_indels > 0:
        logger.info(
            "Excluded %d indels from %s (kept %d SNPs)",
            n_indels, study_id, len(df),
        )

    # Exclude multi-allelic positions: any (chromosome, position) reported
    # with more than one alt-allele contrast within this study.
    before = len(df)
    pos_counts = df.groupby(["chromosome", "position"]).size()
    multi_positions = pos_counts.index[pos_counts > 1]
    if len(multi_positions) > 0:
        multi_idx = pd.MultiIndex.from_arrays(
            [df["chromosome"], df["position"]]
        )
        df = df[~multi_idx.isin(multi_positions)]
        n_multi = before - len(df)
        logger.info(
            "Excluded %d variants at %d multi-allelic positions from %s",
            n_multi, len(multi_positions), study_id,
        )

    # Build canonical variant_id (vectorized) ---------------------------------
    allele_a = df["effect_allele"]
    allele_b = df["other_allele"]
    # Sort alleles alphabetically per row
    sorted_first = np.where(allele_a <= allele_b, allele_a, allele_b)
    sorted_second = np.where(allele_a <= allele_b, allele_b, allele_a)
    df["variant_id"] = (
        "chr" + df["chromosome"].astype(str)
        + ":" + df["position"].astype(str)
        + ":" + sorted_first + ":" + sorted_second
    )

    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_harmonized_file(
    path: Path,
    study_id: str,
    valid_hm_codes: list[int] | None = None,
    maf_threshold: float = 0.01,
    max_abs_beta: float = 10.0,
    max_se: float = 10.0,
) -> StudySummaryStats:
    """Read a GWAS Catalog harmonised summary-statistics file.

    Parameters
    ----------
    path:
        Path to a ``.h.tsv.gz`` (or plain ``.tsv``) file.
    study_id:
        Identifier to attach to the resulting :class:`StudySummaryStats`.
    valid_hm_codes:
        Harmonisation codes to keep.  Defaults to
        :data:`DEFAULT_VALID_HM_CODES` (``[10, 11, 12, 13]``).
    maf_threshold:
        Minimum minor allele frequency.  Variants with MAF below this
        threshold are excluded.  Set to ``0.0`` to disable.  Default ``0.01``.
    max_abs_beta:
        Maximum plausible absolute effect size.  Variants with
        ``|beta| > max_abs_beta`` are dropped as likely parsing errors.
        Set to ``0.0`` to disable.  Default ``10.0``.
    max_se:
        Maximum plausible standard error.  Variants with ``SE > max_se``
        are dropped.  Set to ``0.0`` to disable.  Default ``10.0``.

    Returns
    -------
    StudySummaryStats
    """
    if valid_hm_codes is None:
        valid_hm_codes = DEFAULT_VALID_HM_CODES

    logger.info("Loading harmonized file for study %s from %s", study_id, path)

    # Read raw file (only columns we need) -----------------------------------
    # Peek at header to auto-detect format (old hm_* vs new GWAS-SSF)
    header_df = pd.read_csv(path, sep="\t", comment="#", nrows=0)
    file_columns = set(header_df.columns)

    if _REQUIRED_COLUMNS_OLD.issubset(file_columns):
        _COLUMN_MAP = _COLUMN_MAP_OLD
        _REQUIRED_COLUMNS = _REQUIRED_COLUMNS_OLD
        logger.info("Detected old (hm_*) column format for %s", path)
    elif _REQUIRED_COLUMNS_NEW.issubset(file_columns):
        _COLUMN_MAP = _COLUMN_MAP_NEW
        _REQUIRED_COLUMNS = _REQUIRED_COLUMNS_NEW
        logger.info("Detected new GWAS-SSF column format for %s", path)
    else:
        raise ValueError(
            f"File {path} has unrecognized column format. "
            f"Expected old format columns {sorted(_REQUIRED_COLUMNS_OLD)} "
            f"or new format columns {sorted(_REQUIRED_COLUMNS_NEW)}, "
            f"but found: {sorted(file_columns)}"
        )

    usecols = [c for c in _COLUMN_MAP if c in file_columns]

    # Use pyarrow CSV reader — multi-threaded parsing, much faster than pandas
    try:
        table = pa_csv.read_csv(
            path,
            read_options=pa_csv.ReadOptions(autogenerate_column_names=False),
            parse_options=pa_csv.ParseOptions(delimiter="\t"),
            convert_options=pa_csv.ConvertOptions(
                include_columns=usecols,
                strings_can_be_null=True,
            ),
        )
        df = table.to_pandas()
        del table
    except Exception:
        # Fallback: pandas (handles edge cases like comment lines)
        logger.warning("pyarrow CSV failed for %s, falling back to pandas", path)
        df = pd.read_csv(path, sep="\t", comment="#", usecols=usecols)

    logger.debug("Raw row count: %d", len(df))

    # Keep only the columns we need (tolerate missing optional ones) ---------
    available = {c for c in _COLUMN_MAP if c in df.columns}
    missing_required = _REQUIRED_COLUMNS - available
    if missing_required:
        raise ValueError(
            f"File {path} is missing required columns: {sorted(missing_required)}"
        )

    # Need at least one of beta or odds_ratio for effect sizes
    # Check using raw column names (before renaming)
    beta_col = next((c for c in ("hm_beta", "beta") if c in available), None)
    or_col = next((c for c in ("hm_odds_ratio", "odds_ratio") if c in available), None)
    has_beta = beta_col is not None
    has_or = or_col is not None
    if not has_beta and not has_or:
        raise ValueError(
            f"File {path} has neither beta nor odds_ratio columns"
        )

    df = df[[c for c in _COLUMN_MAP if c in df.columns]].copy()
    df.rename(columns={k: v for k, v in _COLUMN_MAP.items() if k in df.columns},
              inplace=True)

    df = apply_variant_qc(
        df, study_id,
        valid_hm_codes=valid_hm_codes,
        maf_threshold=maf_threshold,
        max_abs_beta=max_abs_beta,
        max_se=max_se,
    )

    logger.info(
        "Study %s: loaded %d variants after filtering", study_id, len(df)
    )

    return StudySummaryStats(study_id=study_id, variants=df)


# ---------------------------------------------------------------------------
# Cross-study alignment
# ---------------------------------------------------------------------------

def align_studies(
    studies: list[StudySummaryStats],
    min_study_count: int = 2,
) -> pd.DataFrame:
    """Align alleles across multiple studies and collect per-variant betas/SEs.

    The first study in the list is treated as the *reference* for allele
    orientation.  Subsequent studies have their beta sign-flipped when their
    effect/other alleles are swapped relative to the reference.  The standard
    error is orientation-independent and left unchanged.  Effect-allele
    frequency is used only for the (orientation-invariant) MAF filter at load
    time and is not carried through alignment.

    Parameters
    ----------
    studies:
        Two or more :class:`StudySummaryStats` objects.
    min_study_count:
        Minimum number of studies a variant must appear in to be included.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with columns
        ``[variant_id, study_id, beta, se]``.
    """
    if len(studies) < 2:
        raise ValueError("align_studies requires at least 2 studies")

    # ---- 1. Find shared variants (O(n) set ops) ----------------------------
    vid_sets = [set(st.variants["variant_id"]) for st in studies]
    if len(vid_sets) == 2:
        shared_vids = vid_sets[0] & vid_sets[1]
    else:
        all_vids = vid_sets[0].union(*vid_sets[1:])
        shared_vids = {v for v in all_vids
                       if sum(v in s for s in vid_sets) >= min_study_count}

    n_total = len(vid_sets[0].union(*vid_sets[1:]))
    logger.info(
        "Variants present in >= %d studies: %d / %d",
        min_study_count, len(shared_vids), n_total,
    )
    del vid_sets

    if not shared_vids:
        logger.info("Aligned variants returned: 0")
        return pd.DataFrame(columns=["variant_id", "study_id", "beta", "se"])

    # ---- 2. Filter + dedup, extract numpy arrays ----------------------------
    _needed = ["variant_id", "effect_allele", "other_allele",
               "beta", "standard_error"]
    study_arrays = []  # (study_id, vids, ea, oa, betas, ses)
    for st in studies:
        df = st.variants[_needed]
        filt = df[df["variant_id"].isin(shared_vids)].drop_duplicates(
            subset="variant_id"
        )
        study_arrays.append((
            st.study_id,
            filt["variant_id"].values,
            filt["effect_allele"].values,
            filt["other_allele"].values,
            filt["beta"].values.astype(np.float64),
            filt["standard_error"].values.astype(np.float64),
        ))
    del shared_vids

    # ---- 3. Reference study -------------------------------------------------
    ref_id, ref_vids, ref_ea, ref_oa, ref_betas, ref_ses = study_arrays[0]

    # Build O(1) lookup: variant_id → ref row index
    ref_lookup = pd.Series(
        np.arange(len(ref_vids), dtype=np.intp), index=ref_vids
    )

    # ---- 4. Align each non-ref study (no pd.merge!) ------------------------
    # Collect aligned data as DataFrame parts (memory-efficient).
    aligned_parts: list[pd.DataFrame] = []

    # Reference data
    aligned_parts.append(pd.DataFrame({
        "variant_id": ref_vids,
        "study_id": ref_id,
        "beta": ref_betas,
        "se": ref_ses,
    }))

    for i in range(1, len(study_arrays)):
        s_id, s_vids, s_ea, s_oa, s_betas, s_ses = study_arrays[i]

        # Index-based alignment: look up ref row for each study variant
        matched = ref_lookup.reindex(s_vids)
        valid = matched.notna().values
        ref_idx = matched.values[valid].astype(np.intp)
        study_idx = np.where(valid)[0]

        if len(ref_idx) == 0:
            continue

        # Allele comparison (pure numpy, no merged DataFrame)
        cur_ea = s_ea[study_idx]
        cur_oa = s_oa[study_idx]
        r_ea = ref_ea[ref_idx]
        r_oa = ref_oa[ref_idx]

        match_same = (cur_ea == r_ea) & (cur_oa == r_oa)
        match_swap = (cur_ea == r_oa) & (cur_oa == r_ea)
        keep = match_same | match_swap

        n_mismatch = int((~keep).sum())
        if n_mismatch > 0:
            logger.warning(
                "%d allele mismatches in study %s -- skipping those variants",
                n_mismatch, s_id,
            )

        if not keep.any():
            continue

        # Apply keep mask and flip swapped betas
        kept_study_idx = study_idx[keep]
        kept_betas = s_betas[kept_study_idx].copy()
        swap_in_kept = match_swap[keep]
        kept_betas[swap_in_kept] = -kept_betas[swap_in_kept]

        aligned_parts.append(pd.DataFrame({
            "variant_id": s_vids[kept_study_idx],
            "study_id": s_id,
            "beta": kept_betas,
            "se": s_ses[kept_study_idx],
        }))

    if not aligned_parts:
        logger.info("Aligned variants returned: 0")
        return pd.DataFrame(columns=["variant_id", "study_id", "beta", "se"])

    # ---- 5. Combine and filter by min_study_count ---------------------------
    combined = pd.concat(aligned_parts, ignore_index=True)
    del aligned_parts

    counts = combined.groupby("variant_id", sort=False)["study_id"].transform("size")
    combined = combined[counts >= min_study_count]

    logger.info("Aligned variants returned: %d", combined["variant_id"].nunique())
    return combined


# ---------------------------------------------------------------------------
# Chromosome-chunked alignment (low peak RAM)
# ---------------------------------------------------------------------------

def _get_chromosomes(studies: list[StudySummaryStats]) -> list[str]:
    """Return sorted list of chromosomes present across all studies."""
    all_chroms: set[str] = set()
    for st in studies:
        all_chroms.update(st.variants["chromosome"].unique())

    # Defensive: drop NaN / empty labels. Upstream dropna should already
    # remove these, but guard here so we never create a chr"nan"/ chunk
    # directory or sort a NaN key.
    all_chroms = {c for c in all_chroms
                  if c is not None and str(c).lower() not in ("", "nan")}

    # Sort: 1-22 numerically, then X, Y, MT
    def _chrom_key(c: str) -> tuple[int, str]:
        try:
            return (0, int(c))  # type: ignore[return-value]
        except ValueError:
            return (1, c)  # type: ignore[return-value]

    return sorted(all_chroms, key=_chrom_key)


def _filter_studies_by_chrom(
    studies: list[StudySummaryStats],
    chrom: str,
) -> list[StudySummaryStats]:
    """Return studies filtered to a single chromosome."""
    filtered = []
    for st in studies:
        chrom_df = st.variants[st.variants["chromosome"] == chrom]
        if not chrom_df.empty:
            filtered.append(StudySummaryStats(
                study_id=st.study_id,
                variants=chrom_df,
            ))
    return filtered


def align_studies_chunked(
    studies: list[StudySummaryStats],
    min_study_count: int = 2,
    progress_callback: "callable | None" = None,
) -> pd.DataFrame:
    """Align studies one chromosome at a time to reduce peak memory.

    Produces the same output as :func:`align_studies` but processes each
    chromosome independently and concatenates the results.  Peak RAM is
    roughly proportional to the largest single chromosome rather than the
    whole genome.

    Parameters
    ----------
    studies:
        Two or more :class:`StudySummaryStats` objects.
    min_study_count:
        Minimum number of studies a variant must appear in.
    progress_callback:
        Optional ``(chrom: str, i: int, total: int) -> None`` called after
        each chromosome is processed (useful for Streamlit progress bars).

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame ``[variant_id, study_id, beta, se]``,
        identical to what :func:`align_studies` would return.
    """
    if len(studies) < 2:
        raise ValueError("align_studies_chunked requires at least 2 studies")

    chroms = _get_chromosomes(studies)
    logger.info(
        "Chunked alignment: processing %d chromosomes sequentially", len(chroms),
    )

    parts: list[pd.DataFrame] = []
    for i, chrom in enumerate(chroms):
        chrom_studies = _filter_studies_by_chrom(studies, chrom)
        if len(chrom_studies) < 2:
            logger.debug("Skipping chr%s — fewer than 2 studies", chrom)
            if progress_callback is not None:
                progress_callback(chrom, i + 1, len(chroms))
            continue

        aligned = align_studies(chrom_studies, min_study_count=min_study_count)
        if not aligned.empty:
            parts.append(aligned)
            logger.info(
                "chr%s: %d variants aligned",
                chrom, aligned["variant_id"].nunique(),
            )

        # Free chromosome-level data immediately
        del chrom_studies, aligned

        if progress_callback is not None:
            progress_callback(chrom, i + 1, len(chroms))

    if not parts:
        return pd.DataFrame(columns=["variant_id", "study_id", "beta", "se"])

    result = pd.concat(parts, ignore_index=True)
    del parts
    logger.info(
        "Chunked alignment complete: %d total variants",
        result["variant_id"].nunique(),
    )
    return result


def iter_align_by_chromosome(
    studies: list[StudySummaryStats],
    min_study_count: int = 2,
) -> "Generator[tuple[str, int, int, pd.DataFrame], None, None]":
    """Yield aligned data one chromosome at a time.

    Unlike :func:`align_studies_chunked`, this generator yields each
    chromosome's aligned DataFrame individually so the caller can run
    meta-analysis and free memory before proceeding to the next chromosome.

    Yields
    ------
    (chrom, index, total, aligned_df) : tuple
        *chrom* is the chromosome label, *index* is the 1-based position
        in the sequence, *total* is the number of chromosomes, and
        *aligned_df* is the long-format DataFrame for that chromosome.
        Only chromosomes with non-empty aligned results are yielded.
    """
    if len(studies) < 2:
        raise ValueError("iter_align_by_chromosome requires at least 2 studies")

    chroms = _get_chromosomes(studies)
    logger.info(
        "Iterative alignment: %d chromosomes to process", len(chroms),
    )

    for i, chrom in enumerate(chroms):
        chrom_studies = _filter_studies_by_chrom(studies, chrom)
        if len(chrom_studies) < 2:
            logger.debug("Skipping chr%s — fewer than 2 studies", chrom)
            continue

        aligned = align_studies(chrom_studies, min_study_count=min_study_count)
        del chrom_studies

        if not aligned.empty:
            logger.info(
                "chr%s: %d variants aligned",
                chrom, aligned["variant_id"].nunique(),
            )
            yield chrom, i + 1, len(chroms), aligned

        del aligned


# ---------------------------------------------------------------------------
# Two-pass disk-based chunking (very low peak RAM for many studies)
# ---------------------------------------------------------------------------

# Columns needed for alignment — skip the rest to save disk/memory
# rsid is carried through so it can be reported alongside meta-analysis hits.
_CHUNK_COLS = [
    "variant_id", "rsid", "chromosome", "effect_allele", "other_allele",
    "beta", "standard_error",
]


def _chunk_single_study(
    study_id: str,
    path: Path,
    chunks_dir: Path,
    valid_hm_codes: list[int] | None,
    maf_threshold: float = 0.01,
    max_abs_beta: float = 10.0,
    max_se: float = 10.0,
) -> "tuple[set[str], object]":
    """Load one study, split by chromosome, write parquet chunks to disk.

    This is an optimised fast-path that avoids the full
    :func:`load_harmonized_file` pipeline.  It reads with pyarrow, filters
    and transforms in pyarrow compute (no pandas intermediate for bulk
    operations), and writes partitioned parquet directly.

    Designed to run in a worker process.  Returns the set of chromosome
    labels found in this study.
    """
    if valid_hm_codes is None:
        valid_hm_codes = DEFAULT_VALID_HM_CODES

    # --- 1. Detect column format from header ---
    header_df = pd.read_csv(path, sep="\t", comment="#", nrows=0)
    file_columns = set(header_df.columns)
    del header_df

    if _REQUIRED_COLUMNS_OLD.issubset(file_columns):
        col_map = _COLUMN_MAP_OLD
    elif _REQUIRED_COLUMNS_NEW.issubset(file_columns):
        col_map = _COLUMN_MAP_NEW
    else:
        raise ValueError(
            f"File {path} has unrecognized column format. "
            f"Found: {sorted(file_columns)}"
        )

    # Only read columns needed for chunking (+ hm_code for filtering,
    # + odds_ratio/p_value as fallbacks for beta/SE derivation)
    internal_needed = {
        "chromosome", "position", "effect_allele", "other_allele",
        "beta", "standard_error", "hm_code", "odds_ratio", "p_value",
        "effect_allele_frequency", "rsid",
    }
    raw_cols = [c for c, v in col_map.items() if v in internal_needed and c in file_columns]
    rename = {c: col_map[c] for c in raw_cols}

    # --- 2. Read CSV with pyarrow ---
    try:
        table = pa_csv.read_csv(
            path,
            read_options=pa_csv.ReadOptions(autogenerate_column_names=False),
            parse_options=pa_csv.ParseOptions(delimiter="\t"),
            convert_options=pa_csv.ConvertOptions(
                include_columns=raw_cols,
                strings_can_be_null=True,
            ),
        )
    except Exception:
        logging.getLogger(__name__).warning(
            "pyarrow CSV failed for %s, falling back to pandas", path,
        )
        df_raw = pd.read_csv(path, sep="\t", comment="#", usecols=raw_cols)
        table = pa.Table.from_pandas(df_raw, preserve_index=False)
        del df_raw

    # Rename columns to internal names
    for old, new in rename.items():
        idx = table.schema.get_field_index(old)
        if idx >= 0:
            table = table.rename_columns(
                [rename.get(c, c) for c in table.column_names]
            )
            break  # rename_columns does all at once

    # --- 3. Convert to pandas and apply the shared per-study QC ---
    # Both loaders funnel through apply_variant_qc, so the QC (hm_code
    # filter, beta/SE derivation, p-vs-Z check, plausibility bounds, MAF,
    # indel and multi-allelic drops, variant_id) has a single
    # implementation and cannot drift between the whole-file and
    # disk-backed paths.
    df = table.to_pandas(self_destruct=True)
    del table
    df = apply_variant_qc(
        df, study_id,
        valid_hm_codes=valid_hm_codes,
        maf_threshold=maf_threshold,
        max_abs_beta=max_abs_beta,
        max_se=max_se,
    )

    # --- 5b. Sentinel-SNP genome-build check ---
    # Probe for well-known variants whose GRCh37 vs GRCh38 positions
    # differ; log a warning if the study looks like GRCh37 instead of
    # the GRCh38 that Catalog harmonisation is supposed to produce.
    from .build_check import check_genome_build as _check_build
    build_verdict = _check_build(df)
    if build_verdict.verdict == "grch37":
        logging.getLogger(__name__).warning(
            "Study %s appears to be GRCh37, not GRCh38 (sentinel matches: "
            "GRCh37=%s, GRCh38=%s). Meta-analysis with GRCh38 studies will "
            "likely yield 0 overlapping variants.",
            study_id, build_verdict.grch37_hits, build_verdict.grch38_hits,
        )
    elif build_verdict.verdict == "mixed":
        logging.getLogger(__name__).warning(
            "Study %s has mixed sentinel positions (GRCh37=%s, GRCh38=%s) — "
            "data may be inconsistent or misannotated.",
            study_id, build_verdict.grch37_hits, build_verdict.grch38_hits,
        )
    elif build_verdict.verdict == "unknown":
        logging.getLogger(__name__).info(
            "Study %s: no sentinel SNPs found; genome build could not be "
            "verified.", study_id,
        )
    else:
        logging.getLogger(__name__).debug(
            "Study %s: %d sentinel SNPs match GRCh38 as expected.",
            study_id, len(build_verdict.grch38_hits),
        )

    # --- 6. Write per-chromosome parquet chunks ---
    chunk_cols = _CHUNK_COLS
    chroms_found: set[str] = set()
    for chrom, chrom_df in df[chunk_cols].groupby("chromosome"):
        chrom_dir = chunks_dir / f"chr{chrom}"
        chrom_dir.mkdir(parents=True, exist_ok=True)
        chrom_df.to_parquet(chrom_dir / f"{study_id}.parquet", index=False)
        chroms_found.add(str(chrom))

    del df
    return chroms_found, build_verdict


def _get_available_ram() -> int | None:
    """Return available RAM in bytes, or None if it cannot be determined."""
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        pass

    import platform
    import subprocess

    system = platform.system()
    try:
        if system == "Darwin":
            # macOS: parse vm_stat (page size × (free + inactive + speculative))
            out = subprocess.check_output(["vm_stat"], text=True)
            page_size = 4096
            free = inactive = spec = 0
            for line in out.splitlines():
                if "Pages free" in line:
                    free = int(line.split(":")[1].strip().rstrip("."))
                elif "Pages inactive" in line:
                    inactive = int(line.split(":")[1].strip().rstrip("."))
                elif "Pages speculative" in line:
                    spec = int(line.split(":")[1].strip().rstrip("."))
            return (free + inactive + spec) * page_size
        elif system == "Linux":
            # Linux: read MemAvailable from /proc/meminfo
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024  # kB → bytes
    except Exception:
        pass

    return None


def _estimate_max_workers(
    file_paths: "dict[str, Path]",
    cpu_cap: int = 6,
    mem_fraction: float = 0.6,
    ram_multiplier: float = 5.0,
    floor: int = 1,
) -> int:
    """Choose worker count based on available RAM and study file sizes.

    Parameters
    ----------
    file_paths:
        Mapping of study_id → path (used to estimate per-worker memory).
    cpu_cap:
        Hard upper bound on workers (never exceed this many cores).
    mem_fraction:
        Fraction of *available* (free) RAM we allow ourselves to use.
        Default 0.6 leaves headroom for the OS and the main process.
    ram_multiplier:
        Estimated ratio of in-memory size to on-disk file size.
        TSV → pyarrow → pandas typically inflates ~4-5×; 5 is conservative.
    floor:
        Minimum number of workers (always run at least this many).
    """
    import os

    cpu_limit = min(os.cpu_count() or 4, cpu_cap)

    # If we can't stat the files, fall back to CPU-only heuristic
    file_sizes = []
    for p in file_paths.values():
        try:
            file_sizes.append(p.stat().st_size)
        except OSError:
            pass

    if not file_sizes:
        return max(floor, cpu_limit)

    avg_file_bytes = sum(file_sizes) / len(file_sizes)
    est_per_worker = avg_file_bytes * ram_multiplier

    available_ram = _get_available_ram()
    if available_ram is None:
        return max(floor, cpu_limit)
    budget = available_ram * mem_fraction

    mem_limit = max(floor, int(budget // est_per_worker)) if est_per_worker > 0 else cpu_limit

    workers = min(cpu_limit, mem_limit)
    logging.getLogger(__name__).info(
        "Memory guard: %.1f GB available, ~%.0f MB/worker estimate → %d workers "
        "(cpu_cap=%d, mem_limit=%d)",
        available_ram / 1e9,
        est_per_worker / 1e6,
        workers,
        cpu_limit,
        mem_limit,
    )
    return workers


def chunk_studies_to_disk(
    file_paths: "dict[str, Path]",
    chunks_dir: Path,
    valid_hm_codes: list[int] | None = None,
    progress_callback: "callable | None" = None,
    max_workers: int | None = None,
    maf_threshold: float = 0.01,
    max_abs_beta: float = 10.0,
    max_se: float = 10.0,
) -> "tuple[list[str], dict[str, object]]":
    """Load studies in parallel, split by chromosome, save to disk.

    Uses a process pool to decompress and parse multiple studies
    simultaneously.  Each chromosome's data is saved as a Parquet file
    under ``{chunks_dir}/chr{X}/{study_id}.parquet``.

    Parameters
    ----------
    file_paths:
        Mapping of study_id → path to harmonized summary-statistics file.
    chunks_dir:
        Directory to write per-chromosome Parquet chunks.
    valid_hm_codes:
        Harmonisation codes to keep (passed to :func:`load_harmonized_file`).
    progress_callback:
        Optional ``(study_id: str, i: int, total: int) -> None``.
    max_workers:
        Max parallel workers.  Defaults to ``min(cpu_count, 6)``.

    Returns
    -------
    tuple[list[str], dict[str, BuildVerdict]]
        Sorted chromosome labels found across all studies, and a
        per-study genome-build verdict from the sentinel-SNP probe.
    """
    import shutil

    # Start fresh
    if chunks_dir.is_dir():
        shutil.rmtree(chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Pre-create chromosome directories to avoid race conditions
    for i in range(1, 23):
        (chunks_dir / f"chr{i}").mkdir(exist_ok=True)
    for c in ("X", "Y", "MT"):
        (chunks_dir / f"chr{c}").mkdir(exist_ok=True)

    all_chroms: set[str] = set()
    total = len(file_paths)

    if max_workers is None:
        max_workers = _estimate_max_workers(file_paths)

    logger.info(
        "Chunking %d studies in parallel (max_workers=%d)", total, max_workers,
    )

    completed = 0
    failed: list[str] = []
    build_verdicts: dict[str, object] = {}

    def _record(sid: str, chroms_found, verdict) -> None:
        all_chroms.update(chroms_found)
        build_verdicts[sid] = verdict
        logger.info("Chunked study %s (%d/%d)", sid, completed, total)

    if max_workers <= 1:
        # Sequential, in-process chunking. When the resource guard already
        # allows only one worker (constrained hosts, e.g. a ~1 GB PaaS
        # container), a process pool adds no parallelism but doubles peak
        # memory — forking copies the parent, and spawn/forkserver re-imports
        # the entry module — which is enough to trigger an OOM kill. Running
        # each study directly keeps peak RAM to a single study and produces
        # byte-identical chunks.
        for sid, path in file_paths.items():
            completed += 1
            try:
                chroms_found, verdict = _chunk_single_study(
                    sid, path, chunks_dir, valid_hm_codes,
                    maf_threshold, max_abs_beta, max_se,
                )
                _record(sid, chroms_found, verdict)
            except Exception:
                logger.exception("Failed to chunk study %s — skipping", sid)
                failed.append(sid)
            if progress_callback is not None:
                progress_callback(sid, completed, total)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_sid = {
                executor.submit(
                    _chunk_single_study, sid, path, chunks_dir, valid_hm_codes,
                    maf_threshold, max_abs_beta, max_se,
                ): sid
                for sid, path in file_paths.items()
            }

            for future in as_completed(future_to_sid):
                sid = future_to_sid[future]
                completed += 1
                try:
                    chroms_found, verdict = future.result()
                    _record(sid, chroms_found, verdict)
                except Exception:
                    logger.exception("Failed to chunk study %s — skipping", sid)
                    failed.append(sid)

                if progress_callback is not None:
                    progress_callback(sid, completed, total)

    if failed:
        logger.warning("Skipped %d studies due to errors: %s", len(failed), ", ".join(failed))
    if not all_chroms:
        raise RuntimeError(
            f"All {total} studies failed during chunking. "
            "Check that downloaded files are not corrupted."
        )

    chroms = _get_chromosomes_from_set(all_chroms)
    logger.info("Chunking complete: %d/%d studies succeeded, %d chromosomes",
                total - len(failed), total, len(chroms))
    return chroms, build_verdicts


class _LightStudy:
    """Minimal stand-in for StudySummaryStats used by align_studies.

    Avoids the full column validation of StudySummaryStats — chunks only
    contain the columns needed for alignment.
    """

    __slots__ = ("study_id", "variants")

    def __init__(self, study_id: str, variants: pd.DataFrame) -> None:
        self.study_id = study_id
        self.variants = variants


def load_chromosome_chunks(
    chunks_dir: Path,
    chrom: str,
) -> list:
    """Load all study chunks for a single chromosome from disk.

    Returns a list of lightweight study objects (compatible with
    :func:`align_studies`) for the requested chromosome.
    """
    chrom_dir = chunks_dir / f"chr{chrom}"
    if not chrom_dir.is_dir():
        return []

    studies = []
    for pq_file in sorted(chrom_dir.glob("*.parquet")):
        study_id = pq_file.stem
        df = pd.read_parquet(pq_file)
        studies.append(_LightStudy(study_id=study_id, variants=df))

    return studies


def _get_chromosomes_from_set(chroms: set[str]) -> list[str]:
    """Sort chromosome labels: 1-22 numerically, then X, Y, MT."""
    # Defensive: drop NaN / empty labels (see _get_chromosomes).
    chroms = {c for c in chroms
              if c is not None and str(c).lower() not in ("", "nan")}

    def _chrom_key(c: str) -> tuple[int, str]:
        try:
            return (0, int(c))  # type: ignore[return-value]
        except ValueError:
            return (1, c)  # type: ignore[return-value]

    return sorted(chroms, key=_chrom_key)
