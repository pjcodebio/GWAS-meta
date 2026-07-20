"""Meta-analysis engine combining fixed-effects, random-effects, and heterogeneity."""

from gwas_meta.meta_analysis.fixed_effects import ivw_fixed_effects
from gwas_meta.meta_analysis.heterogeneity import cochrans_q, i_squared, tau_squared_dl
from gwas_meta.meta_analysis.leave_one_out import leave_one_out_max_p
from gwas_meta.meta_analysis.models import MetaAnalysisInput, MetaAnalysisResult
from gwas_meta.meta_analysis.random_effects import dl_random_effects


def run_meta_analysis(inp: MetaAnalysisInput) -> MetaAnalysisResult:
    """Run a complete meta-analysis for a single variant.

    Combines IVW fixed-effects, DerSimonian-Laird random-effects,
    and heterogeneity statistics (Cochran's Q, I-squared, tau-squared).

    Parameters
    ----------
    inp : MetaAnalysisInput
        Input data containing betas, standard errors, and study identifiers.

    Returns
    -------
    MetaAnalysisResult
        Full result object with fixed-effects, random-effects, and
        heterogeneity measures.
    """
    fixed = ivw_fixed_effects(inp)
    random = dl_random_effects(inp)

    k = len(inp.betas)
    q = cochrans_q(inp.betas, inp.standard_errors)
    i2 = i_squared(q, k)

    return MetaAnalysisResult(
        variant_id=inp.variant_id,
        beta_fixed=fixed["beta"],
        se_fixed=fixed["se"],
        z_fixed=fixed["z"],
        p_fixed=fixed["p"],
        beta_random=random["beta"],
        se_random=random["se"],
        z_random=random["z"],
        p_random=random["p"],
        q_stat=q,
        i_squared=i2,
        tau_squared=random["tau_squared"],
        n_studies=k,
        study_ids=list(inp.study_ids),
    )


def run_meta_analysis_batch(
    aligned: "pd.DataFrame",
    as_dataframe: bool = False,
) -> "list[MetaAnalysisResult] | pd.DataFrame":
    """Run meta-analysis on all variants at once using vectorized numpy ops.

    This is much faster than calling :func:`run_meta_analysis` in a loop
    because it processes all variants in a single pass with array operations.

    Parameters
    ----------
    aligned :
        Output from :func:`~gwas_meta.data.harmonize.align_studies` --
        a long-format DataFrame with columns
        ``[variant_id, study_id, beta, se]``.
    as_dataframe :
        If True, return a pandas DataFrame instead of a list of
        MetaAnalysisResult objects. Much faster and more memory-efficient
        for genome-scale runs (avoids creating millions of Python objects).
        Columns: variant_id, beta_fixed, se_fixed, z_fixed, p_fixed,
        beta_random, se_random, z_random, p_random, q_stat, i_squared,
        tau_squared, n_studies. Sorted by p_fixed ascending.

    Returns
    -------
    list[MetaAnalysisResult] or pd.DataFrame
        Results sorted by fixed-effects p-value (ascending).
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import norm

    if aligned.empty:
        return pd.DataFrame() if as_dataframe else []

    # Sort by (variant_id, study_id) so consecutive k rows = one variant
    adf = aligned.sort_values(["variant_id", "study_id"]).reset_index(drop=True)
    n_rows = len(adf)
    n_unique = adf["variant_id"].nunique()

    # Check if all variants have the same k (common case: 2 studies)
    k_uniform = n_rows // n_unique
    uniform = (n_rows == n_unique * k_uniform)

    if uniform:
        k_groups = [(
            adf["variant_id"].values[::k_uniform],
            adf["beta"].values.reshape(n_unique, k_uniform),
            adf["se"].values.reshape(n_unique, k_uniform),
            k_uniform,
        )]
    else:
        # Mixed k values: group by study count
        counts = adf.groupby("variant_id", sort=False).size()
        k_groups = []
        for k_val in np.unique(counts.values):
            k = int(k_val)
            k_vids = set(counts[counts == k].index)
            mask = adf["variant_id"].isin(k_vids)
            sub = adf.loc[mask].sort_values(["variant_id", "study_id"]).reset_index(drop=True)
            n = len(sub) // k
            k_groups.append((
                sub["variant_id"].values[::k],
                sub["beta"].values.reshape(n, k),
                sub["se"].values.reshape(n, k),
                k,
            ))

    df_parts: list = []
    for variant_ids, betas, ses, k in k_groups:
        df_parts.append(_vectorized_meta(variant_ids, betas, ses, k, norm))

    result_df = pd.concat(df_parts, ignore_index=True)
    result_df = result_df.sort_values("p_fixed").reset_index(drop=True)

    if as_dataframe:
        return result_df

    # Convert to MetaAnalysisResult objects (used by Streamlit UI)
    results = []
    for row in result_df.itertuples(index=False):
        results.append(MetaAnalysisResult(
            variant_id=row.variant_id,
            beta_fixed=row.beta_fixed, se_fixed=row.se_fixed,
            z_fixed=row.z_fixed, p_fixed=row.p_fixed,
            beta_random=row.beta_random, se_random=row.se_random,
            z_random=row.z_random, p_random=row.p_random,
            q_stat=row.q_stat, i_squared=row.i_squared,
            tau_squared=row.tau_squared, n_studies=int(row.n_studies),
            study_ids=[],
        ))
    return results


def _vectorized_meta(variant_ids, betas, ses, k, norm):
    """Shared vectorized IVW + DL computation. Returns a DataFrame."""
    import numpy as np
    import pandas as pd

    n = betas.shape[0]

    # --- Fixed effects (IVW) ---
    weights = 1.0 / (ses ** 2)
    sum_w = weights.sum(axis=1)
    beta_fixed = (weights * betas).sum(axis=1) / sum_w
    se_fixed = 1.0 / np.sqrt(sum_w)
    z_fixed = np.where(se_fixed > 0, beta_fixed / se_fixed, 0.0)
    p_fixed = 2.0 * norm.sf(np.abs(z_fixed))

    # --- Heterogeneity ---
    q_stat = (weights * (betas - beta_fixed[:, np.newaxis]) ** 2).sum(axis=1)
    if k > 1:
        i2 = np.maximum(0.0, (q_stat - (k - 1)) / q_stat * 100.0)
        i2 = np.where(q_stat == 0.0, 0.0, i2)
        c = sum_w - (weights ** 2).sum(axis=1) / sum_w
        tau2 = np.maximum(0.0, np.where(c == 0.0, 0.0, (q_stat - (k - 1)) / c))
    else:
        i2 = np.zeros(n)
        tau2 = np.zeros(n)

    # --- Random effects (DL) ---
    w_star = 1.0 / (ses ** 2 + tau2[:, np.newaxis])
    sw_star = w_star.sum(axis=1)
    beta_random = (w_star * betas).sum(axis=1) / sw_star
    se_random = 1.0 / np.sqrt(sw_star)
    z_random = np.where(se_random > 0, beta_random / se_random, 0.0)
    p_random = 2.0 * norm.sf(np.abs(z_random))

    return pd.DataFrame({
        "variant_id":  variant_ids,
        "beta_fixed":  beta_fixed,  "se_fixed":  se_fixed,
        "z_fixed":     z_fixed,     "p_fixed":   p_fixed,
        "beta_random": beta_random, "se_random": se_random,
        "z_random":    z_random,    "p_random":  p_random,
        "q_stat":      q_stat,      "i_squared": i2,
        "tau_squared":  tau2,        "n_studies":  k,
    })


__all__ = [
    "MetaAnalysisInput",
    "MetaAnalysisResult",
    "cochrans_q",
    "dl_random_effects",
    "i_squared",
    "ivw_fixed_effects",
    "leave_one_out_max_p",
    "run_meta_analysis",
    "run_meta_analysis_batch",
    "tau_squared_dl",
]
