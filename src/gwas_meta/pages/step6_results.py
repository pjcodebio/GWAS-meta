"""Step 6: Display results table, LLM summary, and export."""

import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from gwas_meta.llm import create_provider
from gwas_meta.llm.prompts import SUMMARY_SYSTEM_PROMPT, SUMMARY_USER_TEMPLATE
from gwas_meta.meta_analysis import MetaAnalysisResult

_RESULTS_CSV = Path("results/meta_results.csv")


def _load_results_csv() -> list[MetaAnalysisResult] | None:
    """Load previously saved results from disk."""
    if not _RESULTS_CSV.is_file():
        return None
    results = []
    with open(_RESULTS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        per_study_cols = [
            c for c in (reader.fieldnames or []) if c.startswith("beta_")
            and c not in {"beta_fixed", "beta_random"}
        ]
        for row in reader:
            per_study_betas: dict[str, float] = {}
            for col in per_study_cols:
                raw = row.get(col, "")
                if raw not in ("", None):
                    try:
                        per_study_betas[col.removeprefix("beta_")] = float(raw)
                    except ValueError:
                        pass
            def _opt_int(k: str) -> int:
                v = row.get(k, "")
                try:
                    return int(v) if v not in ("", None) else 0
                except (TypeError, ValueError):
                    return 0

            def _opt_float(k: str) -> float:
                v = row.get(k, "")
                try:
                    return float(v) if v not in ("", None) else float("nan")
                except (TypeError, ValueError):
                    return float("nan")

            results.append(MetaAnalysisResult(
                variant_id=row["variant_id"],
                beta_fixed=float(row["beta_fixed"]),
                se_fixed=float(row["se_fixed"]),
                z_fixed=float(row["z_fixed"]),
                p_fixed=float(row["p_fixed"]),
                beta_random=float(row["beta_random"]),
                se_random=float(row["se_random"]),
                z_random=float(row["z_random"]),
                p_random=float(row["p_random"]),
                q_stat=float(row["q_stat"]),
                i_squared=float(row["i_squared"]),
                tau_squared=float(row["tau_squared"]),
                n_studies=int(row["n_studies"]),
                study_ids=row["study_ids"].split(";"),
                per_study_betas=per_study_betas,
                rsid=row.get("rsid", "") or "",
                n_pos=_opt_int("n_pos"),
                n_neg=_opt_int("n_neg"),
                n_zero=_opt_int("n_zero"),
                loo_max_p=_opt_float("loo_max_p"),
                loo_worst_dropped=row.get("loo_worst_dropped", "") or "",
            ))
    return results if results else None


def _collect_study_ids(results: list[MetaAnalysisResult]) -> list[str]:
    """Return a sorted list of study_ids seen across per_study_betas dicts.

    Prefers the canonical session-state list captured at meta-analysis time,
    falling back to whatever appears in the result rows (e.g. after CSV reload).
    """
    sids = st.session_state.get("included_study_ids")
    if sids:
        return list(sids)
    seen: set[str] = set()
    for r in results:
        if r.per_study_betas:
            seen.update(r.per_study_betas.keys())
    return sorted(seen)


def _build_results_df(
    results: list[MetaAnalysisResult],
    sig_threshold: float,
    study_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Build the per-variant results DataFrame.

    Per-study beta columns are appended at the far right, one per study,
    so the top-N table and the AI summary both reflect input-level effects.
    """
    if study_ids is None:
        study_ids = _collect_study_ids(results)
    import math as _math
    rows = []
    for r in results:
        row = {
            "Variant": r.variant_id,
            "rsID": r.rsid or "",
            "Beta (FE)": r.beta_fixed,
            "SE (FE)": r.se_fixed,
            "P (FE)": r.p_fixed,
            "Beta (RE)": r.beta_random,
            "SE (RE)": r.se_random,
            "P (RE)": r.p_random,
            "Q": r.q_stat,
            "I²": r.i_squared,
            "τ²": r.tau_squared,
            "N studies": r.n_studies,
            # Direction of effect: "+/-/0" summary
            "Dir (+/-/0)": f"{r.n_pos}/{r.n_neg}/{r.n_zero}",
            # LOO worst-case p: blank for non-hits or k < 3
            "LOO max P": (r.loo_max_p if not _math.isnan(r.loo_max_p)
                          else np.nan),
            "LOO dropped": r.loo_worst_dropped or "",
            "Significant": r.p_fixed < sig_threshold,
        }
        per_study = r.per_study_betas or {}
        for sid in study_ids:
            row[f"Beta [{sid}]"] = per_study.get(sid, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_top_hits(results: list[MetaAnalysisResult]) -> list[dict]:
    """Serialize the top-20 results exactly as sent to the LLM.

    Single source of truth for the summary payload: both the transparency
    display and :func:`_generate_summary` call this, so what the user is shown
    before the call is byte-for-byte what the provider sends.
    """
    return [
        {
            "variant_id": r.variant_id,
            "beta_fixed": round(r.beta_fixed, 6),
            "p_fixed": r.p_fixed,
            "i_squared": round(r.i_squared, 1),
            "n_studies": r.n_studies,
        }
        for r in results[:20]
    ]


def _summary_user_message(
    results: list[MetaAnalysisResult], sig_threshold: float
) -> str:
    """Return the exact user message assembled for the summarization call."""
    n_sig = sum(1 for r in results if r.p_fixed < sig_threshold)
    return SUMMARY_USER_TEMPLATE.format(
        research_question=st.session_state.research_question,
        n_variants=len(results),
        n_significant=n_sig,
        top_hits_json=json.dumps(_summary_top_hits(results), indent=2),
    )


def _generate_summary(results: list[MetaAnalysisResult], sig_threshold: float) -> str:
    config = st.session_state.config
    llm_cfg = config.get("llm", {})
    provider_name = llm_cfg.get("provider", "anthropic")
    provider_cfg = llm_cfg.get(provider_name, {})
    provider = create_provider(provider_name, **provider_cfg)

    n_sig = sum(1 for r in results if r.p_fixed < sig_threshold)
    return provider.summarize_results(
        question=st.session_state.research_question,
        n_variants=len(results),
        n_significant=n_sig,
        top_hits=_summary_top_hits(results),
    )


def _build_rank_fig(
    df: pd.DataFrame, sig_threshold: float
) -> "plotly.graph_objects.Figure":
    """Build significance rank plot (variants sorted by p-value)."""
    plot_df = df.head(min(5000, len(df))).copy()
    plot_df["-log10(P)"] = -np.log10(plot_df["P (FE)"].clip(lower=1e-300))
    plot_df["Chromosome"] = plot_df["Variant"].str.extract(r"chr(\w+):")[0]
    plot_df["Rank"] = range(len(plot_df))

    fig = px.scatter(
        plot_df,
        x="Rank",
        y="-log10(P)",
        color="Chromosome",
        hover_data=["Variant", "P (FE)", "Beta (FE)"],
        title="Significance Rank Plot",
    )
    fig.add_hline(
        y=-np.log10(sig_threshold),
        line_dash="dash",
        line_color="red",
        annotation_text="Genome-wide significance",
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(xaxis_title="Variant rank (by p-value)")
    return fig


def _build_manhattan_fig(
    results: list["MetaAnalysisResult"], sig_threshold: float
) -> "plotly.graph_objects.Figure":
    """Build a true Manhattan plot (genomic position on x-axis).

    Subsamples non-significant variants to keep Plotly responsive.
    All significant and suggestive variants are always included.
    """
    import plotly.graph_objects as go

    # Parse variant IDs into chrom, pos and collect p-values
    chroms = []
    positions = []
    pvals = []
    variant_ids = []
    betas = []

    for r in results:
        parts = r.variant_id.split(":")
        if len(parts) < 2:
            continue
        chrom_raw = parts[0].replace("chr", "")
        try:
            pos = int(parts[1])
        except ValueError:
            continue
        chroms.append(chrom_raw)
        positions.append(pos)
        pvals.append(r.p_fixed)
        variant_ids.append(r.variant_id)
        betas.append(r.beta_fixed)

    if not chroms:
        # Fallback: empty figure
        return go.Figure()

    df = pd.DataFrame({
        "chrom_raw": chroms,
        "pos": positions,
        "p": pvals,
        "variant": variant_ids,
        "beta": betas,
    })
    df["-log10(P)"] = -np.log10(df["p"].clip(lower=1e-300))

    # Map chromosomes to integers for sorting
    chrom_order = [str(c) for c in range(1, 23)] + ["X", "Y"]
    chrom_to_int = {c: i for i, c in enumerate(chrom_order)}
    df["chrom_int"] = df["chrom_raw"].map(chrom_to_int)
    df = df.dropna(subset=["chrom_int"])
    df["chrom_int"] = df["chrom_int"].astype(int)
    df = df.sort_values(["chrom_int", "pos"])

    # Subsample non-significant variants for performance
    sig_mask = df["p"] < 1e-5
    df_sig = df[sig_mask]
    df_nonsig = df[~sig_mask]
    max_nonsig = 20_000
    if len(df_nonsig) > max_nonsig:
        df_nonsig = df_nonsig.sample(n=max_nonsig, random_state=42)
    df_plot = pd.concat([df_sig, df_nonsig]).sort_values(["chrom_int", "pos"])

    # Compute cumulative genomic position
    chrom_offsets = {}
    chrom_centers = {}
    cumulative = 0
    gap = 5_000_000  # 5 Mb gap between chromosomes
    for chrom_int in sorted(df_plot["chrom_int"].unique()):
        chrom_df = df_plot[df_plot["chrom_int"] == chrom_int]
        min_pos = chrom_df["pos"].min()
        max_pos = chrom_df["pos"].max()
        chrom_offsets[chrom_int] = cumulative - min_pos
        chrom_centers[chrom_int] = cumulative + (max_pos - min_pos) / 2
        cumulative += (max_pos - min_pos) + gap

    df_plot["genome_pos"] = df_plot["pos"] + df_plot["chrom_int"].map(chrom_offsets)

    # Alternating colors by chromosome
    color_even = "#2E86C1"
    color_odd = "#1B2A4A"
    df_plot["color"] = df_plot["chrom_int"].apply(
        lambda c: color_even if c % 2 == 0 else color_odd
    )

    # Build figure with go for performance
    fig = go.Figure()

    for chrom_int in sorted(df_plot["chrom_int"].unique()):
        cdf = df_plot[df_plot["chrom_int"] == chrom_int]
        chrom_label = chrom_order[chrom_int] if chrom_int < len(chrom_order) else str(chrom_int)
        color = color_even if chrom_int % 2 == 0 else color_odd
        fig.add_trace(go.Scattergl(
            x=cdf["genome_pos"],
            y=cdf["-log10(P)"],
            mode="markers",
            marker=dict(size=3, color=color, opacity=0.6),
            name=f"Chr {chrom_label}",
            text=cdf["variant"],
            customdata=np.stack([cdf["p"], cdf["beta"]], axis=-1),
            hovertemplate=(
                "Variant: %%{text}<br>"
                "P (FE): %%{customdata[0]:.2e}<br>"
                "Beta (FE): %%{customdata[1]:.6f}<br>"
                "<extra>Chr %s</extra>" % chrom_label
            ),
            showlegend=False,
        ))

    # Significance line
    fig.add_hline(
        y=-np.log10(sig_threshold),
        line_dash="dash",
        line_color="red",
        annotation_text="Genome-wide significance",
    )

    # Chromosome labels on x-axis
    tick_vals = [chrom_centers[c] for c in sorted(chrom_centers)]
    tick_text = [
        chrom_order[c] if c < len(chrom_order) else str(c)
        for c in sorted(chrom_centers)
    ]
    fig.update_layout(
        title="Manhattan Plot",
        xaxis=dict(
            title="Chromosome",
            tickvals=tick_vals,
            ticktext=tick_text,
            showgrid=False,
        ),
        yaxis_title="-log₁₀(P)",
        plot_bgcolor="white",
        height=500,
    )

    return fig


def _fig_to_png_bytes(fig, width: int = 1600, height: int = 600) -> bytes | None:
    """Render a Plotly figure to PNG. Returns None if kaleido is missing."""
    try:
        return fig.to_image(format="png", width=width, height=height, scale=2)
    except Exception:
        return None


def _plot_to_pdf_bytes(fig, title: str) -> bytes:
    """Render a single Plotly figure as a one-page landscape PDF."""
    from fpdf import FPDF

    pdf = FPDF(orientation="L", format="A4")
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT", align="C")

    png = _fig_to_png_bytes(fig, width=1800, height=800)
    if png is None:
        pdf.set_font("Helvetica", "I", 10)
        pdf.multi_cell(
            0, 6,
            "Plot could not be rendered (the 'kaleido' package is required "
            "to export Plotly figures as PNG / PDF).",
        )
    else:
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(png)
            tmp_path = tmp.name
        try:
            pdf.image(tmp_path, w=pdf.w - 20)
        finally:
            os.unlink(tmp_path)

    out = pdf.output(dest="S")
    return bytes(out) if not isinstance(out, bytes) else out


def _generate_pdf(
    df: pd.DataFrame,
    manhattan_fig,
    rank_fig,
    sig_threshold: float,
    summary_text: str,
    research_question: str,
    study_ids: list[str],
    n_total: int | None = None,
    n_significant: int | None = None,
    n_suggestive: int | None = None,
) -> bytes:
    """Generate a full PDF report and return raw bytes.

    Uses landscape A4 so the wide hits table (variant + meta stats +
    one beta column per input study) fits without truncation.
    """
    from fpdf import FPDF

    pdf = FPDF(orientation="L", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    usable_w = pdf.w - pdf.l_margin - pdf.r_margin

    # --- Title ---
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "GWAS Meta-Analysis Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {date.today().isoformat()}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)

    # --- Research question ---
    if research_question:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Research Question", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(usable_w, 5, research_question)
        pdf.ln(4)

    # --- Summary metrics ---
    # Use caller-provided totals (computed from full results) when available,
    # otherwise fall back to the (possibly truncated) DataFrame.
    if n_total is None:
        n_total = len(df)
    if n_significant is None:
        n_significant = int(df["Significant"].sum())
    if n_suggestive is None:
        n_suggestive = int((df["P (FE)"] < 1e-5).sum())

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Summary Metrics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Total variants analysed: {n_total:,}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Genome-wide significant (p < {sig_threshold:.0e}): {n_significant:,}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Suggestive (p < 1e-5): {n_suggestive:,}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # --- Plots (Manhattan + Rank, each on a fresh page) ---
    def _embed_plot(fig, title: str) -> None:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT", align="C")
        png = _fig_to_png_bytes(fig, width=1800, height=800) if fig is not None else None
        if png is None:
            pdf.set_font("Helvetica", "I", 10)
            pdf.multi_cell(
                0, 6,
                f"({title} omitted -- install 'kaleido' to include plots in the report)",
            )
            return
        import os
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(png)
            tmp_path = tmp.name
        try:
            pdf.image(tmp_path, w=pdf.w - 20)
        finally:
            os.unlink(tmp_path)

    _embed_plot(manhattan_fig, "Manhattan Plot")
    _embed_plot(rank_fig, "Significance Rank Plot")

    # --- Significant hits table (on a fresh page so per-study cols fit) ---
    pdf.add_page()
    sig_df = df[df["Significant"]].copy()
    table_label = "Genome-wide Significant Hits"
    if len(sig_df) == 0:
        sig_df = df.head(20).copy()
        table_label = "Top 20 Variants (none reached genome-wide significance)"
    elif len(sig_df) > 50:
        sig_df = sig_df.head(50)
        table_label = f"Top 50 Genome-wide Significant Hits (of {n_significant:,} total)"

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, table_label, new_x="LMARGIN", new_y="NEXT")

    # Dynamic columns: fixed meta stats + one beta column per input study.
    # Widths are computed to fill the page so nothing gets cut off.
    base_cols = [
        ("Variant", None),
        ("rsID", None),
        ("Beta (FE)", "{:.4f}"),
        ("P (FE)", "{:.2e}"),
        ("I\u00b2", "{:.1f}"),
        ("N studies", "{:d}"),
        ("Dir (+/-/0)", None),
        ("LOO max P", "{:.2e}"),
    ]
    per_study_cols = [(f"Beta [{sid}]", "{:.4f}") for sid in study_ids]
    all_cols = base_cols + per_study_cols

    # Allocate width: Variant gets a fixed 50mm, rsID 22mm, other base cols
    # 14-22mm each, remaining usable width spreads across per-study beta cols.
    fixed_widths = [50, 22, 20, 22, 14, 18, 18, 20]
    base_used = sum(fixed_widths)
    n_studies = max(1, len(per_study_cols))
    per_study_w = max(16, (usable_w - base_used) / n_studies)
    col_widths = fixed_widths + [per_study_w] * len(per_study_cols)

    pdf.set_font("Helvetica", "B", 7)
    for (label, _), w in zip(all_cols, col_widths):
        pdf.cell(w, 6, label, border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 6)
    for _, row in sig_df.iterrows():
        cells: list[str] = []
        for (label, fmt), w in zip(all_cols, col_widths):
            val = row.get(label)
            if label == "Variant":
                cells.append(str(val))
            elif label == "rsID":
                cells.append(str(val) if val else "--")
            elif label == "N studies":
                cells.append(f"{int(val)}")
            elif label == "Dir (+/-/0)":
                cells.append(str(val) if val else "--")
            elif label == "LOO dropped":
                cells.append(str(val) if val else "--")
            elif val is None or (isinstance(val, float) and np.isnan(val)):
                cells.append("--")
            else:
                cells.append(fmt.format(val))
        for txt, w in zip(cells, col_widths):
            pdf.cell(w, 5, txt, border=1)
        pdf.ln()
    pdf.ln(4)

    # --- Studies included ---
    selected_studies = st.session_state.get("selected_studies", [])
    if selected_studies:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(usable_w, 8, "Studies Included", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        for s in selected_studies:
            title = s.title[:70] + ("..." if len(s.title) > 70 else "")
            text = f"{s.study_id}: {title}"
            text = text.encode("latin-1", errors="replace").decode("latin-1")
            pdf.cell(usable_w, 5, text, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    # --- AI Summary ---
    if summary_text:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "AI Summary", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        summary_safe = summary_text.encode("latin-1", errors="replace").decode("latin-1")
        pdf.multi_cell(usable_w, 5, summary_safe)

    return pdf.output()


def render():
    st.header("Step 6: Results & Summary")

    results = st.session_state.meta_results
    if results is None:
        # Try to reload from disk (e.g. after session loss)
        results = _load_results_csv()
        if results is not None:
            st.session_state.meta_results = results
            st.info(f"Restored {len(results):,} results from disk.")
        else:
            st.warning("No results. Go back to Step 5.")
            return

    sig_threshold = st.session_state.config.get("meta_analysis", {}).get(
        "significance_threshold", 5e-8
    )

    # Sort results by p-value first so we can build the DF from
    # a pre-sorted list (avoids sorting an 8M-row DataFrame).
    if not getattr(st.session_state, "_results_sorted", False):
        with st.spinner("Sorting results by p-value..."):
            results.sort(key=lambda r: r.p_fixed)
            st.session_state._results_sorted = True

    # Build top-N DataFrame for display (fast) instead of full 8M-row DF
    display_cap = 5000
    top_results = results[:display_cap]
    study_ids = _collect_study_ids(results)
    df_top = _build_results_df(top_results, sig_threshold, study_ids=study_ids)

    # Compute summary metrics directly from the result objects (no full DF needed)
    n_total = len(results)
    n_sig = sum(1 for r in results if r.p_fixed < sig_threshold)
    n_suggestive = sum(1 for r in results if r.p_fixed < 1e-5)

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total variants", f"{n_total:,}")
    col2.metric("Genome-wide significant", f"{n_sig:,}")
    col3.metric("Suggestive (p < 1e-5)", f"{n_suggestive:,}")

    # Results table (only top N)
    st.subheader("Top Results")
    show_n = st.slider("Show top N variants", 10, min(500, len(df_top)), 50)
    fmt = {
        "Beta (FE)": "{:.4f}",
        "SE (FE)": "{:.4f}",
        "P (FE)": "{:.2e}",
        "Beta (RE)": "{:.4f}",
        "SE (RE)": "{:.4f}",
        "P (RE)": "{:.2e}",
        "Q": "{:.2f}",
        "I²": "{:.1f}",
        "τ²": "{:.6f}",
    }
    for sid in study_ids:
        fmt[f"Beta [{sid}]"] = "{:.4f}"
    st.dataframe(
        df_top.head(show_n).style.format(fmt, na_rep="—"),
        width="stretch",
        hide_index=True,
    )
    if study_ids:
        st.caption(
            "Far-right columns show the per-study beta for each input study "
            "(blank = variant absent from that study)."
        )

    # Plot section with toggle
    st.subheader("Visualization")
    plot_type = st.radio(
        "Plot type",
        ["Manhattan plot", "Rank plot"],
        horizontal=True,
        help="Manhattan: variants by genomic position. Rank: variants sorted by p-value.",
    )

    if plot_type == "Manhattan plot":
        if "manhattan_fig" not in st.session_state:
            with st.spinner("Building Manhattan plot..."):
                st.session_state.manhattan_fig = _build_manhattan_fig(results, sig_threshold)
        st.plotly_chart(st.session_state.manhattan_fig, width="stretch", key="plot_manhattan")
    else:
        if "rank_fig" not in st.session_state:
            with st.spinner("Building Rank plot..."):
                st.session_state.rank_fig = _build_rank_fig(df_top, sig_threshold)
        st.plotly_chart(st.session_state.rank_fig, width="stretch", key="plot_rank")

    # --- Download plots ---
    st.markdown("**Download plots**")
    dl_man, dl_rank = st.columns(2)
    with dl_man:
        if st.button("Prepare Manhattan PDF", key="prep_manhattan_pdf"):
            with st.spinner("Rendering Manhattan plot..."):
                if "manhattan_fig" not in st.session_state:
                    st.session_state.manhattan_fig = _build_manhattan_fig(
                        results, sig_threshold,
                    )
                st.session_state._manhattan_pdf = _plot_to_pdf_bytes(
                    st.session_state.manhattan_fig, "Manhattan Plot",
                )
        if st.session_state.get("_manhattan_pdf"):
            st.download_button(
                "Download Manhattan plot (PDF)",
                data=st.session_state._manhattan_pdf,
                file_name="manhattan_plot.pdf",
                mime="application/pdf",
                key="dl_manhattan_pdf",
            )
    with dl_rank:
        if st.button("Prepare Rank PDF", key="prep_rank_pdf"):
            with st.spinner("Rendering rank plot..."):
                if "rank_fig" not in st.session_state:
                    st.session_state.rank_fig = _build_rank_fig(df_top, sig_threshold)
                st.session_state._rank_pdf = _plot_to_pdf_bytes(
                    st.session_state.rank_fig, "Significance Rank Plot",
                )
        if st.session_state.get("_rank_pdf"):
            st.download_button(
                "Download rank plot (PDF)",
                data=st.session_state._rank_pdf,
                file_name="rank_plot.pdf",
                mime="application/pdf",
                key="dl_rank_pdf",
            )

    # LLM summary
    st.subheader("AI Summary")

    # Transparency: show the exact model input before the call, mirroring Step 1.
    # `_summary_user_message` is the same code path `_generate_summary` sends, so
    # what the user sees here is byte-for-byte what goes to the provider.
    with st.expander("Prompt sent to the AI (read-only)"):
        st.caption(
            "This is the exact request sent when you click Generate: the top-20 "
            "variants (by fixed-effects p-value), aggregate counts, and your "
            "research question, alongside a fixed system instruction. The returned "
            "text is an AI-generated interpretation to evaluate critically, not a "
            "definitive conclusion."
        )
        st.markdown("**User message**")
        st.text_area(
            "Summary user message",
            value=_summary_user_message(results, sig_threshold),
            height=240,
            disabled=True,
            label_visibility="collapsed",
        )
        st.markdown("**System instruction**")
        st.text_area(
            "Summary system prompt",
            value=SUMMARY_SYSTEM_PROMPT,
            height=220,
            disabled=True,
            label_visibility="collapsed",
        )

    if not st.session_state.summary_text:
        if st.button("Generate AI Summary"):
            with st.spinner("Generating summary..."):
                try:
                    summary = _generate_summary(results, sig_threshold)
                    st.session_state.summary_text = summary
                    st.rerun()
                except Exception as e:
                    st.error(f"Summary generation failed: {e}")
    else:
        st.markdown(st.session_state.summary_text)

    # Export
    st.subheader("Export")
    col_csv, col_pdf, col_prov = st.columns(3)

    with col_csv:
        if _RESULTS_CSV.is_file():
            size_mb = _RESULTS_CSV.stat().st_size / (1024 * 1024)
            csv_path = _RESULTS_CSV.resolve()
            st.markdown(f"**Full results CSV** ({size_mb:,.0f} MB)")
            st.code(str(csv_path), language=None)
            st.caption("File is too large for in-browser download. Open the path above in Finder or terminal.")
        else:
            st.button("Prepare full CSV", disabled=True)
            st.caption("No results CSV found on disk.")

    with col_pdf:
        if st.button("Prepare PDF report"):
            with st.spinner("Generating PDF report..."):
                # Ensure both plots exist so the report bundles them.
                if "manhattan_fig" not in st.session_state:
                    st.session_state.manhattan_fig = _build_manhattan_fig(
                        results, sig_threshold,
                    )
                if "rank_fig" not in st.session_state:
                    st.session_state.rank_fig = _build_rank_fig(
                        df_top, sig_threshold,
                    )
                pdf_bytes = _generate_pdf(
                    df=df_top,
                    manhattan_fig=st.session_state.manhattan_fig,
                    rank_fig=st.session_state.rank_fig,
                    sig_threshold=sig_threshold,
                    summary_text=st.session_state.summary_text,
                    research_question=st.session_state.research_question,
                    study_ids=study_ids,
                    n_total=n_total,
                    n_significant=n_sig,
                    n_suggestive=n_suggestive,
                )
                st.session_state._pdf_data = bytes(pdf_bytes)
                st.rerun()

        if st.session_state.get("_pdf_data"):
            st.download_button(
                "Download report (PDF)",
                data=st.session_state._pdf_data,
                file_name="meta_analysis_report.pdf",
                mime="application/pdf",
            )

    with col_prov:
        prov = st.session_state.get("provenance")
        if prov is not None:
            from datetime import datetime as _dt
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            st.markdown("**Session provenance**")
            st.caption("JSON + Markdown bundled in one zip.")
            st.download_button(
                "Download provenance record",
                data=prov.to_zip_bytes(),
                file_name=f"gwas_meta_provenance_{ts}.zip",
                mime="application/zip",
            )

    if st.session_state.summary_text:
        st.download_button(
            "Download summary (TXT)",
            data=st.session_state.summary_text,
            file_name="meta_analysis_summary.txt",
            mime="text/plain",
        )
