"""Session provenance logger for the GWAS meta-analysis Streamlit app.

Records every step of an interactive session (research question, LLM-proposed
vs. user-edited search terms, study selection, QC, alignment, meta-analysis,
plus compute time of each backend step) and serializes the result as a single
ZIP containing both a machine-readable JSON and a human-readable Markdown
rendering suitable for a paper supplement or lab-meeting slide.

Timing policy: compute time is measured around the *backend* call only
(``time_block`` uses ``perf_counter`` on entry/exit) so it never includes
the seconds the user spends reviewing, editing, or selecting.
"""

from __future__ import annotations

import dataclasses
import io
import json
import platform
import subprocess
import time
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Iterator


def _git_commit(repo_hint: str | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_hint,
            check=True,
            capture_output=True,
            timeout=2,
        )
        return out.stdout.decode().strip()
    except Exception:
        return None


def _tool_version() -> str:
    try:
        from importlib.metadata import version
        return version("gwas_meta")
    except Exception:
        return "0.1.0"


def _to_jsonable(obj: Any) -> Any:
    """Convert dataclasses, Paths, sets, and other non-JSON types to JSON-safe forms."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_to_jsonable(v) for v in obj)
    if hasattr(obj, "__fspath__"):
        return str(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


class ProvenanceLogger:
    """Append-only timestamped log held in ``st.session_state``.

    Events are not written to disk during the session; they're serialized into
    a single ZIP when the user clicks the Step 6 download button.
    """

    def __init__(self, gwas_catalog_url: str | None = None, repo_hint: str | None = None):
        now = datetime.now(timezone.utc)
        self.header: dict[str, Any] = {
            "tool": "gwas-meta",
            "tool_version": _tool_version(),
            "git_commit": _git_commit(repo_hint),
            "session_start_utc": now.isoformat(),
            "gwas_catalog_url": gwas_catalog_url,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
        }
        self.events: list[dict[str, Any]] = []
        # Monotonic baseline used only for relative offsets in the readable log.
        self._t0 = time.perf_counter()

    # ---- writing -------------------------------------------------------

    def event(self, kind: str, payload: dict[str, Any] | None = None,
              compute_seconds: float | None = None) -> None:
        rec = {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "payload": _to_jsonable(payload or {}),
        }
        if compute_seconds is not None:
            rec["compute_seconds"] = round(float(compute_seconds), 4)
        self.events.append(rec)

    @contextmanager
    def time_block(self, kind: str, payload_factory: Callable[[], dict[str, Any]] | None = None
                   ) -> Iterator[dict[str, Any]]:
        """Time a backend call. ``payload_factory`` is called *after* the block returns,
        so it can read post-call state (e.g. result counts).

        Yields a mutable dict the caller may also write into; on exit, the
        ``payload_factory`` result (if any) is merged on top.
        """
        scratch: dict[str, Any] = {}
        t0 = time.perf_counter()
        try:
            yield scratch
        finally:
            dt = time.perf_counter() - t0
            payload = scratch
            if payload_factory is not None:
                try:
                    payload = {**scratch, **(payload_factory() or {})}
                except Exception as exc:  # never let logging break the app
                    payload = {**scratch, "_payload_factory_error": repr(exc)}
            self.event(kind, payload, compute_seconds=dt)

    def has_event(self, kind: str) -> bool:
        return any(e["kind"] == kind for e in self.events)

    # ---- serializing ---------------------------------------------------

    def to_json_bytes(self) -> bytes:
        doc = {"header": self.header, "events": self.events}
        return json.dumps(doc, indent=2, ensure_ascii=False).encode("utf-8")

    def to_markdown_bytes(self) -> bytes:
        return _render_markdown(self.header, self.events).encode("utf-8")

    def to_zip_bytes(self) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("provenance_log.json", self.to_json_bytes())
            zf.writestr("provenance_log.md", self.to_markdown_bytes())
        return buf.getvalue()


# =====================================================================
# Markdown rendering
# =====================================================================

def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_(none)_\n"
    sep = "|".join(["---"] * len(headers))
    out = ["| " + " | ".join(headers) + " |", f"|{sep}|"]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(out) + "\n"


def _proposed_vs_final(prop: dict, final: dict) -> str:
    fields = [
        ("trait_description", "Trait description"),
        ("efo_terms", "EFO terms"),
        ("inclusion_criteria", "Inclusion criteria"),
        ("exclusion_criteria", "Exclusion criteria"),
        ("ancestry_preference", "Ancestry preference"),
        ("min_sample_size", "Minimum sample size"),
    ]
    rows = []
    for k, label in fields:
        pv = prop.get(k)
        fv = final.get(k)
        same = pv == fv
        rows.append([
            label,
            _fmt_field(pv),
            _fmt_field(fv),
            "—" if same else "**edited**",
        ])
    return _md_table(["Field", "Proposed (LLM)", "Final (user)", "Changed?"], rows)


def _fmt_field(v: Any) -> str:
    if v is None or v == "":
        return "_(none)_"
    if isinstance(v, list):
        return "<br>".join(str(x) for x in v) if v else "_(none)_"
    return str(v)


def _render_markdown(header: dict, events: list[dict]) -> str:  # noqa: PLR0915, PLR0912
    out: list[str] = []
    out.append("# GWAS meta-analysis — provenance record\n")
    out.append("## Session header\n")
    out.append(_md_table(["Field", "Value"], [
        ["Tool", f"{header.get('tool')} v{header.get('tool_version')}"],
        ["Git commit", header.get("git_commit") or "_(unavailable)_"],
        ["Session start (UTC)", header.get("session_start_utc")],
        ["GWAS Catalog endpoint", header.get("gwas_catalog_url") or "_(default)_"],
        ["Platform", header.get("platform")],
        ["Python", header.get("python_version")],
        ["Machine", header.get("machine")],
    ]))

    by_kind: dict[str, list[dict]] = {}
    for e in events:
        by_kind.setdefault(e["kind"], []).append(e)

    # Research question
    q_events = by_kind.get("research_question", [])
    if q_events:
        out.append("\n## Research question\n")
        for e in q_events:
            out.append(f"> {e['payload'].get('question', '')}\n")

    # Search terms
    prop_events = by_kind.get("search_terms_proposed", [])
    final_events = by_kind.get("search_terms_final", [])
    if prop_events or final_events:
        out.append("\n## Search terms — proposed vs. final\n")
        if prop_events:
            p = prop_events[-1]["payload"]
            provider = p.get("provider")
            model = p.get("model")
            if provider or model:
                out.append(f"_LLM provider: **{provider or '?'}**, model: **{model or '?'}**_\n")
        prop = (prop_events[-1]["payload"].get("criteria") if prop_events else {}) or {}
        final = (final_events[-1]["payload"].get("criteria") if final_events else {}) or {}
        out.append(_proposed_vs_final(prop, final))

    # Catalog results
    cat_events = by_kind.get("catalog_search_results", [])
    if cat_events:
        out.append("\n## Studies returned by GWAS Catalog search\n")
        studies = cat_events[-1]["payload"].get("studies", []) or []
        out.append(f"**{len(studies)}** studies returned.\n\n")
        out.append(_md_table(
            ["Study ID", "Trait(s)", "Sample size", "Ancestry", "Journal", "Date"],
            [[s.get("study_id"), s.get("traits"), s.get("sample_size"),
              s.get("ancestry"), s.get("journal"), s.get("date")] for s in studies],
        ))

    # Manual uploads
    upl_events = by_kind.get("manual_upload", [])
    if upl_events:
        out.append("\n## Manual data uploads\n")
        rows = []
        for e in upl_events:
            for f in e["payload"].get("files", []):
                rows.append([f.get("study_id"), f.get("filename"), f.get("source"), e["ts_utc"]])
        out.append(_md_table(["Study ID", "Filename", "Source", "Timestamp (UTC)"], rows))

    # Studies selected
    sel_events = by_kind.get("studies_selected", [])
    if sel_events:
        out.append("\n## Studies — include / exclude decisions\n")
        p = sel_events[-1]["payload"]
        included = p.get("included", [])
        excluded = p.get("excluded", [])
        out.append(f"**Included: {len(included)} · Excluded: {len(excluded)}**\n\n")
        rows = []
        for s in included:
            rows.append([s if isinstance(s, str) else s.get("study_id"), "✓ included"])
        for s in excluded:
            rows.append([s if isinstance(s, str) else s.get("study_id"), "✗ excluded"])
        out.append(_md_table(["Study ID", "Decision"], rows))

    # QC
    dl_events = by_kind.get("download_complete", [])
    if dl_events:
        out.append("\n## Download\n")
        p = dl_events[-1]["payload"]
        out.append(_md_table(["Metric", "Value"], [
            ["Studies requested", p.get("n_requested")],
            ["Studies downloaded", p.get("n_downloaded")],
            ["Failures", p.get("n_errors")],
            ["Compute time (s)", dl_events[-1].get("compute_seconds")],
        ]))

    qc_events = by_kind.get("qc_lambda_gc", [])
    if qc_events:
        out.append("\n## QC — genomic inflation (λGC)\n")
        lam = qc_events[-1]["payload"].get("lambda_gc", {}) or {}
        rows = [[sid, f"{v:.3f}" if isinstance(v, (int, float)) else v]
                for sid, v in lam.items()]
        out.append(_md_table(["Study ID", "λGC"], rows))

    exc_events = by_kind.get("qc_exclusions", [])
    if exc_events:
        out.append("\n### QC exclusions (user)\n")
        excl = exc_events[-1]["payload"].get("excluded", []) or []
        if excl:
            out.append("- " + "\n- ".join(excl) + "\n")
        else:
            out.append("_(no exclusions)_\n")

    # Meta-analysis settings + counts
    set_events = by_kind.get("meta_settings", [])
    if set_events:
        out.append("\n## Meta-analysis settings\n")
        s = set_events[-1]["payload"]
        out.append(_md_table(["Setting", "Value"], [
            ["Model", "IVW fixed-effects + DerSimonian-Laird random-effects"],
            ["min_study_count", s.get("min_study_count")],
            ["Valid hm_codes (harmonisation filter)", s.get("valid_hm_codes")],
            ["MAF threshold (loader QC)", s.get("maf_threshold", "≥ 0.01")],
            ["abs(β) max (loader QC)", s.get("beta_max", "≤ 10")],
            ["SE max (loader QC)", s.get("se_max", "≤ 10")],
            ["Heterogeneity Q p-value threshold", s.get("q_threshold")],
            ["Genome-wide significance threshold", s.get("sig_threshold")],
        ]))

    ck_events = by_kind.get("chunking_complete", [])
    am_events = by_kind.get("alignment_meta_complete", [])
    if ck_events or am_events:
        out.append("\n## Alignment & meta-analysis — per-chromosome\n")
        per_chrom = (am_events[-1]["payload"].get("per_chrom", []) if am_events else [])
        # Totals first (prominent)
        if am_events:
            ap = am_events[-1]["payload"]
            out.append(_md_table(["Metric", "Value"], [
                ["Chromosomes processed", ap.get("n_chroms_processed")],
                ["Chromosomes skipped (too few studies)", ap.get("n_chroms_skipped")],
                ["Total aligned variants", ap.get("total_aligned_variants")],
                ["Total variants after meta-analysis", ap.get("total_meta_variants")],
                ["Alignment + meta compute time (s)", am_events[-1].get("compute_seconds")],
                ["Chunking compute time (s)", ck_events[-1].get("compute_seconds") if ck_events else None],
            ]))
        # Per-chrom: collapsible so it doesn't dominate the document.
        if per_chrom:
            out.append("\n<details><summary>Per-chromosome counts ({} rows)</summary>\n\n".format(len(per_chrom)))
            out.append(_md_table(
                ["chrom", "n_studies", "aligned variants", "meta variants"],
                [[r.get("chrom"), r.get("n_studies"),
                  r.get("aligned_variants"), r.get("meta_variants")] for r in per_chrom],
            ))
            out.append("\n</details>\n")

    # Meta-analysis result totals
    mr_events = by_kind.get("meta_results_summary", [])
    if mr_events:
        out.append("\n## Meta-analysis results\n")
        p = mr_events[-1]["payload"]
        rows = [
            ["Total variants analysed", p.get("n_total")],
            ["Genome-wide significant (p < 5e-8)", p.get("n_significant")],
            ["Suggestive (p < 1e-5)", p.get("n_suggestive")],
            ["Heterogeneity-filtered out", p.get("n_heterogeneity_removed")],
        ]
        if p.get("n_independent_loci") is not None:
            rows.append(["Independent loci (≥1 Mb apart)", p["n_independent_loci"]])
        out.append(_md_table(["Metric", "Value"], rows))

    # Compute-time summary
    timed = [e for e in events if "compute_seconds" in e]
    if timed:
        out.append("\n## Compute-time summary\n")
        out.append("_Backend compute time only — excludes user review/edit time._\n\n")
        rows = [[e["kind"], f"{e['compute_seconds']:.3f}"] for e in timed]
        rows.append(["**Total backend compute**",
                     f"**{sum(e['compute_seconds'] for e in timed):.3f}**"])
        out.append(_md_table(["Step", "Compute seconds"], rows))

    # Full event timeline
    out.append("\n## Event timeline\n")
    rows = []
    for e in events:
        rows.append([
            e["ts_utc"],
            e["kind"],
            f"{e.get('compute_seconds', '')}" if "compute_seconds" in e else "",
        ])
    out.append(_md_table(["Timestamp (UTC)", "Event", "Compute s"], rows))

    return "".join(out)
