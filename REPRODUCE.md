# Reproducing the Results in this Thesis

This document walks through every empirical result in the manuscript
and explains how to reproduce it from this repository. It is intended
for readers who want to spot-check the numbers rather than take them
on faith.

You do **not** need to run everything. Each section is independent.
The unit-test smoke check (§1) and one case study (§3, §4, or §5) is
enough to convince a skeptical reader that the tool does what the
thesis claims.

---

## Claim index — every quantitative claim → command → expected value

Each row is a number that appears in the manuscript, where it appears,
the command that regenerates it (sections refer to this document), and
the artifact field or table to read it from. Timings are hardware-
dependent and are **not** expected to match to the second; every other
number should match exactly. Rows marked *cited* are facts taken from
the literature, not outputs of this tool, and are not reproducible from
this repository — their source is the citation in the manuscript.

### Abstract & Conclusion (§5)

| Thesis claim | Number | Reproduce with | Read from |
|---|---|---|---|
| max abs Δ(β) vs metafor | ≤ 3.11 × 10⁻¹⁵ | REPRODUCE §2 | `metafor_comparison_synthetic_SUMMARY.csv` |
| max abs Δ(I²) vs metafor | ≤ 1.64 × 10⁻¹⁴ | REPRODUCE §2 | same |
| gout β correlation | Pearson r = 0.999997 | REPRODUCE §3 | `benchmark_summary.json → table_3_2_2a.all_matched.pearson_r_beta` |
| gout direction concordance | 99.96 % | REPRODUCE §3 | same → `direction_concordance` |
| gout matched variants | 7,603,652 | REPRODUCE §3 | same → `table_3_2_2b.aligned_overlap` |
| speedup vs per-variant loop | "substantial" (order ~10²) | — | indicative; not benchmarked as a headline number |
| peak memory | 4.0 GB | REPRODUCE §3 | `benchmark_summary.json → memory_gb.pipeline_peak`; hardware-dependent |

### §2 Materials & Methods (design constants)

| Thesis claim | Number | Status |
|---|---|---|
| χ² median under null (λGC denominator) | 0.4549 | definitional (0.5-quantile of χ²₁) |
| p-value clip range | (10⁻³⁰⁰, 1−10⁻¹⁰) | source constant, `src/gwas_meta/…/effect_size` |
| effect-size plausibility bound | \|β\| or SE > 10 excluded | source constant; QC counts visible in run log |
| worker memory inflation factor / cap | ×5, ≤ 6 workers, 60 % RAM | source constant, `adaptive` resource guard |
| LLM cost per run | a few euro-cents | *cited* pricing; token counts ~1,200 in / ~1,500 out |
| unit tests | 105 / 105 pass | REPRODUCE §1 (`pytest tests/ -q`) |

### §3.2.1 — Tier 1 (synthetic vs metafor)

| Thesis claim | Number | Reproduce with | Read from |
|---|---|---|---|
| synthetic variants compared | 3,847 | REPRODUCE §2 | generator output row count |
| nine per-variant max abs diffs | see Table 3.2.1 | REPRODUCE §2 | `…_SUMMARY.csv` (matches the table in §2 below) |
| k distribution | 642/642/641/641/640/641 | REPRODUCE §2 | generator log |

### §3.2.2 — Tier 2 (gout, end-to-end)

| Thesis claim | Number | Reproduce with | Read from |
|---|---|---|---|
| raw input rows (FinnGen / UKBB / ref) | 21,294,541 / 23,731,281 / 32,353,705 | REPRODUCE §3 preprocess | `benchmark/gout_inputs/provenance.json` |
| post-QC (FinnGen / UKBB) | 8,685,374 / 8,603,008 | REPRODUCE §3 | `benchmark_summary.json → table_3_2_2b.after_qc` |
| aligned overlap | 7,603,652 | REPRODUCE §3 | same → `aligned_overlap` |
| concordance (r, dir, Δβ, GWS) | Table 3.2.2a | REPRODUCE §3 | same → `table_3_2_2a` |
| canonical loci β / p (×4) | Table 3.2.2c | REPRODUCE §3 | `canonical_loci.csv` |
| reference-only gap | 24,638,083 | REPRODUCE §3 `--full-accounting` | `table_3_2_2_full_accounting` |
| in raw, removed by QC | 22,557,005 (91.55 %) | same | same |
| single-cohort | 2,081,078 (8.45 %) | same | same |
| in-neither / missed / engine-only | 0 / 0 / 0 | same | same + `variant_accounting_full.csv` |

### §3.3 — LDL cholesterol case study

| Thesis claim | Number | Reproduce with | Read from |
|---|---|---|---|
| aligned / meta-analysed variants | 6,840,875 | REPRODUCE §4 | `ldl_summary.json → aligned_variants` |
| expected loci recovered | 30 / 30 (15 + 15) | REPRODUCE §4 | `ldl_summary.json → recovery`, `ldl_recovery.csv` |
| genome-wide significant (post-Q) | 40,645 | REPRODUCE §4 **`--q-filter 1e-6`** | `ldl_summary.json → gws_variants` |
| removed for extreme heterogeneity | 14,728 | REPRODUCE §4 `--q-filter 1e-6` | same → `het_variants_removed` |
| suggestive (p < 1e-5, post-Q) | 73,965 | REPRODUCE §4 `--q-filter 1e-6` | `ldl_summary.json → suggestive_variants` |
| LPA window GWS variants (raw meta) | 1,120 | REPRODUCE §4 (any run) | `ldl_recovery.csv → n_sig_in_window` |
| λGC (UKB / MVP) | 1.358 / 1.626 | REPRODUCE §4 | `ldl_summary.json → lambda_gc` |
| direction concordance (raw GWS set) | 97.4 % | REPRODUCE §4 | `ldl_summary.json → direction_concordance_gws` |
| median I² across raw GWS set | 73 % | REPRODUCE §4 | `ldl_summary.json → median_i2_gws_percent` |
| weakest locus (ANXA9-CERS2) | 1 variant, p = 5.0 × 10⁻⁹ | REPRODUCE §4 | `ldl_recovery.csv` |
| PCSK9 random-effects p (RE illustration) | 1.6 × 10⁻¹⁰ | REPRODUCE §4 | `ldl_recovery.csv → lead_p_random` (PCSK9 row) (see note) |
| Table 3.3.1 β UKB / β MVP / β Meta | per-study + pooled, all SAME | REPRODUCE §4 | `ldl_recovery.csv → beta_GCST90019512 / beta_GCST90475420 / lead_beta_fixed / sign` |

> The per-locus recovery table (`ldl_recovery.csv`) and the two heterogeneity
> numbers are computed on the **raw (pre-Q) meta**, matching manuscript
> Table 3.3.1 and the §3.3 heterogeneity discussion; only the headline
> GWS / suggestive / het-removed counts use the post-Q filter. At the five
> loci whose fixed-effects p floors to 0 (PCSK9, SORT1, APOB, LDLR, APOE),
> the "lead" variant is a tie-break among many floored variants, so the
> lead's `p_random` shown in `ldl_recovery.csv` is convention-dependent. The
> manuscript's illustrative figure (PCSK9, P_RE = 1.6 × 10⁻¹⁰) is the lead
> variant `chr1:55039974:G:T`, which the wrapper's deterministic tie-break
> selects consistently; read `lead_p_random` for the PCSK9 row in
> `ldl_recovery.csv` to confirm.

### §3.4 — Retinitis pigmentosa case study

| Thesis claim | Number | Reproduce with | Read from |
|---|---|---|---|
| variants meta-analysed | 6,867,143 | REPRODUCE §5 | `rp_summary.json → meta_rows` |
| genome-wide significant total | 376 | REPRODUCE §5 | same → `gws_variants` |
| GWS off chromosome 6 | 0 | REPRODUCE §5 | same → `gws_off_chr6` |
| EYS lead variant | chr6:64,990,459:A:G | REPRODUCE §5 | same → `eys.lead_variant` |
| EYS lead OR / p | 3.948 / 1.173 × 10⁻¹³ | REPRODUCE §5 | same → `eys.lead_or`, `eys.lead_p_fixed` |
| risk / protective split | 347 / 29 | REPRODUCE §5 | derived from `meta_results.csv` |
| median I² across GWS set | 12.9 | REPRODUCE §5 | derived from `meta_results.csv` |
| λGC (stage 1 / stage 2) | 1.06 / 1.03 | REPRODUCE §5 | run log (per-study QC lines) |

### Cited facts (not reproducible from this repo)

| Thesis claim | Number | Source (see manuscript citation) |
|---|---|---|
| clinical-trial failure rate | 86.2 % | Wong et al. 2019 |
| genetic-support → approval odds | ~2× | Nelson 2015; King 2019 |
| GWAS Catalog scale | ~7,700 pubs / >1.1 M associations | Cerezo 2025 / live Catalog stats |
| 8q24 / MYC enhancer biology | — | Ahmadiyeh 2010; Sur 2012 |

> Numbers marked "derived from `meta_results.csv`" are present in the
> full per-variant output but not surfaced as a summary field; a one-line
> pandas expression over that CSV reproduces them. If you want these
> promoted into the summary JSON, that is a small wrapper change.

---

## 0. Setup

**Requirements**: Python ≥ 3.11, ~4 GB free RAM for a full-genome
case study, ~10 GB free disk for cached summary statistics. The Tier-1
metafor check (§2) additionally needs R with the `metafor` package —
§2 opens with commands to check for it and install it if missing; every
other step is Python-only.

```bash
git clone https://github.com/pjcodebio/GWAMA_tool.git
cd GWAMA_tool
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Download budget** — every case study caches its inputs *inside the
clone* (under `.cache/` or `benchmark/*/inputs/`), all git-ignored so
nothing is accidentally committed. Running every section from scratch
downloads approximately:

| Section | Downloads | Total |
|---|---|---|
| §2 metafor | none (synthetic, generated locally) | 0 |
| §3 gout | FinnGen R12 + pan-UKBB combined meta (1.77 GB primary + ~1.9 GB per-cohort splits) | ~3.7 GB |
| §4 LDL | GCST90019512 + GCST90475420 harmonised files | ~842 MB |
| §5 RP | GCST90011892 + GCST90011893 harmonised files | ~447 MB |

Downloads are cached, so re-running is free. If you want cached data on
an external drive, every wrapper accepts `--cache-dir <path>` (LDL, RP)
or `--out-dir <path>` (gout preprocess) to redirect.

**No API key is required** to reproduce the results. The tool uses a
large language model for two convenience features only — free-text
question parsing in Step 2 and results summarisation in Step 6 —
which can be skipped entirely by using the upload / manual-selection
mode described below.

If you *do* want to exercise the LLM path, copy `.env.example` to
`.env` and populate `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`; the
provider is configured in `config/settings.yaml`).

---

## Quick recipe — run everything, top to bottom

Copy-paste this whole block after the one-time setup above. Each step
prints a `HEADLINE` section and (for steps 1, 3, 4, 5) **returns exit
code 0 on pass, 1 on fail** — so `echo "exit=$?"` after each command is
the fastest check. Expected values are in the comments; the detailed
per-step tables further down explain every number. Total compute is
~20 min on an M1 laptop, plus first-run downloads (see the budget
table above; nothing re-downloads once cached).

```bash
# ── Step 1 — engine unit tests (~2 s) ────────────────────────────────
pytest tests/ -q
#   expect: 84 passed

# ── Step 2 — Tier-1 synthetic vs R metafor (~10 min, needs R+metafor) ─
python tests/validation/generate_synthetic_metafor_set.py
python tests/validation/run_engine_on_synthetic.py \
    --per_study tests/validation/synthetic/synthetic_per_study_long.csv \
    --out       tests/validation/synthetic/engine_output_synthetic.csv
Rscript tests/validation/metafor_validate.R \
    --per_study tests/validation/synthetic/synthetic_per_study_long.csv \
    --engine    tests/validation/synthetic/engine_output_synthetic.csv \
    --out       tests/validation/synthetic/metafor_comparison_synthetic.csv \
    --tier      synthetic
#   expect: all 9 statistics max|Δ| ≤ 1e-11 (Table 3.2.1)

# ── Step 3 — Tier-2 gout benchmark (~7 min w/ --full-accounting; 1.77 GB download 1st run) ─
python benchmark/preprocess_gout_inputs.py --out-dir benchmark/gout_inputs
python benchmark/run_benchmark.py \
    --inputs-dir benchmark/gout_inputs \
    --out-dir    benchmark/gout_output \
    --full-accounting
#   expect: r(β)=0.999997, dir 99.96 %, and the gap reconciles
#           24,638,083 = 22,557,005 + 2,081,078 + 0 + 0

# ── Step 4 — LDL recovery case study (~3.3 min warm; +~5 min for 842 MB download 1st run) ─
python benchmark/run_ldl_case_study.py \
    --out-dir benchmark/ldl_case_study/out \
    --q-filter 1e-6
#   expect: 30/30 loci, 40,645 GWS, dir 97.4 %, median I² 73 %,
#           λGC 1.36 / 1.63

# ── Step 5 — RP specificity case study (~2.8 min warm; +~2 min for 447 MB download 1st run)
python benchmark/run_rp_case_study.py \
    --out-dir benchmark/rp_case_study/out
#   expect: 376 GWS all on chr6, 0 off-chr6, EYS OR 3.95,
#           347/29 risk/protective, median I² 12.9, λGC 1.06 / 1.03
```

Then open the JSON/CSV artifact for each step and check the fields
against the **Claim index** above — every headline number is a named
field (`ldl_summary.json`, `rp_summary.json`, `benchmark_summary.json`).
The per-step sections below give the full expected tables and explain
what each check proves.

---

## 1. Unit tests — engine correctness (§2.6.1)

Fastest check. Verifies the meta-analysis engine against
hand-calculated reference values for a three-study synthetic set.

```bash
pytest tests/ -q
```

**Expected result**: 84 / 84 pass in ~2 seconds. If any test fails,
stop and open an issue — the numeric agreement claimed in the thesis
is falsified.

---

## 2. Engine vs R `metafor` — Tier 1 synthetic validation (§3.2.1)

Compares the engine to the reference R package (`metafor` 4.8.0)
across 3,847 synthetic variants covering the full range of study
counts (k = 2 – 8) and heterogeneity regimes.

### Prerequisite: R + `metafor` (this is the only step that needs R)

Every other step is Python-only, so you can skip this step entirely if
you would rather not install R.

**First, check whether you already have them** — run these two lines:

```bash
Rscript --version                                            # prints an R version, or "command not found"
Rscript -e 'cat(as.character(packageVersion("metafor")))'    # prints e.g. 4.8.0, or an error
```

If the first prints an R version and the second prints a `metafor`
version (any recent version works; the thesis used 4.8.0), you are
ready — skip to **How to run** below.

**Only if either command failed, install them (one time):**

```bash
# macOS (Homebrew — https://brew.sh):
brew install r
Rscript -e 'install.packages("metafor", repos="https://cloud.r-project.org")'

# Debian / Ubuntu Linux:
sudo apt-get update && sudo apt-get install -y r-base
Rscript -e 'install.packages("metafor", repos="https://cloud.r-project.org")'
```

The `install.packages(...)` line downloads `metafor` from CRAN (~1–2
min, one time). Re-run the two check commands above to confirm, then
continue.

**How to run** (three steps, from the repo root):

```bash
# 1. Generate the 3,847-variant synthetic per-study table (seed = 20260606).
python tests/validation/generate_synthetic_metafor_set.py

# 2. Run the shipped engine on that table.
python tests/validation/run_engine_on_synthetic.py \
    --per_study tests/validation/synthetic/synthetic_per_study_long.csv \
    --out       tests/validation/synthetic/engine_output_synthetic.csv

# 3. Cross-validate against metafor and print the summary.
Rscript tests/validation/metafor_validate.R \
    --per_study tests/validation/synthetic/synthetic_per_study_long.csv \
    --engine    tests/validation/synthetic/engine_output_synthetic.csv \
    --out       tests/validation/synthetic/metafor_comparison_synthetic.csv \
    --tier      synthetic
```

The synthetic set is a k × I² × effect-regime grid
(k ∈ {2, 3, 4, 5, 6, 8}, target I² ∈ {0, 0.02, 0.08, 0.30},
μ ∈ {0, 0.03, 0.10, 0.25}, 40 reps per cell) plus 7 adversarial edge
cases (k=2 identical, opposite signs, one-study dominated, extreme
z, near-null, wide-SE spread, big-k high-heterogeneity).
Requires R with `metafor` installed.

**Expected max absolute differences (from Table 3.2.1)**:

| Statistic | Max \|Δ\| |
|---|---|
| `beta_fixed` | 3.11 × 10⁻¹⁵ |
| `se_fixed` | 9.37 × 10⁻¹⁷ |
| `p_fixed` | 9.08 × 10⁻¹⁴ |
| `beta_random` | 1.40 × 10⁻¹⁵ |
| `se_random` | 3.41 × 10⁻¹⁵ |
| `p_random` | 2.01 × 10⁻¹⁴ |
| `q_stat` | 4.82 × 10⁻¹¹ |
| `tau_squared` | 1.24 × 10⁻¹⁴ |
| `i_squared` | 1.64 × 10⁻¹⁴ |

All statistics agree with `metafor` to within machine precision (any
deviation is at the ULP level of double-precision floating point).
Per-variant deltas and a k-stratified summary are written to
`tests/validation/synthetic/metafor_comparison_synthetic.csv` and
`_SUMMARY.csv`.

---

## 3. Gout benchmark — Tier 2 end-to-end pipeline (§3.2.2)

Reproduces a published genome-scale meta-analysis: FinnGen R12
GOUT_STRICT plus pan-UKBB gout. This is the strongest correctness
claim in the thesis.

### Inputs

Three harmonized `.h.tsv.gz` files, all derived from a **single
primary source** whose exact release is pinned below (URL, date, and
an MD5 recorded in `provenance.json` at preprocess time), so a reader
can fetch the byte-identical file: the FinnGen R12 + pan-UKBB combined
meta-analysis output at

```
https://storage.googleapis.com/finngen-public-data-r12/meta_analysis/ukbb/summary_stats/finngen_R12_GOUT_STRICT_meta_out.tsv.gz
```

(~1.77 GB, GRCh38, forward strand, released 2024-10-22). This one
file contains, per variant, the FinnGen single-cohort estimate
(`FINNGEN_*` columns), the pan-UKBB single-cohort estimate
(`UKBB_*` columns), and the IVW meta result (`all_inv_var_meta_*`
columns). The three benchmark inputs are column-wise slices of it.

**Preprocess** (one command):

```bash
python benchmark/preprocess_gout_inputs.py \
    --out-dir benchmark/gout_inputs
```

This downloads the 1.77 GB primary source (cached under
`.cache/gout/`), splits it into `input_finngen.h.tsv.gz`,
`input_ukbb.h.tsv.gz`, and `expected_meta.h.tsv.gz` under
`benchmark/gout_inputs/`, and writes a `provenance.json` recording
the source URL, MD5, and per-file row counts. Both directories are
git-ignored. Pass a different `--out-dir` (e.g. an external drive)
if you prefer to keep the split files elsewhere.

Regenerated files are byte-content-identical to the appendix at
every canonical locus and match the manuscript's variant counts
exactly (FinnGen 21,294,541 raw, pan-UKBB 23,731,281 raw, reference
32,353,705 raw). File sizes match the appendix to within ~40 KB of
gzip-metadata noise.

Build (GRCh38) and forward-strand correctness are guaranteed by the
FinnGen R12 release convention and are validated retrospectively by
the 99.9638 % direction concordance and Pearson r = 0.999997 that
the benchmark produces against the meta slice.

### Run

```bash
python benchmark/run_benchmark.py \
    --inputs-dir benchmark/gout_inputs \
    --out-dir    benchmark/gout_output
```

The two-pass pipeline (chunk + align + meta) runs in ~2.2 min on an
M1 laptop when the inputs are on local SSD; peak memory 4.0 GB
(`benchmark_summary.json → memory_gb.pipeline_peak`). Reading the
inputs from an external HDD adds ~1 min. The harness additionally
loads the reference file and joins engine ↔ reference; that step is
not part of the shipped pipeline and adds a further ~1.5 min (and
`--full-accounting` adds ~2–3 min more). Timings are hardware-
dependent; the memory and accuracy figures are not.

The command writes three artifacts into `--out-dir`:
`benchmark_summary.json`, `variant_accounting.csv`,
`canonical_loci.csv` (a fourth, `variant_accounting_full.csv`, is
added with `--full-accounting`; see below). The most recent reference
outputs from this run are checked in at `benchmark/gout_output/` so a
reviewer can inspect the expected result without downloading the
~1.9 GB of inputs.

### UI-driven alternative

If you prefer to reproduce this via the app rather than the CLI:

```bash
streamlit run src/gwas_meta/app.py
```

Navigate through Steps 3 → 6, picking the same two input files via
Step 3's "Local folder path" tab. This is the mode where the k = 2
guardrail banner is visible at the top of Step 5 (see below).

### Expected results (Table 3.2.2a — concordance vs published)

| Metric | All variants | Reference-significant |
|---|---|---|
| Pearson r (β) | 0.999997 | 0.999999 |
| Pearson r (−log10 p) | 1.0000 | — |
| Direction concordance | 99.9638 % | 100.0000 % |
| Mean \|Δβ\| | 4.26 × 10⁻⁵ | 2.76 × 10⁻⁴ |
| Max \|Δβ\| | 1.70 × 10⁻³ | 1.05 × 10⁻³ |
| GWS variants (engine / ref / shared) | 3,184 / 3,185 / 3,182 | — |

### Expected canonical loci (Table 3.2.2c)

| Locus | Lead rsID | β (engine) | p (engine) |
|---|---|---|---|
| SLC2A9 | rs6449137 | −0.4417 | 3.8 × 10⁻¹²⁷ |
| ABCG2 Q141K | rs2231142 | +0.6097 | 2.8 × 10⁻¹⁴⁸ |
| GCKR | rs1260326 | −0.1499 | 4.3 × 10⁻²¹ |
| SLC22A11 / SLC22A12 | rs2164495 | −0.1054 | 6.1 × 10⁻¹¹ |

(SLC22A11 and SLC22A12 share the same lead variant.)

### Variant accounting (Table 3.2.2b)

| Study | Raw rows | After QC |
|---|---|---|
| FinnGen | 21,294,541 | 8,685,374 |
| pan-UKBB | 23,731,281 | 8,603,008 |
| Aligned overlap | — | 7,603,652 |

### Full variant accounting — the §3.2.2 gap table (Table 3.2.2)

Add `--full-accounting` to the command above to also partition the
reference-only variants (the ~24.6 M canonical keys present in the
published reference but not in the engine output) into the four
mutually-exclusive categories reported in §3.2.2:

```bash
python benchmark/run_benchmark.py \
    --inputs-dir benchmark/gout_inputs \
    --out-dir    benchmark/gout_output \
    --full-accounting
```

This streams the two raw inputs and the reference a second time and
works one chromosome at a time (raw/reference key universes include
indels and multi-allelic sites, so variants the engine drops as
indel/multi-allelic surface under "removed by QC"; the engine and
post-QC key sets stay SNP-only). It adds ~2–3 min and writes a fourth
artifact, `variant_accounting_full.csv` (one row per chromosome), plus
a `table_3_2_2_full_accounting` block in `benchmark_summary.json`.

| Category | Count | % of gap |
|---|---|---|
| Reference-only gap (unique keys) | 24,638,083 | — |
| In raw input, removed by QC | 22,557,005 | 91.55 % |
| Single-cohort (dropped by min_study_count = 2) | 2,081,078 | 8.45 % |
| In neither raw input | 0 | 0.00 % |
| In both post-QC, missed (engine bug if > 0) | 0 | 0.00 % |
| Engine-only (must be 0) | 0 | — |

The two zero rows are the load-bearing checks: **missed = 0** confirms
the engine drops no variant that passed QC in both cohorts, verified on
every chromosome, and **engine-only = 0** confirms every engine output
variant is present in the reference.

### Guardrails on this run

- **k = 2 banner** at top of Step 5 (UI only) — expected, this run
  has k = 2. Fixed-effects results are unaffected; the random-effects
  τ² / p_random should not be interpreted at face value.
- Sample-overlap check: should stay silent (FinnGen ≠ UKBB).
- Trait-mismatch check: should stay silent (both target gout).
- Genome-build check: should stay silent (both are GRCh38).
- LOO: blank on every row (k < 3).

---

## 4. LDL cholesterol case study (§3.3)

Recovery of established LDL loci from cohorts disjoint from the
GLGC-2013 anchor study. Two inputs (k = 2, ~5 minute run):

- UK Biobank biomarker GWAS — **GCST90019512**
- Million Veteran Program lipids GWAS — **GCST90475420** (Verma et al. 2024, *Science*)

### Run (single command)

```bash
# Reproduces the manuscript's headline LDL numbers (post-Cochran-Q).
python benchmark/run_ldl_case_study.py \
    --out-dir benchmark/ldl_case_study/out \
    --q-filter 1e-6
```

Downloads both inputs from GWAS Catalog FTP (cached to
`.cache/summary_stats/`), runs load → align → IVW meta, applies the
Cochran-Q heterogeneity filter (`p_Q < 1e-6`, matching the UI and the
manuscript), scores recovery against 30 expected loci
(`benchmark/ldl_case_study/expected_loci.csv`), and writes
`meta_results.csv`, `ldl_recovery.csv`, `ldl_summary.json` into
`--out-dir`. Wall-clock ~3.3 min warm cache on an M1 laptop (the two-pass
pipeline itself is ~1.7 min: chunk 44 s + align/meta 56 s); the first-run
842 MB download adds ~4–6 min. Timings are hardware- and network-dependent;
the recovery, GWS, λGC, and concordance figures are not.

**Pre-Q vs post-Q (important).** The manuscript mixes the two: the
per-locus recovery table (Table 3.3.1) and the heterogeneity discussion
(median I² = 73 %, direction concordance 97.4 %) are computed on the
**raw meta**, while the headline GWS / suggestive / het-removed counts
(Table 3.3.2) use the **post-Q** filter. The wrapper mirrors this:
`ldl_recovery.csv`, `direction_concordance_gws`, and
`median_i2_gws_percent` are always raw-meta values (so LPA shows 1,120
regardless of `--q-filter`), while `--q-filter 1e-6` sets the reported
`gws_variants` (40,645), `suggestive_variants` (73,965), and
`het_variants_removed` (14,728). `--q-filter` defaults to `0`, which
leaves those three counts at their pre-Q values (47,402 / 81,666 / 0).
Use `--q-filter 1e-6` to match the manuscript's headline counts.

### Reference provenance (for reviewer diff)

The provenance JSON from the manuscript's LDL run is checked in at
`benchmark/ldl_case_study/provenance_reference.json`. A reviewer's
own run auto-writes its own provenance record; diffing the two on
the study-selection and search-criteria sections tells them whether
the reproduction matches the manuscript's setup.

### Expected results (from Tables 3.3.1 / 3.3.2)

- **6,840,875** aligned / meta-analysed variants (`aligned_variants`).
- **30 / 30** expected loci recovered (15 canonical + 15 GLGC-2013 new).
- **40,645** genome-wide-significant variants post-Q (`p_fixed < 5 × 10⁻⁸`
  after `--q-filter 1e-6`); **14,728** removed for extreme heterogeneity;
  **73,965** suggestive (`p < 1e-5`). Without `--q-filter` these are the
  pre-Q values (47,402 / — / 81,666).
- LPA window contains **1,120** genome-wide-significant variants in the
  raw meta (`ldl_recovery.csv`; independent of `--q-filter`).
- Direction concordance across the raw GWS set: **97.4 %**
  (`direction_concordance_gws`); median I² across the raw GWS set:
  **73 %** (`median_i2_gws_percent`).
- λGC (UK Biobank) ≈ **1.36**; λGC (MVP) ≈ **1.63**.
- Weakest recovery (ANXA9-CERS2): a single variant at p ≈ 5 × 10⁻⁹.
- APOE random-effects p-value ≈ 1.7 × 10⁻⁵ (large I² driven by
  UKB / MVP scale difference — UKB is on biomarker units, MVP is
  inverse-normal-transformed).

### What to look at in the output

- `ldl_recovery.csv` should read `recovered=True` on all 26 rows.
- `ldl_summary.json → direction_concordance_gws` should be ≈ 0.974
  (both cohorts agree in direction at 97.4 % of raw GWS variants). A
  value near 0.5 would indicate a systematic allele-orientation flip —
  the red flag this check exists to catch.

The biological interpretation of the recovered loci (what each gene
does, why the disjoint-from-GLGC-2013 selection matters, how APOE
splits under fixed vs random effects) lives in manuscript §3.3, not
here.

---

## 5. Retinitis pigmentosa specificity case study (§3.4)

Tests the converse property: the tool should recover the single locus
where a common-variant RP signal genuinely exists (EYS on chr6) and
stay silent across the ~80 Mendelian RP genes.

Two inputs (k = 2, ~2.8 min warm run):

- Nishiguchi 2021 stage 1 — **GCST90011892** (432 cases / 603 controls)
- Nishiguchi 2021 stage 2 — **GCST90011893** (208 cases / 287 controls)

Combined 640 cases / 890 controls, Japanese ancestry. Both from
Nishiguchi et al. 2021 (Commun Biol 4:140).

### Run (single command)

```bash
python benchmark/run_rp_case_study.py --out-dir benchmark/rp_case_study/out
```

Downloads both inputs from GWAS Catalog FTP (cached to
`.cache/summary_stats/`), then runs the **same disk-backed two-pass
pipeline the Streamlit tool uses** (`chunk_studies_to_disk` →
per-chromosome `load_chromosome_chunks` → `align_studies` →
`run_meta_analysis_batch`) — i.e. the wrapper is the tool minus the UI,
sharing both the QC code path and the `config/settings.yaml` settings
(`valid_hm_codes`, `min_study_count`, `significance_threshold`, recorded
under `config_resolved` in the summary). It scores EYS locus recovery and
chromosome-wise GWS specificity, and writes `meta_results.csv`,
`rp_recovery.csv`, `rp_summary.json` into `--out-dir`. Wall-clock ~2.8 min
warm cache on an M1 laptop (the two-pass pipeline itself is ~1.5 min:
chunk 35 s + align/meta 56 s); the first-run 447 MB download adds ~2 min.
Timings are hardware- and network-dependent; the GWS-specificity, EYS
lead, and λGC figures are not.

### Reference provenance

`benchmark/rp_case_study/provenance_reference.json` holds the
manuscript's original UI-driven run provenance for reviewer diff.

### Expected results (from §3.4, reproduced 2026-07-02)

- **6,867,143** variants meta-analysed genome-wide (exact match).
- **1** genome-wide-significant locus: EYS on chr6.
- Lead variant **chr6:64,990,459:A:G** — OR = **3.948**,
  P = **1.173 × 10⁻¹³** (published: OR 3.95, P 1.18 × 10⁻¹³).
- **376** GWS variants total, **all on chr6**. Zero GWS on any other
  chromosome — that is the specificity claim in the thesis.

### Sanity note

Nishiguchi's two stages share a PubMed ID (both are from Nishiguchi
2021 Commun Biol 4:140), and the tool's shared-cohort finder correctly
flags this at Step 5 / in `rp_summary.json`. Sharing a paper is not
the same as sharing samples: Nishiguchi 2021 is a classical two-stage
GWAS in which the discovery (432 cases / 603 controls) and replication
(208 / 287) cohorts are independently recruited, and the sample sizes
are exactly additive to the reported combined totals (640 / 890). The
flag is therefore a fact worth verifying rather than a defect; reading
the source paper's methods resolves it. This is the intended
behaviour of the finder — it surfaces evidence worth checking and
leaves the resolution to the user, which is exactly what happened
here.

---

## 6. Where to find run artifacts

Every run writes:

- **`meta_results.csv`** — the full per-variant output, one row per
  variant with columns `variant_id, beta_fixed, se_fixed, z_fixed,
  p_fixed, beta_random, se_random, z_random, p_random, q_stat,
  i_squared, tau_squared, n_studies`. This is the raw (pre-Q) meta.
- **`logs/run_<timestamp>.log`** — full run log including any
  guardrail warnings (k = 2, sample overlap, trait mismatch, genome
  build sentinel probe).
- **`results/provenance_<timestamp>.json`** — machine-readable log of
  every user choice, filter setting, and per-stage timing for the
  run. This is the reproducibility anchor: any run described in the
  thesis can be identified by its provenance JSON.
- **`results/meta_report_<timestamp>.pdf`** — PDF export of the top
  hits, search criteria, and study list.

---

## 7. Sanity checks the tool prints and what they mean

Warnings that may appear at Step 5 and their interpretation:

- **"Only 2 studies included (k = 2)"** — fires whenever exactly two
  studies are meta-analysed. Fixed-effects results are unaffected;
  the random-effects τ² / p_random should not be interpreted at face
  value. This warning is expected in the gout, LDL, and RP case
  studies.
- **"Possible sample overlap detected"** — fires when two studies'
  sample descriptions both mention a well-known biobank (UK Biobank,
  FinnGen, 23andMe, etc.). None of the case studies in the thesis
  trigger this. If you construct a run that does, the warning links
  to standard corrections (MetaSubtract, bivariate LDSC) that are
  out of scope for this tool.
- **"Trait mismatch detected"** — fires when studies do not share any
  EFO trait. Should stay silent on all case studies.
- **"Genome-build mismatch"** — fires when the tool's sentinel-SNP
  probe finds GRCh37 positions instead of GRCh38. All GWAS Catalog
  harmonized files should be GRCh38, so this should stay silent on
  Catalog-fetched runs. On user-uploaded files it acts as a
  guardrail.

---

## Reporting problems

If any number in this document does not reproduce, please open an
issue at https://github.com/pjcodebio/GWAMA_tool/issues with:

- Which section (1–5).
- The specific number that disagreed and what you got instead.
- The run's `provenance_<timestamp>.json` (auto-generated at
  `results/`).
- Your Python version and OS.
