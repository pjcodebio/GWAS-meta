# gwas-meta

An open-source web tool that automates end-to-end GWAS meta-analysis
while preserving expert oversight at key decision points. Built as
the code artifact of a master's thesis.

The tool integrates programmatic access to the NHGRI-EBI GWAS
Catalog, automated data harmonization, per-study QC, and fixed- and
random-effects meta-analysis in a Streamlit wizard. A two-pass
disk-based architecture handles genome-scale runs (tens of millions
of variants across 50+ studies) on ~4 GB RAM.

## Reproducing the thesis results

See [`REPRODUCE.md`](./REPRODUCE.md) for a step-by-step walkthrough of
every empirical result in the thesis (unit tests, engine validation
against R `metafor`, the full-genome gout benchmark, and the LDL /
retinitis pigmentosa case studies) with expected numbers and the
GCST inputs for each run.

## Install

```bash
git clone https://github.com/pjcodebio/GWAS-meta.git
cd GWAS-meta
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Requirements: Python ≥ 3.11, ~4 GB RAM for a full-genome run.

## Run

```bash
streamlit run src/gwas_meta/app.py
```

Opens the wizard in your browser (default: http://localhost:8501).

An LLM API key (`.env` from `.env.example`) is required for the default
guided workflow: Step 2 parses your free-text research question into GWAS
Catalog search criteria via the LLM, so that step will not run without one
(the key is also used for the optional results summary in Step 6). To run
without a key, use the manual **"Upload datasets yourself"** path — supply
your own summary-statistics files and select studies directly.

## Tests

```bash
pytest tests/ -q
```
