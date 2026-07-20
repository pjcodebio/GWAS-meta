#!/usr/bin/env Rscript
# =============================================================================
# metafor cross-validation of the gwas-meta engine (Tier 1 synthetic + Tier 2 chr22)
#
# For every variant in a long-format per-study table, this script computes the
# reference meta-analysis with metafor under the SAME specification the engine uses:
#     fixed-effects IVW         ->  rma(yi, vi, method = "FE")
#     random-effects DL, z-test ->  rma(yi, vi, method = "DL", test = "z")
# and compares NINE statistics against the engine's own output.
#
# Inputs (CSV):
#   --per_study   long table: variant_id, study_id, beta, standard_error  (+others ignored)
#   --engine      engine output, one row per variant, with the nine statistics (see COLS below)
#   --out         path for the per-variant comparison CSV
#   --tier        label written into the output ("synthetic" or "chr22_lungcancer")
#
# Usage:
#   Rscript metafor_validate.R \
#     --per_study synthetic/synthetic_per_study_long.csv \
#     --engine    synthetic/engine_output_synthetic.csv \
#     --out       synthetic/metafor_comparison_synthetic.csv \
#     --tier      synthetic
#
# Requires: install.packages("metafor")
# =============================================================================

suppressMessages({
  if (!requireNamespace("metafor", quietly = TRUE))
    stop("Install metafor: install.packages('metafor')")
  library(metafor)
})

# ---- tiny arg parser --------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag, default = NULL) {
  i <- which(args == flag)
  if (length(i) == 1 && i < length(args)) return(args[i + 1])
  default
}
per_study_path <- getarg("--per_study")
engine_path    <- getarg("--engine")
out_path       <- getarg("--out")
tier           <- getarg("--tier", "unspecified")
stopifnot(!is.null(per_study_path), !is.null(engine_path), !is.null(out_path))

# ---- engine output column names (EDIT HERE if your engine names differ) -----
# Map: canonical statistic -> column name in your engine's per-variant output CSV.
COLS <- list(
  beta_fe = "beta_fe",   se_fe = "se_fe",   z_fe = "z_fe",   p_fe = "p_fe",
  beta_re = "beta_re",   se_re = "se_re",   p_re = "p_re",
  Q       = "Q",         tau2  = "tau2",    I2    = "I2"
)
# Nine compared statistics: beta_fe, se_fe, p_fe, beta_re, se_re, p_re, Q, tau2, I2
# (z_fe optional; included if present.)

# ---- load -------------------------------------------------------------------
ps  <- read.csv(per_study_path, stringsAsFactors = FALSE)
eng <- read.csv(engine_path,    stringsAsFactors = FALSE)
stopifnot(all(c("variant_id","beta","standard_error") %in% names(ps)))

# log-space two-sided p from z (avoids underflow to 0)
p_from_z <- function(z) exp(log(2) + pnorm(abs(z), lower.tail = FALSE, log.p = TRUE))

vids <- unique(ps$variant_id)
n    <- length(vids)
message(sprintf("[%s] %d variants to validate against metafor ...", tier, n))

res <- vector("list", n)
pb_every <- max(1, floor(n / 20))

for (idx in seq_along(vids)) {
  vid <- vids[idx]
  sub <- ps[ps$variant_id == vid, ]
  yi  <- sub$beta
  vi  <- sub$standard_error^2
  k   <- nrow(sub)

  row <- list(variant_id = vid, k = k, tier = tier)

  # ---- metafor reference (FE + DL/z) ----
  fe <- tryCatch(rma(yi = yi, vi = vi, method = "FE"),               error = function(e) NULL)
  re <- tryCatch(rma(yi = yi, vi = vi, method = "DL", test = "z"),   error = function(e) NULL)

  if (!is.null(fe)) {
    row$mf_beta_fe <- as.numeric(fe$beta)
    row$mf_se_fe   <- fe$se
    row$mf_z_fe    <- fe$zval
    row$mf_p_fe    <- p_from_z(fe$zval)   # recompute in log-space for tiny p
  }
  if (!is.null(re)) {
    row$mf_beta_re <- as.numeric(re$beta)
    row$mf_se_re   <- re$se
    row$mf_p_re    <- p_from_z(re$zval)
    row$mf_Q       <- re$QE
    row$mf_tau2    <- re$tau2
    row$mf_I2      <- re$I2 / 100        # metafor reports I2 as %, engine likely 0-1 -> normalise; ADJUST if engine uses %
  }

  res[[idx]] <- row
  if (idx %% pb_every == 0) message(sprintf("  ... %d / %d", idx, n))
}

mf <- do.call(rbind, lapply(res, function(r) as.data.frame(r, stringsAsFactors = FALSE)))

# ---- join engine output and compute diffs ----------------------------------
need_eng <- unlist(COLS)
miss <- setdiff(c("variant_id", need_eng), names(eng))
if (length(miss) > 0)
  warning("Engine CSV missing columns (edit COLS map): ", paste(miss, collapse = ", "))

m <- merge(mf, eng, by = "variant_id", all.x = TRUE)

# pairs: (metafor column, engine column, label)
pairs <- list(
  c("mf_beta_fe", COLS$beta_fe, "beta_fe"),
  c("mf_se_fe",   COLS$se_fe,   "se_fe"),
  c("mf_p_fe",    COLS$p_fe,    "p_fe"),
  c("mf_beta_re", COLS$beta_re, "beta_re"),
  c("mf_se_re",   COLS$se_re,   "se_re"),
  c("mf_p_re",    COLS$p_re,    "p_re"),
  c("mf_Q",       COLS$Q,       "Q"),
  c("mf_tau2",    COLS$tau2,    "tau2"),
  c("mf_I2",      COLS$I2,      "I2")
)

abs_diff <- function(a, b) abs(a - b)
rel_diff <- function(a, b) ifelse(pmax(abs(a), abs(b)) > 0,
                                  abs(a - b) / pmax(abs(a), abs(b)), 0)

for (p in pairs) {
  mfc <- p[1]; ec <- p[2]; lab <- p[3]
  if (mfc %in% names(m) && ec %in% names(m)) {
    m[[paste0("absdiff_", lab)]] <- abs_diff(m[[mfc]], m[[ec]])
    m[[paste0("reldiff_", lab)]] <- rel_diff(m[[mfc]], m[[ec]])
  }
}

# ---- write per-variant comparison ------------------------------------------
dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
write.csv(m, out_path, row.names = FALSE)
message("Wrote per-variant comparison: ", out_path)

# ---- summary: max abs/rel diff per statistic, overall and by k -------------
summ_path <- sub("\\.csv$", "_SUMMARY.csv", out_path)
labs <- sapply(pairs, `[`, 3)
overall <- do.call(rbind, lapply(labs, function(lab) {
  ac <- paste0("absdiff_", lab); rc <- paste0("reldiff_", lab)
  if (!ac %in% names(m)) return(NULL)
  data.frame(statistic = lab, k = "ALL",
             n = sum(is.finite(m[[ac]])),
             max_abs_diff = max(m[[ac]], na.rm = TRUE),
             max_rel_diff = max(m[[rc]], na.rm = TRUE))
}))
by_k <- do.call(rbind, lapply(sort(unique(m$k)), function(kk) {
  sk <- m[m$k == kk, ]
  do.call(rbind, lapply(labs, function(lab) {
    ac <- paste0("absdiff_", lab); rc <- paste0("reldiff_", lab)
    if (!ac %in% names(sk)) return(NULL)
    data.frame(statistic = lab, k = as.character(kk),
               n = sum(is.finite(sk[[ac]])),
               max_abs_diff = max(sk[[ac]], na.rm = TRUE),
               max_rel_diff = max(sk[[rc]], na.rm = TRUE))
  }))
}))
summary_tbl <- rbind(overall, by_k)
write.csv(summary_tbl, summ_path, row.names = FALSE)
message("Wrote summary: ", summ_path)

cat("\n==== max ABSOLUTE difference per statistic (ALL k) ====\n")
print(overall[, c("statistic","n","max_abs_diff","max_rel_diff")], row.names = FALSE)
cat("\nInterpretation: machine-precision agreement is ~1e-12 or smaller for beta/se/Q/tau2/I2.\n")
cat("p-values: compare on the log scale if any are extremely small (see note in 3.2.1).\n")
