# R laGP Benchmark: Borehole Example
#
# Classic 8D borehole function benchmark for comparing R laGP performance
# against Julia laGP.jl. The borehole function models water flow through
# a borehole with 8 input parameters.
#
# Reference: Surrogates: Gaussian Process Modeling, Design and Optimization
# by Robert Gramacy (Appendix A)
#
# Run with: Rscript benchmarks/benchmark_borehole.R

library(laGP)
library(lhs)

cat(rep("=", 70), "\n", sep = "")
cat("Borehole Example: R laGP Benchmark\n")
cat(rep("=", 70), "\n", sep = "")

set.seed(42)

# ============================================================================
# Borehole Function
# ============================================================================

#' Borehole function - models water flow through a borehole.
#'
#' All 8 inputs are coded to [0, 1] and transformed internally to natural units.
#'
#' @param x Vector of 8 inputs in [0,1]
#' @return Water flow rate (m^3/yr)
borehole <- function(x) {
  # Transform coded [0,1] inputs to natural units
  rw <- x[1] * (0.15 - 0.05) + 0.05
  r  <- x[2] * (50000 - 100) + 100
  Tu <- x[3] * (115600 - 63070) + 63070
  Hu <- x[4] * (1110 - 990) + 990
  Tl <- x[5] * (116 - 63.1) + 63.1
  Hl <- x[6] * (820 - 700) + 700
  L  <- x[7] * (1680 - 1120) + 1120
  Kw <- x[8] * (12045 - 9855) + 9855

  # Borehole flow equation
  m1 <- 2 * pi * Tu * (Hu - Hl)
  m2 <- log(r / rw)
  m3 <- 1 + 2 * L * Tu / (m2 * rw^2 * Kw) + Tu / Tl

  return(m1 / m2 / m3)
}

# ============================================================================
# Configuration
# ============================================================================

n_train <- 10000
n_test <- 10000
noise_sd <- 1.0

# aGP parameters
agp_start <- 6
agp_end <- 50
agp_close <- 1000

cat("\nConfiguration:\n")
cat("  Training points:", n_train, "(8D)\n")
cat("  Test points:", n_test, "\n")
cat("  Noise: sd =", noise_sd, "\n")
cat("  aGP parameters: start =", agp_start, ", end =", agp_end, ", close =", agp_close, "\n")
cat("  OMP threads:", parallel::detectCores(), "\n\n")

# ============================================================================
# Data Generation
# ============================================================================

cat("Generating data...\n")
t_start <- Sys.time()

X_train <- randomLHS(n_train, 8)
X_test <- randomLHS(n_test, 8)

Z_train_true <- apply(X_train, 1, borehole)
Z_test_true <- apply(X_test, 1, borehole)

# Add noise to training data
Z_train <- Z_train_true + noise_sd * rnorm(n_train)

t_data <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))

cat(sprintf("  Data generation: %.2f seconds\n", t_data))
cat(sprintf("  Training response range: [%.2f, %.2f]\n", min(Z_train), max(Z_train)))
cat(sprintf("  Test response range: [%.2f, %.2f]\n", min(Z_test_true), max(Z_test_true)))
cat("\n")

# Get hyperparameter defaults
da <- darg(NULL, X_train)
ga <- garg(list(mle = TRUE), Z_train)

# ============================================================================
# PART 1: Full Separable GP on subset (n=1000)
# ============================================================================

cat(rep("-", 70), "\n", sep = "")
cat("PART 1: Full Separable GP (n=1000 subset, MLE)\n")
cat(rep("-", 70), "\n", sep = "")

n_subset <- 1000
X_subset <- X_train[1:n_subset, ]
Z_subset <- Z_train[1:n_subset]

cat("  Fitting separable GP with MLE...")
t_start <- Sys.time()

gp_sep <- newGPsep(X_subset, Z_subset, d = da$start, g = ga$start, dK = TRUE)
mle_result <- jmleGPsep(gp_sep, drange = c(da$min, da$max), grange = c(ga$min, ga$max), verb = 0)
pred_sep <- predGPsep(gp_sep, X_test, lite = TRUE)

t_gpsep <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
cat(sprintf(" %.2f seconds\n", t_gpsep))

rmse_gpsep <- sqrt(mean((pred_sep$mean - Z_test_true)^2))
cat(sprintf("  RMSE: %.4f\n", rmse_gpsep))
cat(sprintf("  Final nugget g: %.6f\n", mle_result$g))
cat("  Final lengthscales d:", paste(round(mle_result[1:8], 4), collapse = ", "), "\n\n")

deleteGPsep(gp_sep)

# ============================================================================
# PART 2: aGP Isotropic (full data)
# ============================================================================

cat(rep("-", 70), "\n", sep = "")
cat(sprintf("PART 2: aGP Isotropic (n=%d, end=%d)\n", n_train, agp_end))
cat(rep("-", 70), "\n", sep = "")

cat("  Running aGP isotropic (ALC method)...")
t_start <- Sys.time()

pred_agp_iso <- aGP(X_train, Z_train, X_test,
                    start = agp_start, end = agp_end, close = agp_close,
                    d = list(start = da$start, max = da$max),
                    g = list(start = ga$start, max = ga$max),
                    method = "alc",
                    omp.threads = parallel::detectCores(),
                    verb = 0)

t_agp_iso <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
cat(sprintf(" %.2f seconds\n", t_agp_iso))

rmse_agp_iso <- sqrt(mean((pred_agp_iso$mean - Z_test_true)^2))
cat(sprintf("  RMSE: %.4f\n\n", rmse_agp_iso))

# ============================================================================
# PART 3: aGP Separable (full data)
# ============================================================================

cat(rep("-", 70), "\n", sep = "")
cat(sprintf("PART 3: aGP Separable (n=%d, end=%d)\n", n_train, agp_end))
cat(rep("-", 70), "\n", sep = "")

cat("  Running aGPsep (ALC method)...")
t_start <- Sys.time()

pred_agp_sep <- aGPsep(X_train, Z_train, X_test,
                       start = agp_start, end = agp_end, close = agp_close,
                       d = list(start = da$start, max = da$max),
                       g = list(start = ga$start, max = ga$max),
                       method = "alc",
                       omp.threads = parallel::detectCores(),
                       verb = 0)

t_agp_sep <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
cat(sprintf(" %.2f seconds\n", t_agp_sep))

rmse_agp_sep <- sqrt(mean((pred_agp_sep$mean - Z_test_true)^2))
cat(sprintf("  RMSE: %.4f\n\n", rmse_agp_sep))

# ============================================================================
# PART 4: aGP NN (nearest neighbor only, for speed comparison)
# ============================================================================

cat(rep("-", 70), "\n", sep = "")
cat(sprintf("PART 4: aGP NN (n=%d, end=%d)\n", n_train, agp_end))
cat(rep("-", 70), "\n", sep = "")

cat("  Running aGP NN (no acquisition)...")
t_start <- Sys.time()

pred_agp_nn <- aGP(X_train, Z_train, X_test,
                   start = agp_start, end = agp_end, close = agp_close,
                   d = list(start = da$start, max = da$max),
                   g = list(start = ga$start, max = ga$max),
                   method = "nn",
                   omp.threads = parallel::detectCores(),
                   verb = 0)

t_agp_nn <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))
cat(sprintf(" %.2f seconds\n", t_agp_nn))

rmse_agp_nn <- sqrt(mean((pred_agp_nn$mean - Z_test_true)^2))
cat(sprintf("  RMSE: %.4f\n\n", rmse_agp_nn))

# ============================================================================
# Results Summary
# ============================================================================

cat(rep("=", 70), "\n", sep = "")
cat("SUMMARY (for Julia comparison)\n")
cat(rep("=", 70), "\n\n", sep = "")

cat(sprintf("%-20s %12s %12s\n", "Method", "Time (s)", "RMSE"))
cat(rep("-", 46), "\n", sep = "")
cat(sprintf("%-20s %12.2f %12.4f\n", "Full GPsep (1000)", t_gpsep, rmse_gpsep))
cat(sprintf("%-20s %12.2f %12.4f\n", sprintf("aGP iso (%d)", n_train), t_agp_iso, rmse_agp_iso))
cat(sprintf("%-20s %12.2f %12.4f\n", sprintf("aGP sep (%d)", n_train), t_agp_sep, rmse_agp_sep))
cat(sprintf("%-20s %12.2f %12.4f\n", sprintf("aGP NN (%d)", n_train), t_agp_nn, rmse_agp_nn))
cat(rep("-", 46), "\n\n", sep = "")

cat("Notes:\n")
cat("  - Full GPsep runs on subset (n=1000) due to O(n^3) complexity\n")
cat(sprintf("  - aGP methods use full dataset (n=%d) with local approximations\n", n_train))
cat("  - Separable GP allows per-dimension lengthscales (ARD)\n")
cat("  - NN method skips acquisition function (fastest, but may be less accurate)\n\n")

cat("To compare with Julia laGP.jl:\n")
cat("  julia --project=. -t auto examples/borehole_example.jl\n")
