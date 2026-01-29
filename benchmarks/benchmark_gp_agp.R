#!/usr/bin/env Rscript
# Benchmark: Full GP vs aGP (ALC/NN/MSPE) using R laGP
#
# This script is designed to compare performance across:
#   - Full GP
#   - aGP with ALC
#   - aGP with NN
#   - aGP with MSPE
#
# Environment variables (optional):
#   LAGP_N_TRAIN="100,500,1000,2000,5000"   # comma-separated list
#   LAGP_N_TEST_GRID="10"          # grid size per dimension
#   LAGP_AGP_START="6"
#   LAGP_AGP_ENDPT="30"
#   LAGP_CLOSE="1000"              # candidate set size for ALC/MSPE
#   LAGP_REPS="1"                  # repeats per method (median time)
#   LAGP_SEED="42"
#   LAGP_OMP_THREADS="<int>"        # defaults to detectCores()
#   LAGP_OUT="benchmarks/results_r.csv"
#   LAGP_SHARED_DIR="benchmarks/shared_data" # load shared CSVs if set

require_or_install <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

require_or_install("laGP")
require_or_install("lhs")

parse_int_list <- function(x, default) {
  if (is.na(x) || x == "") return(default)
  parts <- unlist(strsplit(x, ","))
  vals <- as.integer(trimws(parts))
  vals <- vals[!is.na(vals)]
  if (length(vals) == 0) default else vals
}

parse_int <- function(x, default) {
  if (is.na(x) || x == "") return(default)
  v <- suppressWarnings(as.integer(x))
  if (is.na(v)) default else v
}

parse_num <- function(x, default) {
  if (is.na(x) || x == "") return(default)
  v <- suppressWarnings(as.numeric(x))
  if (is.na(v)) default else v
}

seed <- parse_int(Sys.getenv("LAGP_SEED", "42"), 42)
set.seed(seed)

# ============================================================================
# Test function: 2D sinusoidal
# ============================================================================

test_function <- function(x1, x2) {
  sin(2 * pi * x1) * cos(2 * pi * x2) + 0.5 * sin(4 * pi * x1)
}

# ============================================================================
# Data generation
# ============================================================================

generate_lhs_data <- function(n, dim = 2) {
  # Use fast random LHS to avoid slow optimization for large n
  X <- randomLHS(n, dim)
  Z <- apply(X, 1, function(row) test_function(row[1], row[2]))
  list(X = X, Z = Z)
}

generate_test_grid <- function(nx, ny) {
  x <- seq(0, 1, length.out = nx)
  y <- seq(0, 1, length.out = ny)
  grid <- expand.grid(x = x, y = y)
  as.matrix(grid)
}

load_shared_data <- function(n, shared_dir) {
  x_path <- file.path(shared_dir, sprintf("train_X_%d.csv", n))
  z_path <- file.path(shared_dir, sprintf("train_Z_%d.csv", n))
  xx_path <- file.path(shared_dir, "test_XX.csv")

  if (!file.exists(x_path) || !file.exists(z_path) || !file.exists(xx_path)) {
    stop("Shared data missing. Expected files: ", x_path, ", ", z_path, ", ", xx_path)
  }

  X <- as.matrix(read.csv(x_path, header = FALSE))
  Z <- as.vector(read.csv(z_path, header = FALSE)[, 1])
  XX <- as.matrix(read.csv(xx_path, header = FALSE))
  list(X = X, Z = Z, XX = XX)
}

# ============================================================================
# Benchmark helpers
# ============================================================================

benchmark_full_gp <- function(X, Z, XX, d, g) {
  gp <- newGP(X, Z, d = d, g = g, dK = TRUE)
  pred <- predGP(gp, XX, lite = TRUE)
  deleteGP(gp)
  pred$mean
}

benchmark_agp <- function(X, Z, XX, d_list, g_list, method,
                          start = 6, endpt = 50, close = 1000, omp_threads = 1) {
  result <- aGP(X, Z, XX,
                start = start, end = endpt, close = close,
                d = d_list,
                g = g_list,
                method = method,
                omp.threads = omp_threads,
                verb = 0)
  result$mean
}

time_reps <- function(fun, reps) {
  times <- numeric(reps)
  result <- NULL
  for (i in seq_len(reps)) {
    t <- system.time({ result <- fun() })
    times[i] <- t[["elapsed"]]
  }
  list(time = median(times), result = result)
}

# ============================================================================
# Main benchmark
# ============================================================================

run_benchmark <- function() {
  n_train_sizes <- parse_int_list(Sys.getenv("LAGP_N_TRAIN", ""), c(100, 500, 1000, 2000, 5000))
  n_test_grid <- parse_int(Sys.getenv("LAGP_N_TEST_GRID", "10"), 10)
  agp_start <- parse_int(Sys.getenv("LAGP_AGP_START", "6"), 6)
  agp_endpt <- parse_int(Sys.getenv("LAGP_AGP_ENDPT", "30"), 30)
  close_n <- parse_int(Sys.getenv("LAGP_CLOSE", "1000"), 1000)
  reps <- parse_int(Sys.getenv("LAGP_REPS", "1"), 1)
  omp_threads <- parse_int(Sys.getenv("LAGP_OMP_THREADS", ""), parallel::detectCores())
  if (is.na(omp_threads) || omp_threads < 1) {
    omp_threads <- 1
  }
  out_path <- Sys.getenv("LAGP_OUT", "")
  shared_dir <- Sys.getenv("LAGP_SHARED_DIR", "")

  cat(strrep("=", 70), "\n")
  cat("Benchmark: Full GP vs aGP (ALC/NN/MSPE) - R laGP\n")
  cat(strrep("=", 70), "\n\n")

  XX <- generate_test_grid(n_test_grid, n_test_grid)
  n_test <- nrow(XX)
  true_vals <- apply(XX, 1, function(row) test_function(row[1], row[2]))

  cat("Configuration:\n")
  cat("  Training sizes:", paste(n_train_sizes, collapse = ", "), "\n")
  cat("  Test points:", n_test, "(", n_test_grid, "x", n_test_grid, "grid )\n")
  cat("  aGP parameters: start =", agp_start, ", endpt =", agp_endpt, ", close =", close_n, "\n")
  cat("  Reps per method:", reps, "\n")
  cat("  OMP threads:", omp_threads, "\n\n")

  results <- data.frame(
    n = integer(),
    full = numeric(),
    alc = numeric(),
    nn = numeric(),
    mspe = numeric(),
    rmse_full = numeric(),
    rmse_alc = numeric(),
    rmse_nn = numeric(),
    rmse_mspe = numeric(),
    stringsAsFactors = FALSE
  )

  for (n_train in n_train_sizes) {
    cat(strrep("-", 50), "\n")
    cat("n_train =", n_train, "\n")
    cat(strrep("-", 50), "\n")

    if (shared_dir != "") {
      data <- load_shared_data(n_train, shared_dir)
      X <- data$X
      Z <- data$Z
      XX <- data$XX
      n_test <- nrow(XX)
      true_vals <- apply(XX, 1, function(row) test_function(row[1], row[2]))
    } else {
      data <- generate_lhs_data(n_train)
      X <- data$X
      Z <- data$Z
    }

    d_args <- darg(d = NULL, X = X)
    g_args <- garg(g = list(mle = TRUE), y = Z)

    d_start <- d_args$start
    g_start <- g_args$start

    d_list <- list(start = d_start, min = d_args$min, max = d_args$max, mle = FALSE)
    g_list <- list(start = g_start, min = g_args$min, max = g_args$max, mle = FALSE)

    cat("  Timing Full GP...")
    full <- time_reps(function() benchmark_full_gp(X, Z, XX, d_start, g_start), reps)
    cat(sprintf(" %.3fs\n", full$time))

    close_eff <- min(close_n, n_train)
    cat("  Timing aGP ALC...")
    alc <- time_reps(function() benchmark_agp(X, Z, XX, d_list, g_list, "alc",
                                              agp_start, agp_endpt, close_eff, omp_threads), reps)
    cat(sprintf(" %.3fs\n", alc$time))

    cat("  Timing aGP NN...")
    nn <- time_reps(function() benchmark_agp(X, Z, XX, d_list, g_list, "nn",
                                             agp_start, agp_endpt, close_eff, omp_threads), reps)
    cat(sprintf(" %.3fs\n", nn$time))

    cat("  Timing aGP MSPE...")
    mspe <- time_reps(function() benchmark_agp(X, Z, XX, d_list, g_list, "mspe",
                                               agp_start, agp_endpt, close_eff, omp_threads), reps)
    cat(sprintf(" %.3fs\n", mspe$time))

    rmse_full <- sqrt(mean((full$result - true_vals)^2))
    rmse_alc <- sqrt(mean((alc$result - true_vals)^2))
    rmse_nn <- sqrt(mean((nn$result - true_vals)^2))
    rmse_mspe <- sqrt(mean((mspe$result - true_vals)^2))

    cat(sprintf("  RMSE: Full=%.4f, ALC=%.4f, NN=%.4f, MSPE=%.4f\n",
                rmse_full, rmse_alc, rmse_nn, rmse_mspe))

    results <- rbind(results, data.frame(
      n = n_train,
      full = full$time,
      alc = alc$time,
      nn = nn$time,
      mspe = mspe$time,
      rmse_full = rmse_full,
      rmse_alc = rmse_alc,
      rmse_nn = rmse_nn,
      rmse_mspe = rmse_mspe
    ))

    cat("\n")
  }

  cat(strrep("=", 70), "\n")
  cat("RESULTS SUMMARY\n")
  cat(strrep("=", 70), "\n\n")

  cat("Timing Results (seconds):\n")
  cat(strrep("-", 80), "\n")
  cat(sprintf("%-10s %12s %12s %12s %12s %12s\n",
              "n_train", "Full GP", "aGP ALC", "aGP NN", "aGP MSPE", "Winner"))
  cat(strrep("-", 80), "\n")

  for (i in seq_len(nrow(results))) {
    r <- results[i, ]
    times <- c(r$full, r$alc, r$nn, r$mspe)
    methods <- c("Full GP", "aGP ALC", "aGP NN", "aGP MSPE")
    winner <- methods[which.min(times)]
    cat(sprintf("%-10d %12.3f %12.3f %12.3f %12.3f %12s\n",
                r$n, r$full, r$alc, r$nn, r$mspe, winner))
  }
  cat(strrep("-", 80), "\n\n")

  cat("Speedup vs Full GP:\n")
  cat(strrep("-", 60), "\n")
  cat(sprintf("%-10s %12s %12s %12s\n", "n_train", "ALC", "NN", "MSPE"))
  cat(strrep("-", 60), "\n")

  for (i in seq_len(nrow(results))) {
    r <- results[i, ]
    cat(sprintf("%-10d %11.2fx %11.2fx %11.2fx\n",
                r$n, r$full / r$alc, r$full / r$nn, r$full / r$mspe))
  }
  cat(strrep("-", 60), "\n\n")

  cat("RMSE (sanity check):\n")
  cat(strrep("-", 80), "\n")
  cat(sprintf("%-10s %12s %12s %12s %12s\n",
              "n_train", "Full GP", "aGP ALC", "aGP NN", "aGP MSPE"))
  cat(strrep("-", 80), "\n")

  for (i in seq_len(nrow(results))) {
    r <- results[i, ]
    cat(sprintf("%-10d %12.4f %12.4f %12.4f %12.4f\n",
                r$n, r$rmse_full, r$rmse_alc, r$rmse_nn, r$rmse_mspe))
  }
  cat(strrep("-", 80), "\n")

  if (out_path != "") {
    write.csv(results, out_path, row.names = FALSE)
    cat("\nWrote CSV:", out_path, "\n")
  }
}

run_benchmark()
