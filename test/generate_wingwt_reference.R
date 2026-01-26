#!/usr/bin/env Rscript
# Generate reference data for laGP.jl Wingwt validation
# Run this script from the test/ directory: Rscript generate_wingwt_reference.R

library(laGP)
library(lhs)
library(jsonlite)

# Ensure output directory exists
dir.create("reference", showWarnings = FALSE)

cat("Generating Wingwt reference data for laGP.jl tests...\n")

# Wing weight function (coded inputs [0,1])
wingwt <- function(X) {
  Sw  <- X[,1] * 50 + 150
  Wfw <- X[,2] * 80 + 220
  A   <- X[,3] * 4 + 6
  L   <- (X[,4] * 20 - 10) * pi / 180
  q   <- X[,5] * 29 + 16
  l   <- X[,6] * 0.5 + 0.5
  Rtc <- X[,7] * 0.1 + 0.08
  Nz  <- X[,8] * 3.5 + 2.5
  Wdg <- X[,9] * 800 + 1700

  W <- 0.036 * Sw^0.758 * Wfw^0.0035
  W <- W * (A / cos(L)^2)^0.6
  W <- W * q^0.006
  W <- W * l^0.04
  W <- W * (100 * Rtc / cos(L))^(-0.3)
  W <- W * (Nz * Wdg)^0.49
  return(W)
}

# ============================================================================
# Generate training data with LHS design
# ============================================================================
cat("  Generating LHS training design...\n")

set.seed(42)
n <- 200
X <- randomLHS(n, 9)
Y <- wingwt(X)

cat("  Response range:", range(Y), "\n")

# ============================================================================
# Fit GP and run MLE
# ============================================================================
cat("  Fitting GP and running MLE...\n")

# Fit GP with fixed initial values
g_init <- 1e-6
fit <- newGPsep(X, Y, d=2, g=g_init, dK=TRUE)

# mleGPsep only optimizes lengthscales d, nugget g stays fixed
# This matches the book example behavior
mle <- mleGPsep(fit, tmax=500)
mle_d <- mle$d
mle_g <- g_init  # g is not optimized by mleGPsep

cat("  MLE d:", mle_d, "\n")
cat("  MLE g:", mle_g, "(fixed)\n")

# Get log-likelihood after MLE
llik <- llikGPsep(fit)
cat("  Log-likelihood:", llik, "\n")

# ============================================================================
# Generate test predictions
# ============================================================================
cat("  Generating test predictions...\n")

set.seed(123)
n_test <- 100
X_test <- randomLHS(n_test, 9)
Y_test <- wingwt(X_test)

pred <- predGPsep(fit, X_test, lite=TRUE)

rmse <- sqrt(mean((Y_test - pred$mean)^2))
cat("  Test RMSE:", rmse, "lb\n")

# Clean up
deleteGPsep(fit)

# ============================================================================
# Save to JSON
# ============================================================================
cat("  Writing wingwt.json...\n")

test_data <- list(
    # Training data
    X = as.vector(t(X)),  # Row-major flattened
    X_nrow = n,
    X_ncol = 9,
    Y = as.vector(Y),
    # Initial hyperparameters
    d_init = 2.0,
    g_init = g_init,
    # MLE results
    mle_d = mle_d,
    mle_g = mle_g,
    llik = llik,
    # Test data
    X_test = as.vector(t(X_test)),  # Row-major flattened
    X_test_nrow = n_test,
    X_test_ncol = 9,
    Y_test = as.vector(Y_test),
    # Predictions
    pred_mean = pred$mean,
    pred_s2 = pred$s2,
    pred_df = pred$df,
    # RMSE for informational purposes
    rmse = rmse
)

write_json(test_data, "reference/wingwt.json",
           auto_unbox = TRUE, digits = 16)

cat("Done! Reference file written to test/reference/wingwt.json\n")
