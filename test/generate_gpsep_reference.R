#!/usr/bin/env Rscript
# Generate reference data for laGP.jl GPsep validation
# Run this script from the test/ directory: Rscript generate_gpsep_reference.R

library(laGP)
library(jsonlite)

# Ensure output directory exists
dir.create("reference", showWarnings = FALSE)

cat("Generating GPsep reference data for laGP.jl tests...\n")

# Manual K computation for verification (separable kernel)
compute_K_sep <- function(X, d, g) {
    n <- nrow(X)
    m <- ncol(X)
    K <- matrix(0, n, n)
    for (i in 1:n) {
        for (j in 1:n) {
            if (i == j) {
                K[i, j] <- 1 + g
            } else {
                dist_sq <- sum((X[i, ] - X[j, ])^2 / d)
                K[i, j] <- exp(-dist_sq)
            }
        }
    }
    return(K)
}

# ============================================================================
# Test case: 10x2 anisotropic data for GPsep
# ============================================================================
cat("  Generating GPsep basic test case...\n")

set.seed(42)
n <- 10
m <- 2
X <- matrix(runif(n * m), ncol = m)
Z <- sin(10 * X[, 1]) + 0.1 * X[, 2]
d <- c(0.1, 1.0)
g <- 1e-4

# Create GP and get values
gpsepi <- newGPsep(X, Z, d, g, dK = TRUE)

# Manually compute internal state for verification
K <- compute_K_sep(X, d, g)
KiZ <- solve(K, Z)
phi <- as.numeric(t(Z) %*% KiZ)
ldetK <- as.numeric(determinant(K, logarithm = TRUE)$modulus)

# Get R laGP values
llik <- llikGPsep(gpsepi)

# Note: laGP R package doesn't expose dllikGPsep for gradients
# Gradients are validated via finite differences in the Julia tests

# Predictions
XX <- matrix(c(0.3, 0.3, 0.5, 0.5, 0.7, 0.7), ncol = m, byrow = TRUE)
pred <- predGPsep(gpsepi, XX, lite = TRUE)

# Clean up this GP
deleteGPsep(gpsepi)

# MLE optimization (fresh GP)
cat("  Running MLE optimization...\n")
gpsep_mle <- newGPsep(X, Z, d, g, dK = TRUE)
mle_result <- jmleGPsep(gpsep_mle)
# jmleGPsep returns d.1, d.2, ..., d.m and g in the result data frame
mle_d <- as.numeric(mle_result[1, grepl("^d\\.", names(mle_result))])
mle_g <- mle_result$g[1]
mle_llik <- llikGPsep(gpsep_mle)

# Clean up
deleteGPsep(gpsep_mle)

# ============================================================================
# Save to JSON
# ============================================================================
cat("  Writing gpsep_basic.json...\n")

test_data <- list(
    # Input data
    X = as.vector(t(X)),  # Row-major flattened
    X_nrow = n,
    X_ncol = m,
    Z = as.vector(Z),
    d = d,
    g = g,
    # Internal state (manually computed for verification)
    K = as.vector(t(K)),  # Row-major flattened
    KiZ = as.vector(KiZ),
    phi = phi,
    ldetK = ldetK,
    # Log-likelihood
    llik = llik,
    # Prediction inputs
    XX = as.vector(t(XX)),  # Row-major flattened
    XX_nrow = nrow(XX),
    XX_ncol = ncol(XX),
    # Prediction outputs
    pred_mean = pred$mean,
    pred_s2 = pred$s2,
    pred_df = pred$df,
    # MLE results
    mle_d = mle_d,
    mle_g = mle_g,
    mle_llik = mle_llik
)

write_json(test_data, "reference/gpsep_basic.json",
           auto_unbox = TRUE, digits = 16)

cat("Done! Reference file written to test/reference/gpsep_basic.json\n")
