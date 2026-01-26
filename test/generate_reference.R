#!/usr/bin/env Rscript
# Generate reference data for laGP.jl tests
# Run this script from the test/ directory

library(laGP)
library(jsonlite)

# Ensure output directory exists
dir.create("reference", showWarnings = FALSE)

cat("Generating reference data for laGP.jl tests...\n")

# ============================================================================
# Test case 1: Basic GP prediction
# ============================================================================
cat("  1. Basic GP prediction...\n")
set.seed(42)
X <- matrix(runif(20), ncol = 2)
Z <- sin(rowSums(X * 3))
gpi <- newGP(X, Z, d = 0.5, g = 1e-3)
XX <- matrix(c(0.3, 0.3, 0.7, 0.7), ncol = 2, byrow = TRUE)
pred <- predGP(gpi, XX, lite = TRUE)
llik_val <- llikGP(gpi)

test_gp_basic <- list(
    X = as.vector(t(X)),  # Row-major flattened
    X_nrow = nrow(X),
    X_ncol = ncol(X),
    Z = as.vector(Z),
    d = 0.5,
    g = 1e-3,
    XX = as.vector(t(XX)),  # Row-major flattened
    XX_nrow = nrow(XX),
    XX_ncol = ncol(XX),
    pred_mean = as.vector(pred$mean),
    pred_s2 = as.vector(pred$s2),
    llik = llik_val,
    df = pred$df
)

deleteGP(gpi)

# ============================================================================
# Test case 2: MLE optimization
# ============================================================================
cat("  2. MLE optimization...\n")
set.seed(42)
X <- matrix(runif(20), ncol = 2)
Z <- sin(rowSums(X * 3))

# MLE for d
gpi_d <- newGP(X, Z, d = 0.1, g = 1e-3, dK = TRUE)
mle_d_result <- mleGP(gpi_d, param = "d", tmin = 0.01, tmax = 10)
mle_d_val <- mle_d_result$d
llik_after_d <- llikGP(gpi_d)
deleteGP(gpi_d)

# MLE for g
gpi_g <- newGP(X, Z, d = 0.5, g = 1e-4, dK = TRUE)
mle_g_result <- mleGP(gpi_g, param = "g", tmin = 1e-6, tmax = 1)
mle_g_val <- mle_g_result$g
llik_after_g <- llikGP(gpi_g)
deleteGP(gpi_g)

# Joint MLE
gpi_jmle <- newGP(X, Z, d = 0.1, g = 1e-4, dK = TRUE)
jmle_result <- jmleGP(gpi_jmle, drange = c(0.01, 10), grange = c(1e-6, 1))
jmle_d_val <- jmle_result$d
jmle_g_val <- jmle_result$g
llik_after_jmle <- llikGP(gpi_jmle)
deleteGP(gpi_jmle)

# darg and garg helper functions
darg_result <- darg(d = NULL, X = X)
garg_result <- garg(g = list(mle = TRUE), y = Z)

test_mle <- list(
    X = as.vector(t(X)),
    X_nrow = nrow(X),
    X_ncol = ncol(X),
    Z = as.vector(Z),
    # MLE d
    d_init = 0.1,
    g_for_d = 1e-3,
    mle_d = mle_d_val,
    llik_after_d = llik_after_d,
    # MLE g
    d_for_g = 0.5,
    g_init = 1e-4,
    mle_g = mle_g_val,
    llik_after_g = llik_after_g,
    # Joint MLE
    jmle_d_init = 0.1,
    jmle_g_init = 1e-4,
    jmle_d = jmle_d_val,
    jmle_g = jmle_g_val,
    llik_after_jmle = llik_after_jmle,
    # darg/garg
    darg_start = darg_result$start,
    darg_max = darg_result$max,
    darg_min = darg_result$min,
    garg_start = garg_result$start,
    garg_max = garg_result$max,
    garg_min = garg_result$min
)

# ============================================================================
# Test case 3: ALC acquisition function
# ============================================================================
cat("  3. ALC acquisition function...\n")
set.seed(42)
X <- matrix(runif(20), ncol = 2)
Z <- sin(rowSums(X * 3))
gpi <- newGP(X, Z, d = 0.5, g = 1e-3, dK = TRUE)

set.seed(123)
Xcand <- matrix(runif(100), ncol = 2)
Xref <- matrix(c(0.5, 0.5), ncol = 2)
alc_vals <- alcGP(gpi, Xcand, Xref)

# Also test MSPE
mspe_vals <- mspeGP(gpi, Xcand, Xref)

test_acquisition <- list(
    X = as.vector(t(X)),
    X_nrow = nrow(X),
    X_ncol = ncol(X),
    Z = as.vector(Z),
    d = 0.5,
    g = 1e-3,
    Xcand = as.vector(t(Xcand)),
    Xcand_nrow = nrow(Xcand),
    Xcand_ncol = ncol(Xcand),
    Xref = as.vector(t(Xref)),
    Xref_nrow = nrow(Xref),
    Xref_ncol = ncol(Xref),
    alc = as.vector(alc_vals),
    mspe = as.vector(mspe_vals)
)

deleteGP(gpi)

# ============================================================================
# Test case 4: Local Approximate GP (laGP/aGP)
# ============================================================================
cat("  4. Local Approximate GP (aGP)...\n")
set.seed(42)
n <- 100
X <- matrix(runif(n * 2), ncol = 2)
Z <- sin(rowSums(X * 3)) + rnorm(n, sd = 0.1)

XX <- matrix(c(0.3, 0.3, 0.5, 0.5, 0.7, 0.7), ncol = 2, byrow = TRUE)

# aGP with ALC method, fixed hyperparameters
out_alc <- aGP(X, Z, XX,
    start = 6, end = 20,
    d = list(start = 0.5, mle = FALSE),
    g = list(start = 1e-3, mle = FALSE),
    method = "alc", verb = 0
)

# aGP with NN method, fixed hyperparameters
out_nn <- aGP(X, Z, XX,
    start = 6, end = 20,
    d = list(start = 0.5, mle = FALSE),
    g = list(start = 1e-3, mle = FALSE),
    method = "nn", verb = 0
)

# aGP with MSPE method, fixed hyperparameters
out_mspe <- aGP(X, Z, XX,
    start = 6, end = 20,
    d = list(start = 0.5, mle = FALSE),
    g = list(start = 1e-3, mle = FALSE),
    method = "mspe", verb = 0
)

# aGP with MLE enabled
out_mle <- aGP(X, Z, XX,
    start = 6, end = 20,
    d = list(start = 0.5, mle = TRUE, min = 0.01, max = 10),
    g = list(start = 1e-3, mle = TRUE, min = 1e-6, max = 1),
    method = "alc", verb = 0
)

test_lagp <- list(
    X = as.vector(t(X)),
    X_nrow = nrow(X),
    X_ncol = ncol(X),
    Z = as.vector(Z),
    XX = as.vector(t(XX)),
    XX_nrow = nrow(XX),
    XX_ncol = ncol(XX),
    start = 6,
    end_size = 20,
    d = 0.5,
    g = 1e-3,
    # ALC method results
    alc_mean = as.vector(out_alc$mean),
    alc_var = as.vector(out_alc$var),
    # NN method results
    nn_mean = as.vector(out_nn$mean),
    nn_var = as.vector(out_nn$var),
    # MSPE method results
    mspe_mean = as.vector(out_mspe$mean),
    mspe_var = as.vector(out_mspe$var),
    # MLE results
    mle_mean = as.vector(out_mle$mean),
    mle_var = as.vector(out_mle$var),
    mle_mle_d = as.vector(out_mle$mle$d),
    mle_mle_g = as.vector(out_mle$mle$g)
)

# ============================================================================
# Write all reference files
# ============================================================================
cat("Writing JSON files...\n")

write_json(test_gp_basic, "reference/gp_basic.json",
    auto_unbox = TRUE, digits = 16
)
write_json(test_mle, "reference/mle.json",
    auto_unbox = TRUE, digits = 16
)
write_json(test_acquisition, "reference/acquisition.json",
    auto_unbox = TRUE, digits = 16
)
write_json(test_lagp, "reference/lagp.json",
    auto_unbox = TRUE, digits = 16
)

cat("Done! Reference files written to test/reference/\n")
