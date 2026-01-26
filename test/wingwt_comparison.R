# Wing Weight GP Comparison Test - R Reference
library(laGP)
library(lhs)

set.seed(42)

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

# LHS design (use smaller n for faster comparison)
n <- 200
X <- randomLHS(n, 9)
Y <- wingwt(X)

cat("Response range:", range(Y), "\n")

# Fit GP with fixed initial values (matching book)
fit <- newGPsep(X, Y, d=2, g=1e-6, dK=TRUE)
mle <- mleGPsep(fit)

cat("\nMLE results:\n")
cat("  d:", mle$d, "\n")
cat("  g:", mle$g, "\n")

# Test predictions on held-out grid
n_test <- 100
X_test <- randomLHS(n_test, 9)
Y_test <- wingwt(X_test)

pred <- predGPsep(fit, X_test, lite=TRUE)
rmse <- sqrt(mean((Y_test - pred$mean)^2))

cat("\nTest RMSE:", rmse, "lb\n")
cat("Test RMSE %:", 100 * rmse / diff(range(Y)), "%\n")

deleteGPsep(fit)
