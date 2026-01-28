# Sinusoidal Example: GP Posterior Sampling
#
# This example demonstrates:
# 1. Fitting an isotropic GP to sparse sinusoidal data
# 2. Using pred_gp(lite=false) to get full posterior covariance
# 3. Drawing posterior samples from the GP
# 4. Visualizing the posterior samples with CairoMakie

using laGP
using Distributions
using PDMats
using LinearAlgebra
using Random
using CairoMakie

# ============================================================================
# Part 1: Generate Training Data
# ============================================================================

# Sparse training points from sin(x) over [0, 2π]
# Match R's seq(0, 2*pi, length=6)
X_train = reshape(collect(range(0, 2π, length=6)), :, 1)
Y_train = sin.(X_train[:, 1])

println("Training data:")
println("  X: ", vec(X_train))
println("  Y: ", Y_train)

# ============================================================================
# Part 2: Fit Isotropic GP with MLE
# ============================================================================

# Initial lengthscale and small nugget (matching R's example)
d_init = 2.0  # lengthscale
g_init = 1e-6  # small nugget for interpolation

# Create GP (matches R: newGP(X, Z, 2, 1e-6, dK=TRUE))
gp = new_gp(X_train, Y_train, d_init, g_init)

# MLE for lengthscale only (nugget stays fixed at 1e-6)
# Matches R's simple API: mleGP(gp, tmax=20)
mle_gp!(gp, :d; tmax=20)

println("\nOptimized hyperparameters:")
println("  d = ", gp.d)
println("  g = ", gp.g)
println("  log-likelihood = ", llik_gp(gp))

# ============================================================================
# Part 3: Posterior Prediction with Full Covariance
# ============================================================================

# Dense test grid - match R's seq(-1, 2*pi+1, length=499)
xx = collect(range(-1, 2π + 1, length=499))
XX = reshape(xx, :, 1)

# Get full posterior (lite=false returns GPPredictionFull with covariance matrix)
pred_full = pred_gp(gp, XX; lite=false)

println("\nPrediction:")
println("  Test points: ", length(xx))
println("  Sigma size: ", size(pred_full.Sigma))

# ============================================================================
# Part 4: Draw Posterior Samples
# ============================================================================

# Draw posterior samples using MvTDist (mirrors R's rmvt)
# Student-t is correct because laGP uses concentrated likelihood to estimate τ²,
# introducing additional uncertainty captured by t-distribution with df = n
Random.seed!(42)
n_samples = 100  # Match R's 100 samples
mvt = MvTDist(pred_full.df, pred_full.mean, PDMat(Symmetric(pred_full.Sigma)))
samples = rand(mvt, n_samples)

println("\nPosterior samples: ", size(samples))

# ============================================================================
# Part 5: Visualization
# ============================================================================

fig = Figure(size=(700, 500))
ax = Axis(fig[1, 1],
          xlabel="x",
          ylabel="Y(x) | θ̂",
          title="Simple Sinusoidal Example: GP Posterior Samples")

# Draw posterior samples (gray, semi-transparent)
for i in 1:n_samples
    lines!(ax, xx, samples[:, i], color=(:gray, 0.3), linewidth=0.5)
end

# Draw posterior mean (blue)
lines!(ax, xx, pred_full.mean, color=:blue, linewidth=2, label="Posterior mean")

# Draw true function (dashed green)
lines!(ax, xx, sin.(xx), color=:green, linewidth=1.5, linestyle=:dash, label="sin(x)")

# Draw training points (black circles)
scatter!(ax, vec(X_train), Y_train, color=:black, markersize=12, label="Training data")

# Add legend
axislegend(ax, position=:lb)

# Set axis limits to match new test range
xlims!(ax, -1, 2π + 1)
ylims!(ax, -2.0, 2.0)

# Save figure
output_path = joinpath(@__DIR__, "sinusoidal_example.png")
save(output_path, fig, px_per_unit=2)
println("\nFigure saved to: ", output_path)

# Display figure
fig
