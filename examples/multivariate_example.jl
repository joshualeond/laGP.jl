# Multivariate Example: Separable GP with ARD
#
# This example demonstrates:
# 1. Fitting a separable GP (ARD) to 2D input data
# 2. Using per-dimension lengthscales to capture anisotropic structure
# 3. Visualizing the prediction surface and uncertainty
#
# Based on the tinygp "multivariate" tutorial:
# Target function: y = sin(x1) * cos(x2 + x1) + noise

using laGP
using Random
using CairoMakie

# ============================================================================
# Part 1: Generate Training Data
# ============================================================================

# Set seed for reproducibility
Random.seed!(48392)

# Generate 100 random 2D points uniformly sampled from [-5, 5]^2
n = 100
X = -5.0 .+ 10.0 .* rand(n, 2)

# Target function with noise (matching tinygp)
yerr = 0.1
y = sin.(X[:, 1]) .* cos.(X[:, 2] .+ X[:, 1]) .+ yerr .* randn(n)

println("Training data:")
println("  n = $n points in 2D")
println("  Input range: [-5, 5]^2")
println("  Noise level: $yerr")

# ============================================================================
# Part 2: Initialize Hyperparameters
# ============================================================================

# Use data-driven initialization for lengthscales and nugget
d_info = darg_sep(X)
g_info = garg(y)

println("\nInitial hyperparameter ranges:")
println("  d: start=$(round(d_info.ranges[1].start, sigdigits=3)), " *
        "min=$(round(d_info.ranges[1].min, sigdigits=3)), " *
        "max=$(round(d_info.ranges[1].max, sigdigits=3))")
println("  g: start=$(round(g_info.start, sigdigits=3)), " *
        "min=$(round(g_info.min, sigdigits=3)), " *
        "max=$(round(g_info.max, sigdigits=3))")

# Initial lengthscales (same for both dimensions initially)
d_init = [d_info.ranges[1].start, d_info.ranges[2].start]

# ============================================================================
# Part 3: Create Separable GP
# ============================================================================

# Create GP with per-dimension lengthscales (ARD)
gp = new_gp_sep(X, y, d_init, g_info.start)

println("\nInitial GP hyperparameters:")
println("  d = ", round.(gp.d, sigdigits=4))
println("  g = ", round(gp.g, sigdigits=4))
println("  log-likelihood = ", round(llik_gp_sep(gp), sigdigits=4))

# ============================================================================
# Part 4: Optimize via Joint MLE
# ============================================================================

# Set up ranges for optimization
drange = (d_info.ranges[1].min, d_info.ranges[1].max)
grange = (g_info.min, g_info.max)

# Run joint MLE optimization
result = jmle_gp_sep!(gp; drange=drange, grange=grange, verb=0)

println("\nOptimized GP hyperparameters:")
println("  d = ", round.(gp.d, sigdigits=4))
println("  g = ", round(gp.g, sigdigits=4))
println("  log-likelihood = ", round(llik_gp_sep(gp), sigdigits=4))
println("  iterations = ", result.tot_its)
println("  status = ", result.msg)

# Show per-dimension lengthscale interpretation
println("\nLengthscale interpretation (ARD):")
println("  d[1] (x1 dimension) = $(round(gp.d[1], sigdigits=4))")
println("  d[2] (x2 dimension) = $(round(gp.d[2], sigdigits=4))")
if gp.d[1] < gp.d[2]
    println("  -> Function varies more rapidly in x1 direction")
else
    println("  -> Function varies more rapidly in x2 direction")
end

# ============================================================================
# Part 5: Create Prediction Grid
# ============================================================================

# Create prediction grid
n_x1, n_x2 = 100, 50
x1_grid = range(-5, 5, length=n_x1)
x2_grid = range(-5, 5, length=n_x2)

# Build test matrix (row-major ordering for heatmap)
XX = Matrix{Float64}(undef, n_x1 * n_x2, 2)
let idx = 1
    for j in 1:n_x2
        for i in 1:n_x1
            XX[idx, 1] = x1_grid[i]
            XX[idx, 2] = x2_grid[j]
            idx += 1
        end
    end
end

println("\nPrediction grid: $(n_x1) x $(n_x2) = $(n_x1 * n_x2) points")

# ============================================================================
# Part 6: Make Predictions
# ============================================================================

# Get predictions (lite=true for efficiency)
pred = pred_gp_sep(gp, XX; lite=true)

# Reshape predictions for plotting
mean_grid = reshape(pred.mean, n_x1, n_x2)
std_grid = reshape(sqrt.(pred.s2), n_x1, n_x2)

println("Prediction complete:")
println("  Mean range: [$(round(minimum(pred.mean), sigdigits=3)), $(round(maximum(pred.mean), sigdigits=3))]")
println("  Std range: [$(round(minimum(sqrt.(pred.s2)), sigdigits=3)), $(round(maximum(sqrt.(pred.s2)), sigdigits=3))]")

# ============================================================================
# Part 7: Visualization
# ============================================================================

fig = Figure(size=(1000, 450))

# Left panel: Predicted mean surface
ax1 = Axis(fig[1, 1],
           xlabel="x1",
           ylabel="x2",
           title="Predicted Mean")

hm1 = heatmap!(ax1, collect(x1_grid), collect(x2_grid), mean_grid',
               colormap=:viridis)

# Overlay training points
scatter!(ax1, X[:, 1], X[:, 2], color=:white, markersize=5,
         strokecolor=:black, strokewidth=0.5)

Colorbar(fig[1, 2], hm1, label="Mean")

# Right panel: Prediction uncertainty (standard deviation)
ax2 = Axis(fig[1, 3],
           xlabel="x1",
           ylabel="x2",
           title="Prediction Uncertainty (Std)")

hm2 = heatmap!(ax2, collect(x1_grid), collect(x2_grid), std_grid',
               colormap=:plasma)

# Overlay training points
scatter!(ax2, X[:, 1], X[:, 2], color=:white, markersize=5,
         strokecolor=:black, strokewidth=0.5)

Colorbar(fig[1, 4], hm2, label="Std")

# Add title with hyperparameters
Label(fig[0, :], "Separable GP (ARD): d1=$(round(gp.d[1], sigdigits=3)), d2=$(round(gp.d[2], sigdigits=3)), g=$(round(gp.g, sigdigits=3))",
      fontsize=16)

# Save figure
output_path = joinpath(@__DIR__, "multivariate_example.png")
save(output_path, fig, px_per_unit=2)
println("\nFigure saved to: ", output_path)

# Display figure
fig
