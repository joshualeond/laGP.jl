# laGP.jl Demo
# Port of elements from the laGP R package demo (ALfhat.R)
#
# This demo shows:
# 1. Building a GP model on a 2D test function
# 2. Making predictions with local approximate GP
# 3. Visualizing predictions and local design selection

using laGP
using Random
using LatinHypercubeSampling
using CairoMakie
using Statistics: mean

# Set random seed for reproducibility
Random.seed!(42)

# Output directory (same as this script)
const OUTPUT_DIR = @__DIR__

# ============================================================================
# Test Functions from laGP R package
# ============================================================================

"""
    f2d(X)

2D test function from the laGP demo.
Modified Ackley-like function with interesting local structure.

Arguments:
- X: Matrix (n x 2) of input points in [-2, 2] x [-2, 2]

Returns:
- Vector of function values
"""
function f2d(X::Matrix)
    if size(X, 2) != 2
        error("f2d requires 2D input")
    end

    x = X[:, 1]
    y = X[:, 2]

    function g(z)
        return exp.(-(z .- 1).^2) .+ exp.(-0.8 .* (z .+ 1).^2) .- 0.05 .* sin.(8 .* (z .+ 0.1))
    end

    return -g(x) .* g(y)
end

# ============================================================================
# Generate Training Data
# ============================================================================

println("Generating training data...")

# Create a Latin Hypercube design in [0, 1]^2
n_train = 100
plan, _ = LHCoptim(n_train, 2, 10)  # 100 points, 2 dims, 10 iterations
X_train = Matrix{Float64}(plan ./ n_train)

# Scale to [-2, 2] for f2d
X_scaled = 4.0 .* (X_train .- 0.5)
Z_train = f2d(X_scaled)

println("Training data: $(n_train) points in [0, 1]^2")
println("Response range: [$(minimum(Z_train)), $(maximum(Z_train))]")

# ============================================================================
# Create Test Grid
# ============================================================================

println("\nCreating test grid...")

n_test_side = 100
x_test = range(0.0, 1.0, length=n_test_side)
X_test = Matrix{Float64}(undef, n_test_side^2, 2)
let idx = 1
    for j in 1:n_test_side
        for i in 1:n_test_side
            X_test[idx, 1] = x_test[i]
            X_test[idx, 2] = x_test[j]
            idx += 1
        end
    end
end

println("Test grid: $(n_test_side) x $(n_test_side) = $(n_test_side^2) points")

# ============================================================================
# True Function Visualization (ground truth)
# ============================================================================

println("\nCreating true function plot...")
true_vals = f2d(4.0 .* (X_test .- 0.5))
Z_true_grid = reshape(true_vals, n_test_side, n_test_side)

fig0 = Figure(size=(600, 500))
ax0 = Axis(fig0[1, 1], xlabel="x₁", ylabel="x₂", title="True Function f2d",
           aspect=DataAspect())
hm0 = heatmap!(ax0, collect(x_test), collect(x_test), Z_true_grid, colormap=:viridis)
contour!(ax0, collect(x_test), collect(x_test), Z_true_grid, color=:white, linewidth=0.5, levels=10)
scatter!(ax0, X_train[:, 1], X_train[:, 2], color=:red, markersize=4,
         strokewidth=0.5, strokecolor=:white, alpha=0.7)
Colorbar(fig0[1, 2], hm0, label="f(x)")
save(joinpath(OUTPUT_DIR, "true_function.png"), fig0)
println("Saved: true_function.png")

# ============================================================================
# Full GP Model (for comparison)
# ============================================================================

println("\n" * "="^60)
println("Building Full GP Model")
println("="^60)

# Estimate hyperparameters
d_range = darg(X_train)
g_range = garg(Z_train)

println("Lengthscale range: $(d_range.min) to $(d_range.max), start=$(d_range.start)")
println("Nugget range: $(g_range.min) to $(g_range.max), start=$(g_range.start)")

# Create and fit GP
gp = new_gp(X_train, Z_train, d_range.start, g_range.start)
println("Initial log-likelihood: $(llik_gp(gp))")

# MLE for hyperparameters
jmle_gp!(gp; drange=(d_range.min, d_range.max), grange=(g_range.min, g_range.max))
println("After MLE: d=$(gp.d), g=$(gp.g)")
println("Final log-likelihood: $(llik_gp(gp))")

# Predict on test grid
pred_full = pred_gp(gp, X_test; lite=true)
println("Full GP prediction done.")

# ============================================================================
# Local Approximate GP (aGP)
# ============================================================================

println("\n" * "="^60)
println("Running Local Approximate GP (aGP)")
println("="^60)

# aGP with ALC method
println("Running aGP with ALC acquisition (start=6, end=50)...")
result_alc = agp(X_train, Z_train, X_test;
    start=6, endpt=50,
    d=(start=gp.d, mle=false),
    g=(start=gp.g, mle=false),
    method=:alc
)
println("Done.")

# aGP with NN method (faster, less accurate)
println("Running aGP with NN method...")
result_nn = agp(X_train, Z_train, X_test;
    start=6, endpt=50,
    d=(start=gp.d, mle=false),
    g=(start=gp.g, mle=false),
    method=:nn
)
println("Done.")

# Compare prediction errors (true_vals already computed above)
rmse_full = sqrt(mean((pred_full.mean .- true_vals).^2))
rmse_alc = sqrt(mean((result_alc.mean .- true_vals).^2))
rmse_nn = sqrt(mean((result_nn.mean .- true_vals).^2))

println("\nPrediction RMSE comparison:")
println("  Full GP: $(round(rmse_full, digits=6))")
println("  aGP ALC: $(round(rmse_alc, digits=6))")
println("  aGP NN:  $(round(rmse_nn, digits=6))")

# ============================================================================
# Visualizations
# ============================================================================

println("\n" * "="^60)
println("Creating Visualizations")
println("="^60)

# 1. GP Surface Plot (mean prediction over 2D grid)
println("Creating GP surface plot...")
resolution = 100
x1_grid = range(0.0, 1.0, length=resolution)
x2_grid = range(0.0, 1.0, length=resolution)
XX_grid = Matrix{Float64}(undef, resolution^2, 2)
let idx = 1
    for j in 1:resolution
        for i in 1:resolution
            XX_grid[idx, 1] = x1_grid[i]
            XX_grid[idx, 2] = x2_grid[j]
            idx += 1
        end
    end
end
pred_grid = pred_gp(gp, XX_grid; lite=true)
Z_mean_grid = reshape(pred_grid.mean, resolution, resolution)

fig1 = Figure(size=(600, 500))
ax1 = Axis(fig1[1, 1], xlabel="x₁", ylabel="x₂", title="GP Mean Prediction",
           aspect=DataAspect())
hm1 = heatmap!(ax1, collect(x1_grid), collect(x2_grid), Z_mean_grid, colormap=:viridis)
contour!(ax1, collect(x1_grid), collect(x2_grid), Z_mean_grid, color=:white, linewidth=0.5, levels=10)
scatter!(ax1, X_train[:, 1], X_train[:, 2], color=:red, markersize=4,
         strokewidth=0.5, strokecolor=:white, alpha=0.7)
Colorbar(fig1[1, 2], hm1, label="Mean")
save(joinpath(OUTPUT_DIR, "gp_surface.png"), fig1)
println("Saved: gp_surface.png")

# 2. GP Variance Plot
println("Creating GP variance plot...")
Z_var_grid = reshape(pred_grid.s2, resolution, resolution)

fig2 = Figure(size=(600, 500))
ax2 = Axis(fig2[1, 1], xlabel="x₁", ylabel="x₂", title="GP Prediction Variance",
           aspect=DataAspect())
hm2 = heatmap!(ax2, collect(x1_grid), collect(x2_grid), Z_var_grid, colormap=:plasma)
contour!(ax2, collect(x1_grid), collect(x2_grid), Z_var_grid, color=:white, linewidth=0.5, levels=10)
scatter!(ax2, X_train[:, 1], X_train[:, 2], color=:cyan, markersize=4,
         strokewidth=0.5, strokecolor=:white, alpha=0.7)
Colorbar(fig2[1, 2], hm2, label="Variance")
save(joinpath(OUTPUT_DIR, "gp_variance.png"), fig2)
println("Saved: gp_variance.png")

# 3. aGP Predictions (mean and variance side by side)
println("Creating aGP predictions plot...")
Z_alc_mean_grid = reshape(result_alc.mean, n_test_side, n_test_side)
Z_alc_var_grid = reshape(result_alc.var, n_test_side, n_test_side)

fig3 = Figure(size=(1000, 450))
ax3a = Axis(fig3[1, 1], xlabel="x₁", ylabel="x₂", title="aGP Mean (ALC)",
            aspect=DataAspect())
hm3a = heatmap!(ax3a, collect(x_test), collect(x_test), Z_alc_mean_grid, colormap=:viridis)
contour!(ax3a, collect(x_test), collect(x_test), Z_alc_mean_grid, color=:white, linewidth=0.5, levels=10)
scatter!(ax3a, X_train[:, 1], X_train[:, 2], color=:red, markersize=3, alpha=0.5)
Colorbar(fig3[1, 2], hm3a, label="Mean")

ax3b = Axis(fig3[1, 3], xlabel="x₁", ylabel="x₂", title="aGP Variance (ALC)",
            aspect=DataAspect())
hm3b = heatmap!(ax3b, collect(x_test), collect(x_test), Z_alc_var_grid, colormap=:plasma)
contour!(ax3b, collect(x_test), collect(x_test), Z_alc_var_grid, color=:white, linewidth=0.5, levels=10)
scatter!(ax3b, X_train[:, 1], X_train[:, 2], color=:cyan, markersize=3, alpha=0.5)
Colorbar(fig3[1, 4], hm3b, label="Variance")
save(joinpath(OUTPUT_DIR, "agp_predictions.png"), fig3)
println("Saved: agp_predictions.png")

# 4. Local Design Selection Visualization
println("Creating local design selection plot...")
Xref = [0.5, 0.5]
lagp_result = lagp(Xref, 6, 30, X_train, Z_train;
    d=gp.d, g=gp.g, method=:alc)

fig4 = Figure(size=(600, 500))
ax4 = Axis(fig4[1, 1], xlabel="x₁", ylabel="x₂", title="Local Design Selection",
           aspect=DataAspect())
# Plot all training points (gray)
scatter!(ax4, X_train[:, 1], X_train[:, 2], color=:gray, markersize=6, alpha=0.4,
         label="All training points")
# Highlight selected local design points (blue)
local_idx = lagp_result.indices
scatter!(ax4, X_train[local_idx, 1], X_train[local_idx, 2], color=:blue, markersize=10,
         strokewidth=1, strokecolor=:black, label="Local design ($(length(local_idx)) pts)")
# Mark reference point (red star)
scatter!(ax4, [Xref[1]], [Xref[2]], color=:red, markersize=15, marker=:star5,
         strokewidth=1, strokecolor=:black, label="Reference point")
axislegend(ax4, position=:lt)
save(joinpath(OUTPUT_DIR, "local_design.png"), fig4)
println("Saved: local_design.png")

# 5. Comparison Plot: Full GP vs aGP ALC vs aGP NN
println("Creating comparison plot...")
fig6 = Figure(size=(1200, 400))

# Reshape predictions to grid
Z_full_grid = reshape(pred_full.mean, n_test_side, n_test_side)
Z_alc_grid = reshape(result_alc.mean, n_test_side, n_test_side)
Z_nn_grid = reshape(result_nn.mean, n_test_side, n_test_side)

ax6a = Axis(fig6[1, 1], xlabel="x₁", ylabel="x₂", title="Full GP",
           aspect=DataAspect())
hm6a = heatmap!(ax6a, collect(x_test), collect(x_test), Z_full_grid, colormap=:viridis)
contour!(ax6a, collect(x_test), collect(x_test), Z_full_grid, color=:white, linewidth=0.5)

ax6b = Axis(fig6[1, 2], xlabel="x₁", ylabel="x₂", title="aGP (ALC)",
           aspect=DataAspect())
hm6b = heatmap!(ax6b, collect(x_test), collect(x_test), Z_alc_grid, colormap=:viridis)
contour!(ax6b, collect(x_test), collect(x_test), Z_alc_grid, color=:white, linewidth=0.5)

ax6c = Axis(fig6[1, 3], xlabel="x₁", ylabel="x₂", title="aGP (NN)",
           aspect=DataAspect())
hm6c = heatmap!(ax6c, collect(x_test), collect(x_test), Z_nn_grid, colormap=:viridis)
contour!(ax6c, collect(x_test), collect(x_test), Z_nn_grid, color=:white, linewidth=0.5)

Colorbar(fig6[1, 4], hm6a, label="Prediction")

save(joinpath(OUTPUT_DIR, "comparison.png"), fig6)
println("Saved: comparison.png")

println("\n" * "="^60)
println("Demo Complete!")
println("="^60)
println("\nGenerated files:")
println("  - true_function.png: True function (ground truth)")
println("  - gp_surface.png: Full GP mean prediction")
println("  - gp_variance.png: Full GP prediction variance")
println("  - agp_predictions.png: aGP mean and variance")
println("  - local_design.png: Local design selection example")
println("  - comparison.png: Full GP vs aGP comparison")
