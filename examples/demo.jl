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

"""
    constraint1(X)

First constraint function (feasible when c1 ≤ 0).
"""
function constraint1(X::Matrix)
    return 1.5 .- X[:, 1] .- 2 .* X[:, 2] .- 0.5 .* sin.(2π .* (X[:, 1].^2 .- 2 .* X[:, 2]))
end

"""
    constraint2(X)

Second constraint function (feasible when c2 ≤ 0).
"""
function constraint2(X::Matrix)
    return sum(X.^2, dims=2)[:] .- 1.5
end

# ============================================================================
# Generate Training Data
# ============================================================================

println("Generating training data...")

# Create a Latin Hypercube design in [0, 1]^2
n_train = 100
plan, _ = LHCoptim(n_train, 2, 10)  # 100 points, 2 dims, 10 iterations
X_train = Matrix{Float64}(plan ./ n_train)

# Scale to [-2, 2] for f2d, then back to [0, 1] for constraint functions
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
jmle_gp(gp; drange=(d_range.min, d_range.max), grange=(g_range.min, g_range.max))
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

# Compare prediction errors
true_vals = f2d(4.0 .* (X_test .- 0.5))
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

# 1. GP Surface Plot
println("Creating GP surface plot...")
fig1 = plot_gp_surface(gp, (0.0, 1.0), (0.0, 1.0); resolution=100)
save(joinpath(OUTPUT_DIR, "gp_surface.png"), fig1)
println("Saved: gp_surface.png")

# 2. GP Variance Plot
println("Creating GP variance plot...")
fig2 = plot_gp_variance(gp, (0.0, 1.0), (0.0, 1.0); resolution=100)
save(joinpath(OUTPUT_DIR, "gp_variance.png"), fig2)
println("Saved: gp_variance.png")

# 3. aGP Predictions
println("Creating aGP predictions plot...")
fig3 = plot_agp_predictions(X_train, Z_train, X_test, result_alc)
save(joinpath(OUTPUT_DIR, "agp_predictions.png"), fig3)
println("Saved: agp_predictions.png")

# 4. Local Design Selection Visualization
println("Creating local design selection plot...")
Xref = [0.5, 0.5]
lagp_result = lagp(Xref, 6, 30, X_train, Z_train;
    d=gp.d, g=gp.g, method=:alc)
fig4 = plot_local_design(X_train, Z_train, Xref, lagp_result.indices)
save(joinpath(OUTPUT_DIR, "local_design.png"), fig4)
println("Saved: local_design.png")

# 5. Constrained Optimization Problem Visualization
println("Creating constrained problem visualization...")

# Function wrappers for plotting (operating on [0,1]^2 domain)
f_plot(X) = f2d(4.0 .* (X .- 0.5))
c1_plot(X) = constraint1(X)
c2_plot(X) = constraint2(X)

fig5 = contour_with_constraints(f_plot, [c1_plot, c2_plot], (0.0, 1.0), (0.0, 1.0);
                                 resolution=150, n_levels=15)
# Add training points
ax = fig5.content[1]
scatter!(ax, X_train[:, 1], X_train[:, 2], color=:blue, markersize=4,
         strokewidth=0.5, strokecolor=:white, alpha=0.7)
save(joinpath(OUTPUT_DIR, "constrained_problem.png"), fig5)
println("Saved: constrained_problem.png")

# 6. Comparison Plot: Full GP vs aGP ALC vs aGP NN
println("Creating comparison plot...")
fig6 = Figure(size=(1200, 400))

# Reshape predictions to grid
Z_full_grid = reshape(pred_full.mean, n_test_side, n_test_side)
Z_alc_grid = reshape(result_alc.mean, n_test_side, n_test_side)
Z_nn_grid = reshape(result_nn.mean, n_test_side, n_test_side)

ax1 = Axis(fig6[1, 1], xlabel="x₁", ylabel="x₂", title="Full GP",
           aspect=DataAspect())
hm1 = heatmap!(ax1, collect(x_test), collect(x_test), Z_full_grid, colormap=:viridis)
contour!(ax1, collect(x_test), collect(x_test), Z_full_grid, color=:white, linewidth=0.5)

ax2 = Axis(fig6[1, 2], xlabel="x₁", ylabel="x₂", title="aGP (ALC)",
           aspect=DataAspect())
hm2 = heatmap!(ax2, collect(x_test), collect(x_test), Z_alc_grid, colormap=:viridis)
contour!(ax2, collect(x_test), collect(x_test), Z_alc_grid, color=:white, linewidth=0.5)

ax3 = Axis(fig6[1, 3], xlabel="x₁", ylabel="x₂", title="aGP (NN)",
           aspect=DataAspect())
hm3 = heatmap!(ax3, collect(x_test), collect(x_test), Z_nn_grid, colormap=:viridis)
contour!(ax3, collect(x_test), collect(x_test), Z_nn_grid, color=:white, linewidth=0.5)

Colorbar(fig6[1, 4], hm1, label="Prediction")

save(joinpath(OUTPUT_DIR, "comparison.png"), fig6)
println("Saved: comparison.png")

println("\n" * "="^60)
println("Demo Complete!")
println("="^60)
println("\nGenerated files:")
println("  - gp_surface.png: Full GP mean prediction")
println("  - gp_variance.png: Full GP prediction variance")
println("  - agp_predictions.png: aGP mean and variance")
println("  - local_design.png: Local design selection example")
println("  - constrained_problem.png: Test function with constraints")
println("  - comparison.png: Full GP vs aGP comparison")
