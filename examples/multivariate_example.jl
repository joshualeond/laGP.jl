# Multivariate Example: Isotropic vs Separable GP Comparison
#
# This example demonstrates:
# 1. Fitting both isotropic and separable (ARD) GPs to 2D input data
# 2. Comparing their lengthscales and log-likelihoods
# 3. Showing why ARD is beneficial for anisotropic functions
# 4. Visualizing the prediction surfaces side-by-side
#
# Target function: y = sin(4*x1) + 0.2*x2 + noise
# This function has extreme anisotropy:
# - Fast oscillations in x1 (needs small lengthscale)
# - Nearly linear in x2 (needs large lengthscale)

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

# Target function with noise
# Extreme anisotropy: fast oscillations in x1, linear in x2
yerr = 0.1
y = sin.(4 .* X[:, 1]) .+ 0.2 .* X[:, 2] .+ yerr .* randn(n)

println("Training data:")
println("  n = $n points in 2D")
println("  Input range: [-5, 5]^2")
println("  Noise level: $yerr")

# ============================================================================
# Part 2: Initialize Hyperparameters
# ============================================================================

# Use data-driven initialization for lengthscales and nugget
d_info_sep = darg_sep(X)  # For separable GP
d_info_iso = darg(X)       # For isotropic GP
g_info = garg(y)

println("\nInitial hyperparameter ranges:")
println("  Separable d: start=$(round(d_info_sep.ranges[1].start, sigdigits=3)), " *
        "min=$(round(d_info_sep.ranges[1].min, sigdigits=3)), " *
        "max=$(round(d_info_sep.ranges[1].max, sigdigits=3))")
println("  Isotropic d: start=$(round(d_info_iso.start, sigdigits=3)), " *
        "min=$(round(d_info_iso.min, sigdigits=3)), " *
        "max=$(round(d_info_iso.max, sigdigits=3))")
println("  g: start=$(round(g_info.start, sigdigits=3)), " *
        "min=$(round(g_info.min, sigdigits=3)), " *
        "max=$(round(g_info.max, sigdigits=3))")

# Set up optimization ranges
grange = (g_info.min, g_info.max)

# ============================================================================
# Part 3: Fit Isotropic GP
# ============================================================================

println("\n" * "="^60)
println("ISOTROPIC GP (single lengthscale)")
println("="^60)

# Create isotropic GP (single lengthscale for all dimensions)
gp_iso = new_gp(X, y, d_info_iso.start, g_info.start)

println("\nInitial isotropic GP hyperparameters:")
println("  d = ", round(gp_iso.d, sigdigits=4))
println("  g = ", round(gp_iso.g, sigdigits=4))
println("  log-likelihood = ", round(llik_gp(gp_iso), sigdigits=4))

# Optimize isotropic GP
drange_iso = (d_info_iso.min, d_info_iso.max)
result_iso = jmle_gp!(gp_iso; drange=drange_iso, grange=grange, verb=0)

println("\nOptimized isotropic GP hyperparameters:")
println("  d = ", round(gp_iso.d, sigdigits=4))
println("  g = ", round(gp_iso.g, sigdigits=4))
println("  log-likelihood = ", round(llik_gp(gp_iso), sigdigits=4))
println("  iterations = ", result_iso.tot_its)
println("  status = ", result_iso.msg)

# ============================================================================
# Part 4: Fit Separable GP
# ============================================================================

println("\n" * "="^60)
println("SEPARABLE GP (per-dimension lengthscales)")
println("="^60)

# Initial lengthscales (same for both dimensions initially)
d_init = [d_info_sep.ranges[1].start, d_info_sep.ranges[2].start]

# Create GP with per-dimension lengthscales (ARD)
gp_sep = new_gp_sep(X, y, d_init, g_info.start)

println("\nInitial separable GP hyperparameters:")
println("  d = ", round.(gp_sep.d, sigdigits=4))
println("  g = ", round(gp_sep.g, sigdigits=4))
println("  log-likelihood = ", round(llik_gp_sep(gp_sep), sigdigits=4))

# Optimize separable GP
drange_sep = (d_info_sep.ranges[1].min, d_info_sep.ranges[1].max)
result_sep = jmle_gp_sep!(gp_sep; drange=drange_sep, grange=grange, verb=0)

println("\nOptimized separable GP hyperparameters:")
println("  d = ", round.(gp_sep.d, sigdigits=4))
println("  g = ", round(gp_sep.g, sigdigits=4))
println("  log-likelihood = ", round(llik_gp_sep(gp_sep), sigdigits=4))
println("  iterations = ", result_sep.tot_its)
println("  status = ", result_sep.msg)

# Show per-dimension lengthscale interpretation
println("\nLengthscale interpretation (ARD):")
println("  d[1] (x1 dimension) = $(round(gp_sep.d[1], sigdigits=4))")
println("  d[2] (x2 dimension) = $(round(gp_sep.d[2], sigdigits=4))")
if gp_sep.d[1] < gp_sep.d[2]
    println("  -> Function varies more rapidly in x1 direction")
else
    println("  -> Function varies more rapidly in x2 direction")
end

# ============================================================================
# Part 5: Compare Log-likelihoods
# ============================================================================

println("\n" * "="^60)
println("COMPARISON")
println("="^60)

llik_iso = llik_gp(gp_iso)
llik_sep = llik_gp_sep(gp_sep)

println("\nLog-likelihood comparison:")
println("  Isotropic:  $(round(llik_iso, sigdigits=5))")
println("  Separable:  $(round(llik_sep, sigdigits=5))")
println("  Difference: $(round(llik_sep - llik_iso, sigdigits=4)) (separable is better)")

println("\nLengthscale comparison:")
println("  Isotropic d = $(round(gp_iso.d, sigdigits=4)) (single value for both dimensions)")
println("  Separable d = $(round.(gp_sep.d, sigdigits=4)) (per-dimension)")
println("  -> Isotropic d is a compromise between d[1]=$(round(gp_sep.d[1], sigdigits=3)) and d[2]=$(round(gp_sep.d[2], sigdigits=3))")

# ============================================================================
# Part 6: Create Prediction Grid
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
# Part 7: Make Predictions for Both GPs
# ============================================================================

# True function (no noise) for reference
y_true = sin.(4 .* XX[:, 1]) .+ 0.2 .* XX[:, 2]
true_grid = reshape(y_true, n_x1, n_x2)

# Isotropic predictions
pred_iso = pred_gp(gp_iso, XX; lite=true)
mean_iso_grid = reshape(pred_iso.mean, n_x1, n_x2)
std_iso_grid = reshape(sqrt.(pred_iso.s2), n_x1, n_x2)

# Separable predictions
pred_sep = pred_gp_sep(gp_sep, XX; lite=true)
mean_sep_grid = reshape(pred_sep.mean, n_x1, n_x2)
std_sep_grid = reshape(sqrt.(pred_sep.s2), n_x1, n_x2)

println("Predictions complete:")
println("  True function range: [$(round(minimum(y_true), sigdigits=3)), $(round(maximum(y_true), sigdigits=3))]")
println("  Isotropic mean range: [$(round(minimum(pred_iso.mean), sigdigits=3)), $(round(maximum(pred_iso.mean), sigdigits=3))]")
println("  Separable mean range: [$(round(minimum(pred_sep.mean), sigdigits=3)), $(round(maximum(pred_sep.mean), sigdigits=3))]")

# ============================================================================
# Part 8: Visualization (3-row comparison with true function)
# ============================================================================

fig = Figure(size=(1100, 1200))

# Use consistent colormap limits based on true function range
clims = (minimum(y_true), maximum(y_true))

# Row 1: True function (reference)
ax_true = Axis(fig[1, 1:3],
               xlabel="x1",
               ylabel="x2",
               title="True Function: y = sin(4x₁) + 0.2x₂")

hm_true = heatmap!(ax_true, collect(x1_grid), collect(x2_grid), true_grid',
                   colormap=:viridis, colorrange=clims)
scatter!(ax_true, X[:, 1], X[:, 2], color=:white, markersize=4,
         strokecolor=:black, strokewidth=0.5)
Colorbar(fig[1, 4], hm_true, label="y")

# Row 2: Isotropic GP
ax1 = Axis(fig[2, 1],
           xlabel="x1",
           ylabel="x2",
           title="Isotropic GP: Mean")

hm1 = heatmap!(ax1, collect(x1_grid), collect(x2_grid), mean_iso_grid',
               colormap=:viridis, colorrange=clims)
scatter!(ax1, X[:, 1], X[:, 2], color=:white, markersize=4,
         strokecolor=:black, strokewidth=0.5)
Colorbar(fig[2, 2], hm1, label="Mean")

ax2 = Axis(fig[2, 3],
           xlabel="x1",
           ylabel="x2",
           title="Isotropic GP: Uncertainty (Std)")

hm2 = heatmap!(ax2, collect(x1_grid), collect(x2_grid), std_iso_grid',
               colormap=:plasma)
scatter!(ax2, X[:, 1], X[:, 2], color=:white, markersize=4,
         strokecolor=:black, strokewidth=0.5)
Colorbar(fig[2, 4], hm2, label="Std")

# Row 3: Separable GP
ax3 = Axis(fig[3, 1],
           xlabel="x1",
           ylabel="x2",
           title="Separable GP (ARD): Mean")

hm3 = heatmap!(ax3, collect(x1_grid), collect(x2_grid), mean_sep_grid',
               colormap=:viridis, colorrange=clims)
scatter!(ax3, X[:, 1], X[:, 2], color=:white, markersize=4,
         strokecolor=:black, strokewidth=0.5)
Colorbar(fig[3, 2], hm3, label="Mean")

ax4 = Axis(fig[3, 3],
           xlabel="x1",
           ylabel="x2",
           title="Separable GP (ARD): Uncertainty (Std)")

hm4 = heatmap!(ax4, collect(x1_grid), collect(x2_grid), std_sep_grid',
               colormap=:plasma)
scatter!(ax4, X[:, 1], X[:, 2], color=:white, markersize=4,
         strokecolor=:black, strokewidth=0.5)
Colorbar(fig[3, 4], hm4, label="Std")

# Add summary labels
iso_label = "Isotropic: d=$(round(gp_iso.d, sigdigits=3)), g=$(round(gp_iso.g, sigdigits=3)), llik=$(round(llik_iso, sigdigits=4))"
sep_label = "Separable: d₁=$(round(gp_sep.d[1], sigdigits=3)), d₂=$(round(gp_sep.d[2], sigdigits=3)), g=$(round(gp_sep.g, sigdigits=3)), llik=$(round(llik_sep, sigdigits=4))"

Label(fig[0, :], "Isotropic vs Separable GP Comparison\n$iso_label\n$sep_label",
      fontsize=14)

# Save figure
output_path = joinpath(@__DIR__, "multivariate_example.png")
save(output_path, fig, px_per_unit=2)
println("\nFigure saved to: ", output_path)

# Display figure
fig
