# CairoMakie GP Example: Complete Workflow with laGP.jl
#
# This example demonstrates:
# 1. Fixed hyperparameter GP (like tinygp)
# 2. MLE-optimized GP (like sklearn)
# 3. Separable/ARD GP (per-dimension lengthscales)
# 4. Side-by-side comparison of all three approaches
#
# Target function: z = sin(sqrt(x² + y²))

using laGP
using CairoMakie
using Random

# ============================================================================
# Generate Sample Data
# ============================================================================

# Set seed for reproducibility
Random.seed!(42)

# Generate 75 random points uniformly sampled from [-5, 5]²
n_samples = 75
x = rand(n_samples) .* 10 .- 5  # uniform(-5, 5)
y = rand(n_samples) .* 10 .- 5
z = sin.(sqrt.(x.^2 .+ y.^2))

# Combine into design matrix (rows = observations, as required by laGP)
X = hcat(x, y)  # 75 × 2 matrix

println("Training data:")
println("  n = $n_samples points in 2D")
println("  Input range: [-5, 5]²")
println("  Target function: z = sin(sqrt(x² + y²))")

# ============================================================================
# Create Prediction Grid (100x100)
# ============================================================================

x_grid = range(-5, 5, length=100)
y_grid = range(-5, 5, length=100)

# Build prediction matrix (row-major ordering for heatmap)
X_pred = Matrix{Float64}(undef, 100 * 100, 2)
let idx = 1
    for yj in y_grid
        for xi in x_grid
            X_pred[idx, 1] = xi
            X_pred[idx, 2] = yj
            idx += 1
        end
    end
end

println("Prediction grid: 100 × 100 = 10000 points")

# ============================================================================
# Option 1: Fixed Hyperparameters (like tinygp example)
# ============================================================================

println("\n" * "="^60)
println("Option 1: Fixed Hyperparameters")
println("="^60)

# Kernel parameterization mapping:
#   laGP kernel: k(x,y) = exp(-||x-y||²/d)
#   tinygp/KernelFunctions: k(x,y) = exp(-||x-y||²/(2ℓ²))
#   Therefore: d = 2ℓ². For scale=1.0 (ℓ=1), d = 2.0
d_fixed = 2.0    # lengthscale parameter
g_fixed = 1e-8   # nugget (small for nearly noise-free data)

gp_fixed = new_gp(X, z, d_fixed, g_fixed)

println("Hyperparameters:")
println("  d = $d_fixed (equivalent to ℓ = $(sqrt(d_fixed/2)) in tinygp)")
println("  g = $g_fixed")
println("  log-likelihood = $(round(llik_gp(gp_fixed), sigdigits=4))")

# Make predictions
pred_fixed = pred_gp(gp_fixed, X_pred; lite=true)

# Reshape for plotting
z_pred_fixed = reshape(pred_fixed.mean, 100, 100)
z_std_fixed = reshape(sqrt.(pred_fixed.s2), 100, 100)

# Plot with CairoMakie
fig1 = Figure(size=(900, 400))

ax1 = Axis(fig1[1, 1], title="Fixed GP Mean (d=$d_fixed)", xlabel="x", ylabel="y")
hm1 = heatmap!(ax1, collect(x_grid), collect(y_grid), z_pred_fixed', colormap=:viridis)
scatter!(ax1, x, y, color=:white, markersize=4, strokecolor=:black, strokewidth=0.5)
Colorbar(fig1[1, 2], hm1)

ax2 = Axis(fig1[1, 3], title="Fixed GP Std Dev", xlabel="x", ylabel="y")
hm2 = heatmap!(ax2, collect(x_grid), collect(y_grid), z_std_fixed', colormap=:plasma)
scatter!(ax2, x, y, color=:white, markersize=4, strokecolor=:black, strokewidth=0.5)
Colorbar(fig1[1, 4], hm2)

# ============================================================================
# Option 2: MLE Optimization (like sklearn)
# ============================================================================

println("\n" * "="^60)
println("Option 2: MLE Optimization")
println("="^60)

# Get data-driven initial ranges using laGP's helper functions
d_range = darg(X)
g_range = garg(z)

println("Initial ranges from data:")
println("  d: start=$(round(d_range.start, sigdigits=3)), " *
        "min=$(round(d_range.min, sigdigits=3)), max=$(round(d_range.max, sigdigits=3))")
println("  g: start=$(round(g_range.start, sigdigits=3)), " *
        "min=$(round(g_range.min, sigdigits=3)), max=$(round(g_range.max, sigdigits=3))")

# Create GP with initial values
gp_mle = new_gp(X, z, d_range.start, g_range.start)

println("\nInitial hyperparameters:")
println("  d = $(round(gp_mle.d, sigdigits=4))")
println("  g = $(round(gp_mle.g, sigdigits=4))")
println("  log-likelihood = $(round(llik_gp(gp_mle), sigdigits=4))")

# Run joint MLE optimization
result_mle = jmle_gp(gp_mle;
    drange=(d_range.min, d_range.max),
    grange=(g_range.min, g_range.max),
    maxit=100
)

println("\nAfter MLE optimization:")
println("  d = $(round(gp_mle.d, sigdigits=4))")
println("  g = $(round(gp_mle.g, sigdigits=4))")
println("  log-likelihood = $(round(llik_gp(gp_mle), sigdigits=4))")
println("  iterations = $(result_mle.tot_its)")
println("  status = $(result_mle.msg)")

# Make predictions
pred_mle = pred_gp(gp_mle, X_pred; lite=true)

# Reshape for plotting
z_pred_mle = reshape(pred_mle.mean, 100, 100)
z_std_mle = reshape(sqrt.(pred_mle.s2), 100, 100)

# Plot
fig2 = Figure(size=(900, 400))

ax1 = Axis(fig2[1, 1], title="MLE GP Mean (d=$(round(gp_mle.d, digits=3)))", xlabel="x", ylabel="y")
hm1 = heatmap!(ax1, collect(x_grid), collect(y_grid), z_pred_mle', colormap=:viridis)
scatter!(ax1, x, y, color=:white, markersize=4, strokecolor=:black, strokewidth=0.5)
Colorbar(fig2[1, 2], hm1)

ax2 = Axis(fig2[1, 3], title="MLE GP Std Dev", xlabel="x", ylabel="y")
hm2 = heatmap!(ax2, collect(x_grid), collect(y_grid), z_std_mle', colormap=:plasma)
scatter!(ax2, x, y, color=:white, markersize=4, strokecolor=:black, strokewidth=0.5)
Colorbar(fig2[1, 4], hm2)

# ============================================================================
# Option 3: Separable/ARD GP (per-dimension lengthscales)
# ============================================================================

println("\n" * "="^60)
println("Option 3: Separable/ARD GP")
println("="^60)

# darg_sep returns (ranges=[...], ab=...) where ranges is a Vector of per-dimension NamedTuples
d_sep_range = darg_sep(X)

# Extract starting values for each dimension
d_starts = [d_sep_range.ranges[i].start for i in 1:2]

println("Initial ranges from data (separable):")
for dim in 1:2
    r = d_sep_range.ranges[dim]
    println("  d[$dim]: start=$(round(r.start, sigdigits=3)), " *
            "min=$(round(r.min, sigdigits=3)), max=$(round(r.max, sigdigits=3))")
end

# Create separable GP
gp_sep = new_gp_sep(X, z, d_starts, g_range.start)

println("\nInitial separable hyperparameters:")
println("  d = $(round.(gp_sep.d, sigdigits=4))")
println("  g = $(round(gp_sep.g, sigdigits=4))")
println("  log-likelihood = $(round(llik_gp_sep(gp_sep), sigdigits=4))")

# Build drange as Vector of Tuples for per-dimension bounds
drange_sep = [(d_sep_range.ranges[i].min, d_sep_range.ranges[i].max) for i in 1:2]

# Run joint MLE optimization for separable GP
result_sep = jmle_gp_sep(gp_sep;
    drange=drange_sep,
    grange=(g_range.min, g_range.max)
)

println("\nAfter MLE optimization (separable):")
println("  d = $(round.(gp_sep.d, sigdigits=4))")
println("  g = $(round(gp_sep.g, sigdigits=4))")
println("  log-likelihood = $(round(llik_gp_sep(gp_sep), sigdigits=4))")
println("  iterations = $(result_sep.tot_its)")
println("  status = $(result_sep.msg)")

# Interpret the lengthscales
println("\nLengthscale interpretation (ARD):")
println("  d[1] (x dimension) = $(round(gp_sep.d[1], sigdigits=4))")
println("  d[2] (y dimension) = $(round(gp_sep.d[2], sigdigits=4))")
if gp_sep.d[1] ≈ gp_sep.d[2]
    println("  -> Function is isotropic (similar lengthscales in both directions)")
else
    slower_dir = gp_sep.d[1] > gp_sep.d[2] ? "x" : "y"
    faster_dir = gp_sep.d[1] > gp_sep.d[2] ? "y" : "x"
    println("  -> Function varies more rapidly in $faster_dir direction")
end

# Make predictions
pred_sep = pred_gp_sep(gp_sep, X_pred; lite=true)

# Reshape for plotting
z_pred_sep = reshape(pred_sep.mean, 100, 100)
z_std_sep = reshape(sqrt.(pred_sep.s2), 100, 100)

# Plot
fig3 = Figure(size=(900, 400))

ax1 = Axis(fig3[1, 1],
           title="Separable GP Mean (d=$(round.(gp_sep.d, digits=3)))",
           xlabel="x", ylabel="y")
hm1 = heatmap!(ax1, collect(x_grid), collect(y_grid), z_pred_sep', colormap=:viridis)
scatter!(ax1, x, y, color=:white, markersize=4, strokecolor=:black, strokewidth=0.5)
Colorbar(fig3[1, 2], hm1)

ax2 = Axis(fig3[1, 3], title="Separable GP Std Dev", xlabel="x", ylabel="y")
hm2 = heatmap!(ax2, collect(x_grid), collect(y_grid), z_std_sep', colormap=:plasma)
scatter!(ax2, x, y, color=:white, markersize=4, strokecolor=:black, strokewidth=0.5)
Colorbar(fig3[1, 4], hm2)

# ============================================================================
# Combined Comparison Plot (all 3 methods)
# ============================================================================

println("\n" * "="^60)
println("Creating combined comparison figure...")
println("="^60)

fig_all = Figure(size=(1000, 900))

# Row 1: Fixed hyperparameters
ax1 = Axis(fig_all[1, 1], title="Fixed (d=2.0)", xlabel="x", ylabel="y")
hm1 = heatmap!(ax1, collect(x_grid), collect(y_grid), z_pred_fixed', colormap=:viridis)
scatter!(ax1, x, y, color=:white, markersize=3, strokecolor=:black, strokewidth=0.5)
Colorbar(fig_all[1, 2], hm1)

ax2 = Axis(fig_all[1, 3], title="Fixed Std Dev", xlabel="x", ylabel="y")
hm2 = heatmap!(ax2, collect(x_grid), collect(y_grid), z_std_fixed', colormap=:plasma)
scatter!(ax2, x, y, color=:white, markersize=3, strokecolor=:black, strokewidth=0.5)
Colorbar(fig_all[1, 4], hm2)

# Row 2: MLE optimized
ax3 = Axis(fig_all[2, 1], title="MLE (d=$(round(gp_mle.d, digits=3)))", xlabel="x", ylabel="y")
hm3 = heatmap!(ax3, collect(x_grid), collect(y_grid), z_pred_mle', colormap=:viridis)
scatter!(ax3, x, y, color=:white, markersize=3, strokecolor=:black, strokewidth=0.5)
Colorbar(fig_all[2, 2], hm3)

ax4 = Axis(fig_all[2, 3], title="MLE Std Dev", xlabel="x", ylabel="y")
hm4 = heatmap!(ax4, collect(x_grid), collect(y_grid), z_std_mle', colormap=:plasma)
scatter!(ax4, x, y, color=:white, markersize=3, strokecolor=:black, strokewidth=0.5)
Colorbar(fig_all[2, 4], hm4)

# Row 3: Separable/ARD
ax5 = Axis(fig_all[3, 1], title="Separable (d=$(round.(gp_sep.d, digits=2)))", xlabel="x", ylabel="y")
hm5 = heatmap!(ax5, collect(x_grid), collect(y_grid), z_pred_sep', colormap=:viridis)
scatter!(ax5, x, y, color=:white, markersize=3, strokecolor=:black, strokewidth=0.5)
Colorbar(fig_all[3, 2], hm5)

ax6 = Axis(fig_all[3, 3], title="Separable Std Dev", xlabel="x", ylabel="y")
hm6 = heatmap!(ax6, collect(x_grid), collect(y_grid), z_std_sep', colormap=:plasma)
scatter!(ax6, x, y, color=:white, markersize=3, strokecolor=:black, strokewidth=0.5)
Colorbar(fig_all[3, 4], hm6)

# Add overall title
Label(fig_all[0, :], "GP Comparison: Fixed vs MLE vs Separable/ARD", fontsize=18)

# ============================================================================
# Save Figures
# ============================================================================

output_dir = @__DIR__

save(joinpath(output_dir, "gp_fixed.png"), fig1, px_per_unit=2)
save(joinpath(output_dir, "gp_mle.png"), fig2, px_per_unit=2)
save(joinpath(output_dir, "gp_separable.png"), fig3, px_per_unit=2)
save(joinpath(output_dir, "gp_comparison.png"), fig_all, px_per_unit=2)

println("\nFigures saved to $output_dir:")
println("  - gp_fixed.png")
println("  - gp_mle.png")
println("  - gp_separable.png")
println("  - gp_comparison.png")

# Display comparison figure
fig_all
