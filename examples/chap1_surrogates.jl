# Chapter 1: Surrogate Modeling with Gaussian Processes
# Port of "Surrogates: Gaussian Process Modeling, Design and Optimization" Chapter 1
# Original R code by Robert Gramacy
# Julia port using laGP.jl
#
# This example demonstrates:
# 1. Response surface methodology (RSM) visualizations
# 2. Aircraft wing weight simulator
# 3. Latin Hypercube Sampling design
# 4. GP surrogate fitting and prediction
# 5. Main effects analysis

using laGP
using Random
using LatinHypercubeSampling
using CairoMakie
using Statistics: mean
using Distributions: TDist, quantile

# Set random seed for reproducibility
Random.seed!(42)

# Output directory (same as this script)
const OUTPUT_DIR = @__DIR__

# ============================================================================
# PART 1: Response Surface Examples
# ============================================================================

println("="^70)
println("PART 1: Response Surface Methodology Visualizations")
println("="^70)

"""
    yield(xi1, xi2)

Banana-shaped yield function - nonlinear response surface.
Classic example from response surface methodology.

Arguments:
- xi1: Time variable (typically 1-8)
- xi2: Temperature variable (typically 100-1000)
"""
function yield(xi1, xi2)
    # Transform to working coordinates
    x1 = 3 * xi1 - 15
    x2 = xi2 / 50 - 13
    # Rotation
    x1_rot = cos(0.5) * x1 - sin(0.5) * x2
    x2_rot = sin(0.5) * x1 + cos(0.5) * x2
    # Banana-shaped response
    y = exp(-x1_rot^2 / 80 - 0.5 * (x2_rot + 0.03 * x1_rot^2 - 40 * 0.03)^2)
    return 100 * y
end

"""
    first_order(x1, x2)

First-order (linear) response surface: y = b0 + b1*x1 + b2*x2
"""
first_order(x1, x2) = 50 + 8 * x1 + 3 * x2

"""
    first_order_interaction(x1, x2)

First-order with interaction: y = b0 + b1*x1 + b2*x2 + b12*x1*x2
"""
first_order_interaction(x1, x2) = 50 + 8 * x1 + 3 * x2 - 4 * x1 * x2

"""
    simple_max(x1, x2)

Second-order with simple maximum (elliptical contours).
"""
simple_max(x1, x2) = 50 + 8 * x1 + 3 * x2 - 7 * x1^2 - 3 * x2^2 - 4 * x1 * x2

"""
    stat_ridge(x1, x2)

Stationary ridge system - eigenvalue of zero in Hessian.
"""
stat_ridge(x1, x2) = 80 + 4 * x1 + 8 * x2 - 3 * x1^2 - 12 * x2^2 - 12 * x1 * x2

"""
    rise_ridge(x1, x2)

Rising ridge system - optimal path extends to boundary.
"""
rise_ridge(x1, x2) = 80 - 4 * x1 + 12 * x2 - 3 * x1^2 - 12 * x2^2 - 12 * x1 * x2

"""
    saddle(x1, x2)

Saddle/minimax surface - local optimum is a saddle point.
"""
saddle(x1, x2) = 80 + 4 * x1 + 8 * x2 - 2 * x1^2 - 12 * x2^2 - 12 * x1 * x2

"""
    plot_rsm_surface(f, x1_range, x2_range; title="", resolution=100, aspect=nothing)

Create perspective and heatmap+contour plots for a response surface.

Arguments:
- f: Function of two variables (x1, x2)
- x1_range: Tuple of (min, max) for x1
- x2_range: Tuple of (min, max) for x2
- title: Plot title
- resolution: Number of grid points per dimension
- aspect: Aspect ratio for heatmap (nothing for auto, DataAspect() for equal scaling)
"""
function plot_rsm_surface(f, x1_range, x2_range; title="", resolution=100, aspect=nothing)
    x1 = range(x1_range..., length=resolution)
    x2 = range(x2_range..., length=resolution)
    z = [f(a, b) for a in x1, b in x2]

    fig = Figure(size=(1000, 400))

    # 3D Surface plot
    ax1 = Axis3(fig[1, 1], xlabel="x₁", ylabel="x₂", zlabel="Response",
                title=title, azimuth=0.9π)
    surface!(ax1, x1, x2, z, colormap=:viridis)

    # Heatmap with contours
    ax2 = if isnothing(aspect)
        Axis(fig[1, 2], xlabel="x₁", ylabel="x₂", title=title)
    else
        Axis(fig[1, 2], xlabel="x₁", ylabel="x₂", title=title, aspect=aspect)
    end
    hm = heatmap!(ax2, x1, x2, z, colormap=:heat)
    contour!(ax2, x1, x2, z, color=:black, linewidth=0.5, levels=10)
    Colorbar(fig[1, 3], hm, label="Response")

    return fig
end

# Plot banana yield function
println("\nPlotting banana yield function...")
xi1_range = (1.0, 8.0)
xi2_range = (100.0, 1000.0)
fig_yield = plot_rsm_surface(yield, xi1_range, xi2_range; title="Banana Yield Function")
save(joinpath(OUTPUT_DIR, "chap1_yield.png"), fig_yield)
println("Saved: chap1_yield.png")

# Plot all RSM examples
println("\nPlotting RSM examples...")
rsm_range = (-2.0, 2.0)

rsm_functions = [
    (first_order, "First-Order (Linear)"),
    (first_order_interaction, "First-Order with Interaction"),
    (simple_max, "Simple Maximum"),
    (stat_ridge, "Stationary Ridge"),
    (rise_ridge, "Rising Ridge"),
    (saddle, "Saddle/Minimax")
]

fig_rsm = Figure(size=(1200, 800))
for (i, (f, title)) in enumerate(rsm_functions)
    row = (i - 1) ÷ 3 + 1
    col = (i - 1) % 3 + 1

    x = range(rsm_range..., length=50)
    z = [f(a, b) for a in x, b in x]

    ax = Axis(fig_rsm[row, col], xlabel="x₁", ylabel="x₂", title=title,
              aspect=DataAspect())
    hm = heatmap!(ax, x, x, z, colormap=:viridis)
    contour!(ax, x, x, z, color=:white, linewidth=0.5, levels=8)
end
save(joinpath(OUTPUT_DIR, "chap1_rsm_examples.png"), fig_rsm)
println("Saved: chap1_rsm_examples.png")

# ============================================================================
# PART 2: Aircraft Wing Weight Simulator (True Function Visualization)
# ============================================================================
# This section evaluates the true simulator directly.
# The GP surrogate-based visualizations are in PART 5.

println("\n" * "="^70)
println("PART 2: Aircraft Wing Weight Simulator (True Function)")
println("="^70)

"""
    wingwt(; Sw=0.48, Wfw=0.4, A=0.38, L=0.5, q=0.62, l=0.344, Rtc=0.4, Nz=0.37, Wdg=0.38)

Aircraft wing weight simulator from NASA Langley.
Default values correspond to Cessna C172 Skyhawk baseline.

All inputs are coded to [0, 1] interval and transformed internally to natural units:
- Sw:  Wing area (150-200 ft²)
- Wfw: Weight of fuel in wing (220-300 lb)
- A:   Aspect ratio (6-10)
- L:   Quarter-chord sweep (-10° to 10°)
- q:   Dynamic pressure at cruise (16-45 lb/ft²)
- l:   Taper ratio (0.5-1.0)
- Rtc: Aerofoil thickness to chord ratio (0.08-0.18)
- Nz:  Ultimate load factor (2.5-6.0)
- Wdg: Flight design gross weight (1700-2500 lb)

Returns: Wing structural weight (lb)
"""
function wingwt(; Sw=0.48, Wfw=0.4, A=0.38, L=0.5, q=0.62,
                  l=0.344, Rtc=0.4, Nz=0.37, Wdg=0.38)
    # Transform coded [0,1] inputs to natural units
    Sw_nat = Sw * (200 - 150) + 150       # ft²
    Wfw_nat = Wfw * (300 - 220) + 220     # lb
    A_nat = A * (10 - 6) + 6              # dimensionless
    L_nat = (L * (10 - (-10)) - 10) * π / 180  # radians
    q_nat = q * (45 - 16) + 16            # lb/ft²
    l_nat = l * (1 - 0.5) + 0.5           # dimensionless
    Rtc_nat = Rtc * (0.18 - 0.08) + 0.08  # dimensionless
    Nz_nat = Nz * (6 - 2.5) + 2.5         # dimensionless
    Wdg_nat = Wdg * (2500 - 1700) + 1700  # lb

    # Wing weight formula (Equation 1.1 from Surrogates book)
    W = 0.036 * Sw_nat^0.758 * Wfw_nat^0.0035
    W *= (A_nat / cos(L_nat)^2)^0.6
    W *= q_nat^0.006
    W *= l_nat^0.04
    W *= (100 * Rtc_nat / cos(L_nat))^(-0.3)
    W *= (Nz_nat * Wdg_nat)^0.49

    return W
end

# Variable names for plotting
var_names = ["Sw", "Wfw", "A", "L", "q", "l", "Rtc", "Nz", "Wdg"]

# Baseline C172 values
baseline = [0.48, 0.4, 0.38, 0.5, 0.62, 0.344, 0.4, 0.37, 0.38]

# Test at baseline
W_baseline = wingwt()
println("\nWing weight at C172 baseline: $(round(W_baseline, digits=2)) lb")

# Create 2D slice visualizations
println("\nCreating 2D slice visualizations of wing weight...")

function wingwt_slice(var1_idx, var2_idx, var1_val, var2_val)
    inputs = copy(baseline)
    inputs[var1_idx] = var1_val
    inputs[var2_idx] = var2_val
    return wingwt(Sw=inputs[1], Wfw=inputs[2], A=inputs[3], L=inputs[4],
                  q=inputs[5], l=inputs[6], Rtc=inputs[7], Nz=inputs[8], Wdg=inputs[9])
end

# A (aspect ratio) vs Nz (load factor) slice
x = range(0.0, 1.0, length=100)
z_A_Nz = [wingwt_slice(3, 8, a, nz) for a in x, nz in x]

fig_slice1 = Figure(size=(500, 400))
ax1 = Axis(fig_slice1[1, 1], xlabel="A (aspect ratio)", ylabel="Nz (load factor)",
           title="Wing Weight: A × Nz Slice", aspect=DataAspect())
hm1 = heatmap!(ax1, x, x, z_A_Nz, colormap=:viridis)
contour!(ax1, x, x, z_A_Nz, color=:white, linewidth=0.5, levels=10)
Colorbar(fig_slice1[1, 2], hm1, label="Weight (lb)")
save(joinpath(OUTPUT_DIR, "chap1_wingwt_A_Nz.png"), fig_slice1)
println("Saved: chap1_wingwt_A_Nz.png")

# λ (taper ratio) vs Wfw (fuel weight) slice
z_l_Wfw = [wingwt_slice(6, 2, ll, wfw) for ll in x, wfw in x]

fig_slice2 = Figure(size=(500, 400))
ax2 = Axis(fig_slice2[1, 1], xlabel="λ (taper ratio)", ylabel="Wfw (fuel weight)",
           title="Wing Weight: λ × Wfw Slice", aspect=DataAspect())
hm2 = heatmap!(ax2, x, x, z_l_Wfw, colormap=:viridis)
contour!(ax2, x, x, z_l_Wfw, color=:white, linewidth=0.5, levels=10)
Colorbar(fig_slice2[1, 2], hm2, label="Weight (lb)")
save(joinpath(OUTPUT_DIR, "chap1_wingwt_l_Wfw.png"), fig_slice2)
println("Saved: chap1_wingwt_l_Wfw.png")

# ============================================================================
# PART 3: Latin Hypercube Sampling Design
# ============================================================================

println("\n" * "="^70)
println("PART 3: Latin Hypercube Sampling Design")
println("="^70)

# Generate LHS design
n = 1000
println("\nGenerating $(n)-point LHS design in 9 dimensions...")
plan, _ = LHCoptim(n, 9, 10)  # 1000 points, 9 dims, 10 optimization iterations
X = Matrix{Float64}(plan ./ n)

println("Design matrix shape: $(size(X))")
println("Design range: [$(minimum(X)), $(maximum(X))]")

# Visualize 2D projection
fig_lhs = Figure(size=(500, 450))
ax_lhs = Axis(fig_lhs[1, 1], xlabel="Sw (wing area)", ylabel="Wfw (fuel weight)",
              title="LHS Design Projection (Sw × Wfw)")
scatter!(ax_lhs, X[:, 1], X[:, 2], markersize=4, color=:blue, alpha=0.6)
# Show space-filling bands
hlines!(ax_lhs, [0.6, 0.8], color=:red, linewidth=2, linestyle=:dash)
save(joinpath(OUTPUT_DIR, "chap1_lhs_design.png"), fig_lhs)
println("Saved: chap1_lhs_design.png")

# Verify space-filling property
inbox = sum((X[:, 1] .> 0.6) .& (X[:, 1] .< 0.8)) / n
println("\nSpace-filling check: $(round(100*inbox, digits=1))% of points in [0.6, 0.8] band")
println("(Should be approximately 20%)")

# ============================================================================
# PART 4: GP Surrogate Fitting (Isotropic vs Separable)
# ============================================================================

println("\n" * "="^70)
println("PART 4: GP Surrogate Fitting (Isotropic vs Separable)")
println("="^70)

# Evaluate simulator at design points
println("\nEvaluating wing weight simulator at $(n) design points...")
Y = Vector{Float64}(undef, n)
for i in 1:n
    Y[i] = wingwt(Sw=X[i,1], Wfw=X[i,2], A=X[i,3], L=X[i,4],
                  q=X[i,5], l=X[i,6], Rtc=X[i,7], Nz=X[i,8], Wdg=X[i,9])
end

println("Response range: [$(round(minimum(Y), digits=2)), $(round(maximum(Y), digits=2))] lb")

# Note: The R example uses Y directly for the GP fit, NOT log(Y)
# The log transform in R was only used for the linear regression comparison

# ---- Isotropic GP ----
println("\n--- Isotropic GP ---")
d_range = darg(X)
g_range = garg(Y)

println("Hyperparameter ranges:")
println("  Lengthscale d: min=$(round(d_range.min, sigdigits=3)), max=$(round(d_range.max, sigdigits=3)), start=$(round(d_range.start, sigdigits=3))")
println("  Nugget g: min=$(round(g_range.min, sigdigits=3)), max=$(round(g_range.max, sigdigits=3)), start=$(round(g_range.start, sigdigits=3))")

println("\nFitting isotropic GP model...")
gp_iso = new_gp(X, Y, d_range.start, g_range.start)
println("Initial log-likelihood: $(round(llik_gp(gp_iso), digits=2))")

jmle_gp!(gp_iso; drange=(d_range.min, d_range.max), grange=(g_range.min, g_range.max))
println("After MLE: d=$(round(gp_iso.d, sigdigits=4)), g=$(round(gp_iso.g, sigdigits=4))")
println("Final log-likelihood: $(round(llik_gp(gp_iso), digits=2))")

# ---- Separable GP ----
println("\n--- Separable GP (matching R's newGPsep) ---")
d_range_sep = darg_sep(X)

println("Per-dimension lengthscale ranges from darg_sep:")
for (j, name) in enumerate(var_names)
    r = d_range_sep.ranges[j]
    println("  $name: min=$(round(r.min, sigdigits=3)), max=$(round(r.max, sigdigits=3)), start=$(round(r.start, sigdigits=3))")
end

# Initial lengthscales and ranges from darg_sep
d_start_sep = [r.start for r in d_range_sep.ranges]
d_ranges_sep = [(r.min, r.max) for r in d_range_sep.ranges]

println("\nFitting separable GP model...")
gp_sep = new_gp_sep(X, Y, d_start_sep, g_range.start)
println("Initial log-likelihood: $(round(llik_gp_sep(gp_sep), digits=2))")

jmle_gp_sep!(gp_sep; drange=d_ranges_sep, grange=(g_range.min, g_range.max))
println("After MLE:")
println("  Lengthscales d:")
for (j, name) in enumerate(var_names)
    println("    $name: $(round(gp_sep.d[j], sigdigits=4))")
end
println("  Nugget g: $(round(gp_sep.g, sigdigits=4))")
println("Final log-likelihood: $(round(llik_gp_sep(gp_sep), digits=2))")

# Note: When all lengthscales converge to similar values, the separable GP
# reduces to the isotropic case. This happens when the function is smooth
# in all dimensions. For problems with true anisotropic behavior (different
# smoothness in different directions), the separable GP will find different
# lengthscales per dimension and may provide better predictions.

# For backward compatibility, keep gp as reference (use separable)
gp = gp_sep

# ============================================================================
# PART 5: GP Surrogate Predictions (2D Slices) - Isotropic vs Separable
# ============================================================================
# This section uses pred_gp and pred_gp_sep to make surrogate predictions,
# mirroring Figure 1.12 from the R example. Compare with PART 2 which uses
# the true function directly.

println("\n" * "="^70)
println("PART 5: GP Surrogate Predictions (Isotropic vs Separable)")
println("="^70)

# Create prediction grid for A × Nz slice
println("\nCreating prediction grid for A × Nz slice...")
n_pred = 100
x_pred = range(0.0, 1.0, length=n_pred)
XX = Matrix{Float64}(undef, n_pred^2, 9)

# Fill with baseline values
for i in 1:n_pred^2
    XX[i, :] .= baseline
end

# Vary A (column 3) and Nz (column 8)
let idx = 1
    for nz in x_pred
        for a in x_pred
            XX[idx, 3] = a
            XX[idx, 8] = nz
            idx += 1
        end
    end
end

# Make predictions with both models
println("Making GP predictions...")

# Isotropic predictions
pred_iso = pred_gp(gp_iso, XX; lite=true)
pred_mean_iso = pred_iso.mean

# Separable predictions
pred_sep = pred_gp_sep(gp_sep, XX; lite=true)
pred_mean_sep = pred_sep.mean

# True values
true_vals = [wingwt_slice(3, 8, a, nz) for a in x_pred, nz in x_pred]

# Calculate RMSE for both
rmse_iso = sqrt(mean((vec(true_vals) .- pred_mean_iso).^2))
rmse_sep = sqrt(mean((vec(true_vals) .- pred_mean_sep).^2))

println("\nA × Nz slice results:")
println("  Isotropic GP RMSE: $(round(rmse_iso, digits=4)) lb")
println("  Separable GP RMSE: $(round(rmse_sep, digits=4)) lb")
println("  Improvement: $(round((rmse_iso - rmse_sep) / rmse_iso * 100, digits=1))%")

# Reshape for plotting
surrogate_vals_iso = reshape(pred_mean_iso, n_pred, n_pred)
surrogate_vals_sep = reshape(pred_mean_sep, n_pred, n_pred)

# Plot comparison: True vs Isotropic vs Separable
fig_compare = Figure(size=(1200, 800))

# Row 1: Surfaces
ax1 = Axis(fig_compare[1, 1], xlabel="A", ylabel="Nz",
           title="True Function", aspect=DataAspect())
hm1 = heatmap!(ax1, collect(x_pred), collect(x_pred), true_vals, colormap=:viridis)
contour!(ax1, collect(x_pred), collect(x_pred), true_vals, color=:white, linewidth=0.5, levels=10)

ax2 = Axis(fig_compare[1, 2], xlabel="A", ylabel="Nz",
           title="Isotropic GP (RMSE=$(round(rmse_iso, digits=3)))", aspect=DataAspect())
hm2 = heatmap!(ax2, collect(x_pred), collect(x_pred), surrogate_vals_iso, colormap=:viridis)
contour!(ax2, collect(x_pred), collect(x_pred), surrogate_vals_iso, color=:white, linewidth=0.5, levels=10)

ax3 = Axis(fig_compare[1, 3], xlabel="A", ylabel="Nz",
           title="Separable GP (RMSE=$(round(rmse_sep, digits=3)))", aspect=DataAspect())
hm3 = heatmap!(ax3, collect(x_pred), collect(x_pred), surrogate_vals_sep, colormap=:viridis)
contour!(ax3, collect(x_pred), collect(x_pred), surrogate_vals_sep, color=:white, linewidth=0.5, levels=10)

Colorbar(fig_compare[1, 4], hm1, label="Weight (lb)")

# Row 2: Errors
error_vals_iso = surrogate_vals_iso .- true_vals
error_vals_sep = surrogate_vals_sep .- true_vals
max_err = max(maximum(abs.(error_vals_iso)), maximum(abs.(error_vals_sep)))

ax4 = Axis(fig_compare[2, 1], xlabel="A", ylabel="Nz",
           title="Isotropic Error", aspect=DataAspect())
hm4 = heatmap!(ax4, collect(x_pred), collect(x_pred), error_vals_iso,
               colormap=:RdBu, colorrange=(-max_err, max_err))
contour!(ax4, collect(x_pred), collect(x_pred), error_vals_iso, color=:black, linewidth=0.5, levels=8)

ax5 = Axis(fig_compare[2, 2], xlabel="A", ylabel="Nz",
           title="Separable Error", aspect=DataAspect())
hm5 = heatmap!(ax5, collect(x_pred), collect(x_pred), error_vals_sep,
               colormap=:RdBu, colorrange=(-max_err, max_err))
contour!(ax5, collect(x_pred), collect(x_pred), error_vals_sep, color=:black, linewidth=0.5, levels=8)

# Error histogram comparison
ax6 = Axis(fig_compare[2, 3], xlabel="Error (lb)", ylabel="Count",
           title="Error Distribution")
hist!(ax6, vec(error_vals_iso), bins=50, color=(:blue, 0.5), label="Isotropic")
hist!(ax6, vec(error_vals_sep), bins=50, color=(:red, 0.5), label="Separable")
axislegend(ax6, position=:rt)

Colorbar(fig_compare[2, 4], hm4, label="Error (lb)")

save(joinpath(OUTPUT_DIR, "chap1_surrogate_comparison.png"), fig_compare)
println("Saved: chap1_surrogate_comparison.png")

# Store separable RMSE for summary
rmse = rmse_sep

# ============================================================================
# PART 6: Main Effects Analysis (using Separable GP)
# ============================================================================

println("\n" * "="^70)
println("PART 6: Main Effects Analysis (using Separable GP)")
println("="^70)

println("\nComputing main effects for all 9 inputs...")

n_me = 100
x_me = range(0.0, 1.0, length=n_me)

# Storage for main effects and confidence bands
me = Matrix{Float64}(undef, n_me, 9)       # Mean predictions
meq_lo = Matrix{Float64}(undef, n_me, 9)   # Lower 90% CI
meq_hi = Matrix{Float64}(undef, n_me, 9)   # Upper 90% CI

# Loop through each input variable
for j in 1:9
    # Create prediction matrix with baseline values
    XX_me = Matrix{Float64}(undef, n_me, 9)
    for i in 1:n_me
        XX_me[i, :] .= baseline
    end

    # Vary the j-th input
    XX_me[:, j] .= collect(x_me)

    # Make predictions using separable GP
    p = pred_gp_sep(gp_sep, XX_me; lite=true)

    # Store mean predictions
    me[:, j] .= p.mean

    # Confidence bands using Student-t distribution
    # 90% CI: 5th and 95th percentiles
    t_dist = TDist(p.df)
    t_lo = quantile(t_dist, 0.05)
    t_hi = quantile(t_dist, 0.95)

    for i in 1:n_me
        sd = sqrt(p.s2[i])
        meq_lo[i, j] = p.mean[i] + t_lo * sd
        meq_hi[i, j] = p.mean[i] + t_hi * sd
    end
end

# Plot main effects
fig_me = Figure(size=(800, 500))
ax_me = Axis(fig_me[1, 1], xlabel="Coded Input [0, 1]", ylabel="Wing Weight (lb)",
             title="Main Effects Analysis")

colors = [:blue, :red, :green, :orange, :purple, :brown, :pink, :gray, :cyan]

for j in 1:9
    lines!(ax_me, collect(x_me), me[:, j], color=colors[j], linewidth=2, label=var_names[j])
    # Add confidence bands (dashed)
    lines!(ax_me, collect(x_me), meq_lo[:, j], color=colors[j], linewidth=1, linestyle=:dash)
    lines!(ax_me, collect(x_me), meq_hi[:, j], color=colors[j], linewidth=1, linestyle=:dash)
end

# Split legend into two rows for readability
Legend(fig_me[1, 2], ax_me, nbanks=1)

save(joinpath(OUTPUT_DIR, "chap1_main_effects.png"), fig_me)
println("Saved: chap1_main_effects.png")

# Report most influential variables
println("\nVariable sensitivity (range of main effect):")
for j in 1:9
    effect_range = maximum(me[:, j]) - minimum(me[:, j])
    println("  $(var_names[j]): $(round(effect_range, digits=2)) lb")
end

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^70)
println("Chapter 1 Example Complete!")
println("="^70)

println("\nGenerated files:")
println("  - chap1_yield.png: Banana yield response surface")
println("  - chap1_rsm_examples.png: RSM function gallery")
println("  - chap1_wingwt_A_Nz.png: Wing weight A×Nz slice (true function)")
println("  - chap1_wingwt_l_Wfw.png: Wing weight λ×Wfw slice (true function)")
println("  - chap1_lhs_design.png: Latin Hypercube design projection")
println("  - chap1_surrogate_comparison.png: Isotropic vs Separable GP (A×Nz slice)")
println("  - chap1_main_effects.png: Main effects with confidence bands (Separable GP)")

println("\nKey findings:")
println("  - Most influential inputs: Sw (wing area), A (aspect ratio), Nz (load factor)")
println("  - Least influential inputs: l (taper ratio), Wfw (fuel weight)")
println("  - Isotropic GP vs Separable GP comparison (A×Nz slice):")
println("      Isotropic RMSE = $(round(rmse_iso, digits=4)) lb, Separable RMSE = $(round(rmse_sep, digits=4)) lb")
println("  - For this smooth function, MLE may converge to similar lengthscales")
println("    for all dimensions, making separable GP behave like isotropic GP.")
println("  - Separable GP benefits are more pronounced for functions with true")
println("    anisotropic behavior (different smoothness in different directions).")
println("  - Confidence bands are negligible (deterministic simulator)")
