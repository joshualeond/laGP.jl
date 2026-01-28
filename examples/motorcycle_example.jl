# Motorcycle Crash Test Example: Full GP vs Local Approximate GP
#
# This example demonstrates:
# 1. Fitting a full GP to the classic motorcycle crash test (mcycle) dataset
# 2. Using local approximate GP (aGP) for the same data
# 3. Comparing predictions and uncertainty quantification
# 4. Scaling to larger datasets with replicated observations
#
# The mcycle dataset contains 133 observations of head acceleration (g)
# measured at various times (ms) after a simulated motorcycle crash.

using laGP
using Distributions
using Random
using CairoMakie

# ============================================================================
# Part 1: Motorcycle Crash Test Data (mcycle)
# ============================================================================

# Embedded mcycle dataset from MASS R package (133 observations)
# Source: https://vincentarelbundock.github.io/Rdatasets/csv/MASS/mcycle.csv
# times: time after impact in milliseconds
# accel: head acceleration in g
const mcycle_times = [
    2.4, 2.6, 3.2, 3.6, 4, 6.2, 6.6, 6.8, 7.8, 8.2, 8.8, 8.8, 9.6,
    10, 10.2, 10.6, 11, 11.4, 13.2, 13.6, 13.8, 14.6, 14.6, 14.6,
    14.6, 14.6, 14.6, 14.8, 15.4, 15.4, 15.4, 15.4, 15.6, 15.6, 15.8,
    15.8, 16, 16, 16.2, 16.2, 16.2, 16.4, 16.4, 16.6, 16.8, 16.8,
    16.8, 17.6, 17.6, 17.6, 17.6, 17.8, 17.8, 18.6, 18.6, 19.2, 19.4,
    19.4, 19.6, 20.2, 20.4, 21.2, 21.4, 21.8, 22, 23.2, 23.4, 24,
    24.2, 24.2, 24.6, 25, 25, 25.4, 25.4, 25.6, 26, 26.2, 26.2,
    26.4, 27, 27.2, 27.2, 27.2, 27.6, 28.2, 28.4, 28.4, 28.6, 29.4,
    30.2, 31, 31.2, 32, 32, 32.8, 33.4, 33.8, 34.4, 34.8, 35.2,
    35.2, 35.4, 35.6, 35.6, 36.2, 36.2, 38, 38, 39.2, 39.4, 40,
    40.4, 41.6, 41.6, 42.4, 42.8, 42.8, 43, 44, 44.4, 45, 46.6,
    47.8, 47.8, 48.8, 50.6, 52, 53.2, 55, 55, 55.4, 57.6
]

const mcycle_accel = [
    0, -1.3, -2.7, 0, -2.7, -2.7, -2.7, -1.3, -2.7, -2.7, -1.3,
    -2.7, -2.7, -2.7, -5.4, -2.7, -5.4, 0, -2.7, -2.7, 0, -13.3,
    -5.4, -5.4, -9.3, -16, -22.8, -2.7, -22.8, -32.1, -53.5, -54.9,
    -40.2, -21.5, -21.5, -50.8, -42.9, -26.8, -21.5, -50.8, -61.7,
    -5.4, -80.4, -59, -71, -91.1, -77.7, -37.5, -85.6, -123.1,
    -101.9, -99.1, -104.4, -112.5, -50.8, -123.1, -85.6, -72.3,
    -127.2, -123.1, -117.9, -134, -101.9, -108.4, -123.1, -123.1,
    -128.5, -112.5, -95.1, -81.8, -53.5, -64.4, -57.6, -72.3, -44.3,
    -26.8, -5.4, -107.1, -21.5, -65.6, -16, -45.6, -24.2, 9.5,
    4, 12, -21.5, 37.5, 46.9, -17.4, 36.2, 75, 8.1, 54.9,
    48.2, 46.9, 16, 45.6, 1.3, 75, -16, -54.9, 69.6, 34.8,
    32.1, -37.5, 22.8, 46.9, 10.7, 5.4, -1.3, -21.5, -13.3, 30.8,
    -10.7, 29.4, 0, -10.7, 14.7, -1.3, 0, 10.7, 10.7, -26.8,
    -14.7, -13.3, 0, 10.7, -14.7, -2.7, 10.7, -2.7, 10.7
]

# Format as matrices (rows as observations)
X = reshape(Float64.(mcycle_times), :, 1)
Z = Float64.(mcycle_accel)
n = length(Z)

println("Motorcycle Crash Test Data")
println("  Observations: ", n)
println("  Time range: ", minimum(X), " - ", maximum(X), " ms")
println("  Accel range: ", minimum(Z), " - ", maximum(Z), " g")

# ============================================================================
# Part 2: Parameter Setup Using Data-Driven Defaults
# ============================================================================

# Get data-driven parameter ranges
da = darg(X)
ga = garg(Z)

println("\nData-driven parameter ranges:")
println("  d: start=", round(da.start, digits=4), ", min=", round(da.min, digits=4),
        ", max=", round(da.max, digits=4))
println("  g: start=", round(ga.start, digits=4), ", min=", round(ga.min, digits=6),
        ", max=", round(ga.max, digits=4))

# ============================================================================
# Part 3: Full GP with Joint MLE
# ============================================================================

println("\n--- Full GP ---")

# Create GP with data-driven initial values
gp = new_gp(X, Z, da.start, ga.start)

# Joint MLE for both d and g
result = jmle_gp!(gp; drange=(da.min, da.max), grange=(ga.min, ga.max))

println("MLE results:")
println("  d = ", round(gp.d, digits=4))
println("  g = ", round(gp.g, digits=4))
println("  iterations = ", result.tot_its)
println("  status = ", result.msg)

# Prediction grid
xx = collect(range(minimum(X) - 2, maximum(X) + 2, length=200))
XX = reshape(xx, :, 1)

# Predict with full GP (lite=true for mean and variance only)
pred_full_gp = pred_gp(gp, XX; lite=true)

# Compute 90% credible intervals
# For Student-t: quantile ≈ 1.645 for large df
z_90 = quantile(Normal(), 0.95)
gp_lower = pred_full_gp.mean .- z_90 .* sqrt.(pred_full_gp.s2)
gp_upper = pred_full_gp.mean .+ z_90 .* sqrt.(pred_full_gp.s2)

# ============================================================================
# Part 4: Local Approximate GP (aGP)
# ============================================================================

println("\n--- Local Approximate GP (aGP) ---")

# Run aGP with MLE for both d and g
# Use NamedTuples to enable MLE optimization
d_agp = (start=da.start, min=da.min, max=da.max, mle=true)
g_agp = (start=ga.start, min=ga.min, max=ga.max, mle=true)

pred_agp = agp(X, Z, XX; endpt=30, d=d_agp, g=g_agp, method=:alc, verb=0)

println("aGP prediction complete")
println("  Test points: ", length(xx))
println("  Local neighborhood size: 30")

# Compute 90% credible intervals for aGP
agp_lower = pred_agp.mean .- z_90 .* sqrt.(pred_agp.var)
agp_upper = pred_agp.mean .+ z_90 .* sqrt.(pred_agp.var)

# ============================================================================
# Part 5: Visualization - Full GP vs aGP (Overlaid, matching R Figure 10)
# ============================================================================

fig1 = Figure(size=(700, 500))

ax1 = Axis(fig1[1, 1],
           xlabel="Time (ms)",
           ylabel="Acceleration (g)",
           title="GP vs aGP Comparison")

# Data points
scatter!(ax1, vec(X), Z, color=:black, markersize=6)

# Full GP: black solid mean, black dashed CIs
lines!(ax1, xx, pred_full_gp.mean, color=:black, linewidth=2, label="GP mean")
lines!(ax1, xx, gp_lower, color=:black, linewidth=1, linestyle=:dash, label="GP 90% CI")
lines!(ax1, xx, gp_upper, color=:black, linewidth=1, linestyle=:dash)

# aGP: red solid mean, red dashed CIs
lines!(ax1, xx, pred_agp.mean, color=:red, linewidth=2, label="aGP mean")
lines!(ax1, xx, agp_lower, color=:red, linewidth=1, linestyle=:dash, label="aGP 90% CI")
lines!(ax1, xx, agp_upper, color=:red, linewidth=1, linestyle=:dash)

axislegend(ax1, position=:rb)

# Save figure
output_path1 = joinpath(@__DIR__, "motorcycle_gp_vs_agp.png")
save(output_path1, fig1, px_per_unit=2)
println("\nFigure 1 saved to: ", output_path1)

# ============================================================================
# Part 6: Scaled Dataset - 10x Replication with Noise
# ============================================================================

println("\n--- Scaled Dataset (10x replication) ---")

# Replicate data 10 times with jitter on X (matching R laGP paper)
Random.seed!(42)
n_rep = 10
X_big = repeat(X, n_rep) .+ randn(n * n_rep) .* 1.0  # Jitter on X (sd=1)
Z_big = repeat(Z, n_rep)                              # No noise on Z

println("Enlarged dataset:")
println("  Observations: ", length(Z_big))

# aGP on enlarged dataset with FIXED hyperparameters from the full GP MLE
# This matches R: aGP(X, Z, XX, end = 30, d = d, g = g, verb = 0)
# Using gp.d and gp.g prevents over-fitting to the enlarged dataset
pred_agp_big = agp(X_big, Z_big, XX; endpt=30, d=gp.d, g=gp.g,
                   method=:alc, verb=0)

# Compute 90% credible intervals
agp_big_lower = pred_agp_big.mean .- z_90 .* sqrt.(pred_agp_big.var)
agp_big_upper = pred_agp_big.mean .+ z_90 .* sqrt.(pred_agp_big.var)

# ============================================================================
# Part 7: Visualization - aGP on Enlarged Dataset
# ============================================================================

fig2 = Figure(size=(700, 500))

ax3 = Axis(fig2[1, 1],
           xlabel="Time (ms)",
           ylabel="Acceleration (g)",
           title="aGP on Enlarged Motorcycle Data (n=$(n * n_rep))")

scatter!(ax3, vec(X_big), Z_big, color=(:gray, 0.3), markersize=4, label="Data (10x replicated)")
lines!(ax3, xx, pred_agp_big.mean, color=:red, linewidth=2, label="aGP mean")
lines!(ax3, xx, agp_big_lower, color=:red, linewidth=1, linestyle=:dash, label="aGP 90% CI")
lines!(ax3, xx, agp_big_upper, color=:red, linewidth=1, linestyle=:dash)
axislegend(ax3, position=:rb)

# Save figure
output_path2 = joinpath(@__DIR__, "motorcycle_agp_scaled.png")
save(output_path2, fig2, px_per_unit=2)
println("Figure 2 saved to: ", output_path2)

println("\nMotorcycle example complete!")
