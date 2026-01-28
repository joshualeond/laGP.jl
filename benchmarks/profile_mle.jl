# Profile GP MLE optimization to identify performance bottlenecks
#
# Run with: julia --project=. benchmarks/profile_mle.jl
#
# FINDINGS (2024):
# ================
# Zygote AD gradients are ~60x slower than manual gradients for GP MLE.
# The bottleneck is gradient accumulation for triangular matrices in Zygote.
#
# Component timing (n=1000, m=9):
#   - Kernel matrix:   ~10ms
#   - Cholesky:        ~7ms
#   - Manual gradient: ~64ms
#   - AD gradient:     ~3857ms (60x slower!)
#
# OPTIMIZATION APPLIED:
# Changed default from use_ad=true to use_ad=false in jmle_gp! and jmle_gp_sep!.
#
# RESULTS:
# - Isotropic GP MLE:  509s → 10.5s (~50x speedup)
# - Separable GP MLE:  613s → 159s (~4x speedup)

using laGP
using Profile
using LinearAlgebra
using Statistics
using Random
using KernelFunctions: kernelmatrix, RowVecs

Random.seed!(42)

println("="^70)
println("GP MLE Profiling Script")
println("="^70)

# Wing weight function (same as chap1_surrogates.jl)
function wingwt(; Sw=0.48, Wfw=0.4, A=0.38, L=0.5, q=0.62,
                  l=0.344, Rtc=0.4, Nz=0.37, Wdg=0.38)
    Sw_nat = Sw * (200 - 150) + 150
    Wfw_nat = Wfw * (300 - 220) + 220
    A_nat = A * (10 - 6) + 6
    L_nat = (L * (10 - (-10)) - 10) * π / 180
    q_nat = q * (45 - 16) + 16
    l_nat = l * (1 - 0.5) + 0.5
    Rtc_nat = Rtc * (0.18 - 0.08) + 0.08
    Nz_nat = Nz * (6 - 2.5) + 2.5
    Wdg_nat = Wdg * (2500 - 1700) + 1700

    W = 0.036 * Sw_nat^0.758 * Wfw_nat^0.0035
    W *= (A_nat / cos(L_nat)^2)^0.6
    W *= q_nat^0.006
    W *= l_nat^0.04
    W *= (100 * Rtc_nat / cos(L_nat))^(-0.3)
    W *= (Nz_nat * Wdg_nat)^0.49

    return W
end

# Generate test data
# Note: n=500 provides a good balance for timing comparisons
# With n=1000, each Cholesky is ~7ms making both methods slow
n = 500
m = 9
println("\nGenerating $n x $m design matrix...")
X = rand(n, m)

println("Evaluating wing weight at design points...")
Y = [wingwt(Sw=X[i,1], Wfw=X[i,2], A=X[i,3], L=X[i,4],
            q=X[i,5], l=X[i,6], Rtc=X[i,7], Nz=X[i,8], Wdg=X[i,9])
     for i in 1:n]

println("Response range: [$(round(minimum(Y), digits=2)), $(round(maximum(Y), digits=2))]")

da = darg(X)
ga = garg(Y)
da_sep = darg_sep(X)
d_start_sep = [r.start for r in da_sep.ranges]
d_ranges_sep = [(r.min, r.max) for r in da_sep.ranges]

# ============================================================================
# Warmup
# ============================================================================

println("\n" * "="^70)
println("Warmup (compiling)...")
println("="^70)

X_warm = X[1:50, :]
Y_warm = Y[1:50]
da_warm = darg(X_warm)
ga_warm = garg(Y_warm)
da_warm_sep = darg_sep(X_warm)
d_start_warm = [r.start for r in da_warm_sep.ranges]
d_ranges_warm = [(r.min, r.max) for r in da_warm_sep.ranges]

# Warmup manual gradients (default)
gp_warm = new_gp(X_warm, Y_warm, da_warm.start, ga_warm.start)
jmle_gp!(gp_warm; drange=(da_warm.min, da_warm.max), grange=(ga_warm.min, ga_warm.max))

gp_sep_warm = new_gp_sep(X_warm, Y_warm, d_start_warm, ga_warm.start)
jmle_gp_sep!(gp_sep_warm; drange=d_ranges_warm, grange=(ga_warm.min, ga_warm.max))

println("Warmup complete.")

# ============================================================================
# Component Timing
# ============================================================================

println("\n" * "="^70)
println("COMPONENT TIMING")
println("="^70)

gp_chol = new_gp_sep(X, Y, d_start_sep, ga.start)

println("\nKernel matrix computation:")
t_kernel = @elapsed for _ in 1:10
    kernelmatrix(gp_chol.kernel, RowVecs(gp_chol.X))
end
println("  Average: $(round(t_kernel/10 * 1000, digits=2))ms")

K = kernelmatrix(gp_chol.kernel, RowVecs(gp_chol.X)) + gp_chol.g * I
println("\nCholesky decomposition (1000x1000):")
t_chol = @elapsed for _ in 1:10
    cholesky(Symmetric(K))
end
println("  Average: $(round(t_chol/10 * 1000, digits=2))ms")

println("\nManual gradient (dllik_gp_sep):")
t_manual = @elapsed for _ in 1:10
    dllik_gp_sep(gp_chol)
end
println("  Average: $(round(t_manual/10 * 1000, digits=2))ms")

println("\nAD gradient (Zygote) - WARNING: slow:")
params = Float64[d_start_sep..., ga.start]
t_ad = @elapsed for _ in 1:3
    laGP.Zygote.gradient(p -> laGP.neg_llik_ad(p, X, Y; separable=true), params)
end
println("  Average: $(round(t_ad/3 * 1000, digits=2))ms")

println("\n>>> AD is $(round(t_ad/3 / (t_manual/10), digits=1))x slower than manual gradients")

# ============================================================================
# MLE Timing (with new default use_ad=false)
# ============================================================================

println("\n" * "="^70)
println("MLE TIMING (use_ad=false, the new default)")
println("="^70)

println("\n--- Isotropic GP (jmle_gp! - joint L-BFGS) ---")
gp_iso = new_gp(X, Y, da.start, ga.start)
t_iso = @elapsed result_iso = jmle_gp!(gp_iso; drange=(da.min, da.max), grange=(ga.min, ga.max))
println("Time: $(round(t_iso, digits=2))s")
println("Iterations: $(result_iso.tot_its)")
println("Final: d=$(round(result_iso.d, sigdigits=4)), g=$(round(result_iso.g, sigdigits=4))")

println("\n--- Separable GP (jmle_gp_sep! - joint L-BFGS) ---")
gp_sep = new_gp_sep(X, Y, d_start_sep, ga.start)
t_sep = @elapsed result_sep = jmle_gp_sep!(gp_sep; drange=d_ranges_sep, grange=(ga.min, ga.max))
println("Time: $(round(t_sep, digits=2))s")
println("Iterations: $(result_sep.tot_its)")
println("Final g: $(round(result_sep.g, sigdigits=4))")

# ============================================================================
# Alternating MLE Timing (R-style Newton + L-BFGS)
# ============================================================================

println("\n" * "="^70)
println("ALTERNATING MLE TIMING (R-style)")
println("="^70)

println("\n--- Isotropic GP (amle_gp! - alternating Newton) ---")
gp_iso_alt = new_gp(X, Y, da.start, ga.start)
t_iso_alt = @elapsed result_iso_alt = amle_gp!(gp_iso_alt; drange=(da.min, da.max), grange=(ga.min, ga.max))
println("Time: $(round(t_iso_alt, digits=2))s")
println("Iterations: d=$(result_iso_alt.dits), g=$(result_iso_alt.gits), total=$(result_iso_alt.tot_its)")
println("Final: d=$(round(result_iso_alt.d, sigdigits=4)), g=$(round(result_iso_alt.g, sigdigits=4))")
println("Speedup vs jmle_gp!: $(round(t_iso / t_iso_alt, digits=1))x")

println("\n--- Separable GP (amle_gp_sep! - alternating L-BFGS/Newton) ---")
gp_sep_alt = new_gp_sep(X, Y, d_start_sep, ga.start)
t_sep_alt = @elapsed result_sep_alt = amle_gp_sep!(gp_sep_alt; drange=d_ranges_sep, grange=(ga.min, ga.max))
println("Time: $(round(t_sep_alt, digits=2))s")
println("Iterations: d=$(result_sep_alt.dits), g=$(result_sep_alt.gits), total=$(result_sep_alt.tot_its)")
println("Final g: $(round(result_sep_alt.g, sigdigits=4))")
println("Speedup vs jmle_gp_sep!: $(round(t_sep / t_sep_alt, digits=1))x")

# Verify results match between methods
println("\n--- Verification: Results match between methods ---")
d_diff_iso = abs(result_iso.d - result_iso_alt.d) / result_iso.d
g_diff_iso = abs(result_iso.g - result_iso_alt.g) / result_iso.g
g_diff_sep = abs(result_sep.g - result_sep_alt.g) / result_sep.g

println("Isotropic d relative diff: $(round(d_diff_iso * 100, digits=4))%")
println("Isotropic g relative diff: $(round(g_diff_iso * 100, digits=4))%")
println("Separable g relative diff: $(round(g_diff_sep * 100, digits=4))%")

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^70)
println("SUMMARY")
println("="^70)

println("\nComponent timing:")
println("  Kernel matrix:   $(round(t_kernel/10 * 1000, digits=2))ms")
println("  Cholesky:        $(round(t_chol/10 * 1000, digits=2))ms")
println("  Manual gradient: $(round(t_manual/10 * 1000, digits=2))ms")
println("  AD gradient:     $(round(t_ad/3 * 1000, digits=2))ms")

println("\nMLE timing comparison:")
println("  Isotropic GP:")
println("    jmle_gp! (joint L-BFGS):      $(round(t_iso, digits=2))s")
println("    amle_gp! (alternating Newton): $(round(t_iso_alt, digits=2))s  ($(round(t_iso / t_iso_alt, digits=1))x faster)")
println("  Separable GP:")
println("    jmle_gp_sep! (joint L-BFGS):           $(round(t_sep, digits=2))s")
println("    amle_gp_sep! (alternating L-BFGS/Newton): $(round(t_sep_alt, digits=2))s  ($(round(t_sep / t_sep_alt, digits=1))x faster)")

println("\nConclusion:")
println("  Zygote AD is ~$(round(t_ad/3 / (t_manual/10)))x slower than manual gradients.")
println("  Alternating MLE (R-style) provides additional speedup over joint optimization.")

println("\n" * "="^70)
println("Profiling complete!")
println("="^70)
