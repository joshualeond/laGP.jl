# Borehole Example: Julia laGP.jl Benchmark
#
# Classic 8D borehole function benchmark for comparing laGP.jl performance
# against the R/C implementation. The borehole function models water flow
# through a borehole with 8 input parameters.
#
# Reference: Surrogates: Gaussian Process Modeling, Design and Optimization
# by Robert Gramacy (Appendix A)
#
# Run with: julia --project=. -t auto examples/borehole_example.jl

using laGP
using Random
using LatinHypercubeSampling
using Printf
using Statistics: mean

# Set random seed for reproducibility
Random.seed!(42)

# Output directory (same as this script)
const OUTPUT_DIR = @__DIR__

# ============================================================================
# Borehole Function
# ============================================================================

"""
    borehole(x::Vector{T}) where {T}

Borehole function - models water flow through a borehole.

All 8 inputs are coded to [0, 1] and transformed internally to natural units:
- x[1] rw:  Borehole radius (0.05-0.15 m)
- x[2] r:   Influence radius (100-50000 m)
- x[3] Tu:  Upper aquifer transmissivity (63070-115600 m²/yr)
- x[4] Hu:  Upper aquifer potentiometric head (990-1110 m)
- x[5] Tl:  Lower aquifer transmissivity (63.1-116 m²/yr)
- x[6] Hl:  Lower aquifer potentiometric head (700-820 m)
- x[7] L:   Borehole length (1120-1680 m)
- x[8] Kw:  Hydraulic conductivity of borehole (9855-12045 m/yr)

Returns: Water flow rate (m³/yr)
"""
function borehole(x::Vector{T}) where {T}
    # Transform coded [0,1] inputs to natural units
    rw = x[1] * (0.15 - 0.05) + 0.05
    r  = x[2] * (50000 - 100) + 100
    Tu = x[3] * (115600 - 63070) + 63070
    Hu = x[4] * (1110 - 990) + 990
    Tl = x[5] * (116 - 63.1) + 63.1
    Hl = x[6] * (820 - 700) + 700
    L  = x[7] * (1680 - 1120) + 1120
    Kw = x[8] * (12045 - 9855) + 9855

    # Borehole flow equation
    m1 = 2π * Tu * (Hu - Hl)
    m2 = log(r / rw)
    m3 = 1 + 2L * Tu / (m2 * rw^2 * Kw) + Tu / Tl

    return m1 / m2 / m3
end

# Vectorized version for matrix input (rows as observations)
function borehole(X::Matrix{T}) where {T}
    n = size(X, 1)
    Z = Vector{T}(undef, n)
    for i in 1:n
        Z[i] = borehole(X[i, :])
    end
    return Z
end

# ============================================================================
# Data Generation
# ============================================================================

"""
    generate_data(n_train, n_test, dim=8; noise_sd=1.0)

Generate training and test data using Latin Hypercube Sampling.

Arguments:
- n_train: Number of training points
- n_test: Number of test points
- dim: Input dimension (default 8 for borehole)
- noise_sd: Standard deviation of Gaussian noise added to training responses

Returns: (X_train, Z_train, X_test, Z_test_true)
"""
function generate_data(n_train::Int, n_test::Int, dim::Int=8; noise_sd::Float64=1.0)
    # Generate training design via randomLHS (no optimization, fast)
    # Use scaleLHC to get values in [0,1]
    plan_train = randomLHC(n_train, dim)
    X_train = Matrix{Float64}(plan_train ./ n_train)

    # Generate test design via randomLHS
    plan_test = randomLHC(n_test, dim)
    X_test = Matrix{Float64}(plan_test ./ n_test)

    # Evaluate true function
    Z_train_true = borehole(X_train)
    Z_test_true = borehole(X_test)

    # Add noise to training data
    Z_train = Z_train_true .+ noise_sd .* randn(n_train)

    return X_train, Z_train, X_test, Z_test_true
end

# ============================================================================
# Main Benchmark
# ============================================================================

function run_benchmark()
    println("=" ^ 70)
    println("Borehole Example: Julia laGP.jl Benchmark")
    println("=" ^ 70)
    println()

    # Configuration
    n_train = 10_000
    n_test = 10_000
    noise_sd = 1.0

    # aGP parameters (matching R defaults)
    agp_start = 6
    agp_endpt = 50
    agp_close = 1000

    println("Configuration:")
    println("  Training points: $(n_train) (8D)")
    println("  Test points: $(n_test)")
    println("  Noise: sd=$(noise_sd)")
    println("  aGP parameters: start=$(agp_start), endpt=$(agp_endpt), close=$(agp_close)")
    println("  Threads: $(Threads.nthreads())")
    println()

    # Generate data
    println("Generating data...")
    time_data = @elapsed begin
        X_train, Z_train, X_test, Z_test_true = generate_data(n_train, n_test; noise_sd=noise_sd)
    end
    @printf("  Data generation: %.2f seconds\n", time_data)
    @printf("  Training response range: [%.2f, %.2f]\n", minimum(Z_train), maximum(Z_train))
    @printf("  Test response range: [%.2f, %.2f]\n", minimum(Z_test_true), maximum(Z_test_true))
    println()

    # Get hyperparameter defaults
    d_args = darg(X_train)
    g_args = garg(Z_train)
    d_args_sep = darg_sep(X_train)

    # Results storage
    results = Dict{String, NamedTuple{(:time, :rmse), Tuple{Float64, Float64}}}()

    # ========================================================================
    # PART 1: Full Separable GP on subset (n=1000)
    # ========================================================================

    println("-" ^ 70)
    println("PART 1: Full Separable GP (n=1000 subset, MLE)")
    println("-" ^ 70)

    n_subset = 10_000
    X_subset = X_train[1:n_subset, :]
    Z_subset = Z_train[1:n_subset]

    d_start_sep = [r.start for r in d_args_sep.ranges]
    d_ranges_sep = [(r.min, r.max) for r in d_args_sep.ranges]

    print("  Fitting separable GP with MLE...")
    time_gpsep = @elapsed begin
        gp_sep = new_gp_sep(X_subset, Z_subset, d_start_sep, g_args.start)
        jmle_gp_sep!(gp_sep; drange=d_ranges_sep, grange=(g_args.min, g_args.max))
        pred_sep = pred_gp_sep(gp_sep, X_test; lite=true)
    end
    @printf(" %.2f seconds\n", time_gpsep)

    rmse_gpsep = sqrt(mean((pred_sep.mean .- Z_test_true).^2))
    @printf("  RMSE: %.4f\n", rmse_gpsep)
    @printf("  Final nugget g: %.6f\n", gp_sep.g)
    println("  Final lengthscales d: ", join([@sprintf("%.4f", d) for d in gp_sep.d], ", "))
    println()

    results["Full GPsep (1000)"] = (time=time_gpsep, rmse=rmse_gpsep)

    # ========================================================================
    # PART 2: aGP Isotropic (full data)
    # ========================================================================

    println("-" ^ 70)
    println("PART 2: aGP Isotropic (n=$(n_train), endpt=$(agp_endpt))")
    println("-" ^ 70)

    print("  Running aGP isotropic (ALC method)...")
    time_agp_iso = @elapsed begin
        pred_agp_iso = agp(X_train, Z_train, X_test;
                          start=agp_start, endpt=agp_endpt, close=agp_close,
                          d=d_args.start, g=g_args.start,
                          method=:alc, parallel=true)
    end
    @printf(" %.2f seconds\n", time_agp_iso)

    rmse_agp_iso = sqrt(mean((pred_agp_iso.mean .- Z_test_true).^2))
    @printf("  RMSE: %.4f\n", rmse_agp_iso)
    println()

    results["aGP iso ($(n_train))"] = (time=time_agp_iso, rmse=rmse_agp_iso)

    # ========================================================================
    # PART 3: aGP Separable (full data)
    # ========================================================================

    println("-" ^ 70)
    println("PART 3: aGP Separable (n=$(n_train), endpt=$(agp_endpt))")
    println("-" ^ 70)

    print("  Running aGP separable (ALC method)...")
    time_agp_sep = @elapsed begin
        pred_agp_sep = agp_sep(X_train, Z_train, X_test;
                               start=agp_start, endpt=agp_endpt, close=agp_close,
                               d=d_start_sep, g=g_args.start,
                               method=:alc, parallel=true)
    end
    @printf(" %.2f seconds\n", time_agp_sep)

    rmse_agp_sep = sqrt(mean((pred_agp_sep.mean .- Z_test_true).^2))
    @printf("  RMSE: %.4f\n", rmse_agp_sep)
    println()

    results["aGP sep ($(n_train))"] = (time=time_agp_sep, rmse=rmse_agp_sep)

    # ========================================================================
    # PART 4: aGP NN (nearest neighbor only, for speed comparison)
    # ========================================================================

    println("-" ^ 70)
    println("PART 4: aGP NN (n=$(n_train), endpt=$(agp_endpt))")
    println("-" ^ 70)

    print("  Running aGP NN (no acquisition)...")
    time_agp_nn = @elapsed begin
        pred_agp_nn = agp(X_train, Z_train, X_test;
                         start=agp_start, endpt=agp_endpt, close=agp_close,
                         d=d_args.start, g=g_args.start,
                         method=:nn, parallel=true)
    end
    @printf(" %.2f seconds\n", time_agp_nn)

    rmse_agp_nn = sqrt(mean((pred_agp_nn.mean .- Z_test_true).^2))
    @printf("  RMSE: %.4f\n", rmse_agp_nn)
    println()

    results["aGP NN ($(n_train))"] = (time=time_agp_nn, rmse=rmse_agp_nn)

    # ========================================================================
    # Results Summary
    # ========================================================================

    println("=" ^ 70)
    println("SUMMARY (for R comparison)")
    println("=" ^ 70)
    println()

    @printf("%-20s %12s %12s\n", "Method", "Time (s)", "RMSE")
    println("-" ^ 46)
    for (method, res) in sort(collect(results), by=x->x[2].time)
        @printf("%-20s %12.2f %12.4f\n", method, res.time, res.rmse)
    end
    println("-" ^ 46)
    println()

    println("Notes:")
    println("  - Full GPsep runs on subset (n=1000) due to O(n³) complexity")
    println("  - aGP methods use full dataset (n=$(n_train)) with local approximations")
    println("  - Separable GP allows per-dimension lengthscales (ARD)")
    println("  - NN method skips acquisition function (fastest, but may be less accurate)")
    println()

    println("To compare with R laGP:")
    println("  Rscript benchmarks/benchmark_borehole.R")

    return results
end

# ============================================================================
# Optional: Visualization
# ============================================================================

# Try to load CairoMakie at top level
const HAS_CAIROMAKIE = try
    @eval using CairoMakie
    true
catch
    false
end

function plot_results(X_test, Z_test_true, pred_agp_iso, pred_agp_sep)
    if !HAS_CAIROMAKIE
        println("\nCairoMakie not available - skipping plot generation")
        return nothing
    end

    fig = Figure(size=(1000, 400))

    # Prediction vs True (isotropic)
    ax1 = Axis(fig[1, 1],
               xlabel="True Value",
               ylabel="Predicted Value",
               title="aGP Isotropic",
               aspect=DataAspect())
    scatter!(ax1, Z_test_true, pred_agp_iso.mean, markersize=2, alpha=0.3)
    lines!(ax1, [minimum(Z_test_true), maximum(Z_test_true)],
           [minimum(Z_test_true), maximum(Z_test_true)],
           color=:red, linewidth=2)

    # Prediction vs True (separable)
    ax2 = Axis(fig[1, 2],
               xlabel="True Value",
               ylabel="Predicted Value",
               title="aGP Separable",
               aspect=DataAspect())
    scatter!(ax2, Z_test_true, pred_agp_sep.mean, markersize=2, alpha=0.3)
    lines!(ax2, [minimum(Z_test_true), maximum(Z_test_true)],
           [minimum(Z_test_true), maximum(Z_test_true)],
           color=:red, linewidth=2)

    # Error histogram
    ax3 = Axis(fig[1, 3],
               xlabel="Prediction Error",
               ylabel="Count",
               title="Error Distribution")
    hist!(ax3, pred_agp_iso.mean .- Z_test_true, bins=50, color=(:blue, 0.5), label="Isotropic")
    hist!(ax3, pred_agp_sep.mean .- Z_test_true, bins=50, color=(:red, 0.5), label="Separable")
    axislegend(ax3, position=:rt)

    save(joinpath(OUTPUT_DIR, "borehole_results.png"), fig)
    println("\nPlot saved to: examples/borehole_results.png")

    return fig
end

# ============================================================================
# Run benchmark
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    results = run_benchmark()
end
