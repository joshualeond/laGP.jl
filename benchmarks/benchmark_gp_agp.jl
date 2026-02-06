#!/usr/bin/env julia
# Benchmark: Full GP vs aGP (ALC/NN/MSPE) using laGP.jl
#
# Environment variables (optional):
#   LAGP_N_TRAIN="100,500,1000,2000,5000"   # comma-separated list
#   LAGP_N_TEST_GRID="10"          # grid size per dimension
#   LAGP_AGP_START="6"
#   LAGP_AGP_ENDPT="30"
#   LAGP_CLOSE="1000"              # candidate set size for ALC/MSPE
#   LAGP_REPS="1"                  # repeats per method (median time)
#   LAGP_SEED="42"
#   LAGP_WARMUP="1"                # 1 to warm up, 0 to skip
#   LAGP_PARALLEL="1"              # 1 to allow threaded aGP, 0 to disable
#   LAGP_OUT="benchmarks/results_jl.csv"
#   LAGP_SHARED_DIR="benchmarks/shared_data" # load shared CSVs if set

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using laGP
using Random
using DelimitedFiles
using Printf
using Statistics: mean, median

parse_int_list(s::String, default::Vector{Int}) = begin
    isempty(s) && return default
    parts = split(s, ',')
    vals = Int[]
    for p in parts
        sp = strip(p)
        isempty(sp) && continue
        try
            push!(vals, parse(Int, sp))
        catch
        end
    end
    isempty(vals) ? default : vals
end

parse_int(s::String, default::Int) = begin
    isempty(s) && return default
    try
        parse(Int, s)
    catch
        default
    end
end

parse_bool(s::String, default::Bool) = begin
    isempty(s) && return default
    s in ("1", "true", "TRUE", "yes", "YES") ? true :
        s in ("0", "false", "FALSE", "no", "NO") ? false : default
end

seed = parse_int(get(ENV, "LAGP_SEED", "42"), 42)
Random.seed!(seed)

# ============================================================================
# Test function: 2D sinusoidal
# ============================================================================

function test_function(x1, x2)
    return sin(2π * x1) * cos(2π * x2) + 0.5 * sin(4π * x1)
end

# ============================================================================
# Data generation
# ============================================================================

function generate_lhs_data(n::Int, dim::Int=2)
    # Fast random LHS to keep data generation cheap vs GP timing
    X = Matrix{Float64}(undef, n, dim)
    for j in 1:dim
        perm = randperm(n)
        X[:, j] = (perm .- rand(n)) ./ n
    end
    Z = [test_function(X[i, 1], X[i, 2]) for i in 1:n]
    return X, Z
end

function generate_test_grid(nx::Int, ny::Int)
    x = range(0.0, 1.0, length=nx)
    y = range(0.0, 1.0, length=ny)
    XX = Matrix{Float64}(undef, nx * ny, 2)
    idx = 1
    for yi in y, xi in x
        XX[idx, 1] = xi
        XX[idx, 2] = yi
        idx += 1
    end
    return XX
end

function load_shared_data(n::Int, shared_dir::String)
    x_path = joinpath(shared_dir, "train_X_$(n).csv")
    z_path = joinpath(shared_dir, "train_Z_$(n).csv")
    xx_path = joinpath(shared_dir, "test_XX.csv")

    if !isfile(x_path) || !isfile(z_path) || !isfile(xx_path)
        error("Shared data missing. Expected files: $(x_path), $(z_path), $(xx_path)")
    end

    X = Array{Float64}(readdlm(x_path, ',', Float64))
    Z = vec(Array{Float64}(readdlm(z_path, ',', Float64)))
    XX = Array{Float64}(readdlm(xx_path, ',', Float64))
    return X, Z, XX
end

# ============================================================================
# Benchmark helpers
# ============================================================================

function benchmark_full_gp(X::Matrix{Float64}, Z::Vector{Float64}, XX::Matrix{Float64};
                           d::Float64, g::Float64)
    gp = new_gp(X, Z, d, g)
    pred = pred_gp(gp, XX; lite=true)
    return pred.mean
end

function benchmark_agp(X::Matrix{Float64}, Z::Vector{Float64}, XX::Matrix{Float64};
                       d::Float64, g::Float64, method::Symbol,
                       start::Int=6, endpt::Int=50, close::Int=1000, parallel::Bool=true)
    result = agp(X, Z, XX; start=start, endpt=endpt, close=close, d=d, g=g,
                 method=method, parallel=parallel)
    return result.mean
end

function time_reps(f, reps::Int)
    times = Vector{Float64}(undef, reps)
    result = nothing
    for i in 1:reps
        t = @elapsed begin
            result = f()
        end
        times[i] = t
    end
    return median(times), result
end

# ============================================================================
# Main benchmark
# ============================================================================

function run_benchmark()
    n_train_sizes = parse_int_list(get(ENV, "LAGP_N_TRAIN", ""), [100, 500, 1000, 2000, 5000, 10_000, 20_000])
    n_test_grid = parse_int(get(ENV, "LAGP_N_TEST_GRID", "10"), 10)
    agp_start = parse_int(get(ENV, "LAGP_AGP_START", "6"), 6)
    agp_endpt = parse_int(get(ENV, "LAGP_AGP_ENDPT", "30"), 30)
    close_n = parse_int(get(ENV, "LAGP_CLOSE", "1000"), 1000)
    reps = parse_int(get(ENV, "LAGP_REPS", "1"), 1)
    warmup = parse_bool(get(ENV, "LAGP_WARMUP", "1"), true)
    parallel = parse_bool(get(ENV, "LAGP_PARALLEL", "1"), true)
    out_path = get(ENV, "LAGP_OUT", "")
    shared_dir = get(ENV, "LAGP_SHARED_DIR", "")

    println("=" ^ 70)
    println("Benchmark: Full GP vs aGP (ALC/NN/MSPE) - laGP.jl")
    println("=" ^ 70)
    println()

    XX = generate_test_grid(n_test_grid, n_test_grid)
    n_test = size(XX, 1)
    true_vals = [test_function(XX[i, 1], XX[i, 2]) for i in 1:n_test]

    println("Configuration:")
    println("  Training sizes: $(n_train_sizes)")
    println("  Test points: $(n_test) ($(n_test_grid)×$(n_test_grid) grid)")
    println("  aGP parameters: start=$(agp_start), endpt=$(agp_endpt), close=$(close_n)")
    println("  Reps per method: $(reps)")
    println("  Threads: $(Threads.nthreads())")
    println("  aGP parallel: $(parallel)")
    println()

    if warmup
        Xw, Zw = generate_lhs_data(50)
        XXw = generate_test_grid(5, 5)
        d_start = darg(Xw).start
        g_start = garg(Zw).start
        benchmark_full_gp(Xw, Zw, XXw; d=d_start, g=g_start)
        benchmark_agp(Xw, Zw, XXw; d=d_start, g=g_start, method=:alc,
                      start=6, endpt=20, close=close_n, parallel=parallel)
        benchmark_agp(Xw, Zw, XXw; d=d_start, g=g_start, method=:nn,
                      start=6, endpt=20, close=close_n, parallel=parallel)
        benchmark_agp(Xw, Zw, XXw; d=d_start, g=g_start, method=:mspe,
                      start=6, endpt=20, close=close_n, parallel=parallel)
        GC.gc()
    end

    results = Vector{NamedTuple{(:n, :full, :alc, :nn, :mspe,
                                 :rmse_full, :rmse_alc, :rmse_nn, :rmse_mspe),
                                Tuple{Int,Float64,Float64,Float64,Float64,
                                      Float64,Float64,Float64,Float64}}}()

    for n_train in n_train_sizes
        println("-" ^ 50)
        println("n_train = $(n_train)")
        println("-" ^ 50)

        if !isempty(shared_dir)
            X, Z, XX = load_shared_data(n_train, shared_dir)
            n_test = size(XX, 1)
            true_vals = [test_function(XX[i, 1], XX[i, 2]) for i in 1:n_test]
        else
            X, Z = generate_lhs_data(n_train)
        end
        d_start = darg(X).start
        g_start = garg(Z).start

        print("  Timing Full GP...")
        time_full, pred_full = time_reps(() -> benchmark_full_gp(X, Z, XX; d=d_start, g=g_start), reps)
        @printf(" %.3fs\n", time_full)

        close_eff = min(close_n, n_train)

        print("  Timing aGP ALC...")
        time_alc, pred_alc = time_reps(() -> benchmark_agp(X, Z, XX; d=d_start, g=g_start, method=:alc,
                                                          start=agp_start, endpt=agp_endpt,
                                                          close=close_eff, parallel=parallel), reps)
        @printf(" %.3fs\n", time_alc)

        print("  Timing aGP NN...")
        time_nn, pred_nn = time_reps(() -> benchmark_agp(X, Z, XX; d=d_start, g=g_start, method=:nn,
                                                        start=agp_start, endpt=agp_endpt,
                                                        close=close_eff, parallel=parallel), reps)
        @printf(" %.3fs\n", time_nn)

        print("  Timing aGP MSPE...")
        time_mspe, pred_mspe = time_reps(() -> benchmark_agp(X, Z, XX; d=d_start, g=g_start, method=:mspe,
                                                            start=agp_start, endpt=agp_endpt,
                                                            close=close_eff, parallel=parallel), reps)
        @printf(" %.3fs\n", time_mspe)

        rmse_full = sqrt(mean((pred_full .- true_vals).^2))
        rmse_alc = sqrt(mean((pred_alc .- true_vals).^2))
        rmse_nn = sqrt(mean((pred_nn .- true_vals).^2))
        rmse_mspe = sqrt(mean((pred_mspe .- true_vals).^2))

        @printf("  RMSE: Full=%.4f, ALC=%.4f, NN=%.4f, MSPE=%.4f\n",
                rmse_full, rmse_alc, rmse_nn, rmse_mspe)

        push!(results, (n=n_train, full=time_full, alc=time_alc, nn=time_nn, mspe=time_mspe,
                        rmse_full=rmse_full, rmse_alc=rmse_alc,
                        rmse_nn=rmse_nn, rmse_mspe=rmse_mspe))

        println()
    end

    println("=" ^ 70)
    println("RESULTS SUMMARY")
    println("=" ^ 70)
    println()

    println("Timing Results (seconds):")
    println("-" ^ 80)
    @printf("%-10s %12s %12s %12s %12s %12s\n",
            "n_train", "Full GP", "aGP ALC", "aGP NN", "aGP MSPE", "Winner")
    println("-" ^ 80)

    for r in results
        times = [r.full, r.alc, r.nn, r.mspe]
        methods = ["Full GP", "aGP ALC", "aGP NN", "aGP MSPE"]
        winner = methods[argmin(times)]
        @printf("%-10d %12.3f %12.3f %12.3f %12.3f %12s\n",
                r.n, r.full, r.alc, r.nn, r.mspe, winner)
    end
    println("-" ^ 80)
    println()

    println("Speedup vs Full GP:")
    println("-" ^ 60)
    @printf("%-10s %12s %12s %12s\n", "n_train", "ALC", "NN", "MSPE")
    println("-" ^ 60)

    for r in results
        @printf("%-10d %11.2fx %11.2fx %11.2fx\n",
                r.n, r.full / r.alc, r.full / r.nn, r.full / r.mspe)
    end
    println("-" ^ 60)
    println()

    println("RMSE (sanity check):")
    println("-" ^ 80)
    @printf("%-10s %12s %12s %12s %12s\n",
            "n_train", "Full GP", "aGP ALC", "aGP NN", "aGP MSPE")
    println("-" ^ 80)

    for r in results
        @printf("%-10d %12.4f %12.4f %12.4f %12.4f\n",
                r.n, r.rmse_full, r.rmse_alc, r.rmse_nn, r.rmse_mspe)
    end
    println("-" ^ 80)

    if !isempty(out_path)
        open(out_path, "w") do io
            println(io, "n,full,alc,nn,mspe,rmse_full,rmse_alc,rmse_nn,rmse_mspe")
            for r in results
                @printf(io, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                        r.n, r.full, r.alc, r.nn, r.mspe,
                        r.rmse_full, r.rmse_alc, r.rmse_nn, r.rmse_mspe)
            end
        end
        println("\nWrote CSV: $(out_path)")
    end
end

run_benchmark()
