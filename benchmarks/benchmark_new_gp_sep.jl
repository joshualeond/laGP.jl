#!/usr/bin/env julia

using laGP
using LinearAlgebra
using Random
using Statistics: mean, median

function summarize_times(label::AbstractString, times::Vector{Float64}, alloc::Int)
    println(label)
    println("  mean:   ", round(mean(times), sigdigits=4), " s")
    println("  median: ", round(median(times), sigdigits=4), " s")
    println("  min:    ", round(minimum(times), sigdigits=4), " s")
    println("  max:    ", round(maximum(times), sigdigits=4), " s")
    println("  alloc:  ", round(alloc / 1024^2, digits=3), " MiB")
end

function bench_times(f::Function; reps::Int=8)
    times = Vector{Float64}(undef, reps)
    for i in 1:reps
        GC.gc()
        times[i] = @elapsed f()
    end
    return times
end

function main()
    Random.seed!(42)

    n = 1000
    m = 7
    n_test = 100
    X = rand(n, m)
    Z = randn(n)
    XX = rand(n_test, m)
    d0 = fill(2.0, m)
    d1 = fill(1.5, m)
    g = 1e-6

    println("Benchmark config: n=", n, ", m=", m, ", n_test=", n_test)
    println("BLAS threads: ", BLAS.get_num_threads())
    println("Julia threads: ", Threads.nthreads())
    println()

    # Warmup
    gp = new_gp_sep(X, Z, d0, g)
    pred_gp_sep(gp, XX; lite=true)
    dllik_gp_sep(gp)
    update_gp_sep!(gp; d=d1)
    update_gp_sep!(gp; d=d0)

    # Constructor
    constructor_times = bench_times(() -> new_gp_sep(X, Z, d0, g))
    constructor_alloc = @allocated new_gp_sep(X, Z, d0, g)
    summarize_times("new_gp_sep(X, Z, d, g)", constructor_times, constructor_alloc)
    println()

    # Prediction
    gp_pred = new_gp_sep(X, Z, d0, g)
    pred_times = bench_times(() -> pred_gp_sep(gp_pred, XX; lite=true))
    pred_alloc = @allocated pred_gp_sep(gp_pred, XX; lite=true)
    summarize_times("pred_gp_sep(gp, XX; lite=true)", pred_times, pred_alloc)
    println()

    # Gradient
    gp_grad = new_gp_sep(X, Z, d0, g)
    grad_times = bench_times(() -> dllik_gp_sep(gp_grad))
    grad_alloc = @allocated dllik_gp_sep(gp_grad)
    summarize_times("dllik_gp_sep(gp)", grad_times, grad_alloc)
    println()

    # Hyperparameter update
    gp_upd = new_gp_sep(X, Z, d0, g)
    upd_times = bench_times(() -> begin
        update_gp_sep!(gp_upd; d=d1)
        update_gp_sep!(gp_upd; d=d0)
    end)
    upd_alloc = @allocated begin
        update_gp_sep!(gp_upd; d=d1)
        update_gp_sep!(gp_upd; d=d0)
    end
    summarize_times("update_gp_sep! toggle d (two updates)", upd_times, upd_alloc)
end

main()
