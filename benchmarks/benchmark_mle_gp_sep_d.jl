#!/usr/bin/env julia

using laGP
using LinearAlgebra
using Random
using Statistics: mean, median

function summarize_times(label::AbstractString, times::Vector{Float64})
    println(label)
    println("  mean:   ", round(mean(times), sigdigits=4), " s")
    println("  median: ", round(median(times), sigdigits=4), " s")
    println("  min:    ", round(minimum(times), sigdigits=4), " s")
    println("  max:    ", round(maximum(times), sigdigits=4), " s")
end

function bench_mle(X, Z, d0, g; reps::Int=3, maxit::Int=200)
    times = Vector{Float64}(undef, reps)
    iters = Vector{Int}(undef, reps)

    for i in 1:reps
        gp = new_gp_sep(X, Z, d0, g)
        tmin = sqrt(eps(eltype(X)))
        tmax = size(X, 2)^2
        GC.gc()
        t = @elapsed result = mle_gp_sep!(gp, :d; tmin=tmin, tmax=tmax, maxit=maxit, dab=nothing)
        times[i] = t
        iters[i] = result.its
    end

    return times, iters
end

function main()
    Random.seed!(42)

    n = 1000
    m = 7
    X = rand(n, m)
    Z = randn(n)
    d0 = fill(2.0, m)
    g = 1e-6

    println("Benchmark config: n=", n, ", m=", m)
    println("BLAS threads: ", BLAS.get_num_threads())
    println("Julia threads: ", Threads.nthreads())
    println()

    # Warmup
    gp = new_gp_sep(X, Z, d0, g)
    mle_gp_sep!(gp, :d; tmin=sqrt(eps(eltype(X))), tmax=m^2, maxit=30, dab=nothing)

    times, iters = bench_mle(X, Z, d0, g; reps=3, maxit=200)
    summarize_times("mle_gp_sep!(gp, :d; dab=nothing)", times)
    println("  optimizer iterations: ", iters)
end

main()
