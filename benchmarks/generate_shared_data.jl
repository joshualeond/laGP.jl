#!/usr/bin/env julia
# Generate shared training/test data so R and Julia benchmarks use identical inputs.
#
# Environment variables (optional):
#   LAGP_SHARED_DIR="benchmarks/shared_data"
#   LAGP_N_TRAIN="100,500,1000,2000,5000"
#   LAGP_N_TEST_GRID="10"
#   LAGP_SEED="42"

using Random
using DelimitedFiles

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

function test_function(x1, x2)
    return sin(2π * x1) * cos(2π * x2) + 0.5 * sin(4π * x1)
end

function generate_lhs_data(n::Int, dim::Int=2)
    # Fast random LHS
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

function main()
    shared_dir = get(ENV, "LAGP_SHARED_DIR", "benchmarks/shared_data")
    n_train_sizes = parse_int_list(get(ENV, "LAGP_N_TRAIN", ""), [100, 500, 1000, 2000, 5000])
    n_test_grid = parse_int(get(ENV, "LAGP_N_TEST_GRID", "10"), 10)
    seed = parse_int(get(ENV, "LAGP_SEED", "42"), 42)

    Random.seed!(seed)

    mkpath(shared_dir)

    XX = generate_test_grid(n_test_grid, n_test_grid)
    writedlm(joinpath(shared_dir, "test_XX.csv"), XX, ',')

    for n in n_train_sizes
        X, Z = generate_lhs_data(n)
        writedlm(joinpath(shared_dir, "train_X_$(n).csv"), X, ',')
        writedlm(joinpath(shared_dir, "train_Z_$(n).csv"), Z, ',')
    end

    println("Wrote shared data to: $(shared_dir)")
end

main()
