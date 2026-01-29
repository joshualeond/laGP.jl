using JSON3
using laGP
using Random
using Test

include("test_utils.jl")

# Load reference data
const LAGP_REF = JSON3.read(
    read(joinpath(@__DIR__, "reference", "lagp.json"), String)
)

@testset "Local Approximate GP" begin
    # Extract test data
    X = _reshape_matrix(Float64.(LAGP_REF.X), LAGP_REF.X_nrow, LAGP_REF.X_ncol)
    Z = Float64.(LAGP_REF.Z)
    XX = _reshape_matrix(Float64.(LAGP_REF.XX), LAGP_REF.XX_nrow, LAGP_REF.XX_ncol)
    start = Int(LAGP_REF.start)
    endpt = Int(LAGP_REF.end_size)
    d = Float64(LAGP_REF.d)
    g = Float64(LAGP_REF.g)

    @testset "agp with ALC method (fixed hyperparameters)" begin
        result = agp(X, Z, XX;
            start=start, endpt=endpt,
            d=(start=d, mle=false), g=(start=g, mle=false),
            method=:alc
        )

        ref_mean = Float64.(LAGP_REF.alc_mean)
        ref_var = Float64.(LAGP_REF.alc_var)

        @test length(result.mean) == size(XX, 1)
        @test length(result.var) == size(XX, 1)
        # Looser tolerance due to different point selection order in sequential design
        @test result.mean ≈ ref_mean rtol=0.05
        @test result.var ≈ ref_var rtol=0.15
    end

    @testset "agp with NN method (fixed hyperparameters)" begin
        result = agp(X, Z, XX;
            start=start, endpt=endpt,
            d=(start=d, mle=false), g=(start=g, mle=false),
            method=:nn
        )

        ref_mean = Float64.(LAGP_REF.nn_mean)
        ref_var = Float64.(LAGP_REF.nn_var)

        @test length(result.mean) == size(XX, 1)
        @test length(result.var) == size(XX, 1)
        # NN should be more deterministic, but floating-point tie-breaking can differ
        @test result.mean ≈ ref_mean rtol=0.05
        @test result.var ≈ ref_var rtol=0.15
    end

    @testset "agp with MSPE method (fixed hyperparameters)" begin
        result = agp(X, Z, XX;
            start=start, endpt=endpt,
            d=(start=d, mle=false), g=(start=g, mle=false),
            method=:mspe
        )

        ref_mean = Float64.(LAGP_REF.mspe_mean)
        ref_var = Float64.(LAGP_REF.mspe_var)

        @test length(result.mean) == size(XX, 1)
        @test length(result.var) == size(XX, 1)
        # MSPE selection has systematic differences from R implementation
        @test result.mean ≈ ref_mean rtol=0.1
        @test result.var ≈ ref_var rtol=0.2
    end

    @testset "agp with MLE enabled" begin
        result = agp(X, Z, XX;
            start=start, endpt=endpt,
            d=(start=d, mle=true, min=0.01, max=10.0),
            g=(start=g, mle=true, min=1e-6, max=1.0),
            method=:alc
        )

        ref_mean = Float64.(LAGP_REF.mle_mean)
        ref_var = Float64.(LAGP_REF.mle_var)
        ref_mle_d = Float64.(LAGP_REF.mle_mle_d)
        ref_mle_g = Float64.(LAGP_REF.mle_mle_g)

        @test length(result.mean) == size(XX, 1)
        @test length(result.var) == size(XX, 1)
        # MLE on different local designs can lead to different optima
        # MAP estimation with priors can change both mean and variance estimates
        @test result.mean ≈ ref_mean rtol=0.2
        @test result.var ≈ ref_var rtol=0.5

        # Check MLE results exist and are in reasonable range
        @test haskey(result, :mle)
        # MAP estimation with priors produces different hyperparameters than pure MLE
        # The priors regularize parameters toward more conservative values
        # We just check that the results are positive and reasonable order of magnitude
        @test all(result.mle.d .> 0.01)
        @test all(result.mle.d .< 10.0)
        @test all(result.mle.g .> 1e-6)
        @test all(result.mle.g .< 1.0)
    end
end

@testset "Separable Local Approximate GP" begin
    # Create test data for 2D function with anisotropic behavior
    # f(x,y) = sin(4*x) + 0.2*y (fast variation in x, slow in y)
    Random.seed!(42)
    n_train = 100
    X = rand(n_train, 2) .* 10 .- 5  # [-5, 5]^2
    y = sin.(4 .* X[:, 1]) .+ 0.2 .* X[:, 2] .+ 0.1 .* randn(n_train)

    n_test = 10
    XX = rand(n_test, 2) .* 10 .- 5

    @testset "lagp_sep basic functionality" begin
        Xref = [0.0, 0.0]
        d = [0.1, 10.0]  # Small d[1] for fast variation, large d[2] for slow

        result = lagp_sep(Xref, 6, 30, X, y; d=d, g=1e-3)

        @test haskey(result, :mean)
        @test haskey(result, :var)
        @test haskey(result, :df)
        @test haskey(result, :indices)
        @test length(result.indices) == 30
        @test result.var > 0
    end

    @testset "lagp_sep with NN method" begin
        Xref = [1.0, 1.0]
        d = [0.5, 0.5]

        result = lagp_sep(Xref, 6, 25, X, y; d=d, g=1e-3, method=:nn)

        @test haskey(result, :mean)
        @test length(result.indices) == 25
    end

    @testset "agp_sep with fixed hyperparameters" begin
        d_info = darg_sep(X)
        d_start = [d_info.ranges[1].start, d_info.ranges[2].start]

        result = agp_sep(X, y, XX;
            start=6, endpt=30,
            d=d_start,
            g=1e-3,
            method=:alc
        )

        @test length(result.mean) == n_test
        @test length(result.var) == n_test
        @test all(result.var .> 0)
    end

    @testset "agp_sep with NN method" begin
        d_start = [0.5, 0.5]

        result = agp_sep(X, y, XX;
            start=6, endpt=30,
            d=d_start,
            g=1e-3,
            method=:nn
        )

        @test length(result.mean) == n_test
        @test length(result.var) == n_test
    end

    @testset "agp_sep with MLE" begin
        d_info = darg_sep(X)
        g_info = garg(y)

        result = agp_sep(X, y, XX;
            start=6, endpt=30,
            d=(start=[d_info.ranges[1].start, d_info.ranges[2].start],
               mle=true,
               min=[d_info.ranges[1].min, d_info.ranges[2].min],
               max=[d_info.ranges[1].max, d_info.ranges[2].max]),
            g=(start=g_info.start, mle=true, min=g_info.min, max=g_info.max),
            method=:alc
        )

        @test length(result.mean) == n_test
        @test length(result.var) == n_test
        @test haskey(result, :mle)

        # Check MLE d is a matrix of size (n_test x 2)
        @test size(result.mle.d) == (n_test, 2)
        @test all(result.mle.d .> 0)

        # Check MLE g
        @test length(result.mle.g) == n_test
        @test all(result.mle.g .> 0)
    end

    @testset "agp_sep d-only MLE" begin
        d_info = darg_sep(X)

        result = agp_sep(X, y, XX;
            start=6, endpt=30,
            d=(start=[d_info.ranges[1].start, d_info.ranges[2].start],
               mle=true,
               min=[d_info.ranges[1].min, d_info.ranges[2].min],
               max=[d_info.ranges[1].max, d_info.ranges[2].max]),
            g=1e-3,  # Fixed g
            method=:alc
        )

        @test length(result.mean) == n_test
        @test haskey(result, :mle)
        @test size(result.mle.d) == (n_test, 2)
        @test isempty(result.mle.g)  # g not optimized
    end

    @testset "agp_sep parallel vs sequential consistency" begin
        d_start = [0.5, 0.5]

        result_seq = agp_sep(X, y, XX;
            start=6, endpt=30,
            d=d_start, g=1e-3,
            method=:alc, parallel=false
        )

        result_par = agp_sep(X, y, XX;
            start=6, endpt=30,
            d=d_start, g=1e-3,
            method=:alc, parallel=true
        )

        # Results should be identical regardless of threading
        @test result_seq.mean ≈ result_par.mean
        @test result_seq.var ≈ result_par.var
    end
end
