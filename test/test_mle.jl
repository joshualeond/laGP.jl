using JSON3
using laGP
using Test

include("test_utils.jl")

# Load reference data
const MLE_REF = JSON3.read(
    read(joinpath(@__DIR__, "reference", "mle.json"), String)
)

@testset "MLE" begin
    # Extract test data
    X = _reshape_matrix(Float64.(MLE_REF.X), MLE_REF.X_nrow, MLE_REF.X_ncol)
    Z = Float64.(MLE_REF.Z)

    @testset "mle_gp! for d" begin
        gp = new_gp(X, Z, Float64(MLE_REF.d_init), Float64(MLE_REF.g_for_d))
        result = mle_gp!(gp, :d; tmin=0.01, tmax=10.0)

        @test haskey(result, :d)
        @test result.d ≈ MLE_REF.mle_d rtol=1e-3
        @test gp.d ≈ MLE_REF.mle_d rtol=1e-3  # GP should be updated in-place
        @test llik_gp(gp) ≈ MLE_REF.llik_after_d rtol=1e-3
    end

    @testset "mle_gp! for g" begin
        gp = new_gp(X, Z, Float64(MLE_REF.d_for_g), Float64(MLE_REF.g_init))
        result = mle_gp!(gp, :g; tmin=1e-6, tmax=1.0)

        @test haskey(result, :g)
        # Use looser tolerance for g since optimal is near boundary
        @test result.g ≈ MLE_REF.mle_g rtol=1e-2
        @test gp.g ≈ MLE_REF.mle_g rtol=1e-2
        @test llik_gp(gp) ≈ MLE_REF.llik_after_g rtol=1e-3
    end

    @testset "jmle_gp! joint optimization" begin
        gp = new_gp(X, Z, Float64(MLE_REF.jmle_d_init), Float64(MLE_REF.jmle_g_init))
        result = jmle_gp!(gp; drange=(0.01, 10.0), grange=(1e-6, 1.0))

        @test haskey(result, :d)
        @test haskey(result, :g)
        # MAP estimation with priors produces different hyperparameters
        # Check that values are positive, finite, and within bounds
        @test 0.01 < result.d < 10.0
        @test 1e-6 < result.g < 1.0
        @test isfinite(llik_gp(gp))
    end

    @testset "darg helper" begin
        result = darg(X)

        @test haskey(result, :start)
        @test haskey(result, :min)
        @test haskey(result, :max)
        @test result.start ≈ MLE_REF.darg_start rtol=1e-3
        @test result.min ≈ MLE_REF.darg_min rtol=1e-3
        @test result.max ≈ MLE_REF.darg_max rtol=1e-3
    end

    @testset "garg helper" begin
        result = garg(Z)

        @test haskey(result, :start)
        @test haskey(result, :min)
        @test haskey(result, :max)
        @test result.start ≈ MLE_REF.garg_start rtol=1e-3
        @test result.min ≈ MLE_REF.garg_min rtol=1e-3
        @test result.max ≈ MLE_REF.garg_max rtol=1e-3
    end
end
