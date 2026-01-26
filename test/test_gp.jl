using JSON3
using laGP
using Test

# Load reference data
const GP_BASIC_REF = JSON3.read(
    read(joinpath(@__DIR__, "reference", "gp_basic.json"), String)
)

function _reshape_matrix(vec::Vector, nrow::Int, ncol::Int)
    # R stores matrices column-major, but we saved row-major (as.vector(t(X)))
    # So we reshape to nrow x ncol directly
    return reshape(vec, ncol, nrow)' |> collect
end

@testset "GP Basic" begin
    # Extract test data from reference
    X = _reshape_matrix(Float64.(GP_BASIC_REF.X), GP_BASIC_REF.X_nrow, GP_BASIC_REF.X_ncol)
    Z = Float64.(GP_BASIC_REF.Z)
    d = Float64(GP_BASIC_REF.d)
    g = Float64(GP_BASIC_REF.g)
    XX = _reshape_matrix(Float64.(GP_BASIC_REF.XX), GP_BASIC_REF.XX_nrow, GP_BASIC_REF.XX_ncol)

    # Reference values
    ref_mean = Float64.(GP_BASIC_REF.pred_mean)
    ref_s2 = Float64.(GP_BASIC_REF.pred_s2)
    ref_llik = Float64(GP_BASIC_REF.llik)
    ref_df = Int(GP_BASIC_REF.df)

    @testset "new_gp construction" begin
        gp = new_gp(X, Z, d, g)
        @test gp isa GP
        @test gp.d == d
        @test gp.g == g
        @test size(gp.X) == (10, 2)
        @test length(gp.Z) == 10
    end

    @testset "pred_gp predictions" begin
        gp = new_gp(X, Z, d, g)
        pred = pred_gp(gp, XX; lite=true)

        @test pred isa GPPrediction
        @test length(pred.mean) == 2
        @test length(pred.s2) == 2
        @test pred.df == ref_df

        @test pred.mean ≈ ref_mean rtol=1e-5
        @test pred.s2 ≈ ref_s2 rtol=1e-5
    end

    @testset "llik_gp log-likelihood" begin
        gp = new_gp(X, Z, d, g)
        llik = llik_gp(gp)

        @test llik isa Real
        @test llik ≈ ref_llik rtol=1e-5
    end
end
