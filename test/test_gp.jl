using JSON3
using laGP
using Test

include("test_utils.jl")

# Load reference data
const GP_BASIC_REF = JSON3.read(
    read(joinpath(@__DIR__, "reference", "gp_basic.json"), String)
)

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

    @testset "extend_gp! incremental update" begin
        # Start with first 5 points
        X_start = X[1:5, :]
        Z_start = Z[1:5]
        gp_extended = new_gp(X_start, Z_start, d, g)

        # Extend with remaining 5 points one at a time
        for i in 6:10
            x_new = X[i, :]
            z_new = Z[i]
            extend_gp!(gp_extended, x_new, z_new)
        end

        # Create reference GP with all 10 points
        gp_full = new_gp(X, Z, d, g)

        # Test that predictions match
        pred_extended = pred_gp(gp_extended, XX; lite=true)
        pred_full = pred_gp(gp_full, XX; lite=true)

        @test pred_extended.mean ≈ pred_full.mean rtol=1e-5
        @test pred_extended.s2 ≈ pred_full.s2 rtol=1e-5

        # Test that log-likelihood matches
        llik_extended = llik_gp(gp_extended)
        llik_full = llik_gp(gp_full)
        @test llik_extended ≈ llik_full rtol=1e-5

        # Test that internal quantities match
        @test gp_extended.phi ≈ gp_full.phi rtol=1e-5
        @test gp_extended.ldetK ≈ gp_full.ldetK rtol=1e-5
        @test gp_extended.KiZ ≈ gp_full.KiZ rtol=1e-5

        # Test that Ki matches (within numerical tolerance)
        @test gp_extended.Ki ≈ gp_full.Ki rtol=1e-4
    end

    @testset "extend_gp! single point extension" begin
        # Test extending by just one point
        X_start = X[1:9, :]
        Z_start = Z[1:9]
        gp = new_gp(X_start, Z_start, d, g)

        # Extend with last point
        extend_gp!(gp, X[10, :], Z[10])

        # Compare with full GP
        gp_full = new_gp(X, Z, d, g)

        @test size(gp.X) == size(gp_full.X)
        @test size(gp.chol) == size(gp_full.chol)
        @test llik_gp(gp) ≈ llik_gp(gp_full) rtol=1e-5
    end
end
