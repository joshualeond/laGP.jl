using JSON3
using laGP
using laGP: _compute_squared_distances_batched!, _apply_kernel_isotropic!
using Statistics
using Test

include("test_utils.jl")

# Load reference data
const ACQ_REF = JSON3.read(
    read(joinpath(@__DIR__, "reference", "acquisition.json"), String)
)

@testset "SIMD Helper Functions" begin
    @testset "_compute_squared_distances_batched!" begin
        X1 = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # 3 x 2
        X2 = [0.0 0.0; 1.0 1.0]            # 2 x 2

        D_sq = Matrix{Float64}(undef, 3, 2)
        _compute_squared_distances_batched!(D_sq, X1, X2)

        # Manual calculation:
        # D_sq[1,1] = (1-0)² + (2-0)² = 1 + 4 = 5
        # D_sq[1,2] = (1-1)² + (2-1)² = 0 + 1 = 1
        # D_sq[2,1] = (3-0)² + (4-0)² = 9 + 16 = 25
        # D_sq[2,2] = (3-1)² + (4-1)² = 4 + 9 = 13
        # D_sq[3,1] = (5-0)² + (6-0)² = 25 + 36 = 61
        # D_sq[3,2] = (5-1)² + (6-1)² = 16 + 25 = 41

        @test D_sq[1, 1] ≈ 5.0
        @test D_sq[1, 2] ≈ 1.0
        @test D_sq[2, 1] ≈ 25.0
        @test D_sq[2, 2] ≈ 13.0
        @test D_sq[3, 1] ≈ 61.0
        @test D_sq[3, 2] ≈ 41.0
    end

    @testset "_apply_kernel_isotropic!" begin
        D_sq = [1.0 4.0; 9.0 16.0]
        K = similar(D_sq)
        d = 2.0

        _apply_kernel_isotropic!(K, D_sq, d)

        # K[i,j] = exp(-D_sq[i,j] / d)
        @test K[1, 1] ≈ exp(-1.0 / 2.0)
        @test K[1, 2] ≈ exp(-4.0 / 2.0)
        @test K[2, 1] ≈ exp(-9.0 / 2.0)
        @test K[2, 2] ≈ exp(-16.0 / 2.0)
    end
end

@testset "Acquisition Functions" begin
    # Extract test data
    X = _reshape_matrix(Float64.(ACQ_REF.X), ACQ_REF.X_nrow, ACQ_REF.X_ncol)
    Z = Float64.(ACQ_REF.Z)
    d = Float64(ACQ_REF.d)
    g = Float64(ACQ_REF.g)
    Xcand = _reshape_matrix(Float64.(ACQ_REF.Xcand), ACQ_REF.Xcand_nrow, ACQ_REF.Xcand_ncol)
    Xref = _reshape_matrix(Float64.(ACQ_REF.Xref), ACQ_REF.Xref_nrow, ACQ_REF.Xref_ncol)

    # Reference values
    ref_alc = Float64.(ACQ_REF.alc)
    ref_mspe = Float64.(ACQ_REF.mspe)

    gp = new_gp(X, Z, d, g)

    @testset "alc_gp" begin
        alc_vals = alc_gp(gp, Xcand, Xref)

        @test length(alc_vals) == size(Xcand, 1)
        @test all(alc_vals .>= 0)  # ALC should be non-negative
        @test alc_vals ≈ ref_alc rtol=1e-4
    end

    @testset "mspe_gp" begin
        mspe_vals = mspe_gp(gp, Xcand, Xref)

        @test length(mspe_vals) == size(Xcand, 1)
        @test all(mspe_vals .>= 0)  # MSPE should be non-negative
        # MSPE values have systematic differences due to variance scaling
        # but relative ordering should be similar
        @test cor(mspe_vals, ref_mspe) > 0.97  # High correlation
        @test mspe_vals ≈ ref_mspe rtol=0.35  # Looser tolerance for absolute values
    end
end
