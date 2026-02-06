using LinearAlgebra
using Random
using Test
using laGP

@testset "Full Covariance Prediction" begin
    Random.seed!(2026)

    n = 18
    m = 3
    n_test = 6
    X = rand(n, m)
    Z = sin.(3 .* X[:, 1]) .+ 0.2 .* X[:, 2] .- 0.1 .* X[:, 3]
    XX = rand(n_test, m)

    @testset "isotropic full covariance" begin
        gp = new_gp(X, Z, 0.4, 1e-4)
        pred_lite = pred_gp(gp, XX; lite=true)
        pred_full = pred_gp(gp, XX; lite=false)

        @test pred_full isa GPPredictionFull
        @test pred_full.mean ≈ pred_lite.mean rtol=1e-10
        @test diag(pred_full.Sigma) ≈ pred_lite.s2 rtol=1e-8
        @test isapprox(pred_full.Sigma, pred_full.Sigma'; rtol=0, atol=1e-10)
        @test minimum(eigvals(Symmetric(pred_full.Sigma))) > -1e-8
    end

    @testset "separable full covariance" begin
        gp = new_gp_sep(X, Z, [0.3, 0.8, 1.5], 1e-4)
        pred_lite = pred_gp_sep(gp, XX; lite=true)
        pred_full = pred_gp_sep(gp, XX; lite=false)

        @test pred_full isa GPPredictionFull
        @test pred_full.mean ≈ pred_lite.mean rtol=1e-10
        @test diag(pred_full.Sigma) ≈ pred_lite.s2 rtol=1e-8
        @test isapprox(pred_full.Sigma, pred_full.Sigma'; rtol=0, atol=1e-10)
        @test minimum(eigvals(Symmetric(pred_full.Sigma))) > -1e-8
    end
end
