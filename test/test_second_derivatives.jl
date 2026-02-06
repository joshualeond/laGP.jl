using Random
using Test
using laGP

@testset "Second Derivative Checks" begin
    @testset "isotropic d and g curvature" begin
        for seed in 1:3
            Random.seed!(1000 + seed)
            n = 20
            m = 3
            X = rand(n, m)
            Z = sin.(3 .* X[:, 1]) .+ 0.2 .* cos.(5 .* X[:, 2]) .+ 0.05 .* randn(n)

            d = 0.2 + 0.6 * rand()
            g = 1e-3 + 2e-3 * rand()
            gp = new_gp(X, Z, d, g)

            h_d = 1e-4
            h_g = min(1e-5, g / 5)

            d2 = d2llik_gp(gp)

            llik_d_plus = llik_gp(new_gp(X, Z, d + h_d, g))
            llik_d = llik_gp(new_gp(X, Z, d, g))
            llik_d_minus = llik_gp(new_gp(X, Z, d - h_d, g))
            fd_d2_d = (llik_d_plus - 2 * llik_d + llik_d_minus) / (h_d * h_d)

            llik_g_plus = llik_gp(new_gp(X, Z, d, g + h_g))
            llik_g = llik_gp(new_gp(X, Z, d, g))
            llik_g_minus = llik_gp(new_gp(X, Z, d, g - h_g))
            fd_d2_g = (llik_g_plus - 2 * llik_g + llik_g_minus) / (h_g * h_g)

            @test d2.d2lld ≈ fd_d2_d rtol=1e-3 atol=1e-2
            @test d2.d2llg ≈ fd_d2_g rtol=1e-3 atol=1e-1
        end
    end

    @testset "separable nugget curvature" begin
        for seed in 1:3
            Random.seed!(2000 + seed)
            n = 20
            m = 3
            X = rand(n, m)
            Z = sin.(2.5 .* X[:, 1]) .+ 0.1 .* X[:, 2] .- 0.2 .* X[:, 3] .+ 0.03 .* randn(n)

            d = [0.25, 0.8, 1.4]
            g = 1e-3 + 2e-3 * rand()
            gp = new_gp_sep(X, Z, d, g)

            h_g = min(1e-5, g / 5)
            fd_g = (llik_gp_sep(new_gp_sep(X, Z, d, g + h_g)) -
                    2 * llik_gp_sep(new_gp_sep(X, Z, d, g)) +
                    llik_gp_sep(new_gp_sep(X, Z, d, g - h_g))) / (h_g * h_g)

            @test d2llik_gp_sep_nug(gp) ≈ fd_g rtol=1e-3 atol=1e-1
        end
    end
end
