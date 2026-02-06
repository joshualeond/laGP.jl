using laGP
using KernelFunctions: kernelmatrix, RowVecs
using Random
using Test

@testset "Separable Kernel Builders" begin
    Random.seed!(2026)

    for T in (Float64, Float32)
        n = 24
        m = 5
        n_test = 9

        X = rand(T, n, m)
        XX = rand(T, n_test, m)
        d = T.(0.2 .+ rand(m))

        kernel = laGP.build_kernel_separable(d)

        K_ref = kernelmatrix(kernel, RowVecs(X))
        K_fast = Matrix{T}(undef, n, n)
        laGP._kernelmatrix_sep_train!(K_fast, X, d)

        @test K_fast ≈ K_ref rtol=5e-6 atol=5e-7
        @test issymmetric(K_fast)
        @test all(diag(K_fast) .≈ one(T))

        Kx_ref = kernelmatrix(kernel, RowVecs(X), RowVecs(XX))
        Kx_fast = Matrix{T}(undef, n, n_test)
        laGP._kernelmatrix_sep_cross!(Kx_fast, X, XX, d)

        @test Kx_fast ≈ Kx_ref rtol=5e-6 atol=5e-7
    end
end
