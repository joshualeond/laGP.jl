using Random
using Test
using laGP

@testset "Argument Helper Semantics" begin
    Random.seed!(314)
    X = rand(25, 3)
    Z = randn(25)

    @testset "darg honors explicit start" begin
        info = darg(X; d=0.123)
        @test info.start ≈ 0.123
        @test info.min > 0
        @test info.max > info.min
    end

    @testset "garg defaults and explicit start" begin
        info_default = garg(Z)
        @test info_default.mle == false
        @test info_default.min > 0

        info_fixed = garg(Z; g=0.456)
        @test info_fixed.start ≈ 0.456
        @test info_fixed.mle == false
    end

    @testset "darg_sep honors scalar/vector starts" begin
        info_scalar = darg_sep(X; d=0.2)
        @test all(r.start ≈ 0.2 for r in info_scalar.ranges)

        info_vec = darg_sep(X; d=[0.3, 0.4, 0.5])
        @test [r.start for r in info_vec.ranges] ≈ [0.3, 0.4, 0.5]

        info_singleton_vec = darg_sep(X; d=[0.25])
        @test all(r.start ≈ 0.25 for r in info_singleton_vec.ranges)
    end
end
