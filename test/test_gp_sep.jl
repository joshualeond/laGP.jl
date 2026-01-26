using laGP
using LinearAlgebra
using Statistics: mean
using Test

@testset "GPsep (Separable GP)" begin
    # Create test data - 2D function with different sensitivities per dimension
    # f(x1, x2) = sin(10*x1) + 0.1*x2 (highly sensitive to x1, less to x2)
    n = 30
    m = 2
    X = rand(n, m)
    Z = sin.(10 .* X[:, 1]) .+ 0.1 .* X[:, 2]

    # Test locations
    n_test = 5
    XX = rand(n_test, m)

    @testset "new_gp_sep construction" begin
        d = [0.1, 1.0]  # Different lengthscales per dimension
        g = 1e-4
        gp = new_gp_sep(X, Z, d, g)

        @test gp isa GPsep
        @test gp.d == d
        @test gp.g == g
        @test size(gp.X) == (n, m)
        @test length(gp.Z) == n
        @test length(gp.d) == m
        @test size(gp.K) == (n, n)
        @test issymmetric(gp.K)
    end

    @testset "new_gp_sep validation" begin
        # d length must match X columns
        @test_throws AssertionError new_gp_sep(X, Z, [0.1], 1e-4)
        @test_throws AssertionError new_gp_sep(X, Z, [0.1, 0.2, 0.3], 1e-4)

        # d must be positive
        @test_throws AssertionError new_gp_sep(X, Z, [0.1, -0.2], 1e-4)
        @test_throws AssertionError new_gp_sep(X, Z, [0.0, 0.2], 1e-4)

        # g must be positive
        @test_throws AssertionError new_gp_sep(X, Z, [0.1, 0.2], -0.01)
        @test_throws AssertionError new_gp_sep(X, Z, [0.1, 0.2], 0.0)
    end

    @testset "pred_gp_sep predictions" begin
        d = [0.1, 1.0]
        g = 1e-4
        gp = new_gp_sep(X, Z, d, g)
        pred = pred_gp_sep(gp, XX; lite=true)

        @test pred isa GPPrediction
        @test length(pred.mean) == n_test
        @test length(pred.s2) == n_test
        @test pred.df == n
        @test all(pred.s2 .>= 0)  # Variances must be non-negative
    end

    @testset "pred_gp_sep at training points" begin
        # Predictions at training points should be close to Z (with small nugget)
        d = [0.1, 1.0]
        g = 1e-6
        gp = new_gp_sep(X, Z, d, g)
        pred = pred_gp_sep(gp, X; lite=true)

        @test pred.mean ≈ Z rtol=0.05  # Should be close to training values
    end

    @testset "llik_gp_sep log-likelihood" begin
        d = [0.1, 1.0]
        g = 1e-4
        gp = new_gp_sep(X, Z, d, g)
        llik = llik_gp_sep(gp)

        @test llik isa Real
        @test isfinite(llik)
    end

    @testset "update_gp_sep! hyperparameters" begin
        d = [0.1, 1.0]
        g = 1e-4
        gp = new_gp_sep(X, Z, d, g)
        llik_orig = llik_gp_sep(gp)

        # Update d
        update_gp_sep!(gp; d=[0.2, 2.0])
        @test gp.d == [0.2, 2.0]
        @test llik_gp_sep(gp) != llik_orig

        # Update g
        update_gp_sep!(gp; g=1e-3)
        @test gp.g == 1e-3

        # Update both
        update_gp_sep!(gp; d=[0.3, 0.5], g=1e-5)
        @test gp.d == [0.3, 0.5]
        @test gp.g == 1e-5
    end

    @testset "mle_gp_sep single parameter" begin
        d = [0.5, 0.5]  # Start with equal lengthscales
        g = 1e-3
        gp = new_gp_sep(X, Z, d, g)

        # Optimize d[1]
        result = mle_gp_sep(gp, :d, 1; tmin=0.01, tmax=10.0)
        @test haskey(result, :d)
        @test haskey(result, :g)
        @test haskey(result, :its)
        @test result.its > 0

        # Optimize g
        result = mle_gp_sep(gp, :g; tmin=1e-8, tmax=1.0)
        @test haskey(result, :g)
        @test result.its > 0
    end

    @testset "dllik_gp_sep gradient" begin
        # Test gradient via centered finite differences for better accuracy
        d = [0.1, 1.0]
        g = 1e-3  # Use larger nugget to avoid numerical issues
        gp = new_gp_sep(X, Z, d, g)
        grad = dllik_gp_sep(gp)

        # Check gradient dimensions
        @test length(grad.dlld) == 2
        @test grad.dllg isa Real

        # Centered finite difference for d[1]
        eps_fd = 1e-5
        d_plus = copy(gp.d); d_plus[1] += eps_fd
        d_minus = copy(gp.d); d_minus[1] -= eps_fd
        gp_plus = new_gp_sep(X, Z, d_plus, gp.g)
        gp_minus = new_gp_sep(X, Z, d_minus, gp.g)
        fd_grad_d1 = (llik_gp_sep(gp_plus) - llik_gp_sep(gp_minus)) / (2 * eps_fd)
        @test grad.dlld[1] ≈ fd_grad_d1 rtol=1e-4

        # Centered finite difference for d[2]
        d_plus = copy(gp.d); d_plus[2] += eps_fd
        d_minus = copy(gp.d); d_minus[2] -= eps_fd
        gp_plus = new_gp_sep(X, Z, d_plus, gp.g)
        gp_minus = new_gp_sep(X, Z, d_minus, gp.g)
        fd_grad_d2 = (llik_gp_sep(gp_plus) - llik_gp_sep(gp_minus)) / (2 * eps_fd)
        @test grad.dlld[2] ≈ fd_grad_d2 rtol=1e-4

        # Centered finite difference for g
        gp_plus = new_gp_sep(X, Z, gp.d, gp.g + eps_fd)
        gp_minus = new_gp_sep(X, Z, gp.d, gp.g - eps_fd)
        fd_grad_g = (llik_gp_sep(gp_plus) - llik_gp_sep(gp_minus)) / (2 * eps_fd)
        @test grad.dllg ≈ fd_grad_g rtol=1e-3  # Use looser tolerance for nugget
    end

    @testset "jmle_gp_sep joint optimization" begin
        d = [0.5, 0.5]  # Start with equal lengthscales
        g = 1e-2
        gp = new_gp_sep(X, Z, d, g)
        llik_init = llik_gp_sep(gp)

        result = jmle_gp_sep(gp; drange=(0.01, 10.0), grange=(1e-8, 1.0))

        @test haskey(result, :d)
        @test haskey(result, :g)
        @test haskey(result, :tot_its)

        # Likelihood should improve
        @test llik_gp_sep(gp) >= llik_init

        # For this problem, d[1] should be smaller than d[2] since f is more
        # sensitive to x1
        @test gp.d[1] < gp.d[2]
    end

    @testset "darg_sep helper" begin
        result = darg_sep(X)

        @test haskey(result, :ranges)
        @test haskey(result, :ab)
        @test length(result.ranges) == m
        for dim in 1:m
            @test haskey(result.ranges[dim], :start)
            @test haskey(result.ranges[dim], :min)
            @test haskey(result.ranges[dim], :max)
            @test haskey(result.ranges[dim], :mle)
            @test result.ranges[dim].min < result.ranges[dim].start < result.ranges[dim].max
        end
    end

    @testset "separable vs isotropic on anisotropic data" begin
        # Test that separable GP outperforms isotropic on anisotropic data
        # Create data where one dimension matters much more
        n_train = 50
        n_test = 20

        X_train = rand(n_train, 2)
        Z_train = sin.(10 .* X_train[:, 1]) .+ 0.01 .* X_train[:, 2]

        X_test = rand(n_test, 2)
        Z_test = sin.(10 .* X_test[:, 1]) .+ 0.01 .* X_test[:, 2]

        # Fit isotropic GP
        da_iso = darg(X_train)
        ga = garg(Z_train)
        gp_iso = new_gp(X_train, Z_train, da_iso.start, ga.start)
        jmle_gp(gp_iso; drange=(da_iso.min, da_iso.max), grange=(ga.min, ga.max))
        pred_iso = pred_gp(gp_iso, X_test; lite=true)
        rmse_iso = sqrt(mean((pred_iso.mean .- Z_test).^2))

        # Fit separable GP
        da_sep = darg_sep(X_train)
        d_start = [r.start for r in da_sep.ranges]
        d_ranges = [(r.min, r.max) for r in da_sep.ranges]
        gp_sep = new_gp_sep(X_train, Z_train, d_start, ga.start)
        jmle_gp_sep(gp_sep; drange=d_ranges, grange=(ga.min, ga.max))
        pred_sep = pred_gp_sep(gp_sep, X_test; lite=true)
        rmse_sep = sqrt(mean((pred_sep.mean .- Z_test).^2))

        # Separable should perform better or similarly on anisotropic data
        @test rmse_sep <= rmse_iso * 1.5  # Allow some slack due to random data
    end
end
