# Wing Weight GP Comparison Test - Julia
using laGP
using LatinHypercubeSampling
using Random
using Statistics: mean

Random.seed!(42)

# Wing weight function (coded inputs [0,1])
function wingwt(X::Matrix{Float64})
    n = size(X, 1)
    Y = Vector{Float64}(undef, n)
    for i in 1:n
        Sw  = X[i,1] * 50 + 150
        Wfw = X[i,2] * 80 + 220
        A   = X[i,3] * 4 + 6
        L   = (X[i,4] * 20 - 10) * π / 180
        q   = X[i,5] * 29 + 16
        l   = X[i,6] * 0.5 + 0.5
        Rtc = X[i,7] * 0.1 + 0.08
        Nz  = X[i,8] * 3.5 + 2.5
        Wdg = X[i,9] * 800 + 1700

        W = 0.036 * Sw^0.758 * Wfw^0.0035
        W *= (A / cos(L)^2)^0.6
        W *= q^0.006
        W *= l^0.04
        W *= (100 * Rtc / cos(L))^(-0.3)
        W *= (Nz * Wdg)^0.49
        Y[i] = W
    end
    return Y
end

# LHS design (use smaller n for faster comparison)
n = 200
plan, _ = LHCoptim(n, 9, 5)
X = Matrix{Float64}(plan ./ n)
Y = wingwt(X)

println("Response range: [$(minimum(Y)), $(maximum(Y))]")

# Fit GP with FIXED initial values (matching R exactly)
d_init = fill(2.0, 9)  # Same as R: d=2
g_init = 1e-6          # Same as R: g=1e-6

gp = new_gp_sep(X, Y, d_init, g_init)
println("\nInitial log-likelihood: $(llik_gp_sep(gp))")

# MLE optimization with ranges matching R's darg/garg defaults
# R uses eps = sqrt(.Machine$double.eps) for lower bound, max d=81
drange = (sqrt(eps(Float64)), 81.0)   # lengthscale range (R default max)
grange = (sqrt(eps(Float64)), 1.0)    # nugget range
jmle_gp_sep(gp; drange=drange, grange=grange)

println("\nMLE results:")
println("  d: $(gp.d)")
println("  g: $(gp.g)")
println("Final log-likelihood: $(llik_gp_sep(gp))")

# Test predictions
Random.seed!(123)
n_test = 100
plan_test, _ = LHCoptim(n_test, 9, 5)
X_test = Matrix{Float64}(plan_test ./ n_test)
Y_test = wingwt(X_test)

pred = pred_gp_sep(gp, X_test; lite=true)
rmse = sqrt(mean((Y_test .- pred.mean).^2))

println("\nTest RMSE: $rmse lb")
println("Test RMSE %: $(100 * rmse / (maximum(Y) - minimum(Y)))%")
