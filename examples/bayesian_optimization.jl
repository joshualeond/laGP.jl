# Bayesian Optimization Example with laGP.jl
#
# This example demonstrates:
# 1. Sequential Bayesian optimization with GP surrogate
# 2. UCB (Upper Confidence Bound) acquisition function
# 3. Two-panel visualization: GP fit + acquisition function
#
# Reproduces the visualization style from BayesianOptimization.jl
# but uses laGP.jl for the GP modeling.

using laGP
using CairoMakie
using Random
using Distributions: TDist, quantile

# ============================================================================
# Objective Function
# ============================================================================

# Classic test function with global maximum near x ≈ 2
f(x) = exp(-(x - 2)^2) + exp(-(x - 6)^2 / 10) + 1 / (x^2 + 1)

# Domain
const X_MIN = -2.0
const X_MAX = 10.0

println("Objective function: f(x) = exp(-(x-2)²) + exp(-(x-6)²/10) + 1/(x²+1)")
println("Domain: [$X_MIN, $X_MAX]")
println("Global maximum: x ≈ 2.0")

# ============================================================================
# UCB Acquisition Function
# ============================================================================

"""
    ucb(mean, variance, κ=5.0)

Upper Confidence Bound acquisition function.

For maximization: UCB(x) = μ(x) + κ * σ(x)

- `mean`: GP posterior mean
- `variance`: GP posterior variance (s2)
- `κ`: exploration-exploitation tradeoff parameter (default: 5.0)
"""
function ucb(mean, variance, κ=5.0)
    return mean .+ κ .* sqrt.(variance)
end

# ============================================================================
# Visualization Function
# ============================================================================

"""
    plot_bo_step(x_grid, pred, X_obs, Y_obs, ucb_vals, x_next, κ; iter=1)

Create two-panel Bayesian optimization visualization.

Upper panel: GP mean, 95% CI, true function, and observations
Lower panel: UCB acquisition function with next point marked
"""
function plot_bo_step(x_grid, pred, X_obs, Y_obs, ucb_vals, x_next, κ; iter=1)
    fig = Figure(size=(700, 600))

    # ---- Upper Panel: GP Fit ----
    ax1 = Axis(fig[1, 1],
        title="Gaussian Process Fit (Iteration $iter)",
        ylabel="f(x)")

    # True function (solid black)
    lines!(ax1, x_grid, f.(x_grid), color=:black, linewidth=1.5, label="True f(x)")

    # GP mean (dashed black)
    lines!(ax1, x_grid, pred.mean, color=:black, linestyle=:dash,
           linewidth=2, label="GP mean")

    # 95% CI using Student-t (correct for concentrated likelihood)
    t_crit = quantile(TDist(pred.df), 0.975)
    std_pred = sqrt.(pred.s2)
    lower = pred.mean .- t_crit .* std_pred
    upper = pred.mean .+ t_crit .* std_pred

    band!(ax1, x_grid, lower, upper, color=(:gray, 0.3), label="95% CI")

    # Observations (red circles)
    scatter!(ax1, vec(X_obs), Y_obs, color=:red, marker=:circle,
             markersize=10, label="Observations")

    axislegend(ax1, position=:rt)
    xlims!(ax1, X_MIN, X_MAX)

    # ---- Lower Panel: Acquisition Function ----
    ax2 = Axis(fig[2, 1],
        title="UCB Acquisition (κ=$κ)",
        xlabel="x", ylabel="UCB(x)")

    lines!(ax2, x_grid, ucb_vals, color=:gray40, linewidth=2)

    # Mark next evaluation point (red star)
    next_idx = argmax(ucb_vals)
    scatter!(ax2, [x_next], [ucb_vals[next_idx]],
             color=:red, marker=:star5, markersize=20, label="Next point")

    axislegend(ax2, position=:rt)
    xlims!(ax2, X_MIN, X_MAX)

    return fig
end

# ============================================================================
# Bayesian Optimization Loop
# ============================================================================

println("\n" * "="^60)
println("Starting Bayesian Optimization")
println("="^60)

# Set seed for reproducibility (seed 11 converges to global max at x ≈ 2)
Random.seed!(11)

# Initial random samples
n_init = 2
X_obs = rand(n_init) .* (X_MAX - X_MIN) .+ X_MIN  # uniform in [X_MIN, X_MAX]
X_obs = reshape(X_obs, :, 1)  # convert to n × 1 matrix
Y_obs = f.(X_obs[:, 1])

println("\nInitial observations:")
for i in 1:n_init
    println("  x = $(round(X_obs[i, 1], digits=3)), f(x) = $(round(Y_obs[i], digits=3))")
end

# Prediction grid
x_grid = collect(range(X_MIN, X_MAX, length=200))
X_pred = reshape(x_grid, :, 1)

# UCB exploration parameter
κ = 5.0

# Number of optimization iterations
n_iters = 7

# Output directory
output_dir = @__DIR__

# Run optimization loop
for iter in 0:n_iters
    println("\n--- Iteration $iter ---")

    # Fit GP with MLE
    d_range = darg(X_obs)
    g_range = garg(Y_obs)

    # Create GP with small nugget (objective is noise-free)
    gp = new_gp(X_obs, Y_obs, d_range.start, 1e-6)

    # MLE for lengthscale (keep nugget small and fixed)
    mle_gp(gp, :d; tmin=d_range.min, tmax=d_range.max)

    println("  GP hyperparameters: d=$(round(gp.d, sigdigits=4)), g=$(gp.g)")
    println("  Log-likelihood: $(round(llik_gp(gp), sigdigits=4))")

    # Predict on grid
    pred = pred_gp(gp, X_pred; lite=true)

    # Compute UCB acquisition
    ucb_vals = ucb(pred.mean, pred.s2, κ)

    # Find next evaluation point
    next_idx = argmax(ucb_vals)
    x_next = x_grid[next_idx]

    println("  Next point: x=$(round(x_next, digits=3))")

    # Create visualization
    fig = plot_bo_step(x_grid, pred, X_obs, Y_obs, ucb_vals, x_next, κ; iter=iter)

    # Save figure
    filename = "bo_step_$iter.png"
    save(joinpath(output_dir, filename), fig, px_per_unit=2)
    println("  Saved: $filename")

    # Add new observation (except on last iteration)
    if iter < n_iters
        y_next = f(x_next)
        global X_obs = vcat(X_obs, [x_next;;])
        global Y_obs = vcat(Y_obs, y_next)
        println("  f($(round(x_next, digits=3))) = $(round(y_next, digits=3))")
    end
end

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^60)
println("Optimization Complete")
println("="^60)

# Find best observation
best_idx = argmax(Y_obs)
println("\nBest observation found:")
println("  x* = $(round(X_obs[best_idx, 1], digits=4))")
println("  f(x*) = $(round(Y_obs[best_idx], digits=4))")

# True optimum
x_true = 2.0
println("\nTrue optimum:")
println("  x* = $x_true")
println("  f(x*) = $(round(f(x_true), digits=4))")

println("\nFigures saved to: $output_dir")
println("  bo_step_0.png through bo_step_$n_iters.png")
