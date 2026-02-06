# Chemical Yield Optimization: RSM + Bayesian Optimization
#
# George Box-inspired chemical manufacturing example demonstrating:
# 1. Modern Response Surface Methodology with GP surrogates
# 2. Expected Improvement-based Bayesian Optimization
# 3. Separable (ARD) GP revealing anisotropic sensitivity
#
# Synthetic 2D surface: Temperature (150-300 C) x Pressure (1-10 atm) -> Yield (%)
# Multi-modal: global peak ~96% at T~232, P~6.5 with two local optima

using laGP
using CairoMakie
using Random
using Distributions: Normal, cdf, pdf
using LatinHypercubeSampling

# Output directory
const OUTPUT_DIR = @__DIR__

# ============================================================================
# Yield Function
# ============================================================================

"""
    chemical_yield(T, P)

Synthetic chemical yield (%) as a function of temperature T (150-300 C)
and pressure P (1-10 atm).

Multi-modal surface with three optima:
- Global: ~96% at T~232, P~6.5 (narrow, anisotropic peak)
- Local 1: ~74% at T~175, P~2.8 (low-T, low-P region)
- Local 2: ~73% at T~275, P~8.5 (high-T, high-P region)

Temperature is ~3x more sensitive than Pressure (motivates ARD).
"""
function chemical_yield(T, P)
    # Normalize to internal coordinates
    t = (T - 150) / 150   # [0, 1]
    p = (P - 1) / 9       # [0, 1]

    # Global optimum: T~230, P~6.5 (t~0.533, p~0.611), yield ~93%
    # Narrow, anisotropic peak — temperature ~3x more sensitive
    t0, p0 = 0.533, 0.611
    global_peak = 45.0 * exp(-25.0 * (t - t0)^2 - 8.0 * (p - p0)^2)

    # Local optimum 1: low-T, low-P — T~175, P~2.8 (t~0.167, p~0.2), yield ~74%
    local1 = 26.0 * exp(-20.0 * (t - 0.167)^2 - 15.0 * (p - 0.20)^2)

    # Local optimum 2: high-T, high-P — T~275, P~8.5 (t~0.833, p~0.833), yield ~70%
    local2 = 22.0 * exp(-18.0 * (t - 0.833)^2 - 12.0 * (p - 0.833)^2)

    # Baseline yield (flat background)
    base = 48.0

    return base + global_peak + local1 + local2
end

# Coded [0,1]^2 version for GP
function yield_coded(x1, x2)
    T = 150.0 + 150.0 * x1
    P = 1.0 + 9.0 * x2
    return chemical_yield(T, P)
end

# ============================================================================
# Expected Improvement (for maximization)
# ============================================================================

"""
    expected_improvement(mu, sigma, f_best)

Expected Improvement acquisition function for maximization.
EI(x) = (mu(x) - f_best) * Phi(z) + sigma(x) * phi(z)
where z = (mu(x) - f_best) / sigma(x).
"""
function expected_improvement(mu, sigma, f_best)
    ei = zeros(length(mu))
    for i in eachindex(mu)
        if sigma[i] > 1e-10
            z = (mu[i] - f_best) / sigma[i]
            ei[i] = (mu[i] - f_best) * cdf(Normal(), z) + sigma[i] * pdf(Normal(), z)
        end
    end
    return ei
end

# ============================================================================
# Section 1: True Yield Surface
# ============================================================================

println("="^60)
println("Section 1: True Yield Surface")
println("="^60)

n_grid = 100
T_grid = range(150, 300, length=n_grid)
P_grid = range(1, 10, length=n_grid)
Z_true = [chemical_yield(T, P) for T in T_grid, P in P_grid]

println("Yield range: $(round(minimum(Z_true), digits=1)) - $(round(maximum(Z_true), digits=1))%")

# Find true optimum
best_idx = argmax(Z_true)
println("True optimum: T=$(round(T_grid[best_idx[1]], digits=1)) C, " *
        "P=$(round(P_grid[best_idx[2]], digits=1)) atm, " *
        "Yield=$(round(Z_true[best_idx], digits=1))%")

fig1 = Figure(size=(1100, 450))

# 3D surface
ax3d = Axis3(fig1[1, 1], xlabel="Temperature (C)", ylabel="Pressure (atm)",
             zlabel="Yield (%)", title="Chemical Yield Surface",
             azimuth=0.9π)
surface!(ax3d, collect(T_grid), collect(P_grid), Z_true, colormap=:viridis)

# 2D contour
ax2d = Axis(fig1[1, 2], xlabel="Temperature (C)", ylabel="Pressure (atm)",
            title="Yield Contours")
hm = heatmap!(ax2d, collect(T_grid), collect(P_grid), Z_true, colormap=:viridis)
contour!(ax2d, collect(T_grid), collect(P_grid), Z_true,
         color=:white, linewidth=0.8, levels=15)
Colorbar(fig1[1, 3], hm, label="Yield (%)")

save(joinpath(OUTPUT_DIR, "chem_true_surface.png"), fig1, px_per_unit=2)
println("Saved: chem_true_surface.png")

# ============================================================================
# Section 2: Initial LHS Design
# ============================================================================

println("\n" * "="^60)
println("Section 2: Initial LHS Design (8 points)")
println("="^60)

Random.seed!(42)

n_init = 8
plan, _ = LHCoptim(n_init, 2, 100)
X_coded = scaleLHC(plan, [(0.0, 1.0), (0.0, 1.0)])

# Evaluate yield
Y_obs = [yield_coded(X_coded[i, 1], X_coded[i, 2]) for i in 1:n_init]

# Convert to natural units for display
T_obs = 150.0 .+ 150.0 .* X_coded[:, 1]
P_obs = 1.0 .+ 9.0 .* X_coded[:, 2]

println("Initial design points:")
for i in 1:n_init
    println("  T=$(round(T_obs[i], digits=1)) C, P=$(round(P_obs[i], digits=1)) atm, " *
            "Yield=$(round(Y_obs[i], digits=1))%")
end
println("Best initial: $(round(maximum(Y_obs), digits=1))%")

fig2 = Figure(size=(600, 500))
ax = Axis(fig2[1, 1], xlabel="Temperature (C)", ylabel="Pressure (atm)",
          title="Initial LHS Design (n=$n_init)")
hm = heatmap!(ax, collect(T_grid), collect(P_grid), Z_true, colormap=:viridis)
contour!(ax, collect(T_grid), collect(P_grid), Z_true,
         color=:white, linewidth=0.5, levels=12)
scatter!(ax, T_obs, P_obs, color=:white, markersize=12,
         strokecolor=:black, strokewidth=1.5)
Colorbar(fig2[1, 2], hm, label="Yield (%)")

save(joinpath(OUTPUT_DIR, "chem_initial_design.png"), fig2, px_per_unit=2)
println("Saved: chem_initial_design.png")

# ============================================================================
# Section 3: Initial GP Surrogate
# ============================================================================

println("\n" * "="^60)
println("Section 3: Initial GP Surrogate (Separable)")
println("="^60)

# Fit separable GP in coded space
d_info = darg_sep(X_coded)
g_info = garg(Y_obs)

d_init = [d_info.ranges[1].start, d_info.ranges[2].start]
gp = new_gp_sep(X_coded, Y_obs, d_init, g_info.start)

drange = (d_info.ranges[1].min, d_info.ranges[1].max)
grange = (g_info.min, g_info.max)
jmle_gp_sep!(gp; drange=drange, grange=grange, verb=0)

println("Fitted hyperparameters:")
println("  d = $(round.(gp.d, sigdigits=4))")
println("  g = $(round(gp.g, sigdigits=4))")
println("  log-likelihood = $(round(llik_gp_sep(gp), sigdigits=4))")

# Prediction grid in coded space
n_g = 80
x1_grid = range(0, 1, length=n_g)
x2_grid = range(0, 1, length=n_g)

XX = Matrix{Float64}(undef, n_g * n_g, 2)
let idx = 1
    for j in 1:n_g
        for i in 1:n_g
            XX[idx, 1] = x1_grid[i]
            XX[idx, 2] = x2_grid[j]
            idx += 1
        end
    end
end

pred = pred_gp_sep(gp, XX; lite=true)
mean_grid = reshape(pred.mean, n_g, n_g)
std_grid = reshape(sqrt.(pred.s2), n_g, n_g)

# Convert grid axes to natural units for display
T_disp = collect(150.0 .+ 150.0 .* x1_grid)
P_disp = collect(1.0 .+ 9.0 .* x2_grid)

fig3 = Figure(size=(1100, 450))

ax_mean = Axis(fig3[1, 1], xlabel="Temperature (C)", ylabel="Pressure (atm)",
               title="GP Mean Prediction")
hm_mean = heatmap!(ax_mean, T_disp, P_disp, mean_grid, colormap=:viridis)
contour!(ax_mean, T_disp, P_disp, mean_grid, color=:white, linewidth=0.5, levels=12)
scatter!(ax_mean, T_obs, P_obs, color=:white, markersize=10,
         strokecolor=:black, strokewidth=1.0)
Colorbar(fig3[1, 2], hm_mean, label="Predicted Yield (%)")

ax_std = Axis(fig3[1, 3], xlabel="Temperature (C)", ylabel="Pressure (atm)",
              title="GP Uncertainty (Std)")
hm_std = heatmap!(ax_std, T_disp, P_disp, std_grid, colormap=:plasma)
scatter!(ax_std, T_obs, P_obs, color=:white, markersize=10,
         strokecolor=:black, strokewidth=1.0)
Colorbar(fig3[1, 4], hm_std, label="Std (%)")

save(joinpath(OUTPUT_DIR, "chem_initial_surrogate.png"), fig3, px_per_unit=2)
println("Saved: chem_initial_surrogate.png")

# ============================================================================
# Section 4: Bayesian Optimization Loop (15 iterations)
# ============================================================================

println("\n" * "="^60)
println("Section 4: Bayesian Optimization (15 iterations)")
println("="^60)

X_bo = copy(X_coded)
Y_bo = copy(Y_obs)
best_history = Float64[]

n_bo = 15
plot_iters = [1, 5, 10, 15]

for iter in 1:n_bo
    # Rebuild GP each iteration
    d_info_i = darg_sep(X_bo)
    g_info_i = garg(Y_bo)

    d_init_i = [d_info_i.ranges[1].start, d_info_i.ranges[2].start]
    gp_i = new_gp_sep(X_bo, Y_bo, d_init_i, g_info_i.start)

    drange_i = (d_info_i.ranges[1].min, d_info_i.ranges[1].max)
    grange_i = (g_info_i.min, g_info_i.max)
    jmle_gp_sep!(gp_i; drange=drange_i, grange=grange_i, verb=0)

    # Predict on grid
    pred_i = pred_gp_sep(gp_i, XX; lite=true)
    mu_i = pred_i.mean
    sigma_i = sqrt.(pred_i.s2)

    # Compute EI
    f_best = maximum(Y_bo)
    ei_i = expected_improvement(mu_i, sigma_i, f_best)

    # Find next point
    next_idx = argmax(ei_i)
    x_next = XX[next_idx, :]

    # Evaluate
    y_next = yield_coded(x_next[1], x_next[2])

    T_next = 150.0 + 150.0 * x_next[1]
    P_next = 1.0 + 9.0 * x_next[2]

    println("Iter $iter: T=$(round(T_next, digits=1)) C, P=$(round(P_next, digits=1)) atm, " *
            "Yield=$(round(y_next, digits=1))%, Best=$(round(max(f_best, y_next), digits=1))%")

    # Plot selected iterations
    if iter in plot_iters
        mean_grid_i = reshape(mu_i, n_g, n_g)
        ei_grid_i = reshape(ei_i, n_g, n_g)

        T_all = 150.0 .+ 150.0 .* X_bo[:, 1]
        P_all = 1.0 .+ 9.0 .* X_bo[:, 2]

        fig_bo = Figure(size=(1100, 450))

        # GP Mean panel
        ax_gp = Axis(fig_bo[1, 1], xlabel="Temperature (C)", ylabel="Pressure (atm)",
                      title="GP Mean (Iteration $iter)")
        hm_gp = heatmap!(ax_gp, T_disp, P_disp, mean_grid_i, colormap=:viridis)
        contour!(ax_gp, T_disp, P_disp, mean_grid_i, color=:white, linewidth=0.5, levels=12)
        scatter!(ax_gp, T_all, P_all, color=:white, markersize=8,
                 strokecolor=:black, strokewidth=0.8)
        scatter!(ax_gp, [T_next], [P_next], color=:red, marker=:star5,
                 markersize=18, strokecolor=:black, strokewidth=1.0)
        Colorbar(fig_bo[1, 2], hm_gp, label="Predicted Yield (%)")

        # EI panel
        ax_ei = Axis(fig_bo[1, 3], xlabel="Temperature (C)", ylabel="Pressure (atm)",
                      title="Expected Improvement")
        hm_ei = heatmap!(ax_ei, T_disp, P_disp, ei_grid_i, colormap=:inferno)
        scatter!(ax_ei, T_all, P_all, color=:white, markersize=8,
                 strokecolor=:black, strokewidth=0.8)
        scatter!(ax_ei, [T_next], [P_next], color=:red, marker=:star5,
                 markersize=18, strokecolor=:black, strokewidth=1.0)
        Colorbar(fig_bo[1, 4], hm_ei, label="EI")

        save(joinpath(OUTPUT_DIR, "chem_bo_iter_$iter.png"), fig_bo, px_per_unit=2)
        println("  Saved: chem_bo_iter_$iter.png")
    end

    # Add observation
    global X_bo = vcat(X_bo, x_next')
    global Y_bo = vcat(Y_bo, y_next)
    push!(best_history, maximum(Y_bo))
end

# ============================================================================
# Section 5: Final Surrogate
# ============================================================================

println("\n" * "="^60)
println("Section 5: Final GP Surrogate")
println("="^60)

# Fit final GP
d_info_f = darg_sep(X_bo)
g_info_f = garg(Y_bo)

d_init_f = [d_info_f.ranges[1].start, d_info_f.ranges[2].start]
gp_final = new_gp_sep(X_bo, Y_bo, d_init_f, g_info_f.start)

drange_f = (d_info_f.ranges[1].min, d_info_f.ranges[1].max)
grange_f = (g_info_f.min, g_info_f.max)
jmle_gp_sep!(gp_final; drange=drange_f, grange=grange_f, verb=0)

pred_final = pred_gp_sep(gp_final, XX; lite=true)
mean_final = reshape(pred_final.mean, n_g, n_g)
std_final = reshape(sqrt.(pred_final.s2), n_g, n_g)

# True surface on same grid
Z_disp = [yield_coded(x1, x2) for x1 in x1_grid, x2 in x2_grid]

println("Final GP: d=$(round.(gp_final.d, sigdigits=4)), g=$(round(gp_final.g, sigdigits=4))")
println("Best yield found: $(round(maximum(Y_bo), digits=1))%")

T_all_final = 150.0 .+ 150.0 .* X_bo[:, 1]
P_all_final = 1.0 .+ 9.0 .* X_bo[:, 2]

clims = (minimum(Z_disp), maximum(Z_disp))

fig5 = Figure(size=(1400, 420))

# True surface
ax_t = Axis(fig5[1, 1], xlabel="Temperature (C)", ylabel="Pressure (atm)",
            title="True Yield")
hm_t = heatmap!(ax_t, T_disp, P_disp, Z_disp, colormap=:viridis, colorrange=clims)
contour!(ax_t, T_disp, P_disp, Z_disp, color=:white, linewidth=0.5, levels=12)

# GP Mean
ax_m = Axis(fig5[1, 2], xlabel="Temperature (C)", ylabel="Pressure (atm)",
            title="GP Mean (n=$(size(X_bo, 1)))")
hm_m = heatmap!(ax_m, T_disp, P_disp, mean_final, colormap=:viridis, colorrange=clims)
contour!(ax_m, T_disp, P_disp, mean_final, color=:white, linewidth=0.5, levels=12)
scatter!(ax_m, T_all_final, P_all_final, color=:white, markersize=8,
         strokecolor=:black, strokewidth=0.8)

# GP Uncertainty
ax_u = Axis(fig5[1, 3], xlabel="Temperature (C)", ylabel="Pressure (atm)",
            title="GP Uncertainty")
hm_u = heatmap!(ax_u, T_disp, P_disp, std_final, colormap=:plasma)
scatter!(ax_u, T_all_final, P_all_final, color=:white, markersize=8,
         strokecolor=:black, strokewidth=0.8)

Colorbar(fig5[1, 4], hm_t, label="Yield (%)")

save(joinpath(OUTPUT_DIR, "chem_final_surrogate.png"), fig5, px_per_unit=2)
println("Saved: chem_final_surrogate.png")

# ============================================================================
# Section 6: Convergence Plot
# ============================================================================

println("\n" * "="^60)
println("Section 6: Convergence")
println("="^60)

# True optimum
true_opt = maximum(Z_true)
n_total = n_init + n_bo

println("Total experiments: $n_total ($n_init initial + $n_bo BO)")
println("Best yield: $(round(maximum(Y_bo), digits=1))%")
println("True optimum: $(round(true_opt, digits=1))%")
println("Gap: $(round(true_opt - maximum(Y_bo), digits=1))%")

fig6 = Figure(size=(700, 450))

ax_conv = Axis(fig6[1, 1], xlabel="Iteration", ylabel="Best Yield (%)",
               title="Bayesian Optimization Convergence")

# Plot convergence
lines!(ax_conv, 1:n_bo, best_history, color=:blue, linewidth=2, label="BO best yield")
scatter!(ax_conv, 1:n_bo, best_history, color=:blue, markersize=8)

# True optimum reference line
hlines!(ax_conv, [true_opt], color=:red, linestyle=:dash, linewidth=1.5,
        label="True optimum ($(round(true_opt, digits=1))%)")

# Initial best reference
hlines!(ax_conv, [maximum(Y_obs)], color=:gray, linestyle=:dot, linewidth=1,
        label="Initial best ($(round(maximum(Y_obs), digits=1))%)")

axislegend(ax_conv, position=:rb)

# Add efficiency annotation
n_grid_equiv = 100  # 10x10 grid
ax_text = Axis(fig6[2, 1], height=60)
hidedecorations!(ax_text)
hidespines!(ax_text)
text!(ax_text, 0.5, 0.5,
      text="BO: $n_total experiments | 10x10 grid: $n_grid_equiv experiments | " *
           "Efficiency: $(round(n_grid_equiv / n_total, digits=1))x fewer experiments",
      align=(:center, :center), fontsize=13)

save(joinpath(OUTPUT_DIR, "chem_convergence.png"), fig6, px_per_unit=2)
println("Saved: chem_convergence.png")

# ============================================================================
# Section 7: ARD Lengthscale Interpretation
# ============================================================================

println("\n" * "="^60)
println("Section 7: ARD Lengthscale Interpretation")
println("="^60)

println("\nFinal separable GP lengthscales:")
println("  d[1] (Temperature) = $(round(gp_final.d[1], sigdigits=4))")
println("  d[2] (Pressure)    = $(round(gp_final.d[2], sigdigits=4))")
println("  Ratio d[2]/d[1]    = $(round(gp_final.d[2] / gp_final.d[1], sigdigits=3))")

if gp_final.d[1] < gp_final.d[2]
    println("\n  -> Smaller d[1] means shorter correlation length in Temperature direction")
    println("  -> Temperature is MORE sensitive than Pressure (yield changes faster with T)")
    println("  -> This matches the physics: thermal degradation creates sharp yield drop")
else
    println("\n  -> Pressure is more sensitive than Temperature")
end

println("\n" * "="^60)
println("All figures saved to: $OUTPUT_DIR")
println("="^60)
