# Plotting utilities for laGP using CairoMakie

using CairoMakie

"""
    _create_grid(x_range, y_range, resolution)

Create a 2D grid for plotting.

# Returns
- Matrix of grid points (n x 2), x1 vector, x2 vector
"""
function _create_grid(x_range::Tuple, y_range::Tuple, resolution::Int)
    x1 = range(x_range[1], x_range[2], length=resolution)
    x2 = range(y_range[1], y_range[2], length=resolution)

    X_grid = Matrix{Float64}(undef, resolution * resolution, 2)
    idx = 1
    for j in 1:resolution
        for i in 1:resolution
            X_grid[idx, 1] = x1[i]
            X_grid[idx, 2] = x2[j]
            idx += 1
        end
    end

    return X_grid, collect(x1), collect(x2)
end

"""
    _reshape_to_grid(vals, resolution)

Reshape a vector of values to a grid matrix for plotting.
"""
function _reshape_to_grid(vals::Vector, resolution::Int)
    return reshape(vals, resolution, resolution)
end

"""
    plot_gp_surface(gp, x_range, y_range; resolution=50, colormap=:viridis)

Create a heatmap of GP predictions over a 2D grid.

# Arguments
- `gp::GP`: Gaussian Process model
- `x_range::Tuple`: (min, max) for x-axis
- `y_range::Tuple`: (min, max) for y-axis
- `resolution::Int`: number of points per dimension
- `colormap::Symbol`: colormap for heatmap

# Returns
- Figure with heatmap and contour overlay
"""
function plot_gp_surface(gp::GP, x_range::Tuple, y_range::Tuple;
                         resolution::Int=50, colormap::Symbol=:viridis)
    X_grid, x1, x2 = _create_grid(x_range, y_range, resolution)
    pred = pred_gp(gp, X_grid; lite=true)
    Z_grid = _reshape_to_grid(pred.mean, resolution)

    fig = Figure(size=(600, 500))
    ax = Axis(fig[1, 1], xlabel="x₁", ylabel="x₂", title="GP Mean Prediction")

    hm = heatmap!(ax, x1, x2, Z_grid, colormap=colormap)
    contour!(ax, x1, x2, Z_grid, color=:white, linewidth=0.5)
    Colorbar(fig[1, 2], hm, label="Mean")

    # Add training points
    scatter!(ax, gp.X[:, 1], gp.X[:, 2], color=:red, markersize=6,
             strokewidth=1, strokecolor=:white)

    return fig
end

"""
    plot_gp_variance(gp, x_range, y_range; resolution=50, colormap=:plasma)

Create a heatmap of GP prediction variance over a 2D grid.

# Arguments
- `gp::GP`: Gaussian Process model
- `x_range::Tuple`: (min, max) for x-axis
- `y_range::Tuple`: (min, max) for y-axis
- `resolution::Int`: number of points per dimension
- `colormap::Symbol`: colormap for heatmap

# Returns
- Figure with variance heatmap
"""
function plot_gp_variance(gp::GP, x_range::Tuple, y_range::Tuple;
                          resolution::Int=50, colormap::Symbol=:plasma)
    X_grid, x1, x2 = _create_grid(x_range, y_range, resolution)
    pred = pred_gp(gp, X_grid; lite=true)
    S2_grid = _reshape_to_grid(pred.s2, resolution)

    fig = Figure(size=(600, 500))
    ax = Axis(fig[1, 1], xlabel="x₁", ylabel="x₂", title="GP Prediction Variance")

    hm = heatmap!(ax, x1, x2, S2_grid, colormap=colormap)
    contour!(ax, x1, x2, S2_grid, color=:white, linewidth=0.5)
    Colorbar(fig[1, 2], hm, label="Variance")

    # Add training points
    scatter!(ax, gp.X[:, 1], gp.X[:, 2], color=:cyan, markersize=6,
             strokewidth=1, strokecolor=:white)

    return fig
end

"""
    plot_local_design(X, Z, Xref, local_indices)

Visualize the local design selection for a reference point.

# Arguments
- `X::Matrix`: full training design (2D only)
- `Z::Vector`: training responses
- `Xref::Vector`: reference point
- `local_indices::Vector{Int}`: indices of selected local design points

# Returns
- Figure showing training data, selected subset, and reference point
"""
function plot_local_design(X::Matrix, Z::Vector, Xref::Vector, local_indices::Vector{Int})
    @assert size(X, 2) == 2 "plot_local_design only supports 2D input"
    @assert length(Xref) == 2 "Reference point must be 2D"

    fig = Figure(size=(600, 500))
    ax = Axis(fig[1, 1], xlabel="x₁", ylabel="x₂", title="Local Design Selection")

    # All training points (gray)
    scatter!(ax, X[:, 1], X[:, 2], color=:gray, markersize=4, alpha=0.5,
             label="Training data")

    # Selected local design (blue)
    X_local = X[local_indices, :]
    scatter!(ax, X_local[:, 1], X_local[:, 2], color=:blue, markersize=8,
             strokewidth=1, strokecolor=:black, label="Local design")

    # Reference point (red star)
    scatter!(ax, [Xref[1]], [Xref[2]], marker=:star5, markersize=15,
             color=:red, strokewidth=1, strokecolor=:black, label="Reference")

    axislegend(ax, position=:rt)

    return fig
end

"""
    plot_agp_predictions(X, Z, XX, result; colormap=:viridis)

Visualize aGP predictions at multiple reference points.

# Arguments
- `X::Matrix`: training design (2D only)
- `Z::Vector`: training responses
- `XX::Matrix`: test/reference points
- `result::NamedTuple`: output from agp function

# Returns
- Figure with predictions and uncertainty
"""
function plot_agp_predictions(X::Matrix, Z::Vector, XX::Matrix, result::NamedTuple;
                              colormap::Symbol=:viridis)
    @assert size(X, 2) == 2 "plot_agp_predictions only supports 2D input"

    fig = Figure(size=(900, 400))

    # Mean predictions
    ax1 = Axis(fig[1, 1], xlabel="x₁", ylabel="x₂", title="aGP Mean Predictions")
    sc1 = scatter!(ax1, XX[:, 1], XX[:, 2], color=result.mean, markersize=8,
                   colormap=colormap)
    scatter!(ax1, X[:, 1], X[:, 2], color=:gray, markersize=3, alpha=0.3)
    Colorbar(fig[1, 2], sc1, label="Mean")

    # Variance
    ax2 = Axis(fig[1, 3], xlabel="x₁", ylabel="x₂", title="aGP Prediction Variance")
    sc2 = scatter!(ax2, XX[:, 1], XX[:, 2], color=result.var, markersize=8,
                   colormap=:plasma)
    scatter!(ax2, X[:, 1], X[:, 2], color=:gray, markersize=3, alpha=0.3)
    Colorbar(fig[1, 4], sc2, label="Variance")

    return fig
end

"""
    contour_with_constraints(f, constraints, x_range, y_range;
                             resolution=100, n_levels=10)

Create a contour plot of objective function with constraint boundaries.

# Arguments
- `f::Function`: objective function (Matrix -> Vector of values)
- `constraints::Vector{Function}`: constraint functions (each: Matrix -> Vector, feasible when ≤ 0)
- `x_range::Tuple`: (min, max) for x-axis
- `y_range::Tuple`: (min, max) for y-axis
- `resolution::Int`: number of points per dimension
- `n_levels::Int`: number of contour levels for objective

# Returns
- Figure with contours and constraint boundaries
"""
function contour_with_constraints(f::Function, constraints::Vector{<:Function},
                                  x_range::Tuple, y_range::Tuple;
                                  resolution::Int=100, n_levels::Int=10)
    X_grid, x1, x2 = _create_grid(x_range, y_range, resolution)

    # Evaluate objective
    f_vals = f(X_grid)
    f_grid = _reshape_to_grid(f_vals, resolution)

    # Create mask for feasible region
    feasible = ones(Bool, size(X_grid, 1))
    for c in constraints
        c_vals = c(X_grid)
        feasible .&= (c_vals .<= 0)
    end

    # Create masked objective (NaN outside feasible)
    f_feasible = copy(f_vals)
    f_feasible[.!feasible] .= NaN
    f_feasible_grid = _reshape_to_grid(f_feasible, resolution)

    # Create masked objective for infeasible region
    f_infeasible = copy(f_vals)
    f_infeasible[feasible] .= NaN
    f_infeasible_grid = _reshape_to_grid(f_infeasible, resolution)

    fig = Figure(size=(700, 600))
    ax = Axis(fig[1, 1], xlabel="x₁", ylabel="x₂",
              title="Objective with Constraints")

    # Feasible region contours (solid green)
    contour!(ax, x1, x2, f_feasible_grid, levels=n_levels, color=:forestgreen,
             linewidth=1.5)

    # Infeasible region contours (dashed red)
    contour!(ax, x1, x2, f_infeasible_grid, levels=n_levels, color=:red,
             linewidth=1.0, linestyle=:dash)

    # Constraint boundaries (thick black)
    for c in constraints
        c_vals = c(X_grid)
        c_grid = _reshape_to_grid(c_vals, resolution)
        contour!(ax, x1, x2, c_grid, levels=[0.0], color=:black, linewidth=2)
    end

    return fig
end
