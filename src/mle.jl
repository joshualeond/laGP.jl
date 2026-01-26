# MLE functions for GP hyperparameter optimization

using Optim: Brent, LBFGS, Fminbox, optimize, minimizer, converged, only_fg!
using SpecialFunctions: loggamma

# ============================================================================
# Inverse-Gamma Prior Functions (matching R's laGP approach)
# ============================================================================

"""
    log_invgamma(x, shape, scale)

Compute log-density of Inverse-Gamma distribution.
For IG(a, b): f(x) = (b^a / Γ(a)) * x^(-a-1) * exp(-b/x)
log f(x) = a*log(b) - log(Γ(a)) - (a+1)*log(x) - b/x
"""
function log_invgamma(x::T, shape::T, scale::T) where {T}
    return shape * log(scale) - loggamma(shape) - (shape + 1) * log(x) - scale / x
end

"""
    dlog_invgamma(x, shape, scale)

Gradient of log-IG with respect to x:
d/dx log_invgamma = -(shape + 1)/x + scale/x^2
"""
function dlog_invgamma(x::T, shape::T, scale::T) where {T}
    return -(shape + 1) / x + scale / (x * x)
end

"""
    compute_prior_scale(median_val, shape)

Compute prior scale parameter to achieve specified median.
For IG(shape, scale), the median is approximately scale/(shape + 1/3).
R uses qgamma to invert; this approximation is equivalent for shape=3/2.

For IG(a, b):
- Mode = b / (a + 1)
- Mean = b / (a - 1) for a > 1
- Median ≈ b / (a + 1/3) (approximation)
"""
function compute_prior_scale(median_val::T, shape::T) where {T}
    return median_val * (shape + T(1/3))
end

"""
    compute_prior_scale_from_start(start_val, shape)

Compute prior scale parameter using the starting value from darg/garg.
This makes the prior centered on the data-adaptive starting value.
"""
function compute_prior_scale_from_start(start_val::T, shape::T) where {T}
    # Set scale so that the mode of IG is at start_val
    # Mode = scale / (shape + 1), so scale = start_val * (shape + 1)
    return start_val * (shape + one(T))
end

"""
    _quantile_type7(x_sorted, p)

Compute quantile using R's default type=7 method with linear interpolation.
Input must be pre-sorted.
"""
function _quantile_type7(x_sorted::Vector{T}, p::Real) where {T}
    n = length(x_sorted)
    if n == 1
        return x_sorted[1]
    end

    # R's type 7: h = (n - 1) * p + 1
    h = (n - 1) * p + 1
    lo = floor(Int, h)
    hi = ceil(Int, h)

    # Clamp indices
    lo = clamp(lo, 1, n)
    hi = clamp(hi, 1, n)

    if lo == hi
        return x_sorted[lo]
    else
        frac = h - lo
        return x_sorted[lo] + frac * (x_sorted[hi] - x_sorted[lo])
    end
end

"""
    mle_gp(gp, param; tmax, tmin=sqrt(eps(T)))

Optimize a single GP hyperparameter via maximum likelihood.

# Arguments
- `gp::GP`: Gaussian Process model (modified in-place)
- `param::Symbol`: parameter to optimize (:d or :g)
- `tmax::Real`: maximum value for parameter (required)
- `tmin::Real`: minimum value for parameter (default: sqrt(eps(T)), matching R's behavior)

# Returns
- `NamedTuple`: (d=..., g=..., its=..., msg=...) optimization result
"""
function mle_gp(gp::GP{T}, param::Symbol; tmax::Real, tmin::Real=sqrt(eps(T))) where {T}
    @assert param in (:d, :g) "param must be :d or :g"
    @assert tmin < tmax "tmin must be less than tmax"

    tmin_T = T(tmin)
    tmax_T = T(tmax)

    # Objective function: negative log-likelihood
    function neg_llik(x)
        if param == :d
            update_gp!(gp; d=x)
        else
            update_gp!(gp; g=x)
        end
        return -llik_gp(gp)
    end

    # Grid search to find approximate minimum region (handles multimodal likelihood)
    # Use log-spaced grid for better coverage
    n_grid = 20
    log_tmin = log(tmin_T)
    log_tmax = log(tmax_T)
    grid_vals = [exp(log_tmin + (log_tmax - log_tmin) * i / (n_grid - 1)) for i in 0:(n_grid - 1)]

    best_val = grid_vals[1]
    best_neg_llik = neg_llik(best_val)
    for val in grid_vals[2:end]
        nll = neg_llik(val)
        if nll < best_neg_llik
            best_neg_llik = nll
            best_val = val
        end
    end

    # Find the range around the best grid point for Brent refinement
    best_idx = findfirst(==(best_val), grid_vals)
    search_min = best_idx > 1 ? grid_vals[best_idx - 1] : tmin_T
    search_max = best_idx < n_grid ? grid_vals[best_idx + 1] : tmax_T

    # Use Brent's method to refine in the local region
    result = optimize(neg_llik, search_min, search_max, Brent())

    # Get optimal value
    opt_val = Optim.minimizer(result)
    its = n_grid + result.iterations

    # Ensure GP is updated with optimal value
    if param == :d
        update_gp!(gp; d=opt_val)
        return (d=gp.d, g=gp.g, its=its, msg="converged")
    else
        update_gp!(gp; g=opt_val)
        return (d=gp.d, g=gp.g, its=its, msg="converged")
    end
end

"""
    jmle_gp(gp; drange, grange, maxit=100, verb=0, dab=(3/2, nothing), gab=(3/2, nothing))

Joint MLE optimization of d and g using L-BFGS-B with analytical gradients
and Inverse-Gamma priors (MAP estimation).

# Arguments
- `gp::GP`: Gaussian Process model (modified in-place)
- `drange::Tuple`: (min, max) range for d
- `grange::Tuple`: (min, max) range for g
- `maxit::Int`: maximum iterations
- `verb::Int`: verbosity level
- `dab::Tuple`: (shape, scale) for d prior; if scale=nothing, computed from range
- `gab::Tuple`: (shape, scale) for g prior; if scale=nothing, computed from range

# Returns
- `NamedTuple`: (d=..., g=..., tot_its=..., msg=...)
"""
function jmle_gp(gp::GP{T}; drange::Tuple{Real,Real}, grange::Tuple{Real,Real},
                 maxit::Int=100, verb::Int=0,
                 dab::Tuple{Real,Union{Real,Nothing}}=(3/2, nothing),
                 gab::Tuple{Real,Union{Real,Nothing}}=(3/2, nothing)) where {T}
    # Compute prior parameters
    d_shape = T(dab[1])
    d_scale = if dab[2] === nothing
        # Use geometric mean of range as prior center
        compute_prior_scale(T(sqrt(drange[1] * drange[2])), d_shape)
    else
        T(dab[2])
    end

    g_shape = T(gab[1])
    g_scale = if gab[2] === nothing
        compute_prior_scale(T(sqrt(grange[1] * grange[2])), g_shape)
    else
        T(gab[2])
    end

    # Parameter vector: [d, g] - optimize in LOG SPACE
    x0 = [log(gp.d), log(gp.g)]

    lower = [log(T(drange[1])), log(T(grange[1]))]
    upper = [log(T(drange[2])), log(T(grange[2]))]

    # Objective: negative log-POSTERIOR (likelihood + prior)
    function neg_posterior!(F, G, x)
        # Transform from log space
        d_new = exp(x[1])
        g_new = exp(x[2])

        update_gp!(gp; d=d_new, g=g_new)

        if G !== nothing
            grad = dllik_gp(gp)
            # Likelihood gradients (with chain rule for log transform)
            G[1] = -grad.dlld * d_new
            G[2] = -grad.dllg * g_new

            # Add prior gradients
            G[1] += -dlog_invgamma(d_new, d_shape, d_scale) * d_new
            G[2] += -dlog_invgamma(g_new, g_shape, g_scale) * g_new
        end

        if F !== nothing
            nll = -llik_gp(gp)
            # Add negative log-prior terms
            nll += -log_invgamma(d_new, d_shape, d_scale)
            nll += -log_invgamma(g_new, g_shape, g_scale)
            return nll
        end
    end

    # L-BFGS-B optimization with relaxed tolerance
    result = optimize(only_fg!(neg_posterior!), lower, upper, x0,
                      Fminbox(LBFGS()),
                      Optim.Options(iterations=maxit, g_tol=T(1e-6),
                                    show_trace=(verb > 0)))

    # Ensure GP is updated with final values
    final_x = minimizer(result)
    update_gp!(gp; d=exp(final_x[1]), g=exp(final_x[2]))

    return (d=gp.d, g=gp.g, tot_its=result.iterations,
           msg=converged(result) ? "converged" : "max iterations reached")
end

"""
    darg(X; d=nothing, ab=(3/2, nothing))

Compute default arguments for lengthscale parameter.

Based on pairwise distances in the design matrix X.

# Arguments
- `X::Matrix`: design matrix
- `d::Union{Nothing,Real}`: user-specified d (optional)
- `ab::Tuple`: (shape, scale) for Inverse-Gamma prior; if scale=nothing, computed from range

# Returns
- `NamedTuple`: (start=..., min=..., max=..., mle=..., ab=...)
"""
function darg(X::Matrix{T}; d::Union{Nothing,Real}=nothing,
              ab::Tuple{Real,Union{Real,Nothing}}=(3/2, nothing)) where {T}
    n = size(X, 1)

    # Compute pairwise squared Euclidean distances
    distances = T[]
    for i in 1:n
        for j in (i + 1):n
            dist_sq = zero(T)
            for k in axes(X, 2)
                diff = X[i, k] - X[j, k]
                dist_sq += diff * diff
            end
            if dist_sq > 0
                push!(distances, dist_sq)
            end
        end
    end

    sort!(distances)

    # Compute quantiles using R's type 7 method
    # start = 10th percentile
    d_start = _quantile_type7(distances, 0.1)

    # max = maximum distance
    d_max = maximum(distances)

    # min = minimum distance / 2, but clamped to sqrt(eps)
    d_min = minimum(distances) / 2
    d_min = max(d_min, sqrt(eps(T)))

    return (start=d_start, min=d_min, max=d_max, mle=true, ab=ab)
end

"""
    garg(Z; g=nothing, ab=(3/2, nothing))

Compute default arguments for nugget parameter.

Based on squared residuals from the mean.

# Arguments
- `Z::Vector`: response values
- `g::Union{Nothing,Real}`: user-specified g (optional)
- `ab::Tuple`: (shape, scale) for Inverse-Gamma prior; if scale=nothing, computed from range

# Returns
- `NamedTuple`: (start=..., min=..., max=..., mle=..., ab=...)
"""
function garg(Z::Vector{T}; g::Union{Nothing,Real}=nothing,
              ab::Tuple{Real,Union{Real,Nothing}}=(3/2, nothing)) where {T}
    # Compute squared residuals from mean
    z_mean = mean(Z)
    r2s = [(z - z_mean)^2 for z in Z]
    sort!(r2s)

    # start = 2.5th percentile of r2s using R's type 7 method
    g_start = _quantile_type7(r2s, 0.025)

    # max = max of r2s (for mle=TRUE)
    g_max = maximum(r2s)

    # min = sqrt(machine epsilon)
    g_min = sqrt(eps(T))

    return (start=g_start, min=g_min, max=g_max, mle=true, ab=ab)
end

# ============================================================================
# Separable GP MLE functions
# ============================================================================

"""
    mle_gp_sep(gp, param, dim; tmax, tmin=sqrt(eps(T)))

Optimize a single hyperparameter of a separable GP via maximum likelihood.

# Arguments
- `gp::GPsep`: Separable Gaussian Process model (modified in-place)
- `param::Symbol`: parameter to optimize (:d or :g)
- `dim::Int`: dimension index for :d (ignored for :g)
- `tmax::Real`: maximum value for parameter (required)
- `tmin::Real`: minimum value for parameter (default: sqrt(eps(T)), matching R's behavior)

# Returns
- `NamedTuple`: (d=..., g=..., its=..., msg=...) optimization result
"""
function mle_gp_sep(gp::GPsep{T}, param::Symbol, dim::Int=1;
                     tmax::Real, tmin::Real=sqrt(eps(T))) where {T}
    @assert param in (:d, :g) "param must be :d or :g"
    @assert tmin < tmax "tmin must be less than tmax"

    if param == :d
        @assert 1 <= dim <= length(gp.d) "dim must be between 1 and $(length(gp.d))"
    end

    tmin_T = T(tmin)
    tmax_T = T(tmax)

    # Objective function: negative log-likelihood
    function neg_llik(x)
        if param == :d
            new_d = copy(gp.d)
            new_d[dim] = x
            update_gp_sep!(gp; d=new_d)
        else
            update_gp_sep!(gp; g=x)
        end
        return -llik_gp_sep(gp)
    end

    # Grid search to find approximate minimum region
    n_grid = 20
    log_tmin = log(tmin_T)
    log_tmax = log(tmax_T)
    grid_vals = [exp(log_tmin + (log_tmax - log_tmin) * i / (n_grid - 1)) for i in 0:(n_grid - 1)]

    best_val = grid_vals[1]
    best_neg_llik = neg_llik(best_val)
    for val in grid_vals[2:end]
        nll = neg_llik(val)
        if nll < best_neg_llik
            best_neg_llik = nll
            best_val = val
        end
    end

    # Find the range around the best grid point for Brent refinement
    best_idx = findfirst(==(best_val), grid_vals)
    search_min = best_idx > 1 ? grid_vals[best_idx - 1] : tmin_T
    search_max = best_idx < n_grid ? grid_vals[best_idx + 1] : tmax_T

    # Use Brent's method to refine in the local region
    result = optimize(neg_llik, search_min, search_max, Brent())

    # Get optimal value
    opt_val = Optim.minimizer(result)
    its = n_grid + result.iterations

    # Ensure GP is updated with optimal value
    if param == :d
        new_d = copy(gp.d)
        new_d[dim] = opt_val
        update_gp_sep!(gp; d=new_d)
    else
        update_gp_sep!(gp; g=opt_val)
    end

    return (d=copy(gp.d), g=gp.g, its=its, msg="converged")
end

"""
    jmle_gp_sep(gp; drange, grange, maxit=100, verb=0, dab=(3/2, nothing), gab=(3/2, nothing))

Joint MLE optimization of all lengthscales and nugget using L-BFGS-B with
analytical gradients and Inverse-Gamma priors (MAP estimation).

# Arguments
- `gp::GPsep`: Separable Gaussian Process model (modified in-place)
- `drange::Tuple`: (min, max) range for each d[k], or Vector of tuples per dimension
- `grange::Tuple`: (min, max) range for g
- `maxit::Int`: maximum iterations
- `verb::Int`: verbosity level
- `dab::Tuple`: (shape, scale) for d prior; if scale=nothing, computed from range
- `gab::Tuple`: (shape, scale) for g prior; if scale=nothing, computed from range

# Returns
- `NamedTuple`: (d=..., g=..., tot_its=..., msg=...)
"""
function jmle_gp_sep(gp::GPsep{T}; drange::Union{Tuple{Real,Real},Vector{<:Tuple{Real,Real}}},
                      grange::Tuple{Real,Real}, maxit::Int=100, verb::Int=0,
                      dab::Tuple{Real,Union{Real,Nothing}}=(3/2, nothing),
                      gab::Tuple{Real,Union{Real,Nothing}}=(3/2, nothing)) where {T}
    m = length(gp.d)

    # Convert drange to per-dimension ranges
    if drange isa Tuple
        d_ranges = [drange for _ in 1:m]
    else
        @assert length(drange) == m "drange must have $(m) elements"
        d_ranges = drange
    end

    # Compute prior parameters
    d_shape = T(dab[1])
    d_scales = if dab[2] === nothing
        # Use geometric mean of range as prior center (like R)
        [compute_prior_scale(T(sqrt(r[1] * r[2])), d_shape) for r in d_ranges]
    else
        fill(T(dab[2]), m)
    end

    g_shape = T(gab[1])
    g_scale = if gab[2] === nothing
        compute_prior_scale(T(sqrt(grange[1] * grange[2])), g_shape)
    else
        T(gab[2])
    end

    # Parameter vector: [d[1], ..., d[m], g]
    # Optimize in LOG SPACE for better conditioning
    x0 = [log.(gp.d); log(gp.g)]

    lower = [log(T(r[1])) for r in d_ranges]
    push!(lower, log(T(grange[1])))

    upper = [log(T(r[2])) for r in d_ranges]
    push!(upper, log(T(grange[2])))

    # Objective: negative log-POSTERIOR (likelihood + prior)
    function neg_posterior!(F, G, x)
        # Transform from log space
        d_new = exp.(x[1:m])
        g_new = exp(x[m+1])

        update_gp_sep!(gp; d=d_new, g=g_new)

        if G !== nothing
            grad = dllik_gp_sep(gp)
            # Likelihood gradients (with chain rule for log transform)
            G[1:m] .= -grad.dlld .* d_new
            G[m+1] = -grad.dllg * g_new

            # Add prior gradients (also with chain rule)
            for k in 1:m
                G[k] += -dlog_invgamma(d_new[k], d_shape, d_scales[k]) * d_new[k]
            end
            G[m+1] += -dlog_invgamma(g_new, g_shape, g_scale) * g_new
        end

        if F !== nothing
            nll = -llik_gp_sep(gp)
            # Add negative log-prior terms
            for k in 1:m
                nll += -log_invgamma(d_new[k], d_shape, d_scales[k])
            end
            nll += -log_invgamma(g_new, g_shape, g_scale)
            return nll
        end
    end

    # L-BFGS-B optimization with relaxed tolerance
    result = optimize(only_fg!(neg_posterior!), lower, upper, x0,
                      Fminbox(LBFGS()),
                      Optim.Options(iterations=maxit, g_tol=T(1e-6),
                                    show_trace=(verb > 0)))

    # Ensure GP is updated with final values
    final_x = minimizer(result)
    update_gp_sep!(gp; d=exp.(final_x[1:m]), g=exp(final_x[m+1]))

    return (d=copy(gp.d), g=gp.g, tot_its=result.iterations,
           msg=converged(result) ? "converged" : "max iterations reached")
end

"""
    darg_sep(X; d=nothing, ab=(3/2, nothing))

Compute default arguments for lengthscale parameters (separable version).

Following R's laGP convention, uses the TOTAL pairwise distance to compute
initial ranges (same for all dimensions), then MLE finds per-dimension scaling.

# Arguments
- `X::Matrix`: design matrix
- `d::Union{Nothing,Vector}`: user-specified d (optional)
- `ab::Tuple`: (shape, scale) for Inverse-Gamma prior; if scale=nothing, computed from range

# Returns
- `NamedTuple`: (ranges=..., ab=...) where ranges is Vector of per-dimension NamedTuples
"""
function darg_sep(X::Matrix{T}; d::Union{Nothing,Vector{<:Real}}=nothing,
                  ab::Tuple{Real,Union{Real,Nothing}}=(3/2, nothing)) where {T}
    n, m = size(X)

    # Compute TOTAL pairwise squared Euclidean distances (like isotropic darg)
    # This gives the same starting point for all dimensions
    distances = T[]
    for i in 1:n
        for j in (i + 1):n
            dist_sq = zero(T)
            for k in 1:m
                diff = X[i, k] - X[j, k]
                dist_sq += diff * diff
            end
            if dist_sq > 0
                push!(distances, dist_sq)
            end
        end
    end

    sort!(distances)

    # Compute quantiles using R's type 7 method
    # start = 10th percentile (divided by m to get per-dimension scale)
    d_start = _quantile_type7(distances, 0.1) / m

    # max: R uses tmax=10 by default for separable; we use max distance
    # scaled appropriately for per-dimension lengthscale
    d_max = maximum(distances)

    # min = sqrt(machine epsilon), matching R's default tmin
    d_min = sqrt(eps(T))

    # Return same range for all dimensions - MLE will differentiate them
    results = Vector{NamedTuple{(:start, :min, :max, :mle),Tuple{T,T,T,Bool}}}(undef, m)
    for dim in 1:m
        results[dim] = (start=d_start, min=d_min, max=d_max, mle=true)
    end

    return (ranges=results, ab=ab)
end

# ============================================================================
# Zygote-based Automatic Differentiation Gradients
# ============================================================================

using Zygote

"""
    neg_llik_ad(params, X, Z; separable=false)

Compute negative log-likelihood using a form suitable for Zygote AD.

This function recomputes the likelihood from scratch given parameters,
making it compatible with automatic differentiation. It avoids in-place
mutations for Zygote compatibility.

# Arguments
- `params`: For isotropic: [d, g]; for separable: [d..., g]
- `X::Matrix`: design matrix
- `Z::Vector`: response values
- `separable::Bool`: if true, params contains per-dimension lengthscales

# Returns
- Negative log-likelihood value
"""
function neg_llik_ad(params::Vector{T}, X::Matrix{T}, Z::Vector{T}; separable::Bool=false) where {T}
    n = size(X, 1)
    m = size(X, 2)

    if separable
        d = params[1:m]
        g = params[m+1]
        kernel = build_kernel_separable(d)
    else
        d = params[1]
        g = params[2]
        kernel = build_kernel_isotropic(d)
    end

    # Compute covariance matrix (without mutation for Zygote compatibility)
    K_base = kernelmatrix(kernel, RowVecs(X))
    # Add nugget using broadcasting instead of in-place mutation
    K = K_base + g * I(n)

    # Cholesky factorization
    chol = cholesky(Symmetric(K))

    # Compute KiZ
    KiZ = chol \ Z

    # Compute phi = Z' * Ki * Z
    phi = dot(Z, KiZ)

    # Log determinant
    ldetK = 2 * sum(log.(diag(chol.L)))

    # Negative log-likelihood
    return T(0.5) * (n * log(T(0.5) * phi) + ldetK)
end

"""
    dllik_ad(params, X, Z; separable=false)

Compute gradient of log-likelihood using Zygote automatic differentiation.

# Arguments
- `params`: For isotropic: [d, g]; for separable: [d..., g]
- `X::Matrix`: design matrix
- `Z::Vector`: response values
- `separable::Bool`: if true, params contains per-dimension lengthscales

# Returns
- Gradient vector (same shape as params)
"""
function dllik_ad(params::Vector{T}, X::Matrix{T}, Z::Vector{T}; separable::Bool=false) where {T}
    grad = Zygote.gradient(p -> neg_llik_ad(p, X, Z; separable=separable), params)[1]
    return -grad  # Return gradient of log-likelihood (not negative)
end

# ============================================================================
# MLE Functions for GPModel (AbstractGPs-backed)
# ============================================================================

"""
    mle_gp_model(gp, param; tmax, tmin=sqrt(eps(T)))

Optimize a single GPModel hyperparameter via maximum likelihood.

# Arguments
- `gp::GPModel`: Gaussian Process model (modified in-place)
- `param::Symbol`: parameter to optimize (:d or :g)
- `tmax::Real`: maximum value for parameter (required)
- `tmin::Real`: minimum value for parameter (default: sqrt(eps(T)), matching R's behavior)

# Returns
- `NamedTuple`: (d=..., g=..., its=..., msg=...) optimization result
"""
function mle_gp_model(gp::GPModel{T}, param::Symbol; tmax::Real, tmin::Real=sqrt(eps(T))) where {T}
    @assert param in (:d, :g) "param must be :d or :g"
    @assert tmin < tmax "tmin must be less than tmax"

    tmin_T = T(tmin)
    tmax_T = T(tmax)

    # Objective function: negative log-likelihood
    function neg_llik(x)
        if param == :d
            update_gp_model!(gp; d=x)
        else
            update_gp_model!(gp; g=x)
        end
        return -llik_gp_model(gp)
    end

    # Grid search + Brent refinement
    n_grid = 20
    log_tmin = log(tmin_T)
    log_tmax = log(tmax_T)
    grid_vals = [exp(log_tmin + (log_tmax - log_tmin) * i / (n_grid - 1)) for i in 0:(n_grid - 1)]

    best_val = grid_vals[1]
    best_neg_llik = neg_llik(best_val)
    for val in grid_vals[2:end]
        nll = neg_llik(val)
        if nll < best_neg_llik
            best_neg_llik = nll
            best_val = val
        end
    end

    best_idx = findfirst(==(best_val), grid_vals)
    search_min = best_idx > 1 ? grid_vals[best_idx - 1] : tmin_T
    search_max = best_idx < n_grid ? grid_vals[best_idx + 1] : tmax_T

    result = optimize(neg_llik, search_min, search_max, Brent())

    opt_val = Optim.minimizer(result)
    its = n_grid + result.iterations

    if param == :d
        update_gp_model!(gp; d=opt_val)
    else
        update_gp_model!(gp; g=opt_val)
    end

    return (d=gp.d, g=gp.g, its=its, msg="converged")
end

"""
    jmle_gp_model(gp; drange, grange, maxit=100, verb=0, use_ad=true, dab=(3/2, nothing), gab=(3/2, nothing))

Joint MLE optimization of d and g for GPModel.

Can use either Zygote AD gradients (use_ad=true) or manual gradients (use_ad=false).

# Arguments
- `gp::GPModel`: Gaussian Process model (modified in-place)
- `drange::Tuple`: (min, max) range for d
- `grange::Tuple`: (min, max) range for g
- `maxit::Int`: maximum iterations
- `verb::Int`: verbosity level
- `use_ad::Bool`: use Zygote automatic differentiation for gradients
- `dab::Tuple`: (shape, scale) for d prior
- `gab::Tuple`: (shape, scale) for g prior

# Returns
- `NamedTuple`: (d=..., g=..., tot_its=..., msg=...)
"""
function jmle_gp_model(gp::GPModel{T}; drange::Tuple{Real,Real}, grange::Tuple{Real,Real},
                        maxit::Int=100, verb::Int=0, use_ad::Bool=true,
                        dab::Tuple{Real,Union{Real,Nothing}}=(3/2, nothing),
                        gab::Tuple{Real,Union{Real,Nothing}}=(3/2, nothing)) where {T}
    # Compute prior parameters
    d_shape = T(dab[1])
    d_scale = if dab[2] === nothing
        compute_prior_scale(T(sqrt(drange[1] * drange[2])), d_shape)
    else
        T(dab[2])
    end

    g_shape = T(gab[1])
    g_scale = if gab[2] === nothing
        compute_prior_scale(T(sqrt(grange[1] * grange[2])), g_shape)
    else
        T(gab[2])
    end

    # Parameter vector: [d, g] - optimize in LOG SPACE
    x0 = [log(gp.d), log(gp.g)]

    lower = [log(T(drange[1])), log(T(grange[1]))]
    upper = [log(T(drange[2])), log(T(grange[2]))]

    # Objective: negative log-POSTERIOR
    function neg_posterior!(F, G, x)
        d_new = exp(x[1])
        g_new = exp(x[2])

        update_gp_model!(gp; d=d_new, g=g_new)

        if G !== nothing
            if use_ad
                # Use Zygote AD
                params = T[d_new, g_new]
                ad_grad = Zygote.gradient(p -> neg_llik_ad(p, gp.X, gp.Z; separable=false), params)[1]
                G[1] = ad_grad[1] * d_new  # Chain rule for log transform
                G[2] = ad_grad[2] * g_new
            else
                # Use manual gradients
                grad = dllik_gp_model(gp)
                G[1] = -grad.dlld * d_new
                G[2] = -grad.dllg * g_new
            end

            # Add prior gradients
            G[1] += -dlog_invgamma(d_new, d_shape, d_scale) * d_new
            G[2] += -dlog_invgamma(g_new, g_shape, g_scale) * g_new
        end

        if F !== nothing
            nll = -llik_gp_model(gp)
            nll += -log_invgamma(d_new, d_shape, d_scale)
            nll += -log_invgamma(g_new, g_shape, g_scale)
            return nll
        end
    end

    result = optimize(only_fg!(neg_posterior!), lower, upper, x0,
                      Fminbox(LBFGS()),
                      Optim.Options(iterations=maxit, g_tol=T(1e-6),
                                    show_trace=(verb > 0)))

    final_x = minimizer(result)
    update_gp_model!(gp; d=exp(final_x[1]), g=exp(final_x[2]))

    return (d=gp.d, g=gp.g, tot_its=result.iterations,
           msg=converged(result) ? "converged" : "max iterations reached")
end

"""
    jmle_gp_model_sep(gp; drange, grange, maxit=100, verb=0, use_ad=true, dab=(3/2, nothing), gab=(3/2, nothing))

Joint MLE optimization of lengthscales and nugget for GPModelSep.

# Arguments
- `gp::GPModelSep`: Separable Gaussian Process model (modified in-place)
- `drange::Union{Tuple,Vector}`: range for d parameters
- `grange::Tuple`: (min, max) range for g
- `maxit::Int`: maximum iterations
- `verb::Int`: verbosity level
- `use_ad::Bool`: use Zygote automatic differentiation for gradients
- `dab::Tuple`: (shape, scale) for d prior
- `gab::Tuple`: (shape, scale) for g prior

# Returns
- `NamedTuple`: (d=..., g=..., tot_its=..., msg=...)
"""
function jmle_gp_model_sep(gp::GPModelSep{T}; drange::Union{Tuple{Real,Real},Vector{<:Tuple{Real,Real}}},
                            grange::Tuple{Real,Real}, maxit::Int=100, verb::Int=0, use_ad::Bool=true,
                            dab::Tuple{Real,Union{Real,Nothing}}=(3/2, nothing),
                            gab::Tuple{Real,Union{Real,Nothing}}=(3/2, nothing)) where {T}
    m = length(gp.d)

    # Convert drange to per-dimension ranges
    if drange isa Tuple
        d_ranges = [drange for _ in 1:m]
    else
        @assert length(drange) == m "drange must have $(m) elements"
        d_ranges = drange
    end

    # Compute prior parameters
    d_shape = T(dab[1])
    d_scales = if dab[2] === nothing
        [compute_prior_scale(T(sqrt(r[1] * r[2])), d_shape) for r in d_ranges]
    else
        fill(T(dab[2]), m)
    end

    g_shape = T(gab[1])
    g_scale = if gab[2] === nothing
        compute_prior_scale(T(sqrt(grange[1] * grange[2])), g_shape)
    else
        T(gab[2])
    end

    # Parameter vector: [d[1], ..., d[m], g] in LOG SPACE
    x0 = [log.(gp.d); log(gp.g)]

    lower = [log(T(r[1])) for r in d_ranges]
    push!(lower, log(T(grange[1])))

    upper = [log(T(r[2])) for r in d_ranges]
    push!(upper, log(T(grange[2])))

    # Objective: negative log-POSTERIOR
    function neg_posterior!(F, G, x)
        d_new = exp.(x[1:m])
        g_new = exp(x[m+1])

        update_gp_model_sep!(gp; d=d_new, g=g_new)

        if G !== nothing
            if use_ad
                # Use Zygote AD
                params = T[d_new..., g_new]
                ad_grad = Zygote.gradient(p -> neg_llik_ad(p, gp.X, gp.Z; separable=true), params)[1]
                G[1:m] .= ad_grad[1:m] .* d_new
                G[m+1] = ad_grad[m+1] * g_new
            else
                # Use manual gradients
                grad = dllik_gp_model_sep(gp)
                G[1:m] .= -grad.dlld .* d_new
                G[m+1] = -grad.dllg * g_new
            end

            # Add prior gradients
            for k in 1:m
                G[k] += -dlog_invgamma(d_new[k], d_shape, d_scales[k]) * d_new[k]
            end
            G[m+1] += -dlog_invgamma(g_new, g_shape, g_scale) * g_new
        end

        if F !== nothing
            nll = -llik_gp_model_sep(gp)
            for k in 1:m
                nll += -log_invgamma(d_new[k], d_shape, d_scales[k])
            end
            nll += -log_invgamma(g_new, g_shape, g_scale)
            return nll
        end
    end

    result = optimize(only_fg!(neg_posterior!), lower, upper, x0,
                      Fminbox(LBFGS()),
                      Optim.Options(iterations=maxit, g_tol=T(1e-6),
                                    show_trace=(verb > 0)))

    final_x = minimizer(result)
    update_gp_model_sep!(gp; d=exp.(final_x[1:m]), g=exp(final_x[m+1]))

    return (d=copy(gp.d), g=gp.g, tot_its=result.iterations,
           msg=converged(result) ? "converged" : "max iterations reached")
end
