# MLE functions for GP hyperparameter optimization

using Optim: Brent, LBFGS, Fminbox, optimize, minimizer, converged, only_fg!
using SpecialFunctions: loggamma
using LoopVectorization: @turbo

# ============================================================================
# SIMD-Optimized Distance Computation
# ============================================================================

"""
    _pairwise_squared_distance!(distances, X, i, j)

Compute squared Euclidean distance between rows i and j of X.
Uses SIMD vectorization for optimal performance.
"""
@inline function _squared_distance_rows(X::Matrix{T}, i::Int, j::Int) where {T}
    m = size(X, 2)
    dist_sq = zero(T)
    @turbo for k in 1:m
        diff = X[i, k] - X[j, k]
        dist_sq += diff * diff
    end
    return dist_sq
end

"""
    _compute_all_pairwise_distances!(distances, X)

Compute all pairwise squared Euclidean distances for upper triangular pairs.
Stores results in pre-allocated `distances` vector of length n*(n-1)/2.
Uses SIMD vectorization for optimal performance.
"""
function _compute_all_pairwise_distances!(distances::Vector{T}, X::Matrix{T}) where {T}
    n = size(X, 1)
    m = size(X, 2)
    idx = 0
    @inbounds for i in 1:n
        for j in (i + 1):n
            dist_sq = zero(T)
            @turbo for k in 1:m
                diff = X[i, k] - X[j, k]
                dist_sq += diff * diff
            end
            idx += 1
            distances[idx] = dist_sq
        end
    end
    return distances
end

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

# ============================================================================
# Isotropic GP MLE functions
# ============================================================================

"""
    mle_gp!(gp, param; tmax, tmin=sqrt(eps(T)))

Optimize a single GP hyperparameter via maximum likelihood.

# Arguments
- `gp::GP`: Gaussian Process model (modified in-place)
- `param::Symbol`: parameter to optimize (:d or :g)
- `tmax::Real`: maximum value for parameter (required)
- `tmin::Real`: minimum value for parameter (default: sqrt(eps(T)), matching R's behavior)

# Returns
- `NamedTuple`: (d=..., g=..., its=..., msg=...) optimization result
"""
function mle_gp!(gp::GP{T}, param::Symbol; tmax::Real, tmin::Real=sqrt(eps(T))) where {T}
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
        update_gp!(gp; d=opt_val)
    else
        update_gp!(gp; g=opt_val)
    end

    return (d=gp.d, g=gp.g, its=its, msg="converged")
end

"""
    jmle_gp!(gp; drange, grange, maxit=100, verb=0, dab=(3/2, nothing), gab=(3/2, nothing))

Joint MLE optimization of d and g for GP.

# Arguments
- `gp::GP`: Gaussian Process model (modified in-place)
- `drange::Tuple`: (min, max) range for d
- `grange::Tuple`: (min, max) range for g
- `maxit::Int`: maximum iterations
- `verb::Int`: verbosity level
- `dab::Tuple`: (shape, scale) for d prior
- `gab::Tuple`: (shape, scale) for g prior

# Returns
- `NamedTuple`: (d=..., g=..., tot_its=..., msg=...)
"""
function jmle_gp!(gp::GP{T}; drange::Tuple{Real,Real}, grange::Tuple{Real,Real},
                  maxit::Int=100, verb::Int=0,
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

        update_gp!(gp; d=d_new, g=g_new)

        if G !== nothing
            # Use manual gradients
            grad = dllik_gp(gp)
            G[1] = -grad.dlld * d_new
            G[2] = -grad.dllg * g_new

            # Add prior gradients
            G[1] += -dlog_invgamma(d_new, d_shape, d_scale) * d_new
            G[2] += -dlog_invgamma(g_new, g_shape, g_scale) * g_new
        end

        if F !== nothing
            nll = -llik_gp(gp)
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

    # Pre-allocate array for pairwise squared Euclidean distances
    n_pairs = div(n * (n - 1), 2)
    distances = Vector{T}(undef, n_pairs)

    # Use SIMD-optimized pairwise distance computation
    _compute_all_pairwise_distances!(distances, X)

    # Filter non-zero distances
    n_nonzero = 0
    @inbounds for i in 1:n_pairs
        if distances[i] > 0
            n_nonzero += 1
            distances[n_nonzero] = distances[i]
        end
    end

    # Resize to only include non-zero distances and sort
    resize!(distances, n_nonzero)
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
    mle_gp_sep!(gp, param, dim; tmax, tmin=sqrt(eps(T)))

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
function mle_gp_sep!(gp::GPsep{T}, param::Symbol, dim::Int=1;
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
    jmle_gp_sep!(gp; drange, grange, maxit=100, verb=0, dab=(3/2, nothing), gab=(3/2, nothing))

Joint MLE optimization of lengthscales and nugget for GPsep.

# Arguments
- `gp::GPsep`: Separable Gaussian Process model (modified in-place)
- `drange::Union{Tuple,Vector}`: range for d parameters
- `grange::Tuple`: (min, max) range for g
- `maxit::Int`: maximum iterations
- `verb::Int`: verbosity level
- `dab::Tuple`: (shape, scale) for d prior
- `gab::Tuple`: (shape, scale) for g prior

# Returns
- `NamedTuple`: (d=..., g=..., tot_its=..., msg=...)
"""
function jmle_gp_sep!(gp::GPsep{T}; drange::Union{Tuple{Real,Real},Vector{<:Tuple{Real,Real}}},
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

        update_gp_sep!(gp; d=d_new, g=g_new)

        if G !== nothing
            # Use manual gradients
            grad = dllik_gp_sep(gp)
            G[1:m] .= -grad.dlld .* d_new
            G[m+1] = -grad.dllg * g_new

            # Add prior gradients
            for k in 1:m
                G[k] += -dlog_invgamma(d_new[k], d_shape, d_scales[k]) * d_new[k]
            end
            G[m+1] += -dlog_invgamma(g_new, g_shape, g_scale) * g_new
        end

        if F !== nothing
            nll = -llik_gp_sep(gp)
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

    # Pre-allocate array for TOTAL pairwise squared Euclidean distances
    n_pairs = div(n * (n - 1), 2)
    distances = Vector{T}(undef, n_pairs)

    # Use SIMD-optimized pairwise distance computation
    _compute_all_pairwise_distances!(distances, X)

    # Filter non-zero distances
    n_nonzero = 0
    @inbounds for i in 1:n_pairs
        if distances[i] > 0
            n_nonzero += 1
            distances[n_nonzero] = distances[i]
        end
    end

    # Resize to only include non-zero distances and sort
    resize!(distances, n_nonzero)
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
# Alternating MLE Optimization (R-style Newton + L-BFGS)
# ============================================================================

"""
    d2log_invgamma(x, shape, scale)

Second derivative of log-IG with respect to x:
d²/dx² log_invgamma = (shape + 1)/x² - 2*scale/x³
"""
function d2log_invgamma(x::T, shape::T, scale::T) where {T}
    return (shape + 1) / (x * x) - 2 * scale / (x * x * x)
end

"""
    newton_gp_d(gp; drange, ab=nothing, maxit=100, tol=sqrt(eps(T)))

Newton's method for optimizing lengthscale d in isotropic GP.
Falls back to Brent if Newton fails (wrong direction or bounds issues).

# Arguments
- `gp::GP`: Gaussian Process model (modified in-place)
- `drange::Tuple`: (min, max) range for d
- `ab::Union{Nothing,Tuple}`: (shape, scale) for Inverse-Gamma prior
- `maxit::Int`: maximum Newton iterations
- `tol::Real`: relative convergence tolerance for parameter change

# Returns
- `NamedTuple`: (val=..., its=..., method=...)
"""
function newton_gp_d(gp::GP{T}; drange::Tuple{Real,Real},
                      ab::Union{Nothing,Tuple{Real,Real}}=nothing,
                      maxit::Int=100, tol::T=sqrt(eps(T))) where {T}
    dmin, dmax = T(drange[1]), T(drange[2])
    th = gp.d
    its = 0

    # Prior parameters
    has_prior = ab !== nothing
    d_shape = has_prior ? T(ab[1]) : zero(T)
    d_scale = has_prior ? T(ab[2]) : zero(T)

    for i in 1:maxit
        # Get first and second derivatives
        grad = dllik_gp(gp; dg=false, dd=true)
        d2 = d2llik_gp(gp; d2g=false, d2d=true)

        dllik = grad.dlld
        d2llik = d2.d2lld

        # Add prior contributions
        if has_prior
            dllik += dlog_invgamma(th, d_shape, d_scale)
            d2llik += d2log_invgamma(th, d_shape, d_scale)
        end

        its += 1
        rat = dllik / d2llik

        # Check direction: for maximization, need d2llik < 0 (concave)
        # At a maximum, dllik=0 and d2llik<0
        # If d2llik > 0 (convex), we're at a minimum or saddle point
        if d2llik >= 0
            return _brent_gp_d(gp; drange=drange, ab=ab)
        end

        # Newton step with bounds checking
        tnew = th - rat
        adj = one(T)
        while (tnew <= dmin || tnew >= dmax) && adj > tol
            adj /= 2
            tnew = th - adj * rat
        end

        if tnew <= dmin || tnew >= dmax
            return _brent_gp_d(gp; drange=drange, ab=ab)
        end

        # Update GP
        update_gp!(gp; d=tnew)

        # Check convergence based on relative parameter change
        rel_change = abs(tnew - th) / max(abs(th), one(T))
        if rel_change < tol
            break
        end
        th = tnew
    end

    return (val=gp.d, its=its, method=:newton)
end

"""
    newton_gp_g(gp; grange, ab=nothing, maxit=100, tol=sqrt(eps(T)))

Newton's method for optimizing nugget g in isotropic GP.
Falls back to Brent if Newton fails.

# Arguments
- `gp::GP`: Gaussian Process model (modified in-place)
- `grange::Tuple`: (min, max) range for g
- `ab::Union{Nothing,Tuple}`: (shape, scale) for Inverse-Gamma prior
- `maxit::Int`: maximum Newton iterations
- `tol::Real`: relative convergence tolerance for parameter change

# Returns
- `NamedTuple`: (val=..., its=..., method=...)
"""
function newton_gp_g(gp::GP{T}; grange::Tuple{Real,Real},
                      ab::Union{Nothing,Tuple{Real,Real}}=nothing,
                      maxit::Int=100, tol::T=sqrt(eps(T))) where {T}
    gmin, gmax = T(grange[1]), T(grange[2])
    th = gp.g
    its = 0

    # Prior parameters
    has_prior = ab !== nothing
    g_shape = has_prior ? T(ab[1]) : zero(T)
    g_scale = has_prior ? T(ab[2]) : zero(T)

    for i in 1:maxit
        # Get first and second derivatives
        grad = dllik_gp(gp; dg=true, dd=false)
        d2 = d2llik_gp(gp; d2g=true, d2d=false)

        dllik = grad.dllg
        d2llik = d2.d2llg

        # Add prior contributions
        if has_prior
            dllik += dlog_invgamma(th, g_shape, g_scale)
            d2llik += d2log_invgamma(th, g_shape, g_scale)
        end

        its += 1
        rat = dllik / d2llik

        # Check direction: for maximization, need d2llik < 0 (concave)
        if d2llik >= 0
            return _brent_gp_g(gp; grange=grange, ab=ab)
        end

        # Newton step with bounds checking
        tnew = th - rat
        adj = one(T)
        while (tnew <= gmin || tnew >= gmax) && adj > tol
            adj /= 2
            tnew = th - adj * rat
        end

        if tnew <= gmin || tnew >= gmax
            return _brent_gp_g(gp; grange=grange, ab=ab)
        end

        # Update GP
        update_gp!(gp; g=tnew)

        # Check convergence based on relative parameter change
        rel_change = abs(tnew - th) / max(abs(th), one(T))
        if rel_change < tol
            break
        end
        th = tnew
    end

    return (val=gp.g, its=its, method=:newton)
end

"""
    newton_gp_sep_g(gp; grange, ab=nothing, maxit=100, tol=sqrt(eps(T)))

Newton's method for optimizing nugget g in separable GP.
Falls back to Brent if Newton fails.

# Arguments
- `gp::GPsep`: Separable Gaussian Process model (modified in-place)
- `grange::Tuple`: (min, max) range for g
- `ab::Union{Nothing,Tuple}`: (shape, scale) for Inverse-Gamma prior
- `maxit::Int`: maximum Newton iterations
- `tol::Real`: relative convergence tolerance for parameter change

# Returns
- `NamedTuple`: (val=..., its=..., method=...)
"""
function newton_gp_sep_g(gp::GPsep{T}; grange::Tuple{Real,Real},
                          ab::Union{Nothing,Tuple{Real,Real}}=nothing,
                          maxit::Int=100, tol::T=sqrt(eps(T))) where {T}
    gmin, gmax = T(grange[1]), T(grange[2])
    th = gp.g
    its = 0

    # Prior parameters
    has_prior = ab !== nothing
    g_shape = has_prior ? T(ab[1]) : zero(T)
    g_scale = has_prior ? T(ab[2]) : zero(T)

    for i in 1:maxit
        # Get first and second derivatives
        grad = dllik_gp_sep(gp; dg=true, dd=false)
        d2llik = d2llik_gp_sep_nug(gp)

        dllik = grad.dllg

        # Add prior contributions
        if has_prior
            dllik += dlog_invgamma(th, g_shape, g_scale)
            d2llik += d2log_invgamma(th, g_shape, g_scale)
        end

        its += 1
        rat = dllik / d2llik

        # Check direction: for maximization, need d2llik < 0 (concave)
        if d2llik >= 0
            return _brent_gp_sep_g(gp; grange=grange, ab=ab)
        end

        # Newton step with bounds checking
        tnew = th - rat
        adj = one(T)
        while (tnew <= gmin || tnew >= gmax) && adj > tol
            adj /= 2
            tnew = th - adj * rat
        end

        if tnew <= gmin || tnew >= gmax
            return _brent_gp_sep_g(gp; grange=grange, ab=ab)
        end

        # Update GP
        update_gp_sep!(gp; g=tnew)

        # Check convergence based on relative parameter change
        rel_change = abs(tnew - th) / max(abs(th), one(T))
        if rel_change < tol
            break
        end
        th = tnew
    end

    return (val=gp.g, its=its, method=:newton)
end

# ============================================================================
# Brent Fallback Functions
# ============================================================================

"""
    _brent_gp_d(gp; drange, ab=nothing)

Brent's method fallback for d optimization in isotropic GP.
"""
function _brent_gp_d(gp::GP{T}; drange::Tuple{Real,Real},
                      ab::Union{Nothing,Tuple{Real,Real}}=nothing) where {T}
    dmin, dmax = T(drange[1]), T(drange[2])

    has_prior = ab !== nothing
    d_shape = has_prior ? T(ab[1]) : zero(T)
    d_scale = has_prior ? T(ab[2]) : zero(T)

    function neg_posterior(x)
        update_gp!(gp; d=x)
        nll = -llik_gp(gp)
        if has_prior
            nll += -log_invgamma(x, d_shape, d_scale)
        end
        return nll
    end

    result = optimize(neg_posterior, dmin, dmax, Brent())
    opt_val = Optim.minimizer(result)
    update_gp!(gp; d=opt_val)

    return (val=gp.d, its=result.iterations, method=:brent)
end

"""
    _brent_gp_g(gp; grange, ab=nothing)

Brent's method fallback for g optimization in isotropic GP.
"""
function _brent_gp_g(gp::GP{T}; grange::Tuple{Real,Real},
                      ab::Union{Nothing,Tuple{Real,Real}}=nothing) where {T}
    gmin, gmax = T(grange[1]), T(grange[2])

    has_prior = ab !== nothing
    g_shape = has_prior ? T(ab[1]) : zero(T)
    g_scale = has_prior ? T(ab[2]) : zero(T)

    function neg_posterior(x)
        update_gp!(gp; g=x)
        nll = -llik_gp(gp)
        if has_prior
            nll += -log_invgamma(x, g_shape, g_scale)
        end
        return nll
    end

    result = optimize(neg_posterior, gmin, gmax, Brent())
    opt_val = Optim.minimizer(result)
    update_gp!(gp; g=opt_val)

    return (val=gp.g, its=result.iterations, method=:brent)
end

"""
    _brent_gp_sep_g(gp; grange, ab=nothing)

Brent's method fallback for g optimization in separable GP.
"""
function _brent_gp_sep_g(gp::GPsep{T}; grange::Tuple{Real,Real},
                          ab::Union{Nothing,Tuple{Real,Real}}=nothing) where {T}
    gmin, gmax = T(grange[1]), T(grange[2])

    has_prior = ab !== nothing
    g_shape = has_prior ? T(ab[1]) : zero(T)
    g_scale = has_prior ? T(ab[2]) : zero(T)

    function neg_posterior(x)
        update_gp_sep!(gp; g=x)
        nll = -llik_gp_sep(gp)
        if has_prior
            nll += -log_invgamma(x, g_shape, g_scale)
        end
        return nll
    end

    result = optimize(neg_posterior, gmin, gmax, Brent())
    opt_val = Optim.minimizer(result)
    update_gp_sep!(gp; g=opt_val)

    return (val=gp.g, its=result.iterations, method=:brent)
end

# ============================================================================
# L-BFGS for d-only Optimization (Separable GP)
# ============================================================================

"""
    _lbfgs_d_only(gp; drange, maxit=100, dab=nothing)

L-BFGS optimization for lengthscales only (keeping g fixed) in separable GP.

# Arguments
- `gp::GPsep`: Separable GP model (modified in-place)
- `drange::Union{Tuple,Vector}`: range for d parameters
- `maxit::Int`: maximum iterations
- `dab::Union{Nothing,Tuple}`: (shape, scale) for d prior

# Returns
- `NamedTuple`: (d=..., its=..., conv=...)
"""
function _lbfgs_d_only(gp::GPsep{T}; drange::Union{Tuple{Real,Real},Vector{<:Tuple{Real,Real}}},
                        maxit::Int=100,
                        dab::Union{Nothing,Tuple{Real,Union{Real,Nothing}}}=nothing) where {T}
    m = length(gp.d)

    # Convert drange to per-dimension ranges
    if drange isa Tuple
        d_ranges = [drange for _ in 1:m]
    else
        @assert length(drange) == m "drange must have $(m) elements"
        d_ranges = drange
    end

    # Compute prior parameters
    has_prior = dab !== nothing && dab[1] !== nothing
    d_shape = has_prior ? T(dab[1]) : zero(T)
    d_scales = if has_prior && dab[2] !== nothing
        fill(T(dab[2]), m)
    elseif has_prior
        [compute_prior_scale(T(sqrt(r[1] * r[2])), d_shape) for r in d_ranges]
    else
        zeros(T, m)
    end

    # Parameter vector in LOG SPACE
    x0 = log.(gp.d)

    lower = [log(T(r[1])) for r in d_ranges]
    upper = [log(T(r[2])) for r in d_ranges]

    # Objective: negative log-POSTERIOR (d only, g fixed)
    function neg_posterior!(F, G, x)
        d_new = exp.(x)
        update_gp_sep!(gp; d=d_new)

        if G !== nothing
            grad = dllik_gp_sep(gp; dg=false, dd=true)
            G .= -grad.dlld .* d_new  # Chain rule for log transform

            # Add prior gradients
            if has_prior
                for k in 1:m
                    G[k] += -dlog_invgamma(d_new[k], d_shape, d_scales[k]) * d_new[k]
                end
            end
        end

        if F !== nothing
            nll = -llik_gp_sep(gp)
            if has_prior
                for k in 1:m
                    nll += -log_invgamma(d_new[k], d_shape, d_scales[k])
                end
            end
            return nll
        end
    end

    result = optimize(only_fg!(neg_posterior!), lower, upper, x0,
                      Fminbox(LBFGS()),
                      Optim.Options(iterations=maxit, g_tol=T(1e-6)))

    final_x = minimizer(result)
    update_gp_sep!(gp; d=exp.(final_x))

    return (d=copy(gp.d), its=result.iterations, conv=converged(result) ? 0 : 1)
end

# ============================================================================
# Alternating MLE Functions
# ============================================================================

"""
    amle_gp!(gp; drange, grange, maxit=100, verb=0, dab=(3/2, nothing), gab=(3/2, nothing))

Alternating MLE optimization for isotropic GP (R-style jmleGP).

Alternates between Newton optimization for d and g until convergence.
This matches R's laGP algorithm where both d and g use Newton's method.

# Arguments
- `gp::GP`: Gaussian Process model (modified in-place)
- `drange::Tuple`: (min, max) range for d
- `grange::Tuple`: (min, max) range for g
- `maxit::Int`: maximum outer iterations (default: 100)
- `verb::Int`: verbosity level
- `dab::Tuple`: (shape, scale) for d prior; if scale=nothing, computed from range
- `gab::Tuple`: (shape, scale) for g prior; if scale=nothing, computed from range

# Returns
- `NamedTuple`: (d=..., g=..., dits=..., gits=..., tot_its=..., msg=...)
"""
function amle_gp!(gp::GP{T}; drange::Tuple{Real,Real}, grange::Tuple{Real,Real},
                  maxit::Int=100, verb::Int=0,
                  dab::Tuple{Real,Union{Real,Nothing}}=(3/2, nothing),
                  gab::Tuple{Real,Union{Real,Nothing}}=(3/2, nothing)) where {T}
    # Compute prior parameters
    d_shape = T(dab[1])
    d_scale = if dab[2] === nothing
        compute_prior_scale(T(sqrt(drange[1] * drange[2])), d_shape)
    else
        T(dab[2])
    end
    d_ab = (d_shape, d_scale)

    g_shape = T(gab[1])
    g_scale = if gab[2] === nothing
        compute_prior_scale(T(sqrt(grange[1] * grange[2])), g_shape)
    else
        T(gab[2])
    end
    g_ab = (g_shape, g_scale)

    dits = 0
    gits = 0

    for outer in 1:maxit
        # Newton for d (1D)
        d_result = newton_gp_d(gp; drange=drange, ab=d_ab)
        dits += d_result.its

        # Newton for g (1D)
        g_result = newton_gp_g(gp; grange=grange, ab=g_ab)
        gits += g_result.its

        if verb > 0
            println("Outer $outer: d=$(round(gp.d, sigdigits=4)), g=$(round(gp.g, sigdigits=4)), " *
                    "d_its=$(d_result.its), g_its=$(g_result.its)")
        end

        # Convergence: both took ≤1 iteration (R's criterion)
        if d_result.its <= 1 && g_result.its <= 1
            break
        end
    end

    return (d=gp.d, g=gp.g, dits=dits, gits=gits, tot_its=dits+gits, msg="converged")
end

"""
    amle_gp_sep!(gp; drange, grange, maxit=100, verb=0, dab=(3/2, nothing), gab=(3/2, nothing))

Alternating MLE optimization for separable GP (R-style jmleGPsep).

Alternates between L-BFGS optimization for all d dimensions and Newton for g.
This matches R's laGP algorithm.

# Arguments
- `gp::GPsep`: Separable Gaussian Process model (modified in-place)
- `drange::Union{Tuple,Vector}`: range for d parameters
- `grange::Tuple`: (min, max) range for g
- `maxit::Int`: maximum outer iterations (default: 100)
- `verb::Int`: verbosity level
- `dab::Tuple`: (shape, scale) for d prior; if scale=nothing, computed from range
- `gab::Tuple`: (shape, scale) for g prior; if scale=nothing, computed from range

# Returns
- `NamedTuple`: (d=..., g=..., dits=..., gits=..., tot_its=..., conv=..., msg=...)
"""
function amle_gp_sep!(gp::GPsep{T}; drange::Union{Tuple{Real,Real},Vector{<:Tuple{Real,Real}}},
                      grange::Tuple{Real,Real}, maxit::Int=100, verb::Int=0,
                      dab::Tuple{Real,Union{Real,Nothing}}=(3/2, nothing),
                      gab::Tuple{Real,Union{Real,Nothing}}=(3/2, nothing)) where {T}
    m = length(gp.d)
    tol = sqrt(eps(T))

    # Convert drange to per-dimension ranges
    if drange isa Tuple
        d_ranges = [drange for _ in 1:m]
    else
        @assert length(drange) == m "drange must have $(m) elements"
        d_ranges = drange
    end

    # Compute prior parameters for g
    g_shape = T(gab[1])
    g_scale = if gab[2] === nothing
        compute_prior_scale(T(sqrt(grange[1] * grange[2])), g_shape)
    else
        T(gab[2])
    end
    g_ab = (g_shape, g_scale)

    dits = 0
    gits = 0
    dconv = 0

    # Track previous values for convergence check
    d_prev = copy(gp.d)
    g_prev = gp.g

    for outer in 1:maxit
        # L-BFGS for ALL d dimensions (d only, g fixed)
        # Use limited iterations per outer loop to alternate more frequently
        d_result = _lbfgs_d_only(gp; drange=d_ranges, maxit=50, dab=dab)
        dits += d_result.its
        dconv = d_result.conv

        # Newton for g (1D)
        g_result = newton_gp_sep_g(gp; grange=grange, ab=g_ab)
        gits += g_result.its

        if verb > 0
            println("Outer $outer: g=$(round(gp.g, sigdigits=4)), " *
                    "d_its=$(d_result.its), g_its=$(g_result.its), d_conv=$(d_result.conv)")
        end

        # Check convergence based on relative parameter change
        d_change = maximum(abs.(gp.d .- d_prev) ./ max.(abs.(d_prev), one(T)))
        g_change = abs(gp.g - g_prev) / max(abs(g_prev), one(T))

        if d_change < tol && g_change < tol
            break
        end

        d_prev .= gp.d
        g_prev = gp.g
    end

    return (d=copy(gp.d), g=gp.g, dits=dits, gits=gits, tot_its=dits+gits,
            conv=dconv, msg=dconv == 0 ? "converged" : "max iterations reached")
end
