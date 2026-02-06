# MLE functions for GP hyperparameter optimization

import Optim
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

Based on pairwise distances in the design matrix X. If `d` is provided,
it is used as the returned starting value.

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
    # start = 10th percentile unless user-specified
    if isnothing(d)
        d_start = _quantile_type7(distances, 0.1)
    else
        @assert d > 0 "d must be positive"
        d_start = T(d)
    end

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

Based on squared residuals from the mean. If `g` is provided,
it is used as the returned starting value.

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

    # start = 2.5th percentile unless user-specified
    if isnothing(g)
        g_start = _quantile_type7(r2s, 0.025)
    else
        @assert g > 0 "g must be positive"
        g_start = T(g)
    end

    # max = max of r2s
    g_max = maximum(r2s)

    # min = sqrt(machine epsilon)
    g_min = sqrt(eps(T))

    # Match laGP behavior: default is fixed nugget unless user explicitly opts-in
    return (start=g_start, min=g_min, max=g_max, mle=false, ab=ab)
end

# ============================================================================
# Separable GP MLE functions
# ============================================================================

"""
    mle_gp_sep!(gp, param, dim; tmax, tmin=sqrt(eps(T)), maxit=100, verb=0, dab=(3/2, nothing))

Optimize separable GP hyperparameters via maximum likelihood.

- If `param == :d` and `dim` is provided, optimizes a *single* lengthscale (1D grid + Brent).
- If `param == :d` and `dim` is `nothing`, optimizes *all* lengthscales jointly (L-BFGS-B).
- If `param == :g`, optimizes the nugget (1D grid + Brent).

# Arguments
- `gp::GPsep`: Separable Gaussian Process model (modified in-place)
- `param::Symbol`: parameter to optimize (:d or :g)
- `dim::Union{Int,Nothing}`: dimension index for :d (ignored for :g). If `nothing`, optimizes all d.
- `tmax`: maximum value(s) for parameter(s). Scalar or vector for `:d`.
- `tmin`: minimum value(s) for parameter(s). Scalar or vector for `:d`.
- `maxit::Int`: maximum iterations for joint L-BFGS-B (when `dim` is `nothing`)
- `verb::Int`: verbosity for joint L-BFGS-B
- `dab`: prior tuple for `d` (pass `nothing` to disable prior)

# Returns
- `NamedTuple`: (d=..., g=..., its=..., msg=...) optimization result
"""
function mle_gp_sep!(gp::GPsep{T}, param::Symbol, dim::Union{Int,Nothing}=nothing;
                     tmax, tmin=sqrt(eps(T)), maxit::Int=100, verb::Int=0,
                     dab::Union{Nothing,Tuple{Real,Union{Real,Nothing}}}=(3/2, nothing)) where {T}
    @assert param in (:d, :g) "param must be :d or :g"

    # Joint optimization over all lengthscales (R-style mleGPsep)
    if param == :d && isnothing(dim)
        m = length(gp.d)

        # Expand bounds to per-dimension vectors (R-compatible behavior)
        function _expand_bounds(val)
            if val isa AbstractVector || val isa Tuple
                v = collect(val)
                if length(v) == m
                    return v
                elseif length(v) == 1
                    return fill(v[1], m)
                elseif length(v) == 2 && m != 2
                    return fill(v[1], m)
                else
                    error("length(tmin/tmax) should be 1 or $m (or 2 when m==2)")
                end
            else
                return fill(val, m)
            end
        end

        tmin_vec = _expand_bounds(tmin)
        tmax_vec = _expand_bounds(tmax)
        @assert all(tmin_vec .< tmax_vec) "tmin must be less than tmax for all dimensions"

        d_ranges = [(tmin_vec[i], tmax_vec[i]) for i in 1:m]
        result = mle_gp_sep_d!(gp; drange=d_ranges, maxit=maxit, verb=verb, dab=dab)
        return (d=copy(gp.d), g=gp.g, its=result.tot_its, msg=result.msg)
    end

    # Single-parameter optimization (1D)
    if param == :d
        @assert dim isa Int "dim must be an Int when optimizing a single lengthscale"
        @assert 1 <= dim <= length(gp.d) "dim must be between 1 and $(length(gp.d))"
    end

    # Allow scalar or vector bounds for single-parameter case
    if param == :g
        tmin_val = (tmin isa AbstractVector || tmin isa Tuple) ? tmin[end] : tmin
        tmax_val = (tmax isa AbstractVector || tmax isa Tuple) ? tmax[end] : tmax
    else
        tmin_val = (tmin isa AbstractVector || tmin isa Tuple) ? tmin[dim] : tmin
        tmax_val = (tmax isa AbstractVector || tmax isa Tuple) ? tmax[dim] : tmax
    end
    @assert tmin_val < tmax_val "tmin must be less than tmax"

    tmin_T = T(tmin_val)
    tmax_T = T(tmax_val)

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

mutable struct _SepDMLEWorkspace{T}
    d_work::Vector{T}
    K_no_nugget::Matrix{T}
    K::Matrix{T}
    Ki::Matrix{T}
    KiZ::Vector{T}
    WK::Matrix{T}
    WK_quad::Matrix{T}
    row_sums::Vector{T}
    row_sums_quad::Vector{T}
    tmp1::Vector{T}
    tmp2::Vector{T}
end

function _init_sep_d_mle_workspace(::Type{T}, n::Int, m::Int) where {T}
    return _SepDMLEWorkspace{T}(
        Vector{T}(undef, m),
        Matrix{T}(undef, n, n),
        Matrix{T}(undef, n, n),
        Matrix{T}(undef, n, n),
        Vector{T}(undef, n),
        Matrix{T}(undef, n, n),
        Matrix{T}(undef, n, n),
        Vector{T}(undef, n),
        Vector{T}(undef, n),
        Vector{T}(undef, n),
        Vector{T}(undef, n),
    )
end

function _set_identity!(A::Matrix{T}) where {T}
    n = size(A, 1)
    @assert size(A, 2) == n "identity target must be square"
    fill!(A, zero(T))
    @inbounds for i in 1:n
        A[i, i] = one(T)
    end
    return A
end

function _eval_sep_d_neg_posterior!(
    F,
    G,
    x::AbstractVector{T},
    gp::GPsep{T},
    ws::_SepDMLEWorkspace{T};
    has_prior::Bool=false,
    d_shape::T=zero(T),
    d_scales::Union{Nothing,Vector{T}}=nothing,
) where {T}
    n = size(gp.X, 1)
    m = size(gp.X, 2)
    dn = T(n)
    compute_grad = G !== nothing

    @inbounds for k in 1:m
        ws.d_work[k] = exp(x[k])
    end

    if compute_grad
        _kernelmatrix_sep_train!(ws.K_no_nugget, gp.X, ws.d_work)
        copyto!(ws.K, ws.K_no_nugget)
    else
        _kernelmatrix_sep_train!(ws.K, gp.X, ws.d_work)
    end
    _add_nugget_diagonal!(ws.K, gp.g)

    chol = cholesky(Symmetric(ws.K))

    _set_identity!(ws.Ki)
    ldiv!(chol, ws.Ki)

    copyto!(ws.KiZ, gp.Z)
    ldiv!(chol, ws.KiZ)

    phi = dot(gp.Z, ws.KiZ)
    ldetK = _compute_logdet_chol(chol)
    nll = -_concentrated_llik(n, phi, ldetK)

    if has_prior
        @assert d_scales !== nothing
        @inbounds for k in 1:m
            nll += -log_invgamma(ws.d_work[k], d_shape, d_scales[k])
        end
    end

    if compute_grad
        @inbounds for i in 1:n
            KiZ_i = ws.KiZ[i]
            s = zero(T)
            sq = zero(T)
            for j in 1:n
                kij = ws.K_no_nugget[i, j]

                w = ws.Ki[i, j] * kij
                ws.WK[i, j] = w
                s += w

                wq = KiZ_i * ws.KiZ[j] * kij
                ws.WK_quad[i, j] = wq
                sq += wq
            end
            ws.row_sums[i] = s
            ws.row_sums_quad[i] = sq
        end

        phi_factor = T(0.5) * (dn / phi)
        @inbounds for k in 1:m
            x_k = @view gp.X[:, k]
            inv_dk_sq = one(T) / (ws.d_work[k] * ws.d_work[k])

            mul!(ws.tmp1, ws.WK, x_k)
            mul!(ws.tmp2, ws.WK_quad, x_k)

            tr_acc1 = zero(T)
            tr_acc2 = zero(T)
            q_acc1 = zero(T)
            q_acc2 = zero(T)
            for i in 1:n
                xi = x_k[i]
                xi_sq = xi * xi
                tr_acc1 += xi_sq * ws.row_sums[i]
                tr_acc2 += xi * ws.tmp1[i]
                q_acc1 += xi_sq * ws.row_sums_quad[i]
                q_acc2 += xi * ws.tmp2[i]
            end

            tr_term = 2 * (tr_acc1 - tr_acc2) * inv_dk_sq
            quad_term = 2 * (q_acc1 - q_acc2) * inv_dk_sq
            dlld_k = -T(0.5) * tr_term + phi_factor * quad_term

            G[k] = -dlld_k * ws.d_work[k]
            if has_prior
                @assert d_scales !== nothing
                G[k] += -dlog_invgamma(ws.d_work[k], d_shape, d_scales[k]) * ws.d_work[k]
            end
        end
    end

    if F !== nothing
        return nll
    end
    return nothing
end

"""
    mle_gp_sep_d!(gp; drange, maxit=100, verb=0, dab=(3/2, nothing))

Optimize separable lengthscales only (d vector) with nugget g fixed.

This mirrors laGP's `mleGPsep` behavior, but without updating g.

# Arguments
- `gp::GPsep`: Separable Gaussian Process model (modified in-place)
- `drange::Union{Tuple,Vector}`: range for d parameters
- `maxit::Int`: maximum iterations
- `verb::Int`: verbosity level
- `dab::Tuple`: (shape, scale) for d prior

# Returns
- `NamedTuple`: (d=..., tot_its=..., msg=...)
"""
function mle_gp_sep_d!(gp::GPsep{T}; drange::Union{Tuple{Real,Real},Vector{<:Tuple{Real,Real}}},
                       maxit::Int=100, verb::Int=0,
                       dab::Union{Nothing,Tuple{Real,Union{Real,Nothing}}}=(3/2, nothing)) where {T}
    m = length(gp.d)
    n = size(gp.X, 1)

    # Convert drange to per-dimension ranges
    if drange isa Tuple
        d_ranges = [drange for _ in 1:m]
    else
        @assert length(drange) == m "drange must have $(m) elements"
        d_ranges = drange
    end

    # Compute prior parameters (allow disabling priors with dab=nothing or dab[1]<=0)
    has_prior = dab !== nothing && dab[1] > 0
    d_shape = has_prior ? T(dab[1]) : zero(T)
    d_scales = if has_prior
        if dab[2] === nothing
            [compute_prior_scale(T(sqrt(r[1] * r[2])), d_shape) for r in d_ranges]
        else
            fill(T(dab[2]), m)
        end
    else
        zeros(T, m)
    end

    # Parameter vector: [d[1], ..., d[m]] in LOG SPACE
    x0 = log.(gp.d)
    lower = [log(T(r[1])) for r in d_ranges]
    upper = [log(T(r[2])) for r in d_ranges]
    workspace = _init_sep_d_mle_workspace(T, n, m)

    # Objective: negative log-POSTERIOR (with fixed g)
    function neg_posterior!(F, G, x)
        return _eval_sep_d_neg_posterior!(
            F, G, x, gp, workspace;
            has_prior=has_prior,
            d_shape=d_shape,
            d_scales=has_prior ? d_scales : nothing,
        )
    end

    result = optimize(only_fg!(neg_posterior!), lower, upper, x0,
                      Fminbox(LBFGS(linesearch=Optim.LineSearches.BackTracking())),
                      Optim.Options(iterations=maxit, g_tol=T(1e-6),
                                    show_trace=(verb > 0)))

    final_x = minimizer(result)
    update_gp_sep!(gp; d=exp.(final_x))

    return (d=copy(gp.d), tot_its=result.iterations,
            msg=converged(result) ? "converged" : "max iterations reached")
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
                      dab::Union{Nothing,Tuple{Real,Union{Real,Nothing}}}=(3/2, nothing),
                      gab::Union{Nothing,Tuple{Real,Union{Real,Nothing}}}=(3/2, nothing)) where {T}
    m = length(gp.d)

    # Convert drange to per-dimension ranges
    if drange isa Tuple
        d_ranges = [drange for _ in 1:m]
    else
        @assert length(drange) == m "drange must have $(m) elements"
        d_ranges = drange
    end

    # Compute prior parameters (allow disabling priors with dab/gab=nothing or shape<=0)
    has_d_prior = dab !== nothing && dab[1] > 0
    d_shape = has_d_prior ? T(dab[1]) : zero(T)
    d_scales = if has_d_prior
        if dab[2] === nothing
            [compute_prior_scale(T(sqrt(r[1] * r[2])), d_shape) for r in d_ranges]
        else
            fill(T(dab[2]), m)
        end
    else
        zeros(T, m)
    end

    has_g_prior = gab !== nothing && gab[1] > 0
    g_shape = has_g_prior ? T(gab[1]) : zero(T)
    g_scale = if has_g_prior
        if gab[2] === nothing
            compute_prior_scale(T(sqrt(grange[1] * grange[2])), g_shape)
        else
            T(gab[2])
        end
    else
        zero(T)
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
            if has_d_prior
                for k in 1:m
                    G[k] += -dlog_invgamma(d_new[k], d_shape, d_scales[k]) * d_new[k]
                end
            end
            if has_g_prior
                G[m+1] += -dlog_invgamma(g_new, g_shape, g_scale) * g_new
            end
        end

        if F !== nothing
            nll = -llik_gp_sep(gp)
            if has_d_prior
                for k in 1:m
                    nll += -log_invgamma(d_new[k], d_shape, d_scales[k])
                end
            end
            if has_g_prior
                nll += -log_invgamma(g_new, g_shape, g_scale)
            end
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

Mirrors laGP's `darg` behavior: uses pairwise squared distances to set
start/min/max, then applies those same ranges to each dimension unless
`d` is user-specified.

# Arguments
- `X::Matrix`: design matrix
- `d::Union{Nothing,Real,Vector}`: user-specified d start(s) (optional)
- `ab::Tuple`: (shape, scale) for Inverse-Gamma prior; if scale=nothing, computed from range

# Returns
- `NamedTuple`: (ranges=..., ab=...) where ranges is Vector of per-dimension NamedTuples
"""
function darg_sep(X::Matrix{T}; d::Union{Nothing,Real,Vector{<:Real}}=nothing,
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

    # Compute default start from quantiles (type 7)
    d_start_default = _quantile_type7(distances, 0.1)

    # max = max pairwise squared distance
    d_max = maximum(distances)

    # min = half of the minimum non-zero distance (bounded below by sqrt(eps))
    d_min = min(distances[1] / 2, d_max)
    if d_min < sqrt(eps(T))
        d_min = sqrt(eps(T))
    end

    # User-specified starts (if provided)
    starts = if isnothing(d)
        fill(d_start_default, m)
    elseif d isa Real
        @assert d > 0 "d must be positive"
        fill(T(d), m)
    else
        d_vec = T.(d)
        if length(d_vec) == 1
            @assert d_vec[1] > 0 "d must be positive"
            fill(d_vec[1], m)
        else
            @assert length(d_vec) == m "d must have length 1 or $m"
            @assert all(d_vec .> 0) "all entries in d must be positive"
            d_vec
        end
    end

    # Return same min/max range for all dimensions; starts may differ
    results = Vector{NamedTuple{(:start, :min, :max, :mle),Tuple{T,T,T,Bool}}}(undef, m)
    for dim in 1:m
        results[dim] = (start=starts[dim], min=d_min, max=d_max, mle=true)
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
    _newton_gp_param(gp; pmin, pmax, th0, ab, maxit, tol, get_derivatives, update_fn!, brent_fallback, get_val)

Generic Newton's method for 1D GP parameter optimization.

Runs Newton iterations with damped steps, bounded to [pmin, pmax].
Falls back to Brent's method if the Hessian indicates non-concavity
or the step cannot stay within bounds.

# Arguments
- `gp::AnyGP`: GP model (modified in-place via `update_fn!`)
- `pmin::T`, `pmax::T`: parameter bounds
- `th0::T`: initial parameter value
- `ab::Union{Nothing,Tuple}`: (shape, scale) for Inverse-Gamma prior
- `maxit::Int`: maximum Newton iterations
- `tol::T`: relative convergence tolerance
- `get_derivatives`: `() -> (dllik::T, d2llik::T)` returns first and second derivatives
- `update_fn!`: `tnew -> ()` updates gp with new parameter value
- `brent_fallback`: `() -> NamedTuple` Brent fallback when Newton fails
- `get_val`: `() -> T` returns current parameter value
"""
function _newton_gp_param(gp::AnyGP{T};
        pmin::T, pmax::T, th0::T,
        ab::Union{Nothing,Tuple{Real,Real}},
        maxit::Int, tol::T,
        get_derivatives,
        update_fn!,
        brent_fallback,
        get_val
    ) where {T}
    th = th0
    its = 0

    # Prior parameters
    has_prior = ab !== nothing
    p_shape = has_prior ? T(ab[1]) : zero(T)
    p_scale = has_prior ? T(ab[2]) : zero(T)

    for i in 1:maxit
        dllik, d2llik = get_derivatives()

        # Add prior contributions
        if has_prior
            dllik += dlog_invgamma(th, p_shape, p_scale)
            d2llik += d2log_invgamma(th, p_shape, p_scale)
        end

        its += 1
        rat = dllik / d2llik

        # Check direction: for maximization, need d2llik < 0 (concave)
        if d2llik >= 0
            return brent_fallback()
        end

        # Newton step with bounds checking
        tnew = th - rat
        adj = one(T)
        while (tnew <= pmin || tnew >= pmax) && adj > tol
            adj /= 2
            tnew = th - adj * rat
        end

        if tnew <= pmin || tnew >= pmax
            return brent_fallback()
        end

        # Update GP
        update_fn!(tnew)

        # Check convergence based on relative parameter change
        rel_change = abs(tnew - th) / max(abs(th), one(T))
        if rel_change < tol
            break
        end
        th = tnew
    end

    return (val=get_val(), its=its, method=:newton)
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
    return _newton_gp_param(gp;
        pmin=T(drange[1]), pmax=T(drange[2]), th0=gp.d,
        ab=ab, maxit=maxit, tol=tol,
        get_derivatives = () -> begin
            grad = dllik_gp(gp; dg=false, dd=true)
            d2 = d2llik_gp(gp; d2g=false, d2d=true)
            (grad.dlld, d2.d2lld)
        end,
        update_fn! = tnew -> update_gp!(gp; d=tnew),
        brent_fallback = () -> _brent_gp_d(gp; drange=drange, ab=ab),
        get_val = () -> gp.d)
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
    return _newton_gp_param(gp;
        pmin=T(grange[1]), pmax=T(grange[2]), th0=gp.g,
        ab=ab, maxit=maxit, tol=tol,
        get_derivatives = () -> begin
            grad = dllik_gp(gp; dg=true, dd=false)
            d2 = d2llik_gp(gp; d2g=true, d2d=false)
            (grad.dllg, d2.d2llg)
        end,
        update_fn! = tnew -> update_gp!(gp; g=tnew),
        brent_fallback = () -> _brent_gp_g(gp; grange=grange, ab=ab),
        get_val = () -> gp.g)
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
    return _newton_gp_param(gp;
        pmin=T(grange[1]), pmax=T(grange[2]), th0=gp.g,
        ab=ab, maxit=maxit, tol=tol,
        get_derivatives = () -> begin
            grad = dllik_gp_sep(gp; dg=true, dd=false)
            d2llik = d2llik_gp_sep_nug(gp)
            (grad.dllg, d2llik)
        end,
        update_fn! = tnew -> update_gp_sep!(gp; g=tnew),
        brent_fallback = () -> _brent_gp_sep_g(gp; grange=grange, ab=ab),
        get_val = () -> gp.g)
end

# ============================================================================
# Brent Fallback Functions
# ============================================================================

"""
    _brent_gp_param(gp; range, ab, update_fn!, llik_fn, get_val)

Generic Brent's method fallback for 1D GP parameter optimization.

# Arguments
- `gp::AnyGP`: GP model (modified in-place via `update_fn!`)
- `range::Tuple`: (min, max) range for the parameter
- `ab::Union{Nothing,Tuple}`: (shape, scale) for Inverse-Gamma prior
- `update_fn!`: `x -> ()` updates gp with new parameter value
- `llik_fn`: `() -> T` returns current log-likelihood
- `get_val`: `() -> T` returns current parameter value
"""
function _brent_gp_param(gp::AnyGP{T};
        range::Tuple{Real,Real},
        ab::Union{Nothing,Tuple{Real,Real}},
        update_fn!,
        llik_fn,
        get_val
    ) where {T}
    pmin, pmax = T(range[1]), T(range[2])

    has_prior = ab !== nothing
    p_shape = has_prior ? T(ab[1]) : zero(T)
    p_scale = has_prior ? T(ab[2]) : zero(T)

    function neg_posterior(x)
        update_fn!(x)
        nll = -llik_fn()
        if has_prior
            nll += -log_invgamma(x, p_shape, p_scale)
        end
        return nll
    end

    result = optimize(neg_posterior, pmin, pmax, Brent())
    update_fn!(Optim.minimizer(result))

    return (val=get_val(), its=result.iterations, method=:brent)
end

"""
    _brent_gp_d(gp; drange, ab=nothing)

Brent's method fallback for d optimization in isotropic GP.
"""
function _brent_gp_d(gp::GP{T}; drange::Tuple{Real,Real},
                      ab::Union{Nothing,Tuple{Real,Real}}=nothing) where {T}
    return _brent_gp_param(gp; range=drange, ab=ab,
        update_fn! = x -> update_gp!(gp; d=x),
        llik_fn = () -> llik_gp(gp),
        get_val = () -> gp.d)
end

"""
    _brent_gp_g(gp; grange, ab=nothing)

Brent's method fallback for g optimization in isotropic GP.
"""
function _brent_gp_g(gp::GP{T}; grange::Tuple{Real,Real},
                      ab::Union{Nothing,Tuple{Real,Real}}=nothing) where {T}
    return _brent_gp_param(gp; range=grange, ab=ab,
        update_fn! = x -> update_gp!(gp; g=x),
        llik_fn = () -> llik_gp(gp),
        get_val = () -> gp.g)
end

"""
    _brent_gp_sep_g(gp; grange, ab=nothing)

Brent's method fallback for g optimization in separable GP.
"""
function _brent_gp_sep_g(gp::GPsep{T}; grange::Tuple{Real,Real},
                          ab::Union{Nothing,Tuple{Real,Real}}=nothing) where {T}
    return _brent_gp_param(gp; range=grange, ab=ab,
        update_fn! = x -> update_gp_sep!(gp; g=x),
        llik_fn = () -> llik_gp_sep(gp),
        get_val = () -> gp.g)
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
    has_prior = dab !== nothing && dab[1] !== nothing && dab[1] > 0
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
                      dab::Union{Nothing,Tuple{Real,Union{Real,Nothing}}}=(3/2, nothing),
                      gab::Union{Nothing,Tuple{Real,Union{Real,Nothing}}}=(3/2, nothing)) where {T}
    m = length(gp.d)
    tol = sqrt(eps(T))

    # Convert drange to per-dimension ranges
    if drange isa Tuple
        d_ranges = [drange for _ in 1:m]
    else
        @assert length(drange) == m "drange must have $(m) elements"
        d_ranges = drange
    end

    # Compute prior parameters for g (allow disabling priors with gab=nothing or shape<=0)
    has_g_prior = gab !== nothing && gab[1] > 0
    g_ab = if has_g_prior
        g_shape = T(gab[1])
        g_scale = if gab[2] === nothing
            compute_prior_scale(T(sqrt(grange[1] * grange[2])), g_shape)
        else
            T(gab[2])
        end
        (g_shape, g_scale)
    else
        nothing
    end

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
