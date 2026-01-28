# Local Approximate GP functions

using LinearAlgebra: mul!, Symmetric
using LoopVectorization: @turbo

"""
    _compute_squared_distances(X, Xref)

Compute squared Euclidean distances from Xref to all rows of X.
Returns a vector of length n where n is the number of rows in X.

Uses SIMD vectorization via LoopVectorization.jl for optimal performance.
"""
function _compute_squared_distances(X::Matrix{T}, Xref::AbstractVector{T}) where {T}
    n = size(X, 1)
    m = length(Xref)
    distances = Vector{T}(undef, n)
    @turbo for i in 1:n
        dist_sq = zero(T)
        for j in 1:m
            diff = X[i, j] - Xref[j]
            dist_sq += diff * diff
        end
        distances[i] = dist_sq
    end
    return distances
end

"""
    lagp(Xref, start, endpt, X, Z; d, g, method=:alc, close=1000, verb=0)

Local Approximate GP prediction at a single reference point.

Builds a local GP by starting with nearest neighbors and sequentially
adding points that maximize the chosen acquisition function.

# Arguments
- `Xref::Vector`: single reference point (length m)
- `start::Int`: initial number of nearest neighbors
- `endpt::Int`: final local design size
- `X::Matrix`: full training design (n x m)
- `Z::Vector`: full training responses
- `d::Real`: lengthscale parameter
- `g::Real`: nugget parameter
- `method::Symbol`: acquisition method (:alc, :mspe, or :nn)
- `close::Int`: max candidates to evaluate for ALC/MSPE (default 1000, matching R's laGP)
- `verb::Int`: verbosity level

# Returns
- `NamedTuple`: (mean=..., var=..., df=..., indices=...)
"""
function lagp(Xref::Vector{T}, start::Int, endpt::Int, X::Matrix{T}, Z::Vector{T};
              d::Real, g::Real, method::Symbol=:alc, close::Int=1000, verb::Int=0) where {T}
    n = size(X, 1)
    m = length(Xref)

    @assert start >= 1 "start must be at least 1"
    @assert endpt >= start "endpt must be >= start"
    @assert endpt <= n "endpt must be <= n (number of training points)"
    @assert method in (:alc, :mspe, :nn) "method must be :alc, :mspe, or :nn"

    # Compute distances from Xref to all training points
    distances = _compute_squared_distances(X, Xref)

    # Use partialsortperm for O(n) selection of nearest neighbors + candidates
    # We need at most endpt + close points (endpt for local design, close for candidates)
    n_needed = min(endpt + close, n)
    sorted_indices = partialsortperm(distances, 1:n_needed)

    # Initialize with nearest neighbors
    local_indices = sorted_indices[1:start]
    available_indices = Set(sorted_indices[(start + 1):n_needed])

    # Pre-allocate vector for available indices
    avail_vec = Vector{Int}(undef, n - start)

    # Build local design matrices (copy needed since GP stores references)
    X_local = X[local_indices, :]
    Z_local = Z[local_indices]

    # Create local GP
    gp = new_gp(X_local, Z_local, T(d), T(g))

    # Reference point as matrix
    Xref_mat = reshape(Xref, 1, m)

    # Sequential design selection using incremental Cholesky updates
    while length(local_indices) < endpt && !isempty(available_indices)
        # Find next point to add
        local best_global_idx::Int
        if method == :nn
            # Just add next nearest neighbor
            for idx in sorted_indices
                if idx in available_indices
                    best_global_idx = idx
                    break
                end
            end
        else
            # Use acquisition function with candidate pre-filtering
            # Only evaluate nearest `close` candidates for efficiency
            n_avail = length(available_indices)
            if n_avail > close
                # Pre-filter to nearest `close` candidates
                idx = 0
                for si in sorted_indices
                    if si in available_indices
                        idx += 1
                        avail_vec[idx] = si
                        idx >= close && break
                    end
                end
                n_cand = idx
            else
                # Use all available candidates
                idx = 0
                for ai in available_indices
                    idx += 1
                    avail_vec[idx] = ai
                end
                n_cand = n_avail
            end
            avail_view = @view avail_vec[1:n_cand]
            Xcand = X[avail_view, :]

            if method == :alc
                acq_vals = alc_gp(gp, Xcand, Xref_mat)
                best_local_idx = argmax(acq_vals)
            else  # method == :mspe
                acq_vals = mspe_gp(gp, Xcand, Xref_mat)
                best_local_idx = argmin(acq_vals)
            end

            best_global_idx = avail_view[best_local_idx]
        end

        # Update tracking
        push!(local_indices, best_global_idx)
        delete!(available_indices, best_global_idx)

        # Extend GP with new point using O(n²) incremental Cholesky update
        # instead of O(n³) full rebuild
        x_new = @view X[best_global_idx, :]
        z_new = Z[best_global_idx]
        extend_gp!(gp, Vector{T}(x_new), z_new)
    end

    # Make final prediction
    pred = pred_gp(gp, Xref_mat; lite=true)

    return (mean=pred.mean[1], var=pred.s2[1], df=pred.df, indices=local_indices)
end

"""
    agp(X, Z, XX; start=6, endpt=50, close=1000, d, g, method=:alc, verb=0, parallel=true)

Approximate GP predictions at multiple reference points.

Calls lagp for each row of XX, optionally in parallel using threads.

# Arguments
- `X::Matrix`: training design (n x m)
- `Z::Vector`: training responses
- `XX::Matrix`: test/reference points (n_test x m)
- `start::Int`: initial number of nearest neighbors
- `endpt::Int`: final local design size
- `close::Int`: max candidates to evaluate for ALC/MSPE (default 1000, matching R's laGP)
- `d::Union{Real,NamedTuple}`: lengthscale parameter or (start, mle, min, max)
- `g::Union{Real,NamedTuple}`: nugget parameter or (start, mle, min, max)
- `method::Symbol`: acquisition method (:alc, :mspe, or :nn)
- `verb::Int`: verbosity level
- `parallel::Bool`: use multi-threading

# Returns
- `NamedTuple`: (mean=..., var=..., df=..., mle=...)
"""
function agp(X::Matrix{T}, Z::Vector{T}, XX::Matrix{T};
             start::Int=6, endpt::Int=50, close::Int=1000,
             d::Union{Real,NamedTuple}=0.5, g::Union{Real,NamedTuple}=1e-3,
             method::Symbol=:alc, verb::Int=0, parallel::Bool=true) where {T}

    n_test = size(XX, 1)

    # Parse d and g parameters
    if d isa Real
        d_start = T(d)
        d_mle = false
        d_min = d_max = d_start
    else
        d_start = T(d.start)
        d_mle = get(d, :mle, false)
        d_min = T(get(d, :min, d_start / 10))
        d_max = T(get(d, :max, d_start * 10))
    end

    if g isa Real
        g_start = T(g)
        g_mle = false
        g_min = g_max = g_start
    else
        g_start = T(g.start)
        g_mle = get(g, :mle, false)
        g_min = T(get(g, :min, g_start / 10))
        g_max = T(get(g, :max, g_start * 10))
    end

    # Allocate output arrays
    means = Vector{T}(undef, n_test)
    vars = Vector{T}(undef, n_test)
    dfs = Vector{Int}(undef, n_test)
    mle_d = d_mle ? Vector{T}(undef, n_test) : T[]
    mle_g = g_mle ? Vector{T}(undef, n_test) : T[]

    # Process each test point
    if parallel && Threads.nthreads() > 1
        Threads.@threads for i in 1:n_test
            means[i], vars[i], dfs[i], mle_d_i, mle_g_i = _agp_single(
                X, Z, @view(XX[i, :]), start, endpt, close,
                d_start, d_mle, d_min, d_max,
                g_start, g_mle, g_min, g_max,
                method, verb
            )
            if d_mle
                mle_d[i] = mle_d_i
            end
            if g_mle
                mle_g[i] = mle_g_i
            end
        end
    else
        for i in 1:n_test
            means[i], vars[i], dfs[i], mle_d_i, mle_g_i = _agp_single(
                X, Z, @view(XX[i, :]), start, endpt, close,
                d_start, d_mle, d_min, d_max,
                g_start, g_mle, g_min, g_max,
                method, verb
            )
            if d_mle
                mle_d[i] = mle_d_i
            end
            if g_mle
                mle_g[i] = mle_g_i
            end
        end
    end

    # Return results
    if d_mle || g_mle
        return (mean=means, var=vars, df=dfs[1],
               mle=(d=mle_d, g=mle_g))
    else
        return (mean=means, var=vars, df=dfs[1])
    end
end

"""
    _agp_single(X, Z, Xref, start, endpt, close, ...)

Internal function to process a single test point for agp.
"""
function _agp_single(X::Matrix{T}, Z::Vector{T}, Xref::AbstractVector{T},
                     start::Int, endpt::Int, close::Int,
                     d_start::T, d_mle::Bool, d_min::T, d_max::T,
                     g_start::T, g_mle::Bool, g_min::T, g_max::T,
                     method::Symbol, verb::Int) where {T}
    n = size(X, 1)
    m = length(Xref)

    # Compute distances from Xref to all training points
    distances = _compute_squared_distances(X, Xref)

    # Use partialsortperm for O(n) selection of nearest neighbors + candidates
    # We need at most endpt + close points (endpt for local design, close for candidates)
    n_needed = min(endpt + close, n)
    sorted_indices = partialsortperm(distances, 1:n_needed)

    # Initialize with nearest neighbors
    local_indices = sorted_indices[1:start]
    available_indices = Set(sorted_indices[(start + 1):n_needed])

    # Pre-allocate vector for available indices
    avail_vec = Vector{Int}(undef, n - start)

    # Build local design matrices (copy needed since GP stores references)
    X_local = X[local_indices, :]
    Z_local = Z[local_indices]

    # Create local GP
    gp = new_gp(X_local, Z_local, d_start, g_start)

    # Reference point as matrix (collect needed for SubArray views)
    Xref_mat = Matrix{T}(undef, 1, m)
    @inbounds for j in 1:m
        Xref_mat[1, j] = Xref[j]
    end

    # Sequential design selection using incremental Cholesky updates
    while length(local_indices) < endpt && !isempty(available_indices)
        # Find next point to add
        local best_global_idx::Int
        if method == :nn
            for idx in sorted_indices
                if idx in available_indices
                    best_global_idx = idx
                    break
                end
            end
        else
            # Use acquisition function with candidate pre-filtering
            # Only evaluate nearest `close` candidates for efficiency
            n_avail = length(available_indices)
            if n_avail > close
                # Pre-filter to nearest `close` candidates
                idx = 0
                for si in sorted_indices
                    if si in available_indices
                        idx += 1
                        avail_vec[idx] = si
                        idx >= close && break
                    end
                end
                n_cand = idx
            else
                # Use all available candidates
                idx = 0
                for ai in available_indices
                    idx += 1
                    avail_vec[idx] = ai
                end
                n_cand = n_avail
            end
            avail_view = @view avail_vec[1:n_cand]
            Xcand = X[avail_view, :]

            if method == :alc
                acq_vals = alc_gp(gp, Xcand, Xref_mat)
                best_local_idx = argmax(acq_vals)
            else
                acq_vals = mspe_gp(gp, Xcand, Xref_mat)
                best_local_idx = argmin(acq_vals)
            end

            best_global_idx = avail_view[best_local_idx]
        end

        # Update tracking
        push!(local_indices, best_global_idx)
        delete!(available_indices, best_global_idx)

        # Extend GP with new point using O(n²) incremental Cholesky update
        # instead of O(n³) full rebuild
        x_new = @view X[best_global_idx, :]
        z_new = Z[best_global_idx]
        extend_gp!(gp, Vector{T}(x_new), z_new)
    end

    # Perform MLE if requested
    final_d = gp.d
    final_g = gp.g
    if d_mle || g_mle
        if d_mle && g_mle
            jmle_gp(gp; drange=(d_min, d_max), grange=(g_min, g_max))
        elseif d_mle
            mle_gp(gp, :d; tmin=d_min, tmax=d_max)
        else
            mle_gp(gp, :g; tmin=g_min, tmax=g_max)
        end
        final_d = gp.d
        final_g = gp.g
    end

    # Make final prediction
    pred = pred_gp(gp, Xref_mat; lite=true)

    return pred.mean[1], pred.s2[1], pred.df, final_d, final_g
end
