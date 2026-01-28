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

    # Build local design matrices (copy needed since GP stores references)
    X_local = X[local_indices, :]
    Z_local = Z[local_indices]

    # Create local GP
    gp = new_gp(X_local, Z_local, T(d), T(g))

    # Reference point as matrix (avoid reshaped views in hot loops)
    Xref_mat = Matrix{T}(undef, 1, m)
    @inbounds for j in 1:m
        Xref_mat[1, j] = Xref[j]
    end

    if method == :nn
        # Pure nearest-neighbor expansion (distance order)
        for pos in (start + 1):endpt
            best_global_idx = sorted_indices[pos]
            push!(local_indices, best_global_idx)
            x_new = @view X[best_global_idx, :]
            z_new = Z[best_global_idx]
            extend_gp!(gp, Vector{T}(x_new), z_new)
        end
    else
        # Candidate pool and availability mask
        cand_pool = @view sorted_indices[(start + 1):n_needed]
        available = falses(n)
        for idx in cand_pool
            available[idx] = true
        end
        n_avail = length(cand_pool)

        max_cand = min(close, n_avail)
        cand_buf = Vector{Int}(undef, max_cand)
        acq_buf = Vector{T}(undef, max_cand)

        # Sequential design selection using incremental Cholesky updates
        while length(local_indices) < endpt && n_avail > 0
            max_take = min(close, n_avail)
            n_cand = 0
            for idx in cand_pool
                if available[idx]
                    n_cand += 1
                    cand_buf[n_cand] = idx
                    n_cand == max_take && break
                end
            end
            cand_view = @view cand_buf[1:n_cand]

            if method == :alc
                _alc_gp_idx!(acq_buf, gp, X, cand_view, Xref_mat)
                best_local_idx = argmax(@view acq_buf[1:n_cand])
            else  # method == :mspe
                _mspe_gp_idx!(acq_buf, gp, X, cand_view, Xref_mat)
                best_local_idx = argmin(@view acq_buf[1:n_cand])
            end

            best_global_idx = cand_view[best_local_idx]
            push!(local_indices, best_global_idx)
            available[best_global_idx] = false
            n_avail -= 1

            # Extend GP with new point using O(n²) incremental Cholesky update
            x_new = @view X[best_global_idx, :]
            z_new = Z[best_global_idx]
            extend_gp!(gp, Vector{T}(x_new), z_new)
        end
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

    if method == :nn
        # Pure nearest-neighbor expansion (distance order)
        for pos in (start + 1):endpt
            best_global_idx = sorted_indices[pos]
            push!(local_indices, best_global_idx)
            x_new = @view X[best_global_idx, :]
            z_new = Z[best_global_idx]
            extend_gp!(gp, Vector{T}(x_new), z_new)
        end
    else
        # Candidate pool and availability mask
        cand_pool = @view sorted_indices[(start + 1):n_needed]
        available = falses(n)
        for idx in cand_pool
            available[idx] = true
        end
        n_avail = length(cand_pool)

        max_cand = min(close, n_avail)
        cand_buf = Vector{Int}(undef, max_cand)
        acq_buf = Vector{T}(undef, max_cand)

        # Sequential design selection using incremental Cholesky updates
        while length(local_indices) < endpt && n_avail > 0
            max_take = min(close, n_avail)
            n_cand = 0
            for idx in cand_pool
                if available[idx]
                    n_cand += 1
                    cand_buf[n_cand] = idx
                    n_cand == max_take && break
                end
            end
            cand_view = @view cand_buf[1:n_cand]

            if method == :alc
                _alc_gp_idx!(acq_buf, gp, X, cand_view, Xref_mat)
                best_local_idx = argmax(@view acq_buf[1:n_cand])
            else
                _mspe_gp_idx!(acq_buf, gp, X, cand_view, Xref_mat)
                best_local_idx = argmin(@view acq_buf[1:n_cand])
            end

            best_global_idx = cand_view[best_local_idx]
            push!(local_indices, best_global_idx)
            available[best_global_idx] = false
            n_avail -= 1

            # Extend GP with new point using O(n²) incremental Cholesky update
            x_new = @view X[best_global_idx, :]
            z_new = Z[best_global_idx]
            extend_gp!(gp, Vector{T}(x_new), z_new)
        end
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
