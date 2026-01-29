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
    @inbounds for i in 1:n
        dist_sq = zero(T)
        @turbo for j in 1:m
            diff = X[i, j] - Xref[j]
            dist_sq += diff * diff
        end
        distances[i] = dist_sq
    end
    return distances
end

@inline function _resolve_close(close::Int, n::Int, endpt::Int)
    close_eff = (close <= 0 || close > n) ? n : close
    if close_eff < endpt
        close_eff = endpt
    end
    return close_eff
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
- `close::Int`: size of closest candidate pool (default 1000, matching laGP)
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
    # laGP semantics: `close` is the total candidate pool size (closest points)
    close_eff = _resolve_close(close, n, endpt)
    sorted_indices = partialsortperm(distances, 1:close_eff)

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
            extend_gp!(gp, x_new, z_new)
        end
    else
        # Candidate pool (distance-ordered)
        cand_pool = collect(@view sorted_indices[(start + 1):close_eff])
        n_pool = length(cand_pool)
        acq_buf = Vector{T}(undef, n_pool)

        alc_work = ALCWorkspace{T}()

        # Sequential design selection using incremental Cholesky updates
        ncand = n_pool
        while length(local_indices) < endpt && ncand > 0
            cand_view = @view cand_pool[1:ncand]

            if method == :alc
                _alc_gp_idx!(acq_buf, gp, X, cand_view, Xref_mat, alc_work)
                best_local_idx = argmax(@view acq_buf[1:ncand])
            else  # method == :mspe
                _mspe_gp_idx!(acq_buf, gp, X, cand_view, Xref_mat, alc_work)
                best_local_idx = argmin(@view acq_buf[1:ncand])
            end

            best_global_idx = cand_pool[best_local_idx]
            push!(local_indices, best_global_idx)
            if best_local_idx != ncand
                cand_pool[best_local_idx] = cand_pool[ncand]
            end
            ncand -= 1

            # Extend GP with new point using O(n²) incremental Cholesky update
            x_new = @view X[best_global_idx, :]
            z_new = Z[best_global_idx]
            extend_gp!(gp, x_new, z_new)
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
- `close::Int`: size of closest candidate pool (default 1000, matching laGP)
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
    # laGP semantics: `close` is the total candidate pool size (closest points)
    close_eff = _resolve_close(close, n, endpt)
    sorted_indices = partialsortperm(distances, 1:close_eff)

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
            extend_gp!(gp, x_new, z_new)
        end
    else
        # Candidate pool (distance-ordered)
        cand_pool = collect(@view sorted_indices[(start + 1):close_eff])
        n_pool = length(cand_pool)
        acq_buf = Vector{T}(undef, n_pool)

        alc_work = ALCWorkspace{T}()

        # Sequential design selection using incremental Cholesky updates
        ncand = n_pool
        while length(local_indices) < endpt && ncand > 0
            cand_view = @view cand_pool[1:ncand]

            if method == :alc
                _alc_gp_idx!(acq_buf, gp, X, cand_view, Xref_mat, alc_work)
                best_local_idx = argmax(@view acq_buf[1:ncand])
            else
                _mspe_gp_idx!(acq_buf, gp, X, cand_view, Xref_mat, alc_work)
                best_local_idx = argmin(@view acq_buf[1:ncand])
            end

            best_global_idx = cand_pool[best_local_idx]
            push!(local_indices, best_global_idx)
            if best_local_idx != ncand
                cand_pool[best_local_idx] = cand_pool[ncand]
            end
            ncand -= 1

            # Extend GP with new point using O(n²) incremental Cholesky update
            x_new = @view X[best_global_idx, :]
            z_new = Z[best_global_idx]
            extend_gp!(gp, x_new, z_new)
        end
    end

    # Perform MLE if requested
    final_d = gp.d
    final_g = gp.g
    if d_mle || g_mle
        if d_mle && g_mle
            jmle_gp!(gp; drange=(d_min, d_max), grange=(g_min, g_max))
        elseif d_mle
            mle_gp!(gp, :d; tmin=d_min, tmax=d_max)
        else
            mle_gp!(gp, :g; tmin=g_min, tmax=g_max)
        end
        final_d = gp.d
        final_g = gp.g
    end

    # Make final prediction
    pred = pred_gp(gp, Xref_mat; lite=true)

    return pred.mean[1], pred.s2[1], pred.df, final_d, final_g
end

# ============================================================================
# Separable Local GP Functions
# ============================================================================

"""
    lagp_sep(Xref, start, endpt, X, Z; d, g, method=:alc, close=1000, verb=0)

Local Approximate GP prediction at a single reference point using separable GP.

Builds a local GP with per-dimension lengthscales by starting with nearest neighbors
and sequentially adding points that maximize the chosen acquisition function.

# Arguments
- `Xref::Vector`: single reference point (length m)
- `start::Int`: initial number of nearest neighbors
- `endpt::Int`: final local design size
- `X::Matrix`: full training design (n x m)
- `Z::Vector`: full training responses
- `d::Union{Real,Vector{<:Real}}`: per-dimension lengthscale parameters (scalar replicated)
- `g::Real`: nugget parameter
- `method::Symbol`: acquisition method (:alc or :nn). Note: :mspe not supported for separable
- `close::Int`: size of closest candidate pool (default 1000)
- `verb::Int`: verbosity level

# Returns
- `NamedTuple`: (mean=..., var=..., df=..., indices=...)
"""
function lagp_sep(Xref::Vector{T}, start::Int, endpt::Int, X::Matrix{T}, Z::Vector{T};
                  d::Union{Real,Vector{<:Real}}, g::Real, method::Symbol=:alc, close::Int=1000, verb::Int=0) where {T}
    n = size(X, 1)
    m = length(Xref)

    @assert start >= 1 "start must be at least 1"
    @assert endpt >= start "endpt must be >= start"
    @assert endpt <= n "endpt must be <= n (number of training points)"
    @assert method in (:alc, :nn) "method must be :alc or :nn (MSPE not supported for separable)"
    d_vec = d isa Real ? fill(T(d), m) : T.(d)
    if length(d_vec) == 1
        d_vec = fill(d_vec[1], m)
    end
    @assert length(d_vec) == m "d must have same length as Xref"

    # Compute Euclidean distances from Xref to all training points
    # (candidate selection uses Euclidean distance for efficiency, like R laGP)
    distances = _compute_squared_distances(X, Xref)

    # Use partialsortperm for O(n) selection of nearest neighbors + candidates
    close_eff = _resolve_close(close, n, endpt)
    sorted_indices = partialsortperm(distances, 1:close_eff)

    # Initialize with nearest neighbors
    local_indices = sorted_indices[1:start]

    # Build local design matrices
    X_local = X[local_indices, :]
    Z_local = Z[local_indices]

    # Create local separable GP
    gp = new_gp_sep(X_local, Z_local, d_vec, T(g))

    # Reference point as matrix
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
            extend_gp_sep!(gp, x_new, z_new)
        end
    else
        # ALC-based selection
        cand_pool = collect(@view sorted_indices[(start + 1):close_eff])
        n_pool = length(cand_pool)
        acq_buf = Vector{T}(undef, n_pool)

        alc_work = ALCWorkspace{T}()

        ncand = n_pool
        while length(local_indices) < endpt && ncand > 0
            cand_view = @view cand_pool[1:ncand]

            # Use separable ALC
            _alc_gp_idx!(acq_buf, gp, X, cand_view, Xref_mat, alc_work)
            best_local_idx = argmax(@view acq_buf[1:ncand])

            best_global_idx = cand_pool[best_local_idx]
            push!(local_indices, best_global_idx)
            if best_local_idx != ncand
                cand_pool[best_local_idx] = cand_pool[ncand]
            end
            ncand -= 1

            # Extend GP with new point
            x_new = @view X[best_global_idx, :]
            z_new = Z[best_global_idx]
            extend_gp_sep!(gp, x_new, z_new)
        end
    end

    # Make final prediction
    pred = pred_gp_sep(gp, Xref_mat; lite=true)

    return (mean=pred.mean[1], var=pred.s2[1], df=pred.df, indices=local_indices)
end

"""
    agp_sep(X, Z, XX; start=6, endpt=50, close=1000, d, g, method=:alc, verb=0, parallel=true)

Approximate GP predictions at multiple reference points using separable GP.

Calls lagp_sep for each row of XX, optionally in parallel using threads.

# Arguments
- `X::Matrix`: training design (n x m)
- `Z::Vector`: training responses
- `XX::Matrix`: test/reference points (n_test x m)
- `start::Int`: initial number of nearest neighbors
- `endpt::Int`: final local design size
- `close::Int`: size of closest candidate pool (default 1000)
- `d::Union{Real,Vector{<:Real},NamedTuple}`: lengthscale parameters or (start, mle, min, max)
- `g::Union{Real,NamedTuple}`: nugget parameter or (start, mle, min, max)
- `method::Symbol`: acquisition method (:alc or :nn)
- `verb::Int`: verbosity level
- `parallel::Bool`: use multi-threading

# Returns
- `NamedTuple`: (mean=..., var=..., df=..., mle=...)
  where mle.d is a Matrix (n_test x m) when MLE enabled
"""
function agp_sep(X::Matrix{T}, Z::Vector{T}, XX::Matrix{T};
                 start::Int=6, endpt::Int=50, close::Int=1000,
                 d::Union{Real,Vector{<:Real},NamedTuple}=T(0.5),
                 g::Union{Real,NamedTuple}=1e-3,
                 method::Symbol=:alc, verb::Int=0, parallel::Bool=true) where {T}

    n_test = size(XX, 1)
    m = size(X, 2)

    # Parse d parameters (vector of per-dimension lengthscales)
    function _expand_d_param(val)
        if val isa Number
            return fill(T(val), m)
        elseif val isa AbstractVector
            v = T.(val)
            if length(v) == 1
                return fill(v[1], m)
            elseif length(v) == m
                return v
            else
                error("d must be scalar or length $m")
            end
        else
            error("d must be scalar or length $m")
        end
    end

    if d isa NamedTuple
        d_start = _expand_d_param(d.start)
        d_mle = get(d, :mle, false)
        d_min = _expand_d_param(get(d, :min, d_start ./ 10))
        d_max = _expand_d_param(get(d, :max, d_start .* 10))
    else
        d_start = _expand_d_param(d)
        d_mle = false
        d_min = d_start ./ 10
        d_max = d_start .* 10
    end

    # Parse g parameters
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
    mle_d = d_mle ? Matrix{T}(undef, n_test, m) : Matrix{T}(undef, 0, 0)
    mle_g = g_mle ? Vector{T}(undef, n_test) : T[]

    # Process each test point
    if parallel && Threads.nthreads() > 1
        Threads.@threads for i in 1:n_test
            means[i], vars[i], dfs[i], mle_d_i, mle_g_i = _agp_single_sep(
                X, Z, @view(XX[i, :]), start, endpt, close,
                copy(d_start), d_mle, d_min, d_max,
                g_start, g_mle, g_min, g_max,
                method, verb
            )
            if d_mle
                mle_d[i, :] .= mle_d_i
            end
            if g_mle
                mle_g[i] = mle_g_i
            end
        end
    else
        for i in 1:n_test
            means[i], vars[i], dfs[i], mle_d_i, mle_g_i = _agp_single_sep(
                X, Z, @view(XX[i, :]), start, endpt, close,
                copy(d_start), d_mle, d_min, d_max,
                g_start, g_mle, g_min, g_max,
                method, verb
            )
            if d_mle
                mle_d[i, :] .= mle_d_i
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
    _agp_single_sep(X, Z, Xref, start, endpt, close, ...)

Internal function to process a single test point for agp_sep.
"""
function _agp_single_sep(X::Matrix{T}, Z::Vector{T}, Xref::AbstractVector{T},
                         start::Int, endpt::Int, close::Int,
                         d_start::Vector{T}, d_mle::Bool, d_min::Vector{T}, d_max::Vector{T},
                         g_start::T, g_mle::Bool, g_min::T, g_max::T,
                         method::Symbol, verb::Int) where {T}
    n = size(X, 1)
    m = length(Xref)

    # Compute Euclidean distances from Xref to all training points
    distances = _compute_squared_distances(X, Xref)

    # Use partialsortperm for O(n) selection of nearest neighbors + candidates
    close_eff = _resolve_close(close, n, endpt)
    sorted_indices = partialsortperm(distances, 1:close_eff)

    # Initialize with nearest neighbors
    local_indices = sorted_indices[1:start]

    # Build local design matrices
    X_local = X[local_indices, :]
    Z_local = Z[local_indices]

    # Create local separable GP
    gp = new_gp_sep(X_local, Z_local, d_start, g_start)

    # Reference point as matrix
    Xref_mat = Matrix{T}(undef, 1, m)
    @inbounds for j in 1:m
        Xref_mat[1, j] = Xref[j]
    end

    if method == :nn
        # Pure nearest-neighbor expansion
        for pos in (start + 1):endpt
            best_global_idx = sorted_indices[pos]
            push!(local_indices, best_global_idx)
            x_new = @view X[best_global_idx, :]
            z_new = Z[best_global_idx]
            extend_gp_sep!(gp, x_new, z_new)
        end
    else
        # ALC-based selection
        cand_pool = collect(@view sorted_indices[(start + 1):close_eff])
        n_pool = length(cand_pool)
        acq_buf = Vector{T}(undef, n_pool)

        alc_work = ALCWorkspace{T}()

        ncand = n_pool
        while length(local_indices) < endpt && ncand > 0
            cand_view = @view cand_pool[1:ncand]

            # Use separable ALC
            _alc_gp_idx!(acq_buf, gp, X, cand_view, Xref_mat, alc_work)
            best_local_idx = argmax(@view acq_buf[1:ncand])

            best_global_idx = cand_pool[best_local_idx]
            push!(local_indices, best_global_idx)
            if best_local_idx != ncand
                cand_pool[best_local_idx] = cand_pool[ncand]
            end
            ncand -= 1

            # Extend GP with new point
            x_new = @view X[best_global_idx, :]
            z_new = Z[best_global_idx]
            extend_gp_sep!(gp, x_new, z_new)
        end
    end

    # Perform MLE if requested
    final_d = copy(gp.d)
    final_g = gp.g
    if d_mle || g_mle
        # Build per-dimension ranges for jmle_gp_sep!
        d_ranges = [(d_min[k], d_max[k]) for k in 1:m]
        if d_mle && g_mle
            jmle_gp_sep!(gp; drange=d_ranges, grange=(g_min, g_max))
        elseif d_mle
            # Optimize d only, keep g fixed
            mle_gp_sep_d!(gp; drange=d_ranges)
        else
            # Optimize g only - use Newton/Brent on nugget
            newton_gp_sep_g(gp; grange=(g_min, g_max))
        end
        final_d = copy(gp.d)
        final_g = gp.g
    end

    # Make final prediction
    pred = pred_gp_sep(gp, Xref_mat; lite=true)

    return pred.mean[1], pred.s2[1], pred.df, final_d, final_g
end
