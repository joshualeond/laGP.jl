# Core GP functions

using KernelFunctions: SqExponentialKernel, with_lengthscale, ARDTransform, kernelmatrix, RowVecs
using LoopVectorization: @turbo

# ============================================================================
# Shared helper functions
# ============================================================================

"""
    _concentrated_llik(n, phi, ldetK)

Compute the concentrated (profile) log-likelihood used by laGP.

Formula: llik = -0.5 * (n * log(0.5 * phi) + ldetK)

This is shared across all GP model types (GP, GPsep).
"""
function _concentrated_llik(n::Int, phi::T, ldetK::T) where {T}
    return -T(0.5) * (n * log(T(0.5) * phi) + ldetK)
end

"""
    _compute_squared_distance_matrix!(D_sq, X)

Compute the squared distance matrix D_sq[i,j] = ||x_i - x_j||² in-place.
Uses SIMD vectorization for optimal performance.
"""
function _compute_squared_distance_matrix!(D_sq::Matrix{T}, X::Matrix{T}) where {T}
    n = size(X, 1)
    m = size(X, 2)

    # Zero the diagonal
    @inbounds for i in 1:n
        D_sq[i, i] = zero(T)
    end

    # Compute upper triangle with SIMD
    @inbounds for i in 1:n
        for j in (i+1):n
            dist_sq = zero(T)
            @turbo for k in 1:m
                diff = X[i, k] - X[j, k]
                dist_sq += diff * diff
            end
            D_sq[i, j] = dist_sq
            D_sq[j, i] = dist_sq
        end
    end
    return D_sq
end

"""
    _kernelmatrix_sep_train!(K, X, d)

Build separable squared-exponential kernel matrix in-place:
K[i,j] = exp(-sum((X[i,k]-X[j,k])^2 / d[k] for k in 1:m))

Uses SIMD vectorization and optional threading for large matrices.
"""
function _kernelmatrix_sep_train!(K::Matrix{T}, X::Matrix{T}, d::Vector{T}) where {T}
    n, m = size(X)
    @assert size(K, 1) == n && size(K, 2) == n "K must be n x n"
    @assert length(d) == m "d must have length m"

    inv_d = Vector{T}(undef, m)
    @inbounds @turbo for k in 1:m
        inv_d[k] = one(T) / d[k]
    end

    @inbounds for i in 1:n
        K[i, i] = one(T)
    end

    if n > 1
        if Threads.nthreads() > 1 && n >= 256
            Threads.@threads for i in 1:(n - 1)
                for j in (i + 1):n
                    weighted_dist_sq = zero(T)
                    @turbo for k in 1:m
                        diff = X[i, k] - X[j, k]
                        weighted_dist_sq += diff * diff * inv_d[k]
                    end
                    kij = exp(-weighted_dist_sq)
                    K[i, j] = kij
                    K[j, i] = kij
                end
            end
        else
            @inbounds for i in 1:(n - 1)
                for j in (i + 1):n
                    weighted_dist_sq = zero(T)
                    @turbo for k in 1:m
                        diff = X[i, k] - X[j, k]
                        weighted_dist_sq += diff * diff * inv_d[k]
                    end
                    kij = exp(-weighted_dist_sq)
                    K[i, j] = kij
                    K[j, i] = kij
                end
            end
        end
    end

    return K
end

"""
    _kernelmatrix_sep_cross!(Kxy, X, XX, d)

Build separable cross-covariance matrix in-place:
Kxy[i,j] = exp(-sum((X[i,k]-XX[j,k])^2 / d[k] for k in 1:m))

Uses SIMD vectorization and optional threading for large cross-products.
"""
function _kernelmatrix_sep_cross!(Kxy::Matrix{T}, X::Matrix{T}, XX::Matrix{T}, d::Vector{T}) where {T}
    n, m = size(X)
    n_test, m_test = size(XX)
    @assert m_test == m "XX must have same number of columns as X"
    @assert size(Kxy, 1) == n && size(Kxy, 2) == n_test "Kxy must be n x n_test"
    @assert length(d) == m "d must have length m"

    inv_d = Vector{T}(undef, m)
    @inbounds @turbo for k in 1:m
        inv_d[k] = one(T) / d[k]
    end

    if Threads.nthreads() > 1 && (n * n_test) >= 200_000
        Threads.@threads for j in 1:n_test
            @inbounds for i in 1:n
                weighted_dist_sq = zero(T)
                @turbo for k in 1:m
                    diff = X[i, k] - XX[j, k]
                    weighted_dist_sq += diff * diff * inv_d[k]
                end
                Kxy[i, j] = exp(-weighted_dist_sq)
            end
        end
    else
        @inbounds for j in 1:n_test
            for i in 1:n
                weighted_dist_sq = zero(T)
                @turbo for k in 1:m
                    diff = X[i, k] - XX[j, k]
                    weighted_dist_sq += diff * diff * inv_d[k]
                end
                Kxy[i, j] = exp(-weighted_dist_sq)
            end
        end
    end

    return Kxy
end

"""
    _add_nugget_diagonal!(K, g)

Add nugget to diagonal of matrix K in-place.
"""
function _add_nugget_diagonal!(K::Matrix{T}, g::T) where {T}
    n = size(K, 1)
    @inbounds for i in 1:n
        K[i, i] += g
    end
    return K
end

"""
    _compute_logdet_chol(chol)

Compute log determinant from Cholesky factorization.

Optimized to avoid allocating a temporary vector for the diagonal.
"""
function _compute_logdet_chol(chol::Cholesky{T}) where {T}
    ldetK = zero(T)
    L = chol.L
    @inbounds for i in axes(L, 1)
        ldetK += log(L[i, i])
    end
    return 2 * ldetK
end

"""
    _compute_gp_cache(K, Z)

Compute and return the shared GP cache quantities from a covariance matrix K and response Z.

Returns a NamedTuple with: chol, Ki, KiZ, phi, ldetK, tr_Ki.

Used by `new_gp`, `new_gp_sep`, `update_gp!`, `update_gp_sep!` to avoid code duplication.
"""
function _compute_gp_cache(K::AbstractMatrix{T}, Z::Vector{T}) where {T}
    chol = cholesky(Symmetric(K))
    Ki = inv(chol)
    KiZ = chol \ Z
    phi = dot(Z, KiZ)
    ldetK = _compute_logdet_chol(chol)
    tr_Ki = tr(Ki)
    return (; chol, Ki, KiZ, phi, ldetK, tr_Ki)
end

"""
    _compute_lite_variance(k, Kik, phi, n, g)

Compute diagonal (lite) variance for GP predictions.

Formula: s2[j] = (phi/n) * (1 + g - k[:,j]' * Ki * k[:,j])

Uses SIMD vectorization for optimal performance.
"""
function _compute_lite_variance(k::Matrix{T}, Kik::Matrix{T}, phi::T, n::Int, g::T) where {T}
    n_test = size(k, 2)
    n_train = size(k, 1)
    s2 = Vector{T}(undef, n_test)
    scale = phi / n
    base = one(T) + g
    @inbounds for j in 1:n_test
        kKik = zero(T)
        @turbo for i in 1:n_train
            kKik += k[i, j] * Kik[i, j]
        end
        s2[j] = scale * (base - kKik)
    end
    return s2
end

# ============================================================================
# Isotropic GP functions
# ============================================================================

"""
    new_gp(X, Z, d, g)

Create a new Gaussian Process model using AbstractGPs.jl backend.

# Arguments
- `X::Matrix`: n x m design matrix (n observations, m dimensions)
- `Z::Vector`: n response values
- `d::Real`: lengthscale parameter (laGP parameterization)
- `g::Real`: nugget parameter

# Returns
- `GP`: Gaussian Process model backed by AbstractGPs
"""
function new_gp(X::Matrix{T}, Z::Vector{T}, d::Real, g::Real) where {T<:Real}
    n = size(X, 1)
    @assert length(Z) == n "Z must have same length as number of rows in X"
    @assert d > 0 "lengthscale d must be positive"
    @assert g > 0 "nugget g must be positive"

    d_T = T(d)
    g_T = T(g)

    # Build kernel using AbstractGPs adapter
    kernel = build_kernel_isotropic(d_T)

    # Compute covariance matrix with nugget
    K = kernelmatrix(kernel, RowVecs(X))
    _add_nugget_diagonal!(K, g_T)

    c = _compute_gp_cache(K, Z)

    return GP{T,typeof(kernel)}(copy(X), copy(Z), kernel, c.chol, c.Ki, c.KiZ, d_T, g_T, c.phi, c.ldetK, c.tr_Ki)
end

"""
    pred_gp(gp, XX; lite=true)

Make predictions at test locations XX using AbstractGPs-backed GP.

# Arguments
- `gp::GP`: Gaussian Process model
- `XX::Matrix`: test locations (n_test x m)
- `lite::Bool`: if true, return only diagonal variances

# Returns
- `GPPrediction`: prediction results with mean, s2, and df
"""
function pred_gp(gp::GP{T}, XX::Matrix{T}; lite::Bool=true) where {T}
    n = size(gp.X, 1)
    n_test = size(XX, 1)

    # Cross-covariance using kernel
    k = kernelmatrix(gp.kernel, RowVecs(gp.X), RowVecs(XX))

    # Mean prediction: mu = k' * Ki * Z
    mean_pred = k' * gp.KiZ

    # Variance computation using Cholesky solve
    Kik = gp.chol \ k

    if lite
        # Diagonal variances only
        s2 = _compute_lite_variance(k, Kik, gp.phi, n, gp.g)
        return GPPrediction{T}(mean_pred, s2, n)
    else
        # Full covariance matrix
        # Sigma = (phi/n) * (K** + g*I - k' * Ki * k)
        scale = gp.phi / n

        # Compute K(XX, XX) + g*I (test-test covariance)
        K_test = kernelmatrix(gp.kernel, RowVecs(XX))
        _add_nugget_diagonal!(K_test, gp.g)

        # Sigma = scale * (K_test - k' * Kik)
        Sigma = scale .* (K_test - k' * Kik)

        return GPPredictionFull{T}(mean_pred, Sigma, n)
    end
end

"""
    llik_gp(gp)

Compute the log-likelihood of the GP.

Uses the concentrated likelihood formula from R laGP:
    llik = -0.5 * (n * log(0.5 * phi) + ldetK)

# Arguments
- `gp::GP`: Gaussian Process model

# Returns
- `Real`: log-likelihood value
"""
function llik_gp(gp::GP{T}) where {T}
    return _concentrated_llik(length(gp.Z), gp.phi, gp.ldetK)
end

"""
    dllik_gp(gp; dg=true, dd=true)

Compute gradient of log-likelihood w.r.t. d (lengthscale) and g (nugget).

# Arguments
- `gp::GP`: Gaussian Process model
- `dg::Bool`: compute gradient w.r.t. nugget g (default: true)
- `dd::Bool`: compute gradient w.r.t. lengthscale d (default: true)

# Returns
- `NamedTuple`: (dllg=..., dlld=...) gradients
"""
function dllik_gp(gp::GP{T}; dg::Bool=true, dd::Bool=true) where {T}
    n = length(gp.Z)
    KiZ = gp.KiZ
    Ki = gp.Ki  # Use cached inverse

    # Gradient w.r.t. nugget g
    dllg = zero(T)
    if dg
        # Use cached trace instead of recomputing
        dllg = -T(0.5) * gp.tr_Ki + T(0.5) * (n / gp.phi) * dot(KiZ, KiZ)
    end

    # Gradient w.r.t. lengthscale d
    dlld = zero(T)
    if dd
        X = gp.X
        # Need kernel matrix without nugget for derivative
        K_no_nugget = kernelmatrix(gp.kernel, RowVecs(X))
        inv_d_sq = one(T) / (gp.d * gp.d)
        phi_factor = T(0.5) * (n / gp.phi)

        # Vectorized gradient computation using BLAS operations
        # dK/dd[i,j] = K[i,j] * ||x_i - x_j||² / d²
        # tr_term = sum(Ki .* dK) = sum(Ki .* K .* D) / d²
        # quad_term = KiZ' * dK * KiZ = sum((KiZ * KiZ') .* K .* D) / d²
        #
        # Using identity: sum(W .* D) = 2 * (dot(row_norms², row_sums) - tr(X' * W * X))
        # where row_sums = sum(W, dims=2)

        # Precompute weighted matrices
        WK = Ki .* K_no_nugget                      # For trace term
        WK_quad = (KiZ * KiZ') .* K_no_nugget       # For quadratic term

        # Row sums (exploiting symmetry)
        WK_row_sums = vec(sum(WK, dims=2))
        WK_quad_row_sums = vec(sum(WK_quad, dims=2))

        # Squared row norms of X
        row_norms_sq = vec(sum(X .^ 2, dims=2))

        # sum(W .* D) = 2 * (dot(row_norms², row_sums) - sum(X .* (W * X)))
        # where the second term equals tr(X' * W * X)
        tr_term = 2 * (dot(row_norms_sq, WK_row_sums) - sum(X .* (WK * X))) * inv_d_sq
        quad_term = 2 * (dot(row_norms_sq, WK_quad_row_sums) - sum(X .* (WK_quad * X))) * inv_d_sq

        dlld = -T(0.5) * tr_term + phi_factor * quad_term
    end

    return (dllg=dllg, dlld=dlld)
end

"""
    d2llik_gp(gp; d2g=true, d2d=true)

Compute second derivatives of log-likelihood w.r.t. d (lengthscale) and g (nugget).

Used by Newton's method for 1D parameter optimization.

# Arguments
- `gp::GP`: Gaussian Process model
- `d2g::Bool`: compute second derivative w.r.t. nugget g (default: true)
- `d2d::Bool`: compute second derivative w.r.t. lengthscale d (default: true)

# Returns
- `NamedTuple`: (d2llg=..., d2lld=...) second derivatives
"""
function d2llik_gp(gp::GP{T}; d2g::Bool=true, d2d::Bool=true) where {T}
    n = length(gp.Z)
    dn = T(n)
    Ki = gp.Ki
    KiZ = gp.KiZ
    phi = gp.phi

    # Second derivative w.r.t. nugget g
    d2llg = zero(T)
    if d2g
        # tr(Ki²) - note Ki is symmetric so Ki .* Ki' = Ki .* Ki
        tr_Ki_sq = sum(Ki .* Ki)

        # KiZ' * Ki * KiZ
        Ki_KiZ = Ki * KiZ
        quad_term = dot(KiZ, Ki_KiZ)

        # (KiZ' * KiZ / phi)²
        KiZ_dot = dot(KiZ, KiZ)
        phirat_sq = (KiZ_dot / phi)^2

        # Matches the original laGP C implementation
        # d²llik/dg² = 0.5*tr(Ki²) - (n/phi)*KiZ'*Ki*KiZ + 0.5*n*phirat²
        d2llg = T(0.5) * tr_Ki_sq - (dn / phi) * quad_term + T(0.5) * dn * phirat_sq
    end

    # Second derivative w.r.t. lengthscale d
    d2lld = zero(T)
    if d2d
        X = gp.X
        d = gp.d

        # Get kernel matrix without nugget
        K_no_nugget = kernelmatrix(gp.kernel, RowVecs(X))

        # Compute squared distance matrix D²[i,j] = ||x_i - x_j||²
        # Uses SIMD-optimized computation for optimal performance
        D_sq = Matrix{T}(undef, n, n)
        _compute_squared_distance_matrix!(D_sq, X)

        inv_d_sq = one(T) / (d * d)

        # dK/dd[i,j] = K[i,j] * D²[i,j] / d²
        dK = K_no_nugget .* D_sq .* inv_d_sq

        # d²K/dd²[i,j] = dK[i,j] * (D²[i,j]/d² - 2/d)
        #              = K[i,j] * D²[i,j] / d² * (D²[i,j]/d² - 2/d)
        d2K = dK .* (D_sq .* inv_d_sq .- 2 / d)

        # Original laGP C expression:
        # d²llik/dd² = -0.5*tr(Ki*(d2K - dK*Ki*dK))
        #              -0.5*(n/phi)*KiZ'*(2*dK*Ki*dK - d2K)*KiZ
        #              +0.5*n*(KiZ'*dK*KiZ/phi)^2
        Ki_dK = Ki * dK
        dKKidK = dK * Ki_dK

        tr_term = sum(Ki .* transpose(d2K .- dKKidK))
        two = 2 .* dKKidK .- d2K
        quad_two = dot(KiZ, two * KiZ)
        phirat = dot(KiZ, dK * KiZ) / phi

        d2lld = -T(0.5) * tr_term -
                T(0.5) * (dn / phi) * quad_two +
                T(0.5) * dn * phirat^2
    end

    return (d2llg=d2llg, d2lld=d2lld)
end

"""
    _extend_gp_core!(gp, k, x_new, z_new)

Shared incremental Cholesky update logic for extending a GP with a new observation.

Given the pre-computed kernel vector `k` between `x_new` and existing points,
performs the O(n²) incremental Cholesky update, Ki block update, and all
cached quantity updates (KiZ, phi, ldetK, tr_Ki).

Called by both `extend_gp!` (isotropic) and `extend_gp_sep!` (separable) after
they compute the kernel vector using their respective kernel parameterizations.
"""
function _extend_gp_core!(gp::AnyGP{T}, k::Vector{T}, x_new::AbstractVector{T}, z_new::T) where {T}
    n = size(gp.X, 1)
    m = size(gp.X, 2)

    # κ = kernel(x_new, x_new) = 1 (self-covariance without nugget)
    κ = one(T)

    # Forward solve: l = L⁻¹ k using existing Cholesky
    # This is O(n²) via forward substitution
    L = gp.chol.L
    l = Vector{T}(undef, n)
    ldiv!(l, L, k)

    # Compute λ = sqrt(κ + g - lᵀl)
    l_dot_l = dot(l, l)
    λ_sq = κ + gp.g - l_dot_l

    # Handle numerical issues - if λ² ≤ 0, the matrix would be non-positive-definite
    if λ_sq <= zero(T)
        λ = sqrt(eps(T))  # Small positive value to maintain positive-definiteness
    else
        λ = sqrt(λ_sq)
    end

    # Build new Cholesky factor L_new = [L 0; lᵀ λ] without block concatenation
    L_new = Matrix{T}(undef, n + 1, n + 1)
    @inbounds begin
        for j in 1:n
            for i in 1:n
                L_new[i, j] = (i >= j) ? L[i, j] : zero(T)
            end
            L_new[j, n + 1] = zero(T)
        end
        for j in 1:n
            L_new[n + 1, j] = l[j]
        end
        L_new[n + 1, n + 1] = λ
    end
    gp.chol = Cholesky(L_new, 'L', 0)

    # Update Ki incrementally using block matrix inversion formula
    # For K_new = [K k; kᵀ κ+g], Ki_new can be computed from Ki:
    # Let v = Ki * k, s = κ + g - kᵀ * v = λ²
    # Then:
    #   Ki_new = [Ki + v*vᵀ/s   -v/s  ]
    #            [   -vᵀ/s       1/s  ]
    v = Vector{T}(undef, n)
    mul!(v, gp.Ki, k)
    s = λ_sq  # Already computed as κ + g - lᵀl (note: lᵀl = kᵀ Ki k when L = chol)

    # Handle case where s is very small
    if abs(s) < eps(T)
        s = eps(T)
    end
    inv_s = one(T) / s

    # Build Ki_new without block concatenation
    v_scaled = v .* inv_s
    Ki_new = Matrix{T}(undef, n + 1, n + 1)
    Ki_ul = @view Ki_new[1:n, 1:n]
    copyto!(Ki_ul, gp.Ki)
    LinearAlgebra.BLAS.ger!(one(T), v, v_scaled, Ki_ul)
    @inbounds for i in 1:n
        Ki_new[i, n + 1] = -v_scaled[i]
        Ki_new[n + 1, i] = -v_scaled[i]
    end
    Ki_new[n + 1, n + 1] = inv_s
    gp.Ki = Ki_new

    # Update X by appending new row
    X_new = Matrix{T}(undef, n + 1, m)
    copyto!(@view(X_new[1:n, 1:m]), gp.X)
    @inbounds for j in 1:m
        X_new[n + 1, j] = x_new[j]
    end
    gp.X = X_new

    # Update Z by appending new value
    Z_new = Vector{T}(undef, n + 1)
    copyto!(Z_new, gp.Z)
    Z_new[n + 1] = z_new
    gp.Z = Z_new

    # Update KiZ incrementally using block structure
    vᵀZ = dot(v, @view gp.Z[1:n])
    gp.KiZ .+= v .* ((vᵀZ - z_new) * inv_s)
    push!(gp.KiZ, (z_new - vᵀZ) * inv_s)

    # Update phi = Z_new' * Ki_new * Z_new
    gp.phi = dot(gp.Z, gp.KiZ)

    # Update log determinant: log|K_new| = log|K| + 2*log(λ)
    gp.ldetK += 2 * log(λ)

    # Update trace of Ki incrementally:
    # tr(Ki_new) = tr(Ki) + tr(v*v'/s) + 1/s
    #            = tr(Ki) + (v'*v)/s + 1/s
    #            = tr(Ki) + (dot(v,v) + 1) / s
    v_dot_v = dot(v, v)
    gp.tr_Ki += (v_dot_v + one(T)) * inv_s

    return gp
end

"""
    extend_gp!(gp, x_new, z_new)

Extend a GP with a new observation using O(n²) incremental Cholesky update.

This is much faster than rebuilding the GP from scratch when sequentially
adding points, as it avoids the O(n³) full Cholesky factorization.

# Mathematical Background
Given existing Cholesky L where K = LLᵀ, when adding a new point:

```
K_new = [K    k  ]
        [kᵀ   κ  ]
```

The updated Cholesky is:
```
L_new = [L    0]
        [lᵀ   λ]
```

Where:
- l = L⁻¹ k (forward solve, O(n²))
- λ = sqrt(κ + g - lᵀl)

# Arguments
- `gp::GP`: Gaussian Process model to extend
- `x_new::AbstractVector`: new input point (length m)
- `z_new::Real`: new output value

# Returns
- `gp`: The modified GP (for convenience, same object as input)
"""
function extend_gp!(gp::GP{T}, x_new::AbstractVector{T}, z_new::T) where {T}
    n = size(gp.X, 1)
    m = size(gp.X, 2)

    @assert length(x_new) == m "x_new must have same dimension as GP inputs"

    # Compute kernel values between new point and existing points
    # k[i] = kernel(X[i,:], x_new) = exp(-||X[i,:] - x_new||² / d)
    # where d is the laGP lengthscale parameter
    k = Vector{T}(undef, n)
    inv_d = one(T) / gp.d
    @inbounds for i in 1:n
        dist_sq = zero(T)
        @turbo for j in 1:m
            diff = gp.X[i, j] - x_new[j]
            dist_sq += diff * diff
        end
        k[i] = exp(-dist_sq * inv_d)
    end

    return _extend_gp_core!(gp, k, x_new, z_new)
end

"""
    update_gp!(gp; d=nothing, g=nothing)

Update GP hyperparameters and recompute internal quantities.

# Arguments
- `gp::GP`: Gaussian Process model
- `d::Real`: new lengthscale (optional)
- `g::Real`: new nugget (optional)
"""
function update_gp!(gp::GP{T}; d::Union{Nothing,Real}=nothing,
                     g::Union{Nothing,Real}=nothing) where {T}
    changed = false

    if !isnothing(d) && T(d) != gp.d
        gp.d = T(d)
        gp.kernel = build_kernel_isotropic(gp.d)
        changed = true
    end

    if !isnothing(g) && T(g) != gp.g
        gp.g = T(g)
        changed = true
    end

    if changed
        # Recompute covariance matrix
        K = kernelmatrix(gp.kernel, RowVecs(gp.X))
        _add_nugget_diagonal!(K, gp.g)

        c = _compute_gp_cache(K, gp.Z)
        gp.chol = c.chol
        gp.Ki .= c.Ki
        gp.KiZ .= c.KiZ
        gp.phi = c.phi
        gp.ldetK = c.ldetK
        gp.tr_Ki = c.tr_Ki
    end

    return gp
end

# ============================================================================
# Separable GP functions
# ============================================================================

"""
    new_gp_sep(X, Z, d, g)

Create a new separable Gaussian Process model using AbstractGPs.jl backend.

# Arguments
- `X::Matrix`: n x m design matrix (n observations, m dimensions)
- `Z::Vector`: n response values
- `d::Vector`: lengthscale parameters (m elements, one per dimension)
- `g::Real`: nugget parameter

# Returns
- `GPsep`: Separable Gaussian Process model backed by AbstractGPs
"""
function new_gp_sep(X::Matrix{T}, Z::Vector{T}, d::Vector{<:Real}, g::Real) where {T<:Real}
    n, m = size(X)
    @assert length(Z) == n "Z must have same length as number of rows in X"
    @assert length(d) == m "d must have same length as number of columns in X"
    @assert all(d .> 0) "all lengthscales in d must be positive"
    @assert g > 0 "nugget g must be positive"

    d_T = T.(d)
    g_T = T(g)

    # Build separable kernel using AbstractGPs adapter
    kernel = build_kernel_separable(d_T)

    # Compute covariance matrix with nugget using specialized separable kernel
    K = Matrix{T}(undef, n, n)
    _kernelmatrix_sep_train!(K, X, d_T)
    _add_nugget_diagonal!(K, g_T)

    c = _compute_gp_cache(K, Z)

    return GPsep{T,typeof(kernel)}(copy(X), copy(Z), kernel, c.chol, c.Ki, c.KiZ, d_T, g_T, c.phi, c.ldetK, c.tr_Ki)
end

"""
    pred_gp_sep(gp, XX; lite=true)

Make predictions at test locations XX using AbstractGPs-backed separable GP.

# Arguments
- `gp::GPsep`: Separable Gaussian Process model
- `XX::Matrix`: test locations (n_test x m)
- `lite::Bool`: if true, return only diagonal variances

# Returns
- `GPPrediction`: prediction results with mean, s2, and df
"""
function pred_gp_sep(gp::GPsep{T}, XX::Matrix{T}; lite::Bool=true) where {T}
    n = size(gp.X, 1)
    n_test = size(XX, 1)

    # Cross-covariance using specialized separable kernel
    k = Matrix{T}(undef, n, n_test)
    _kernelmatrix_sep_cross!(k, gp.X, XX, gp.d)

    # Mean prediction: mu = k' * Ki * Z
    mean_pred = k' * gp.KiZ

    # Variance computation using Cholesky solve
    Kik = gp.chol \ k

    if lite
        # Diagonal variances only
        s2 = _compute_lite_variance(k, Kik, gp.phi, n, gp.g)
        return GPPrediction{T}(mean_pred, s2, n)
    else
        # Full covariance matrix
        # Sigma = (phi/n) * (K** + g*I - k' * Ki * k)
        scale = gp.phi / n

        # Compute K(XX, XX) + g*I (test-test covariance)
        K_test = Matrix{T}(undef, n_test, n_test)
        _kernelmatrix_sep_train!(K_test, XX, gp.d)
        _add_nugget_diagonal!(K_test, gp.g)

        # Sigma = scale * (K_test - k' * Kik)
        Sigma = scale .* (K_test - k' * Kik)

        return GPPredictionFull{T}(mean_pred, Sigma, n)
    end
end

"""
    llik_gp_sep(gp)

Compute the log-likelihood of the GPsep.

# Arguments
- `gp::GPsep`: Separable Gaussian Process model

# Returns
- `Real`: log-likelihood value
"""
function llik_gp_sep(gp::GPsep{T}) where {T}
    return _concentrated_llik(length(gp.Z), gp.phi, gp.ldetK)
end

"""
    dllik_gp_sep(gp; dg=true, dd=true)

Compute gradient of log-likelihood w.r.t. d (lengthscales) and g (nugget).

# Arguments
- `gp::GPsep`: Separable Gaussian Process model
- `dg::Bool`: compute gradient w.r.t. nugget g (default: true)
- `dd::Bool`: compute gradient w.r.t. lengthscales d (default: true)

# Returns
- `NamedTuple`: (dllg=..., dlld=...) gradients
"""
function dllik_gp_sep(gp::GPsep{T}; dg::Bool=true, dd::Bool=true) where {T}
    n = length(gp.Z)
    m = length(gp.d)
    KiZ = gp.KiZ
    Ki = gp.Ki  # Use cached inverse

    # Gradient w.r.t. nugget g
    dllg = zero(T)
    if dg
        # Use cached trace instead of recomputing
        dllg = -T(0.5) * gp.tr_Ki + T(0.5) * (n / gp.phi) * dot(KiZ, KiZ)
    end

    # Gradient w.r.t. each lengthscale d[k]
    dlld = zeros(T, m)
    if dd
        X = gp.X
        K_no_nugget = Matrix{T}(undef, n, n)
        _kernelmatrix_sep_train!(K_no_nugget, X, gp.d)
        phi_factor = T(0.5) * (n / gp.phi)

        # Vectorized gradient computation using BLAS operations
        # dK/dd_k[i,j] = K[i,j] * (X[i,k] - X[j,k])² / d[k]²
        # tr_term = sum(Ki .* dK_k) = sum(Ki .* K .* D_k) / d[k]²
        # quad_term = KiZ' * dK_k * KiZ = sum((KiZ * KiZ') .* K .* D_k) / d[k]²
        #
        # Using identity: sum(W .* D_k) = 2 * (dot(x_k², row_sums) - x_k' * W * x_k)
        # where D_k[i,j] = (X[i,k] - X[j,k])² and row_sums = sum(W, dims=2)

        # Precompute weighted matrices (done once, not per dimension)
        WK = Ki .* K_no_nugget                      # For trace term
        WK_quad = (KiZ * KiZ') .* K_no_nugget       # For quadratic term

        # Row sums (exploiting symmetry of WK and WK_quad)
        WK_row_sums = vec(sum(WK, dims=2))
        WK_quad_row_sums = vec(sum(WK_quad, dims=2))

        # Per-dimension gradient using vectorized operations
        for k in 1:m
            inv_dk_sq = one(T) / (gp.d[k] * gp.d[k])
            x_k = @view X[:, k]
            x_k_sq = x_k .^ 2

            # sum(W .* D_k) = 2 * (dot(x_k², row_sums) - x_k' * W * x_k)
            # The dot(x_k, W * x_k) term is equivalent to x_k' * W * x_k
            WK_x_k = WK * x_k
            WK_quad_x_k = WK_quad * x_k

            tr_term = 2 * (dot(x_k_sq, WK_row_sums) - dot(x_k, WK_x_k)) * inv_dk_sq
            quad_term = 2 * (dot(x_k_sq, WK_quad_row_sums) - dot(x_k, WK_quad_x_k)) * inv_dk_sq

            dlld[k] = -T(0.5) * tr_term + phi_factor * quad_term
        end
    end

    return (dllg=dllg, dlld=dlld)
end

"""
    d2llik_gp_sep_nug(gp)

Compute second derivative of log-likelihood w.r.t. nugget g for separable GP.

Used by Newton's method for 1D nugget optimization.

# Arguments
- `gp::GPsep`: Separable Gaussian Process model

# Returns
- `Real`: second derivative d²llik/dg²
"""
function d2llik_gp_sep_nug(gp::GPsep{T}) where {T}
    n = length(gp.Z)
    dn = T(n)
    Ki = gp.Ki
    KiZ = gp.KiZ
    phi = gp.phi

    # tr(Ki²) - note Ki is symmetric so Ki .* Ki' = Ki .* Ki
    tr_Ki_sq = sum(Ki .* Ki)

    # KiZ' * Ki * KiZ
    Ki_KiZ = Ki * KiZ
    quad_term = dot(KiZ, Ki_KiZ)

    # (KiZ' * KiZ / phi)²
    KiZ_dot = dot(KiZ, KiZ)
    phirat_sq = (KiZ_dot / phi)^2

    # Matches the original laGP C implementation
    # d²llik/dg² = 0.5*tr(Ki²) - (n/phi)*KiZ'*Ki*KiZ + 0.5*n*phirat²
    return T(0.5) * tr_Ki_sq - (dn / phi) * quad_term + T(0.5) * dn * phirat_sq
end

"""
    update_gp_sep!(gp; d=nothing, g=nothing)

Update GPsep hyperparameters and recompute internal quantities.

# Arguments
- `gp::GPsep`: Separable Gaussian Process model
- `d::Vector{Real}`: new lengthscales (optional)
- `g::Real`: new nugget (optional)
"""
function update_gp_sep!(gp::GPsep{T}; d::Union{Nothing,Vector{<:Real}}=nothing,
                         g::Union{Nothing,Real}=nothing) where {T}
    changed = false

    if !isnothing(d)
        @assert length(d) == length(gp.d) "d must have same length as gp.d"
        d_T = T.(d)
        if d_T != gp.d
            gp.d .= d_T
            gp.kernel = build_kernel_separable(gp.d)
            changed = true
        end
    end

    if !isnothing(g) && T(g) != gp.g
        gp.g = T(g)
        changed = true
    end

    if changed
        # Recompute covariance matrix
        n = size(gp.X, 1)
        K = Matrix{T}(undef, n, n)
        _kernelmatrix_sep_train!(K, gp.X, gp.d)
        _add_nugget_diagonal!(K, gp.g)

        c = _compute_gp_cache(K, gp.Z)
        gp.chol = c.chol
        gp.Ki .= c.Ki
        gp.KiZ .= c.KiZ
        gp.phi = c.phi
        gp.ldetK = c.ldetK
        gp.tr_Ki = c.tr_Ki
    end

    return gp
end

"""
    extend_gp_sep!(gp, x_new, z_new)

Extend a separable GP with a new observation using O(n²) incremental Cholesky update.

This is much faster than rebuilding the GP from scratch when sequentially
adding points, as it avoids the O(n³) full Cholesky factorization.

# Mathematical Background
Given existing Cholesky L where K = LLᵀ, when adding a new point:

```
K_new = [K    k  ]
        [kᵀ   κ  ]
```

The updated Cholesky is:
```
L_new = [L    0]
        [lᵀ   λ]
```

Where:
- l = L⁻¹ k (forward solve, O(n²))
- λ = sqrt(κ + g - lᵀl)

# Arguments
- `gp::GPsep`: Separable Gaussian Process model to extend
- `x_new::AbstractVector`: new input point (length m)
- `z_new::Real`: new output value

# Returns
- `gp`: The modified GP (for convenience, same object as input)
"""
function extend_gp_sep!(gp::GPsep{T}, x_new::AbstractVector{T}, z_new::T) where {T}
    n = size(gp.X, 1)
    m = size(gp.X, 2)

    @assert length(x_new) == m "x_new must have same dimension as GP inputs"

    # Compute kernel values between new point and existing points
    # Separable kernel: k[i] = exp(-sum((X[i,j] - x_new[j])^2 / d[j] for j in 1:m))
    k = Vector{T}(undef, n)
    inv_d = similar(gp.d)
    @inbounds @turbo for j in 1:m
        inv_d[j] = one(T) / gp.d[j]
    end
    @inbounds for i in 1:n
        weighted_dist_sq = zero(T)
        @turbo for j in 1:m
            diff = gp.X[i, j] - x_new[j]
            weighted_dist_sq += diff * diff * inv_d[j]
        end
        k[i] = exp(-weighted_dist_sq)
    end

    return _extend_gp_core!(gp, k, x_new, z_new)
end
