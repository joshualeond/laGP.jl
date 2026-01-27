# Core GP functions

using AbstractGPs: GP as AbstractGP, posterior, mean, var
using KernelFunctions: SqExponentialKernel, with_lengthscale, ARDTransform, kernelmatrix!, kernelmatrix, RowVecs

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
    _compute_lite_variance(k, Kik, phi, n, g)

Compute diagonal (lite) variance for GP predictions.

Formula: s2[j] = (phi/n) * (1 + g - k[:,j]' * Ki * k[:,j])
"""
function _compute_lite_variance(k::Matrix{T}, Kik::Matrix{T}, phi::T, n::Int, g::T) where {T}
    n_test = size(k, 2)
    n_train = size(k, 1)
    s2 = Vector{T}(undef, n_test)
    scale = phi / n
    @inbounds for j in 1:n_test
        kKik = zero(T)
        for i in 1:n_train
            kKik += k[i, j] * Kik[i, j]
        end
        s2[j] = scale * (one(T) + g - kKik)
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

    # Cholesky factorization
    chol = cholesky(Symmetric(K))

    # Compute Ki * Z
    KiZ = chol \ Z

    # Compute phi = Z' * Ki * Z
    phi = dot(Z, KiZ)

    # Log determinant
    ldetK = _compute_logdet_chol(chol)

    return GP{T,typeof(kernel)}(copy(X), copy(Z), kernel, chol, KiZ, d_T, g_T, phi, ldetK)
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
    Ki = inv(gp.chol)  # Compute inverse from Cholesky

    # Gradient w.r.t. nugget g
    dllg = zero(T)
    if dg
        tr_Ki = tr(Ki)
        dllg = -T(0.5) * tr_Ki + T(0.5) * (n / gp.phi) * dot(KiZ, KiZ)
    end

    # Gradient w.r.t. lengthscale d
    dlld = zero(T)
    if dd
        X = gp.X
        # Need kernel matrix without nugget for derivative
        K_no_nugget = kernelmatrix(gp.kernel, RowVecs(X))
        d_sq = gp.d^2
        tr_term = zero(T)
        quad_term = zero(T)

        @inbounds for j in 1:n
            for i in 1:n
                if i != j
                    dist_sq = zero(T)
                    for m in axes(X, 2)
                        diff = X[i, m] - X[j, m]
                        dist_sq += diff * diff
                    end
                    dK_ij = K_no_nugget[i, j] * dist_sq / d_sq
                    tr_term += Ki[i, j] * dK_ij
                    quad_term += KiZ[i] * dK_ij * KiZ[j]
                end
            end
        end

        dlld = -T(0.5) * tr_term + T(0.5) * (n / gp.phi) * quad_term
    end

    return (dllg=dllg, dlld=dlld)
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

        # Recompute Cholesky
        gp.chol = cholesky(Symmetric(K))

        # Recompute KiZ
        gp.KiZ .= gp.chol \ gp.Z

        # Recompute phi
        gp.phi = dot(gp.Z, gp.KiZ)

        # Recompute log determinant
        gp.ldetK = _compute_logdet_chol(gp.chol)
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

    # Compute covariance matrix with nugget
    K = kernelmatrix(kernel, RowVecs(X))
    _add_nugget_diagonal!(K, g_T)

    # Cholesky factorization
    chol = cholesky(Symmetric(K))

    # Compute Ki * Z
    KiZ = chol \ Z

    # Compute phi = Z' * Ki * Z
    phi = dot(Z, KiZ)

    # Log determinant
    ldetK = _compute_logdet_chol(chol)

    return GPsep{T,typeof(kernel)}(copy(X), copy(Z), kernel, chol, KiZ, d_T, g_T, phi, ldetK)
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
    Ki = inv(gp.chol)

    # Gradient w.r.t. nugget g
    dllg = zero(T)
    if dg
        tr_Ki = zero(T)
        @inbounds for i in 1:n
            tr_Ki += Ki[i, i]
        end
        dllg = -T(0.5) * tr_Ki + T(0.5) * (n / gp.phi) * dot(KiZ, KiZ)
    end

    # Gradient w.r.t. each lengthscale d[k]
    dlld = zeros(T, m)
    if dd
        X = gp.X
        K_no_nugget = kernelmatrix(gp.kernel, RowVecs(X))
        phi_factor = T(0.5) * (n / gp.phi)

        for k in 1:m
            inv_dk_sq = one(T) / (gp.d[k] * gp.d[k])
            tr_term = zero(T)
            quad_term = zero(T)

            @inbounds for j in 1:n
                KiZ_j = KiZ[j]
                for i in 1:n
                    if i != j
                        diff = X[i, k] - X[j, k]
                        dK_ij = K_no_nugget[i, j] * diff * diff * inv_dk_sq
                        tr_term += Ki[i, j] * dK_ij
                        quad_term += KiZ[i] * dK_ij * KiZ_j
                    end
                end
            end

            dlld[k] = -T(0.5) * tr_term + phi_factor * quad_term
        end
    end

    return (dllg=dllg, dlld=dlld)
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
        K = kernelmatrix(gp.kernel, RowVecs(gp.X))
        _add_nugget_diagonal!(K, gp.g)

        # Recompute Cholesky
        gp.chol = cholesky(Symmetric(K))

        # Recompute KiZ
        gp.KiZ .= gp.chol \ gp.Z

        # Recompute phi
        gp.phi = dot(gp.Z, gp.KiZ)

        # Recompute log determinant
        gp.ldetK = _compute_logdet_chol(gp.chol)
    end

    return gp
end
