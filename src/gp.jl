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

    # Cache inverse for gradient computations
    Ki = inv(chol)

    # Compute Ki * Z
    KiZ = chol \ Z

    # Compute phi = Z' * Ki * Z
    phi = dot(Z, KiZ)

    # Log determinant
    ldetK = _compute_logdet_chol(chol)

    return GP{T,typeof(kernel)}(copy(X), copy(Z), kernel, chol, Ki, KiZ, d_T, g_T, phi, ldetK)
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
        tr_Ki = tr(Ki)
        dllg = -T(0.5) * tr_Ki + T(0.5) * (n / gp.phi) * dot(KiZ, KiZ)
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

        # (KiZ' * KiZ / phi)² = (phi / phi)² = 1 -- wait, that's not right
        # KiZ' * KiZ = Z' * Ki' * Ki * Z, but phi = Z' * Ki * Z
        # So phirat = KiZ' * KiZ / phi
        KiZ_dot = dot(KiZ, KiZ)
        phirat_sq = (KiZ_dot / phi)^2

        # d²llik/dg² = -0.5 * tr(Ki²) + (n/phi) * KiZ'*Ki*KiZ - 0.5*n*phirat²
        d2llg = -T(0.5) * tr_Ki_sq + (n / phi) * quad_term - T(0.5) * n * phirat_sq
    end

    # Second derivative w.r.t. lengthscale d
    d2lld = zero(T)
    if d2d
        X = gp.X
        d = gp.d

        # Get kernel matrix without nugget
        K_no_nugget = kernelmatrix(gp.kernel, RowVecs(X))

        # Compute squared distance matrix D²[i,j] = ||x_i - x_j||²
        # We can recover this from K: K[i,j] = exp(-D²[i,j]/d), so D²[i,j] = -d * log(K[i,j])
        # But for numerical stability, compute directly
        D_sq = zeros(T, n, n)
        @inbounds for i in 1:n
            for j in (i+1):n
                dist_sq = zero(T)
                for k in axes(X, 2)
                    diff = X[i, k] - X[j, k]
                    dist_sq += diff * diff
                end
                D_sq[i, j] = dist_sq
                D_sq[j, i] = dist_sq
            end
        end

        inv_d_sq = one(T) / (d * d)

        # dK/dd[i,j] = K[i,j] * D²[i,j] / d²
        dK = K_no_nugget .* D_sq .* inv_d_sq

        # d²K/dd²[i,j] = dK[i,j] * (D²[i,j]/d² - 2/d)
        #              = K[i,j] * D²[i,j] / d² * (D²[i,j]/d² - 2/d)
        d2K = dK .* (D_sq .* inv_d_sq .- 2 / d)

        # First partial of phi w.r.t. d: dφ/dd = -2 * KiZ' * dK * KiZ
        dK_KiZ = dK * KiZ
        dlphi = -2 * dot(KiZ, dK_KiZ)

        # Second partial of phi w.r.t. d: d²φ/dd² = 2*KiZ'*dK*Ki*dK*KiZ - 2*KiZ'*d²K*KiZ
        Ki_dK_KiZ = Ki * dK_KiZ
        d2phi = 2 * dot(dK_KiZ, Ki_dK_KiZ) - 2 * dot(KiZ, d2K * KiZ)

        # tr(Ki * dK * Ki * dK) = sum((Ki * dK) .* (Ki * dK)')
        Ki_dK = Ki * dK
        tr_KidK_sq = sum(Ki_dK .* Ki_dK')

        # tr(Ki * d²K)
        tr_Ki_d2K = sum(Ki .* d2K')

        # d²llik/dd² = 0.5*tr(Ki*dK*Ki*dK) - 0.5*tr(Ki*d²K) + 0.5*(n/phi)*(d²φ - (dφ)²/phi)
        d2lld = T(0.5) * tr_KidK_sq - T(0.5) * tr_Ki_d2K +
                T(0.5) * (n / phi) * (d2phi - dlphi^2 / phi)
    end

    return (d2llg=d2llg, d2lld=d2lld)
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

        # Update cached inverse
        gp.Ki .= inv(gp.chol)

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

    # Cache inverse for gradient computations
    Ki = inv(chol)

    # Compute Ki * Z
    KiZ = chol \ Z

    # Compute phi = Z' * Ki * Z
    phi = dot(Z, KiZ)

    # Log determinant
    ldetK = _compute_logdet_chol(chol)

    return GPsep{T,typeof(kernel)}(copy(X), copy(Z), kernel, chol, Ki, KiZ, d_T, g_T, phi, ldetK)
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
    Ki = gp.Ki  # Use cached inverse

    # Gradient w.r.t. nugget g
    dllg = zero(T)
    if dg
        tr_Ki = tr(Ki)
        dllg = -T(0.5) * tr_Ki + T(0.5) * (n / gp.phi) * dot(KiZ, KiZ)
    end

    # Gradient w.r.t. each lengthscale d[k]
    dlld = zeros(T, m)
    if dd
        X = gp.X
        K_no_nugget = kernelmatrix(gp.kernel, RowVecs(X))
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

    # d²llik/dg² = -0.5 * tr(Ki²) + (n/phi) * KiZ'*Ki*KiZ - 0.5*n*phirat²
    return -T(0.5) * tr_Ki_sq + (n / phi) * quad_term - T(0.5) * n * phirat_sq
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

        # Update cached inverse
        gp.Ki .= inv(gp.chol)

        # Recompute KiZ
        gp.KiZ .= gp.chol \ gp.Z

        # Recompute phi
        gp.phi = dot(gp.Z, gp.KiZ)

        # Recompute log determinant
        gp.ldetK = _compute_logdet_chol(gp.chol)
    end

    return gp
end
