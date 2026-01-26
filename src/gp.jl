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

This is shared across all GP model types (GP, GPsep, GPModel, GPModelSep).
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
"""
function _compute_logdet_chol(chol::Cholesky{T}) where {T}
    return 2 * sum(log.(diag(chol.L)))
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

"""
    _compute_covariance!(K, X, d, g)

Compute the covariance matrix K in-place.

Uses the isotropic squared-exponential kernel:
    K[i,j] = exp(-||x_i - x_j||^2 / d)

with nugget g added to the diagonal.

Uses KernelFunctions.jl for optimized kernel matrix computation.
"""
function _compute_covariance!(K::Matrix{T}, X::Matrix{T}, d::T, g::T) where {T}
    n = size(X, 1)

    # KernelFunctions uses: exp(-||x-y||²/(2ℓ²))
    # Our formula uses: exp(-||x-y||²/d)
    # Mapping: d = 2ℓ², so ℓ = sqrt(d/2)
    ℓ = sqrt(d / 2)
    kernel = with_lengthscale(SqExponentialKernel(), ℓ)

    # Compute kernel matrix using optimized KernelFunctions.jl
    # RowVecs treats rows as data points
    kernelmatrix!(K, kernel, RowVecs(X))

    # Add nugget to diagonal
    _add_nugget_diagonal!(K, g)

    return K
end

"""
    _cross_covariance(X, XX, d)

Compute cross-covariance between training X and test XX.

Returns k where k[i,j] = exp(-||X[i,:] - XX[j,:]||^2 / d)
Size: (n_train x n_test)

Uses KernelFunctions.jl for optimized kernel matrix computation.
"""
function _cross_covariance(X::Matrix{T}, XX::Matrix{T}, d::T) where {T}
    # KernelFunctions uses: exp(-||x-y||²/(2ℓ²))
    # Our formula uses: exp(-||x-y||²/d)
    # Mapping: d = 2ℓ², so ℓ = sqrt(d/2)
    ℓ = sqrt(d / 2)
    kernel = with_lengthscale(SqExponentialKernel(), ℓ)

    # Compute cross-covariance using KernelFunctions.jl
    # Returns matrix where k[i,j] is kernel between X[i,:] and XX[j,:]
    return kernelmatrix(kernel, RowVecs(X), RowVecs(XX))
end

"""
    new_gp(X, Z, d, g)

Create a new Gaussian Process model.

# Arguments
- `X::Matrix`: n x m design matrix (n observations, m dimensions)
- `Z::Vector`: n response values
- `d::Real`: lengthscale parameter
- `g::Real`: nugget parameter

# Returns
- `GP`: Gaussian Process model
"""
function new_gp(X::Matrix{T}, Z::Vector{T}, d::Real, g::Real) where {T<:Real}
    n = size(X, 1)
    @assert length(Z) == n "Z must have same length as number of rows in X"
    @assert d > 0 "lengthscale d must be positive"
    @assert g > 0 "nugget g must be positive"

    d_T = T(d)
    g_T = T(g)

    # Allocate and compute covariance matrix
    K = Matrix{T}(undef, n, n)
    _compute_covariance!(K, X, d_T, g_T)

    # Cholesky factorization
    chol = cholesky(Symmetric(K))

    # Compute K inverse (cached for gradient computation)
    Ki = inv(chol)

    # Compute Ki * Z
    KiZ = chol \ Z

    # Compute phi = Z' * Ki * Z
    phi = dot(Z, KiZ)

    # Log determinant
    ldetK = _compute_logdet_chol(chol)

    return GP{T}(copy(X), copy(Z), K, chol, Ki, KiZ, d_T, g_T, phi, ldetK)
end

"""
    pred_gp(gp, XX; lite=true)

Make predictions at test locations XX.

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

    # Cross-covariance: k(X, XX)
    k = _cross_covariance(gp.X, XX, gp.d)

    # Mean prediction: mu = k' * Ki * Z
    mean = k' * gp.KiZ

    # Variance computation: Ki * k
    Kik = gp.chol \ k

    if lite
        # Diagonal variances only
        s2 = _compute_lite_variance(k, Kik, gp.phi, n, gp.g)
        return GPPrediction{T}(mean, s2, n)
    else
        # Full covariance matrix
        # Sigma = (phi/n) * (K** + g*I - k' * Ki * k)
        scale = gp.phi / n

        # Compute K(XX, XX) + g*I (test-test covariance)
        K_test = Matrix{T}(undef, n_test, n_test)
        _compute_covariance!(K_test, XX, gp.d, gp.g)

        # Sigma = scale * (K_test - k' * Kik)
        Sigma = scale .* (K_test - k' * Kik)

        return GPPredictionFull{T}(mean, Sigma, n)
    end
end

"""
    llik_gp(gp)

Compute the log-likelihood of the GP model.

Uses the formula from R laGP (proportional to likelihood):
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

Returns: NamedTuple (dllg, dlld) where both are scalars.
Based on R's laGP dllikGP.

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
    # dK/dg = I (derivative of nugget term)
    # dll/dg = -0.5 * tr(Ki) + 0.5 * (n/phi) * (KiZ' * KiZ)
    dllg = zero(T)
    if dg
        tr_Ki = tr(Ki)
        dllg = -T(0.5) * tr_Ki + T(0.5) * (n / gp.phi) * dot(KiZ, KiZ)
    end

    # Gradient w.r.t. lengthscale d
    # Optimized: compute inline to avoid O(n³) matrix multiplication
    dlld = zero(T)
    if dd
        X = gp.X
        K = gp.K
        d_sq = gp.d^2
        tr_term = zero(T)
        quad_term = zero(T)

        # dK[i,j]/dd = K[i,j] * ||x_i - x_j||² / d²  (for i ≠ j)
        @inbounds for j in 1:n
            for i in 1:n
                if i != j
                    dist_sq = zero(T)
                    for m in axes(X, 2)
                        diff = X[i, m] - X[j, m]
                        dist_sq += diff * diff
                    end
                    dK_ij = K[i, j] * dist_sq / d_sq
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
        changed = true
    end

    if !isnothing(g) && T(g) != gp.g
        gp.g = T(g)
        changed = true
    end

    if changed
        # Recompute covariance matrix
        _compute_covariance!(gp.K, gp.X, gp.d, gp.g)

        # Recompute Cholesky
        gp.chol = cholesky(Symmetric(gp.K))

        # Recompute K inverse (assign directly)
        gp.Ki = inv(gp.chol)

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
    _compute_covariance_sep!(K, X, d, g)

Compute the covariance matrix K in-place using separable kernel.

Uses the separable (anisotropic) squared-exponential kernel:
    K[i,j] = exp(-Σ_k (x_i[k] - x_j[k])² / d[k])

with nugget g added to the diagonal. Each dimension has its own lengthscale.

Uses KernelFunctions.jl with ARD (Automatic Relevance Determination) for
optimized kernel matrix computation.
"""
function _compute_covariance_sep!(K::Matrix{T}, X::Matrix{T},
                                   d::Vector{T}, g::T) where {T}
    # KernelFunctions with ARDTransform(v) applies: x -> v .* x
    # SqExponentialKernel: exp(-||x-y||²/2)
    # Combined: exp(-0.5 * Σ_k v[k]² * (x[k] - y[k])²)
    # Our formula: exp(-Σ_k (x[k] - y[k])² / d[k])
    # Matching: 0.5 * v[k]² = 1/d[k], so v[k] = sqrt(2/d[k])
    scales = sqrt.(2 ./ d)
    kernel = SqExponentialKernel() ∘ ARDTransform(scales)

    # Compute kernel matrix using optimized KernelFunctions.jl
    kernelmatrix!(K, kernel, RowVecs(X))

    # Add nugget to diagonal
    _add_nugget_diagonal!(K, g)

    return K
end

"""
    _cross_covariance_sep(X, XX, d)

Compute cross-covariance between training X and test XX using separable kernel.

Returns k where k[i,j] = exp(-Σ_m (X[i,m] - XX[j,m])² / d[m])
Size: (n_train x n_test)

Uses KernelFunctions.jl with ARD for optimized computation.
"""
function _cross_covariance_sep(X::Matrix{T}, XX::Matrix{T}, d::Vector{T}) where {T}
    # KernelFunctions with ARDTransform(v) applies: x -> v .* x
    # SqExponentialKernel: exp(-||x-y||²/2)
    # Combined: exp(-0.5 * Σ_k v[k]² * (x[k] - y[k])²)
    # Our formula: exp(-Σ_k (x[k] - y[k])² / d[k])
    # Matching: 0.5 * v[k]² = 1/d[k], so v[k] = sqrt(2/d[k])
    scales = sqrt.(2 ./ d)
    kernel = SqExponentialKernel() ∘ ARDTransform(scales)

    # Compute cross-covariance using KernelFunctions.jl
    return kernelmatrix(kernel, RowVecs(X), RowVecs(XX))
end

"""
    new_gp_sep(X, Z, d, g)

Create a new separable Gaussian Process model.

# Arguments
- `X::Matrix`: n x m design matrix (n observations, m dimensions)
- `Z::Vector`: n response values
- `d::Vector`: lengthscale parameters (m elements, one per dimension)
- `g::Real`: nugget parameter

# Returns
- `GPsep`: Separable Gaussian Process model
"""
function new_gp_sep(X::Matrix{T}, Z::Vector{T}, d::Vector{<:Real}, g::Real) where {T<:Real}
    n, m = size(X)
    @assert length(Z) == n "Z must have same length as number of rows in X"
    @assert length(d) == m "d must have same length as number of columns in X"
    @assert all(d .> 0) "all lengthscales in d must be positive"
    @assert g > 0 "nugget g must be positive"

    d_T = T.(d)
    g_T = T(g)

    # Allocate and compute covariance matrix
    K = Matrix{T}(undef, n, n)
    _compute_covariance_sep!(K, X, d_T, g_T)

    # Cholesky factorization
    chol = cholesky(Symmetric(K))

    # Compute K inverse (cached for gradient computation)
    Ki = inv(chol)

    # Compute Ki * Z
    KiZ = chol \ Z

    # Compute phi = Z' * Ki * Z
    phi = dot(Z, KiZ)

    # Log determinant
    ldetK = _compute_logdet_chol(chol)

    return GPsep{T}(copy(X), copy(Z), K, chol, Ki, KiZ, d_T, g_T, phi, ldetK)
end

"""
    update_gp_sep!(gp; d=nothing, g=nothing)

Update separable GP hyperparameters and recompute internal quantities.

# Arguments
- `gp::GPsep`: Separable Gaussian Process model
- `d::Vector{Real}`: new lengthscales (optional, must be same length as gp.d)
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
            changed = true
        end
    end

    if !isnothing(g) && T(g) != gp.g
        gp.g = T(g)
        changed = true
    end

    if changed
        # Recompute covariance matrix
        _compute_covariance_sep!(gp.K, gp.X, gp.d, gp.g)

        # Recompute Cholesky
        gp.chol = cholesky(Symmetric(gp.K))

        # Recompute K inverse (assign directly, no copy)
        gp.Ki = inv(gp.chol)

        # Recompute KiZ using efficient triangular solve
        gp.KiZ .= gp.chol \ gp.Z

        # Recompute phi
        gp.phi = dot(gp.Z, gp.KiZ)

        # Recompute log determinant
        gp.ldetK = _compute_logdet_chol(gp.chol)
    end

    return gp
end

"""
    pred_gp_sep(gp, XX; lite=true)

Make predictions at test locations XX using separable GP.

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

    # Cross-covariance: k(X, XX)
    k = _cross_covariance_sep(gp.X, XX, gp.d)

    # Mean prediction: mu = k' * Ki * Z
    mean = k' * gp.KiZ

    # Variance computation: Ki * k
    Kik = gp.chol \ k

    if lite
        # Diagonal variances only
        s2 = _compute_lite_variance(k, Kik, gp.phi, n, gp.g)
        return GPPrediction{T}(mean, s2, n)
    else
        # Full covariance matrix
        # Sigma = (phi/n) * (K** + g*I - k' * Ki * k)
        scale = gp.phi / n

        # Compute K(XX, XX) + g*I (test-test covariance)
        K_test = Matrix{T}(undef, n_test, n_test)
        _compute_covariance_sep!(K_test, XX, gp.d, gp.g)

        # Sigma = scale * (K_test - k' * Kik)
        Sigma = scale .* (K_test - k' * Kik)

        return GPPredictionFull{T}(mean, Sigma, n)
    end
end

"""
    llik_gp_sep(gp)

Compute the log-likelihood of the separable GP model.

Uses the formula from R laGP (proportional to likelihood):
    llik = -0.5 * (n * log(0.5 * phi) + ldetK)

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

Returns: NamedTuple (dllg, dlld) where dllg is scalar and dlld is Vector{T}
Based on R's laGP dllikGPsep.

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
    # dK/dg = I (derivative of nugget term)
    # dll/dg = -0.5 * tr(Ki) + 0.5 * (n/phi) * (KiZ' * KiZ)
    dllg = zero(T)
    if dg
        tr_Ki = zero(T)
        @inbounds for i in 1:n
            tr_Ki += Ki[i, i]
        end
        dllg = -T(0.5) * tr_Ki + T(0.5) * (n / gp.phi) * dot(KiZ, KiZ)
    end

    # Gradient w.r.t. each lengthscale d[k]
    # Iterate over dimensions separately for better cache locality on Ki and K
    dlld = zeros(T, m)
    if dd
        X = gp.X
        K = gp.K
        phi_factor = T(0.5) * (n / gp.phi)

        for k in 1:m
            inv_dk_sq = one(T) / (gp.d[k] * gp.d[k])
            tr_term = zero(T)
            quad_term = zero(T)

            # Compute tr(Ki * dK/d(d[k])) and KiZ' * dK/d(d[k]) * KiZ
            @inbounds for j in 1:n
                KiZ_j = KiZ[j]
                for i in 1:n
                    if i != j
                        diff = X[i, k] - X[j, k]
                        dK_ij = K[i, j] * diff * diff * inv_dk_sq
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

# ============================================================================
# AbstractGPs-backed GP functions (GPModel)
# ============================================================================

"""
    new_gp_model(X, Z, d, g)

Create a new Gaussian Process model using AbstractGPs.jl backend.

# Arguments
- `X::Matrix`: n x m design matrix (n observations, m dimensions)
- `Z::Vector`: n response values
- `d::Real`: lengthscale parameter (laGP parameterization)
- `g::Real`: nugget parameter

# Returns
- `GPModel`: Gaussian Process model backed by AbstractGPs
"""
function new_gp_model(X::Matrix{T}, Z::Vector{T}, d::Real, g::Real) where {T<:Real}
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

    return GPModel{T,typeof(kernel)}(copy(X), copy(Z), kernel, chol, KiZ, d_T, g_T, phi, ldetK)
end

"""
    pred_gp_model(gp, XX; lite=true)

Make predictions at test locations XX using AbstractGPs-backed GP.

# Arguments
- `gp::GPModel`: Gaussian Process model
- `XX::Matrix`: test locations (n_test x m)
- `lite::Bool`: if true, return only diagonal variances

# Returns
- `GPPrediction`: prediction results with mean, s2, and df
"""
function pred_gp_model(gp::GPModel{T}, XX::Matrix{T}; lite::Bool=true) where {T}
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
    llik_gp_model(gp)

Compute the log-likelihood of the GPModel.

Uses the concentrated likelihood formula from R laGP:
    llik = -0.5 * (n * log(0.5 * phi) + ldetK)

# Arguments
- `gp::GPModel`: Gaussian Process model

# Returns
- `Real`: log-likelihood value
"""
function llik_gp_model(gp::GPModel{T}) where {T}
    return _concentrated_llik(length(gp.Z), gp.phi, gp.ldetK)
end

"""
    dllik_gp_model(gp; dg=true, dd=true)

Compute gradient of log-likelihood w.r.t. d (lengthscale) and g (nugget).

# Arguments
- `gp::GPModel`: Gaussian Process model
- `dg::Bool`: compute gradient w.r.t. nugget g (default: true)
- `dd::Bool`: compute gradient w.r.t. lengthscale d (default: true)

# Returns
- `NamedTuple`: (dllg=..., dlld=...) gradients
"""
function dllik_gp_model(gp::GPModel{T}; dg::Bool=true, dd::Bool=true) where {T}
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
    update_gp_model!(gp; d=nothing, g=nothing)

Update GPModel hyperparameters and recompute internal quantities.

# Arguments
- `gp::GPModel`: Gaussian Process model
- `d::Real`: new lengthscale (optional)
- `g::Real`: new nugget (optional)
"""
function update_gp_model!(gp::GPModel{T}; d::Union{Nothing,Real}=nothing,
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
# AbstractGPs-backed Separable GP functions (GPModelSep)
# ============================================================================

"""
    new_gp_model_sep(X, Z, d, g)

Create a new separable Gaussian Process model using AbstractGPs.jl backend.

# Arguments
- `X::Matrix`: n x m design matrix (n observations, m dimensions)
- `Z::Vector`: n response values
- `d::Vector`: lengthscale parameters (m elements, one per dimension)
- `g::Real`: nugget parameter

# Returns
- `GPModelSep`: Separable Gaussian Process model backed by AbstractGPs
"""
function new_gp_model_sep(X::Matrix{T}, Z::Vector{T}, d::Vector{<:Real}, g::Real) where {T<:Real}
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

    return GPModelSep{T,typeof(kernel)}(copy(X), copy(Z), kernel, chol, KiZ, d_T, g_T, phi, ldetK)
end

"""
    pred_gp_model_sep(gp, XX; lite=true)

Make predictions at test locations XX using AbstractGPs-backed separable GP.

# Arguments
- `gp::GPModelSep`: Separable Gaussian Process model
- `XX::Matrix`: test locations (n_test x m)
- `lite::Bool`: if true, return only diagonal variances

# Returns
- `GPPrediction`: prediction results with mean, s2, and df
"""
function pred_gp_model_sep(gp::GPModelSep{T}, XX::Matrix{T}; lite::Bool=true) where {T}
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
    llik_gp_model_sep(gp)

Compute the log-likelihood of the GPModelSep.

# Arguments
- `gp::GPModelSep`: Separable Gaussian Process model

# Returns
- `Real`: log-likelihood value
"""
function llik_gp_model_sep(gp::GPModelSep{T}) where {T}
    return _concentrated_llik(length(gp.Z), gp.phi, gp.ldetK)
end

"""
    dllik_gp_model_sep(gp; dg=true, dd=true)

Compute gradient of log-likelihood w.r.t. d (lengthscales) and g (nugget).

# Arguments
- `gp::GPModelSep`: Separable Gaussian Process model
- `dg::Bool`: compute gradient w.r.t. nugget g (default: true)
- `dd::Bool`: compute gradient w.r.t. lengthscales d (default: true)

# Returns
- `NamedTuple`: (dllg=..., dlld=...) gradients
"""
function dllik_gp_model_sep(gp::GPModelSep{T}; dg::Bool=true, dd::Bool=true) where {T}
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
    update_gp_model_sep!(gp; d=nothing, g=nothing)

Update GPModelSep hyperparameters and recompute internal quantities.

# Arguments
- `gp::GPModelSep`: Separable Gaussian Process model
- `d::Vector{Real}`: new lengthscales (optional)
- `g::Real`: new nugget (optional)
"""
function update_gp_model_sep!(gp::GPModelSep{T}; d::Union{Nothing,Vector{<:Real}}=nothing,
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
