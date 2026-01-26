# AbstractGPs adapter for laGP.jl
#
# Provides conversion functions between laGP parameterization and
# the JuliaGaussianProcesses ecosystem (AbstractGPs.jl, KernelFunctions.jl).

using AbstractGPs
using KernelFunctions: SqExponentialKernel, with_lengthscale, ARDTransform, kernelmatrix, RowVecs

# Re-export key AbstractGPs types for convenience
export PosteriorGP

"""
    build_kernel_isotropic(d)

Build an isotropic squared-exponential kernel using laGP parameterization.

laGP kernel: k(x,y) = exp(-||x-y||²/d)
KernelFunctions kernel: k(x,y) = exp(-||x-y||²/(2ℓ²))

Mapping: d = 2ℓ², so ℓ = sqrt(d/2)

# Arguments
- `d::Real`: laGP lengthscale parameter

# Returns
- Kernel from KernelFunctions.jl with appropriate lengthscale
"""
function build_kernel_isotropic(d::Real)
    ℓ = sqrt(d / 2)
    return with_lengthscale(SqExponentialKernel(), ℓ)
end

"""
    build_kernel_separable(d::Vector)

Build a separable (anisotropic) squared-exponential kernel using laGP parameterization.

laGP separable kernel: k(x,y) = exp(-Σ_k (x[k]-y[k])² / d[k])
KernelFunctions with ARDTransform(v): k(x,y) = exp(-0.5 * Σ_k v[k]² * (x[k]-y[k])²)

Matching: 0.5 * v[k]² = 1/d[k], so v[k] = sqrt(2/d[k])

# Arguments
- `d::Vector{Real}`: laGP per-dimension lengthscale parameters

# Returns
- ARD kernel from KernelFunctions.jl
"""
function build_kernel_separable(d::Vector{<:Real})
    scales = sqrt.(2 ./ d)
    return SqExponentialKernel() ∘ ARDTransform(scales)
end

"""
    build_gp_prior(kernel)

Build an AbstractGPs GP prior with the given kernel.

# Arguments
- `kernel`: Kernel from KernelFunctions.jl

# Returns
- GP prior from AbstractGPs.jl
"""
function build_gp_prior(kernel)
    return GP(kernel)
end

"""
    build_finite_gp(X, kernel, g)

Build a FiniteGP at training points X with nugget g.

# Arguments
- `X::Matrix`: Training points (n x m), rows are observations
- `kernel`: Kernel from KernelFunctions.jl
- `g::Real`: Nugget (observation noise variance)

# Returns
- FiniteGP from AbstractGPs.jl
"""
function build_finite_gp(X::Matrix{T}, kernel, g::Real) where {T}
    f = GP(kernel)
    return f(RowVecs(X), T(g))
end

"""
    build_posterior(X, Z, kernel, g)

Build an AbstractGPs posterior given training data.

# Arguments
- `X::Matrix`: Training points (n x m)
- `Z::Vector`: Training responses
- `kernel`: Kernel from KernelFunctions.jl
- `g::Real`: Nugget (observation noise variance)

# Returns
- PosteriorGP from AbstractGPs.jl
"""
function build_posterior(X::Matrix{T}, Z::Vector{T}, kernel, g::Real) where {T}
    fx = build_finite_gp(X, kernel, g)
    return posterior(fx, Z)
end

"""
    extract_cholesky(X, kernel, g)

Compute the Cholesky factorization of the covariance matrix K + g*I.

This is needed for laGP-specific computations (phi, ldetK, KiZ).

# Arguments
- `X::Matrix`: Training points (n x m)
- `kernel`: Kernel from KernelFunctions.jl
- `g::Real`: Nugget (observation noise variance)

# Returns
- Cholesky factorization of K + g*I
"""
function extract_cholesky(X::Matrix{T}, kernel, g::Real) where {T}
    K = kernelmatrix(kernel, RowVecs(X))
    n = size(X, 1)
    # Add nugget to diagonal
    for i in 1:n
        K[i, i] += T(g)
    end
    return cholesky(Symmetric(K))
end

"""
    compute_lagp_quantities(X, Z, kernel, g)

Compute the laGP-specific quantities needed for the concentrated likelihood formula.

Returns phi (Z'*Ki*Z), ldetK (log determinant of K), and KiZ (K\\Z).

# Arguments
- `X::Matrix`: Training points (n x m)
- `Z::Vector`: Training responses
- `kernel`: Kernel from KernelFunctions.jl
- `g::Real`: Nugget

# Returns
- NamedTuple (phi, ldetK, KiZ, chol) where chol is the Cholesky factorization
"""
function compute_lagp_quantities(X::Matrix{T}, Z::Vector{T}, kernel, g::Real) where {T}
    chol = extract_cholesky(X, kernel, g)

    # Compute K⁻¹Z via Cholesky
    KiZ = chol \ Z

    # Compute phi = Z' * Ki * Z
    phi = dot(Z, KiZ)

    # Log determinant: 2 * sum(log(diag(L)))
    ldetK = 2 * sum(log.(diag(chol.L)))

    return (phi=phi, ldetK=ldetK, KiZ=KiZ, chol=chol)
end

"""
    predict_with_lagp_variance(post, XX, phi, n, g)

Make predictions using AbstractGPs posterior but with laGP variance scaling.

AbstractGPs variance: var(post(x)) = 1 + g - k'*Ki*k
laGP variance: (phi/n) * (1 + g - k'*Ki*k)

# Arguments
- `post`: PosteriorGP from AbstractGPs
- `XX::Matrix`: Test points (n_test x m)
- `phi::Real`: Z'*Ki*Z from training
- `n::Int`: Number of training points
- `g::Real`: Nugget

# Returns
- NamedTuple (mean, var) with laGP-scaled variances
"""
function predict_with_lagp_variance(post, XX::Matrix{T}, phi::T, n::Int, g::T) where {T}
    # Get AbstractGPs predictions at test points
    post_xx = post(RowVecs(XX), g)

    # Mean from AbstractGPs
    μ = mean(post_xx)

    # AbstractGPs variance is (1 + g - k'*Ki*k) for our kernel
    # We need to scale by phi/n
    scale = phi / n
    σ² = scale .* var(post_xx)

    return (mean=μ, var=σ²)
end

"""
    crosscov_matrix(post, X_train, X_new, kernel)

Compute cross-covariance matrix between training and new points.

Uses KernelFunctions.jl kernelmatrix for efficient computation.

# Arguments
- `X_train::Matrix`: Training points (n x m)
- `X_new::Matrix`: New points (n_new x m)
- `kernel`: Kernel from KernelFunctions.jl

# Returns
- Matrix (n x n_new) of cross-covariances
"""
function crosscov_matrix(X_train::Matrix{T}, X_new::Matrix{T}, kernel) where {T}
    return kernelmatrix(kernel, RowVecs(X_train), RowVecs(X_new))
end
