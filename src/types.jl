# Types for laGP.jl

"""
    GPPrediction{T<:Real}

Result of GP prediction.

# Fields
- `mean::Vector{T}`: predicted mean values
- `s2::Vector{T}`: predicted variances (if lite=true) or full covariance
- `df::Int`: degrees of freedom (n observations)
"""
struct GPPrediction{T<:Real}
    mean::Vector{T}
    s2::Vector{T}
    df::Int
end

"""
    GPPredictionFull{T<:Real}

Result of GP prediction with full covariance matrix.

# Fields
- `mean::Vector{T}`: predicted mean values
- `Sigma::Matrix{T}`: full posterior covariance matrix (n_test x n_test)
- `df::Int`: degrees of freedom (n observations)
"""
struct GPPredictionFull{T<:Real}
    mean::Vector{T}
    Sigma::Matrix{T}
    df::Int
end

"""
    GP{T<:Real, K}

Gaussian Process model backed by AbstractGPs.jl with isotropic squared-exponential kernel.

This type uses AbstractGPs for the posterior computation while preserving
laGP-specific quantities needed for the concentrated likelihood formula.

# Fields
- `X::Matrix{T}`: n x m design matrix (n observations, m dimensions)
- `Z::Vector{T}`: n response values
- `kernel::K`: Kernel from KernelFunctions.jl
- `chol::Cholesky{T}`: Cholesky factorization of K + g*I
- `KiZ::Vector{T}`: K \\ Z (precomputed for prediction)
- `d::T`: lengthscale parameter (laGP parameterization)
- `g::T`: nugget parameter
- `phi::T`: Z' * Ki * Z (used for variance scaling)
- `ldetK::T`: log determinant of K (used for likelihood)

# Notes
The AbstractGPs posterior can be reconstructed from (X, Z, kernel, g) when needed.
We cache the Cholesky and derived quantities for efficient repeated computations.
"""
mutable struct GP{T<:Real, K}
    X::Matrix{T}
    Z::Vector{T}
    kernel::K
    chol::Cholesky{T,Matrix{T}}
    KiZ::Vector{T}
    d::T
    g::T
    phi::T
    ldetK::T
end

"""
    GPsep{T<:Real, K}

Separable Gaussian Process model backed by AbstractGPs.jl with anisotropic kernel.

Uses a vector of lengthscales (one per input dimension) to capture varying
input sensitivities.

# Fields
- `X::Matrix{T}`: n x m design matrix (n observations, m dimensions)
- `Z::Vector{T}`: n response values
- `kernel::K`: ARD kernel from KernelFunctions.jl
- `chol::Cholesky{T}`: Cholesky factorization of K + g*I
- `KiZ::Vector{T}`: K \\ Z (precomputed for prediction)
- `d::Vector{T}`: lengthscale parameters (m elements, one per dimension)
- `g::T`: nugget parameter
- `phi::T`: Z' * Ki * Z (used for variance scaling)
- `ldetK::T`: log determinant of K (used for likelihood)
"""
mutable struct GPsep{T<:Real, K}
    X::Matrix{T}
    Z::Vector{T}
    kernel::K
    chol::Cholesky{T,Matrix{T}}
    KiZ::Vector{T}
    d::Vector{T}
    g::T
    phi::T
    ldetK::T
end

# Union type for any GP model (useful for generic functions)
const AnyGP{T} = Union{GP{T}, GPsep{T}} where T
