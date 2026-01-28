# AbstractGPs adapter for laGP.jl
#
# Provides conversion functions between laGP parameterization and
# the JuliaGaussianProcesses ecosystem (AbstractGPs.jl, KernelFunctions.jl).

using AbstractGPs
using KernelFunctions: SqExponentialKernel, with_lengthscale, ARDTransform, kernelmatrix, RowVecs

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
