# Acquisition functions for sequential design

using KernelFunctions: kernelmatrix, RowVecs
using LinearAlgebra: mul!, Symmetric
using LoopVectorization: @turbo

# ============================================================================
# SIMD-optimized batched kernel computation
# ============================================================================

"""
    _compute_squared_distances_batched!(D_sq, X1, X2)

Compute squared distances between all rows of X1 and X2 using SIMD.
D_sq[i,j] = ||X1[i,:] - X2[j,:]||²

# Arguments
- `D_sq::Matrix`: pre-allocated output matrix (n1 x n2)
- `X1::Matrix`: first set of points (n1 x m)
- `X2::Matrix`: second set of points (n2 x m)
"""
function _compute_squared_distances_batched!(D_sq::Matrix{T}, X1::Matrix{T}, X2::Matrix{T}) where {T}
    n1, m = size(X1)
    n2 = size(X2, 1)
    @inbounds for j in 1:n2
        for i in 1:n1
            dist_sq = zero(T)
            @turbo for k in 1:m
                diff = X1[i, k] - X2[j, k]
                dist_sq += diff * diff
            end
            D_sq[i, j] = dist_sq
        end
    end
    return D_sq
end

"""
    _apply_kernel_isotropic!(K, D_sq, d)

Apply isotropic squared exponential kernel in-place using SIMD.
K[i,j] = exp(-D_sq[i,j] / d)

# Arguments
- `K::Matrix`: output matrix (modified in-place)
- `D_sq::Matrix`: squared distance matrix
- `d::Real`: lengthscale parameter (laGP parameterization)
"""
function _apply_kernel_isotropic!(K::Matrix{T}, D_sq::Matrix{T}, d::T) where {T}
    inv_d = one(T) / d
    @turbo for idx in eachindex(K)
        K[idx] = exp(-D_sq[idx] * inv_d)
    end
    return K
end

# ============================================================================
# ALC inner computation
# ============================================================================

"""
    _alc_inner_sum(k_ref, gvec, kxy, mui, inv_mui, phi, df, n, n_ref)

Compute the ALC sum using SIMD-optimized loops.
"""
function _alc_inner_sum(k_ref::Matrix{T}, gvec::Vector{T}, kxy::AbstractVector{T},
                        mui::T, inv_mui::T, phi::T, df::T, n::Int, n_ref::Int) where {T}
    alc_sum = zero(T)
    @inbounds for j in 1:n_ref
        kg = zero(T)
        @turbo for k in 1:n
            kg += k_ref[k, j] * gvec[k]
        end
        kxy_j = kxy[j]
        ktKikx_j = kg * kg * mui + 2 * kg * kxy_j + kxy_j * kxy_j * inv_mui
        alc_sum += phi * ktKikx_j / df
    end
    return alc_sum
end

"""
    alc_gp(gp, Xcand, Xref)

Compute Active Learning Cohn (ALC) acquisition values.

ALC measures expected variance reduction at reference points Xref
if we were to add each candidate point from Xcand to the design.

# Arguments
- `gp::GP`: Gaussian Process model
- `Xcand::Matrix`: candidate points (n_cand x m)
- `Xref::Matrix`: reference points (n_ref x m)

# Returns
- `Vector`: ALC values for each candidate point
"""
function alc_gp(gp::GP{T}, Xcand::Matrix{T}, Xref::Matrix{T}) where {T}
    n = size(gp.X, 1)
    n_cand = size(Xcand, 1)
    n_ref = size(Xref, 1)
    df = T(n)

    # Pre-compute k(X, Xref) using kernel
    k_ref = kernelmatrix(gp.kernel, RowVecs(gp.X), RowVecs(Xref))

    # Use cached Ki (already computed and stored in GP struct)
    # Wrap in Symmetric for efficient BLAS operations (uses DSYMV instead of DGEMV)
    Ki = Symmetric(gp.Ki)

    # ========================================================================
    # Batched kernel computation using SIMD
    # Pre-compute all kernel values outside the loop
    # ========================================================================

    # K_X_cand[i,j] = k(X[i,:], Xcand[j,:]) - kernel between training and candidates
    D_sq_X_cand = Matrix{T}(undef, n, n_cand)
    _compute_squared_distances_batched!(D_sq_X_cand, gp.X, Xcand)
    K_X_cand = similar(D_sq_X_cand)
    _apply_kernel_isotropic!(K_X_cand, D_sq_X_cand, gp.d)

    # K_cand_ref[i,j] = k(Xcand[i,:], Xref[j,:]) - kernel between candidates and reference
    D_sq_cand_ref = Matrix{T}(undef, n_cand, n_ref)
    _compute_squared_distances_batched!(D_sq_cand_ref, Xcand, Xref)
    K_cand_ref = similar(D_sq_cand_ref)
    _apply_kernel_isotropic!(K_cand_ref, D_sq_cand_ref, gp.d)

    # Allocate output
    alc = Vector{T}(undef, n_cand)

    # Pre-allocate workspace vectors
    Kikx = Vector{T}(undef, n)
    gvec = Vector{T}(undef, n)

    # Scaling factor
    df_rat = df / (df - 2)

    for i in 1:n_cand
        # kx = k(X, Xcand[i,:]) - use column view from pre-computed matrix
        kx = @view K_X_cand[:, i]

        # Kikx = Ki * kx (in-place)
        mul!(Kikx, Ki, kx)

        # mui = 1 + g - kx' * Ki * kx
        mui = one(T) + gp.g - dot(kx, Kikx)

        # Skip if numerical problems
        if mui <= sqrt(eps(T))
            alc[i] = T(-Inf)
            continue
        end

        # gvec = -Kikx / mui (in-place)
        inv_mui = one(T) / mui
        @inbounds for j in 1:n
            gvec[j] = -Kikx[j] * inv_mui
        end

        # kxy = k(Xcand[i,:], Xref) - use row view from pre-computed matrix
        kxy = @view K_cand_ref[i, :]

        # Compute ALC contribution using SIMD-optimized inner sum
        alc_sum = _alc_inner_sum(k_ref, gvec, kxy, mui, inv_mui, gp.phi, df, n, n_ref)

        alc[i] = alc_sum * df_rat / n_ref
    end

    return alc
end

"""
    mspe_gp(gp, Xcand, Xref)

Compute Mean Squared Prediction Error (MSPE) acquisition values.

MSPE is related to ALC and includes the current prediction variance.

# Arguments
- `gp::GP`: Gaussian Process model
- `Xcand::Matrix`: candidate points (n_cand x m)
- `Xref::Matrix`: reference points (n_ref x m)

# Returns
- `Vector`: MSPE values for each candidate point
"""
function mspe_gp(gp::GP{T}, Xcand::Matrix{T}, Xref::Matrix{T}) where {T}
    n = size(gp.X, 1)
    n_cand = size(Xcand, 1)
    df = T(n)

    # Compute ALC first
    alc_vals = alc_gp(gp, Xcand, Xref)

    # Predict at reference locations
    pred_ref = pred_gp(gp, Xref; lite=true)
    s2avg = mean(pred_ref.s2)

    # Compute MSPE scaling factors
    dnp = (df + one(T)) / (df - one(T))
    dnp2 = dnp * (df - 2) / df

    # MSPE = dnp * s2avg - dnp2 * alc
    mspe = Vector{T}(undef, n_cand)
    for i in 1:n_cand
        mspe[i] = dnp * s2avg - dnp2 * alc_vals[i]
    end

    return mspe
end
