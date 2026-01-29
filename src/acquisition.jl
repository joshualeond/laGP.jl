# Acquisition functions for sequential design

using LinearAlgebra: mul!, Symmetric
using LoopVectorization: @turbo

# Workspace to avoid allocations in tight ALC loops
mutable struct ALCWorkspace{T}
    k_ref_vec::Vector{T}
    k_ref::Matrix{T}
    kx::Vector{T}
    Kikx::Vector{T}
    gvec::Vector{T}
    kxy::Vector{T}
end

ALCWorkspace{T}() where {T} = ALCWorkspace{T}(T[], Matrix{T}(undef, 0, 0), T[], T[], T[], T[])

function _ensure_alc_workspace!(work::ALCWorkspace{T}, n::Int, n_ref::Int) where {T}
    if length(work.kx) != n
        resize!(work.kx, n)
        resize!(work.Kikx, n)
        resize!(work.gvec, n)
        resize!(work.k_ref_vec, n)
    end
    if n_ref > 1
        if size(work.k_ref, 1) != n || size(work.k_ref, 2) != n_ref
            work.k_ref = Matrix{T}(undef, n, n_ref)
        end
        if length(work.kxy) != n_ref
            resize!(work.kxy, n_ref)
        end
    elseif length(work.kxy) != 0
        resize!(work.kxy, 0)
    end
    return work
end

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

@inline function _matvec_small!(y::AbstractVector{T}, A::AbstractMatrix{T},
                                x::AbstractVector{T}) where {T}
    n = size(A, 1)
    @inbounds for i in 1:n
        acc = zero(T)
        @turbo for j in 1:n
            acc += A[i, j] * x[j]
        end
        y[i] = acc
    end
    return y
end

# ============================================================================
# Fused kernel computation helpers (avoid intermediate distance matrices)
# ============================================================================

"""
    _compute_kernel_isotropic!(K, X1, X2, d)

Compute isotropic squared exponential kernel between rows of X1 and X2.
K[i,j] = exp(-||X1[i,:] - X2[j,:]||² / d)
"""
function _compute_kernel_isotropic!(K::AbstractMatrix{T}, X1::AbstractMatrix{T},
                                    X2::AbstractMatrix{T}, d::T) where {T}
    n1, m = size(X1)
    n2 = size(X2, 1)
    inv_d = one(T) / d
    @inbounds for j in 1:n2
        for i in 1:n1
            dist_sq = zero(T)
            @turbo for k in 1:m
                diff = X1[i, k] - X2[j, k]
                dist_sq += diff * diff
            end
            K[i, j] = exp(-dist_sq * inv_d)
        end
    end
    return K
end

"""
    _compute_kernel_vector!(k, X, x, d)

Compute isotropic squared exponential kernel between rows of X and vector x.
k[i] = exp(-||X[i,:] - x||² / d)
"""
function _compute_kernel_vector!(k::AbstractVector{T}, X::AbstractMatrix{T},
                                 x::AbstractVector{T}, d::T) where {T}
    n, m = size(X)
    inv_d = one(T) / d
    @inbounds for i in 1:n
        dist_sq = zero(T)
        @turbo for j in 1:m
            diff = X[i, j] - x[j]
            dist_sq += diff * diff
        end
        k[i] = exp(-dist_sq * inv_d)
    end
    return k
end

"""
    _compute_kernel_vector_row!(k, X1, X2, row, d)

Compute isotropic kernel between rows of X1 and a specific row of X2.
k[i] = exp(-||X1[i,:] - X2[row,:]||² / d)
"""
function _compute_kernel_vector_row!(k::AbstractVector{T}, X1::AbstractMatrix{T},
                                     X2::AbstractMatrix{T}, row::Int, d::T) where {T}
    n, m = size(X1)
    inv_d = one(T) / d
    @inbounds for i in 1:n
        dist_sq = zero(T)
        @turbo for j in 1:m
            diff = X1[i, j] - X2[row, j]
            dist_sq += diff * diff
        end
        k[i] = exp(-dist_sq * inv_d)
    end
    return k
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
    _alc_gp_idx!(alc, gp, Xcand, cand_idx, Xref)

Internal low-allocation ALC kernel that computes criteria for rows in `Xcand`
specified by `cand_idx`. Results are written into `alc`.
"""
function _alc_gp_idx!(alc::AbstractVector{T}, gp::GP{T},
                      Xcand::AbstractMatrix{T},
                      cand_idx::AbstractVector{<:Integer},
                      Xref::AbstractMatrix{T}) where {T}
    work = ALCWorkspace{T}()
    return _alc_gp_idx!(alc, gp, Xcand, cand_idx, Xref, work)
end

function _alc_gp_idx!(alc::AbstractVector{T}, gp::GP{T},
                      Xcand::AbstractMatrix{T},
                      cand_idx::AbstractVector{<:Integer},
                      Xref::AbstractMatrix{T},
                      work::ALCWorkspace{T}) where {T}
    n = size(gp.X, 1)
    n_ref = size(Xref, 1)
    df = T(n)

    _ensure_alc_workspace!(work, n, n_ref)

    # Pre-compute k(X, Xref)
    if n_ref == 1
        k_ref_vec = work.k_ref_vec
        _compute_kernel_vector_row!(k_ref_vec, gp.X, Xref, 1, gp.d)
    else
        k_ref = work.k_ref
        _compute_kernel_isotropic!(k_ref, gp.X, Xref, gp.d)
    end

    # Use cached Ki (already computed and stored in GP struct)
    Ki = Symmetric(gp.Ki)

    # Workspace vectors
    kx = work.kx
    Kikx = work.Kikx
    gvec = work.gvec
    kxy = work.kxy

    df_rat = df / (df - 2)
    inv_d = one(T) / gp.d
    m = size(gp.X, 2)

    @inbounds for (i, idx) in enumerate(cand_idx)
        # kx = k(X, Xcand[idx,:])
        _compute_kernel_vector_row!(kx, gp.X, Xcand, idx, gp.d)

        # Kikx = Ki * kx (in-place)
        if n <= 64
            _matvec_small!(Kikx, gp.Ki, kx)
        else
            mul!(Kikx, Ki, kx)
        end

        # mui = 1 + g - kx' * Ki * kx
        dot_kx = zero(T)
        @turbo for j in 1:n
            dot_kx += kx[j] * Kikx[j]
        end
        mui = one(T) + gp.g - dot_kx

        if mui <= sqrt(eps(T))
            alc[i] = T(-Inf)
            continue
        end

        inv_mui = one(T) / mui
        @turbo for j in 1:n
            gvec[j] = -Kikx[j] * inv_mui
        end

        if n_ref == 1
            # kxy scalar
            dist_sq = zero(T)
            @turbo for j in 1:m
                diff = Xref[1, j] - Xcand[idx, j]
                dist_sq += diff * diff
            end
            kxy_val = exp(-dist_sq * inv_d)

            kg = zero(T)
            k_ref_vec = work.k_ref_vec
            @turbo for j in 1:n
                kg += k_ref_vec[j] * gvec[j]
            end
            ktKikx = kg * kg * mui + 2 * kg * kxy_val + kxy_val * kxy_val * inv_mui
            alc[i] = (gp.phi / df) * df_rat * ktKikx
        else
            # kxy = k(Xcand[idx,:], Xref)
            _compute_kernel_vector_row!(kxy, Xref, Xcand, idx, gp.d)
            k_ref = work.k_ref
            alc_sum = _alc_inner_sum(k_ref, gvec, kxy, mui, inv_mui, gp.phi, df, n, n_ref)
            alc[i] = alc_sum * df_rat / n_ref
        end
    end

    return alc
end

"""
    alc_gp(gp, Xcand, Xref)

Compute Active Learning Cohn (ALC) acquisition values.

ALC measures expected variance reduction at reference points Xref
if we were to add each candidate point from Xcand to the design.
"""
function alc_gp(gp::GP{T}, Xcand::AbstractMatrix{T}, Xref::AbstractMatrix{T}) where {T}
    n_cand = size(Xcand, 1)
    alc = Vector{T}(undef, n_cand)
    _alc_gp_idx!(alc, gp, Xcand, 1:n_cand, Xref)
    return alc
end

"""
    _mspe_gp_idx!(mspe, gp, Xcand, cand_idx, Xref)

Internal low-allocation MSPE kernel for candidate indices.
"""
function _mspe_gp_idx!(mspe::AbstractVector{T}, gp::GP{T},
                       Xcand::AbstractMatrix{T},
                       cand_idx::AbstractVector{<:Integer},
                       Xref::AbstractMatrix{T}) where {T}
    n = size(gp.X, 1)
    df = T(n)

    # Reuse mspe buffer to store ALC first
    _alc_gp_idx!(mspe, gp, Xcand, cand_idx, Xref)

    # Predict at reference locations
    pred_ref = pred_gp(gp, Xref; lite=true)
    s2avg = mean(pred_ref.s2)

    # Compute MSPE scaling factors
    dnp = (df + one(T)) / (df - one(T))
    dnp2 = dnp * (df - 2) / df

    # MSPE = dnp * s2avg - dnp2 * alc
    @inbounds for i in 1:length(cand_idx)
        mspe[i] = dnp * s2avg - dnp2 * mspe[i]
    end

    return mspe
end

function _mspe_gp_idx!(mspe::AbstractVector{T}, gp::GP{T},
                       Xcand::AbstractMatrix{T},
                       cand_idx::AbstractVector{<:Integer},
                       Xref::AbstractMatrix{T},
                       work::ALCWorkspace{T}) where {T}
    n = size(gp.X, 1)
    df = T(n)

    # Reuse mspe buffer to store ALC first
    _alc_gp_idx!(mspe, gp, Xcand, cand_idx, Xref, work)

    # Predict at reference locations
    pred_ref = pred_gp(gp, Xref; lite=true)
    s2avg = mean(pred_ref.s2)

    # Compute MSPE scaling factors
    dnp = (df + one(T)) / (df - one(T))
    dnp2 = dnp * (df - 2) / df

    # MSPE = dnp * s2avg - dnp2 * alc
    @inbounds for i in 1:length(cand_idx)
        mspe[i] = dnp * s2avg - dnp2 * mspe[i]
    end

    return mspe
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
function mspe_gp(gp::GP{T}, Xcand::AbstractMatrix{T}, Xref::AbstractMatrix{T}) where {T}
    n_cand = size(Xcand, 1)
    mspe = Vector{T}(undef, n_cand)
    _mspe_gp_idx!(mspe, gp, Xcand, 1:n_cand, Xref)
    return mspe
end
