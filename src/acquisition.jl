# Acquisition functions for sequential design

using KernelFunctions: kernelmatrix, RowVecs
using LinearAlgebra: mul!

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

    # Get Ki from Cholesky
    Ki = inv(gp.chol)

    # Allocate output
    alc = Vector{T}(undef, n_cand)

    # Pre-allocate workspace vectors
    Kikx = Vector{T}(undef, n)
    gvec = Vector{T}(undef, n)

    # Scaling factor
    df_rat = df / (df - 2)

    for i in 1:n_cand
        # kx = k(X, x_cand) - use view to avoid allocation
        x_cand = @view Xcand[i:i, :]
        kx = vec(kernelmatrix(gp.kernel, RowVecs(gp.X), RowVecs(x_cand)))

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

        # kxy = k(x_cand, Xref)
        kxy = vec(kernelmatrix(gp.kernel, RowVecs(x_cand), RowVecs(Xref)))

        # Compute ALC contribution
        alc_sum = zero(T)
        @inbounds for j in 1:n_ref
            kg = zero(T)
            for k in 1:n
                kg += k_ref[k, j] * gvec[k]
            end
            ktKikx_j = kg * kg * mui + 2 * kg * kxy[j] + kxy[j] * kxy[j] * inv_mui
            alc_sum += gp.phi * ktKikx_j / df
        end

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
