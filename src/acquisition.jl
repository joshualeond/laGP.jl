# Acquisition functions for sequential design

using KernelFunctions: kernelmatrix, RowVecs

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

    # Pre-compute k(X, Xref) - covariance between training and reference
    k_ref = _cross_covariance(gp.X, Xref, gp.d)  # n x n_ref

    # Pre-compute Ki for ALC calculation
    Ki = inv(gp.chol)

    # Allocate output
    alc = Vector{T}(undef, n_cand)

    # Scaling factor: df / (df - 2)
    df_rat = df / (df - 2)

    for i in 1:n_cand
        # Extract candidate point
        x_cand = Xcand[i, :]

        # kx = k(X, x_cand) - covariance between training and candidate
        kx = Vector{T}(undef, n)
        for j in 1:n
            dist_sq = zero(T)
            for m in axes(gp.X, 2)
                diff = gp.X[j, m] - x_cand[m]
                dist_sq += diff * diff
            end
            kx[j] = exp(-dist_sq / gp.d)
        end

        # Kikx = Ki * kx (precomputed vector)
        Kikx = Ki * kx

        # mui = 1 + g - kx' * Ki * kx
        mui = one(T) + gp.g - dot(kx, Kikx)

        # Skip if numerical problems
        if mui <= sqrt(eps(T))
            alc[i] = T(-Inf)
            continue
        end

        # gvec = -Kikx / mui
        gvec = -Kikx / mui

        # kxy = k(x_cand, Xref) - covariance between candidate and reference
        kxy = Vector{T}(undef, n_ref)
        for j in 1:n_ref
            dist_sq = zero(T)
            for m in axes(Xref, 2)
                diff = x_cand[m] - Xref[j, m]
                dist_sq += diff * diff
            end
            kxy[j] = exp(-dist_sq / gp.d)
        end

        # Compute ktKikx for each reference point
        # ktKikx[j] = (k[:,j]' * gvec)^2 * mui + 2*(k[:,j]' * gvec)*kxy[j] + kxy[j]^2/mui
        alc_sum = zero(T)
        for j in 1:n_ref
            kg = dot(k_ref[:, j], gvec)
            ktKikx_j = kg^2 * mui + 2 * kg * kxy[j] + kxy[j]^2 / mui

            # ALC contribution: phi * ktKikx / df * df_rat
            alc_sum += gp.phi * ktKikx_j / df
        end

        # Average over reference points and apply scaling
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

    # Predict at reference locations to get s2avg
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

# ============================================================================
# Acquisition functions for GPModel (AbstractGPs-backed)
# ============================================================================

"""
    alc_gp_model(gp, Xcand, Xref)

Compute Active Learning Cohn (ALC) acquisition values for GPModel.

# Arguments
- `gp::GPModel`: Gaussian Process model
- `Xcand::Matrix`: candidate points (n_cand x m)
- `Xref::Matrix`: reference points (n_ref x m)

# Returns
- `Vector`: ALC values for each candidate point
"""
function alc_gp_model(gp::GPModel{T}, Xcand::Matrix{T}, Xref::Matrix{T}) where {T}
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

    # Scaling factor
    df_rat = df / (df - 2)

    for i in 1:n_cand
        # kx = k(X, x_cand)
        x_cand = Xcand[i:i, :]  # Keep as matrix for kernelmatrix
        kx = vec(kernelmatrix(gp.kernel, RowVecs(gp.X), RowVecs(x_cand)))

        # Kikx = Ki * kx
        Kikx = Ki * kx

        # mui = 1 + g - kx' * Ki * kx
        mui = one(T) + gp.g - dot(kx, Kikx)

        # Skip if numerical problems
        if mui <= sqrt(eps(T))
            alc[i] = T(-Inf)
            continue
        end

        # gvec = -Kikx / mui
        gvec = -Kikx / mui

        # kxy = k(x_cand, Xref)
        kxy = vec(kernelmatrix(gp.kernel, RowVecs(x_cand), RowVecs(Xref)))

        # Compute ALC contribution
        alc_sum = zero(T)
        for j in 1:n_ref
            kg = dot(k_ref[:, j], gvec)
            ktKikx_j = kg^2 * mui + 2 * kg * kxy[j] + kxy[j]^2 / mui
            alc_sum += gp.phi * ktKikx_j / df
        end

        alc[i] = alc_sum * df_rat / n_ref
    end

    return alc
end

"""
    mspe_gp_model(gp, Xcand, Xref)

Compute Mean Squared Prediction Error (MSPE) acquisition values for GPModel.

# Arguments
- `gp::GPModel`: Gaussian Process model
- `Xcand::Matrix`: candidate points (n_cand x m)
- `Xref::Matrix`: reference points (n_ref x m)

# Returns
- `Vector`: MSPE values for each candidate point
"""
function mspe_gp_model(gp::GPModel{T}, Xcand::Matrix{T}, Xref::Matrix{T}) where {T}
    n = size(gp.X, 1)
    n_cand = size(Xcand, 1)
    df = T(n)

    # Compute ALC first
    alc_vals = alc_gp_model(gp, Xcand, Xref)

    # Predict at reference locations
    pred_ref = pred_gp_model(gp, Xref; lite=true)
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
