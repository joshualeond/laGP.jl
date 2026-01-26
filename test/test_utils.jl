# Shared test utilities for laGP.jl tests
# This file uses an include guard to prevent redefinition warnings

if !@isdefined(_LAGP_TEST_UTILS_LOADED)
    const _LAGP_TEST_UTILS_LOADED = true

    """
        _reshape_matrix(vec, nrow, ncol)

    Reshape a vector from R (stored row-major as as.vector(t(X))) back to a matrix.
    R stores matrices column-major, but we saved row-major so we reshape to nrow x ncol directly.
    """
    function _reshape_matrix(vec::Vector, nrow::Int, ncol::Int)
        return reshape(vec, ncol, nrow)' |> collect
    end
end
