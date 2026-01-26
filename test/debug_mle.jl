using JSON3
using laGP

# Access internal function
import laGP: update_gp!

# Load reference data
const MLE_REF = JSON3.read(read(joinpath(@__DIR__, "reference", "mle.json"), String))

function _reshape_matrix(vec::Vector, nrow::Int, ncol::Int)
    return reshape(vec, ncol, nrow)' |> collect
end

X = _reshape_matrix(Float64.(MLE_REF.X), MLE_REF.X_nrow, MLE_REF.X_ncol)
Z = Float64.(MLE_REF.Z)

# Create GP with initial d and g
gp = new_gp(X, Z, Float64(MLE_REF.d_init), Float64(MLE_REF.g_for_d))

# Check log-likelihood at various d values
println("Checking log-likelihood landscape for d (g=$(MLE_REF.g_for_d)):")
for d in [0.01, 0.1, 0.2, 0.29, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
    update_gp!(gp; d=d)
    ll = llik_gp(gp)
    println("  d=$d: llik=$ll")
end

println("\nExpected MLE d = $(MLE_REF.mle_d)")
println("Expected llik after MLE = $(MLE_REF.llik_after_d)")
