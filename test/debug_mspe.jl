using JSON3
using Statistics
using laGP

const ACQ_REF = JSON3.read(read(joinpath(@__DIR__, "reference", "acquisition.json"), String))

function _reshape_matrix(vec::Vector, nrow::Int, ncol::Int)
    return reshape(vec, ncol, nrow)' |> collect
end

X = _reshape_matrix(Float64.(ACQ_REF.X), ACQ_REF.X_nrow, ACQ_REF.X_ncol)
Z = Float64.(ACQ_REF.Z)
d = Float64(ACQ_REF.d)
g = Float64(ACQ_REF.g)
Xcand = _reshape_matrix(Float64.(ACQ_REF.Xcand), ACQ_REF.Xcand_nrow, ACQ_REF.Xcand_ncol)
Xref = _reshape_matrix(Float64.(ACQ_REF.Xref), ACQ_REF.Xref_nrow, ACQ_REF.Xref_ncol)

gp = new_gp(X, Z, d, g)

# Check ALC first
alc_vals = alc_gp(gp, Xcand, Xref)
ref_alc = Float64.(ACQ_REF.alc)
println("ALC[1:5]: ", alc_vals[1:5])
println("R ALC[1:5]: ", ref_alc[1:5])

# Check MSPE intermediate values
n = size(X, 1)
pred_ref = pred_gp(gp, Xref; lite=true)
s2avg = mean(pred_ref.s2)
println("\ns2avg = ", s2avg)
println("pred_ref.s2 = ", pred_ref.s2)
println("df = ", n)

dnp = (n + 1.0) / (n - 1.0)
dnp2 = dnp * (n - 2.0) / n
println("dnp = ", dnp)
println("dnp2 = ", dnp2)

mspe_vals = mspe_gp(gp, Xcand, Xref)
ref_mspe = Float64.(ACQ_REF.mspe)
println("\nMSPE[1:5]: ", mspe_vals[1:5])
println("R MSPE[1:5]: ", ref_mspe[1:5])

# Manual MSPE calculation
println("\nManual MSPE[1]: dnp*s2avg - dnp2*alc = ", dnp * s2avg - dnp2 * alc_vals[1])
println("Expected MSPE[1]: ", ref_mspe[1])
