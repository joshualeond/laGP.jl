"""
    laGP

Local Approximate Gaussian Process (laGP) regression for Julia.

Port of the R laGP package by Robert Gramacy.
"""
module laGP

using AbstractGPs
using KernelFunctions
using LinearAlgebra
using Optim
using Statistics

# Types
export GP
export GPsep
export GPPrediction
export GPPredictionFull

# Core GP functions (isotropic)
export new_gp
export pred_gp
export llik_gp
export dllik_gp
export d2llik_gp
export update_gp!

# Core GP functions (separable)
export new_gp_sep
export pred_gp_sep
export llik_gp_sep
export dllik_gp_sep
export d2llik_gp_sep_nug
export update_gp_sep!

# MLE functions (isotropic)
export mle_gp
export jmle_gp
export amle_gp
export darg
export garg

# MLE functions (separable)
export mle_gp_sep
export jmle_gp_sep
export amle_gp_sep
export darg_sep

# Acquisition functions
export alc_gp
export mspe_gp

# Local GP functions
export lagp
export agp

include("abstractgps_adapter.jl")
include("types.jl")
include("gp.jl")
include("mle.jl")
include("acquisition.jl")
include("local_gp.jl")

end # module laGP
