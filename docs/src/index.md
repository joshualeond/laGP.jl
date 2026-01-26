# laGP.jl

*Local Approximate Gaussian Process Regression for Julia*

## Overview

laGP.jl is a Julia implementation of Local Approximate Gaussian Process (laGP) regression, ported from the [R laGP package](https://cran.r-project.org/package=laGP) by Robert Gramacy. It enables scalable GP predictions for large datasets by building local GP models at each prediction point.

### Key Features

- **Scalable Predictions**: Handle datasets with thousands of observations by building local GP models
- **Dual Implementation**: Choose between legacy direct matrix computations or AbstractGPs.jl backend
- **Isotropic & Separable Kernels**: Single lengthscale or per-dimension ARD lengthscales
- **Acquisition Functions**: ALC (Active Learning Cohn) and MSPE for intelligent point selection
- **MLE with Priors**: Maximum likelihood estimation with Inverse-Gamma priors (MAP)
- **Multi-threaded**: Parallel predictions across test points

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/joshualeond/laGP.jl")
```

## Quick Example

```julia
using laGP
using Random

Random.seed!(42)

# Generate training data
n = 100
X = rand(n, 2)
Z = sin.(2π * X[:, 1]) .* cos.(2π * X[:, 2]) + 0.1 * randn(n)

# Estimate hyperparameters from data
d_range = darg(X)
g_range = garg(Z)

# Create GP with initial parameters
gp = new_gp(X, Z, d_range.start, g_range.start)

# Optimize hyperparameters via joint MLE
jmle_gp(gp; drange=(d_range.min, d_range.max), grange=(g_range.min, g_range.max))

println("Optimized lengthscale: ", gp.d)
println("Optimized nugget: ", gp.g)

# Make predictions
X_test = rand(10, 2)
pred = pred_gp(gp, X_test)

println("Predictions: ", pred.mean)
println("Variances: ", pred.s2)
```

## Package Structure

The package provides two parallel implementations:

| Type | Description | Functions |
|------|-------------|-----------|
| `GP`, `GPsep` | Legacy types with direct matrix computations | `new_gp`, `pred_gp`, `llik_gp`, etc. |
| `GPModel`, `GPModelSep` | AbstractGPs.jl backend | `new_gp_model`, `pred_gp_model`, etc. |

Both share the same API pattern and produce equivalent results.

## Design Matrix Convention

All design matrices use **rows as observations**:
- `X` is `n × m` where `n` is number of points and `m` is dimensionality
- This matches the R laGP convention

## Contents

```@contents
Pages = [
    "getting_started.md",
    "theory.md",
    "examples/demo.md",
    "examples/sinusoidal.md",
    "examples/surrogates.md",
    "examples/satellite.md",
    "api.md",
]
Depth = 2
```

## References

- Gramacy, R. B. (2016). laGP: Large-Scale Spatial Modeling via Local Approximate Gaussian Processes in R. *Journal of Statistical Software*, 72(1), 1-46.
- Gramacy, R. B. (2020). *Surrogates: Gaussian Process Modeling, Design and Optimization for the Applied Sciences*. Chapman Hall/CRC.
