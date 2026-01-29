# laGP.jl

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://joshualeond.github.io/laGP.jl/dev/)

A Julia implementation of Local Approximate Gaussian Process (laGP) regression for scalable GP predictions on large datasets.

## Overview

laGP.jl is a port of the [R laGP package](https://cran.r-project.org/package=laGP) by Robert Gramacy. It provides efficient Gaussian Process regression by building local GP models at each prediction point using nearest neighbors and acquisition functions.

**Key Features:**

- Scalable GP predictions for large datasets via local approximation
- Both isotropic and separable (ARD) kernel implementations
- Active Learning Cohn (ALC) and MSPE acquisition functions
- Maximum likelihood estimation with Inverse-Gamma priors
- Dual implementation: legacy direct matrix operations and AbstractGPs.jl backend
- Multi-threaded prediction support

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/joshualeond/laGP.jl")
```

## Quick Start

```julia
using laGP
using Random

Random.seed!(42)

# Generate training data
n = 100
X = rand(n, 2)
Z = sin.(2π * X[:, 1]) .* cos.(2π * X[:, 2]) + 0.1 * randn(n)

# Estimate hyperparameters
d_range = darg(X)
g_range = garg(Z)

# Create and fit GP
gp = new_gp(X, Z, d_range.start, g_range.start)
jmle_gp!(gp; drange=(d_range.min, d_range.max), grange=(g_range.min, g_range.max))

# Make predictions
X_test = rand(10, 2)
pred = pred_gp(gp, X_test)
println("Predictions: ", pred.mean)
println("Variances: ", pred.s2)
```

## Local Approximate GP

For large datasets, use the local approximate GP functions:

```julia
using laGP

# Large training set
X_train = rand(10000, 2)
Z_train = sin.(2π * X_train[:, 1]) .* cos.(2π * X_train[:, 2])

# Test points
X_test = rand(100, 2)

# Get hyperparameter ranges
d_range = darg(X_train)
g_range = garg(Z_train)

# Local approximate GP predictions
result = agp(X_train, Z_train, X_test;
    start=6, endpt=50,
    d=(start=d_range.start, mle=false),
    g=(start=g_range.start, mle=false),
    method=:alc  # or :mspe, :nn
)

println("Mean predictions: ", result.mean)
println("Variance estimates: ", result.var)
```

## Documentation

For full documentation, including tutorials and API reference, see the [documentation](https://joshualeond.github.io/laGP.jl/dev/).

## References

- Gramacy, R. B. (2016). laGP: Large-Scale Spatial Modeling via Local Approximate Gaussian Processes in R. *Journal of Statistical Software*, 72(1), 1-46.
- Gramacy, R. B. (2020). *Surrogates: Gaussian Process Modeling, Design and Optimization for the Applied Sciences*. Chapman Hall/CRC.

## License

LGPL-3.0
