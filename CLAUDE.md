# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

laGP.jl is a Julia port of the R laGP package by Robert Gramacy for Local Approximate Gaussian Process regression. It provides scalable GP predictions for large datasets by building local GP models at each prediction point using nearest neighbors and acquisition functions.

## Build and Test Commands

```bash
# Run all tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Run a single test file
julia --project=. test/test_gp.jl

# Run Julia REPL with project environment
julia --project=.

# Inside REPL, load the package
using laGP
```

## Architecture

### GP Types

The codebase provides two GP model types backed by AbstractGPs.jl:

- **`GP`** - Isotropic GP with single lengthscale for all dimensions
- **`GPsep`** - Separable (ARD) GP with per-dimension lengthscales

Both share the same API pattern: `new_gp*`, `pred_gp*`, `llik_gp*`, `dllik_gp*`, `update_gp*!`

### Core Modules

- **`types.jl`** - Type definitions for GP models
- **`gp.jl`** - Core GP operations: covariance computation, prediction, likelihood, gradients
- **`mle.jl`** - Maximum likelihood estimation using Optim.jl with Inverse-Gamma priors
- **`acquisition.jl`** - ALC (Active Learning Cohn) and MSPE acquisition functions
- **`local_gp.jl`** - `lagp`/`agp` functions for local approximate GP prediction
- **`abstractgps_adapter.jl`** - Kernel conversion between laGP and KernelFunctions.jl parameterization

### Key Parameterization Differences

laGP uses a different kernel parameterization than KernelFunctions.jl:
- laGP kernel: `k(x,y) = exp(-||x-y||²/d)`
- KernelFunctions: `k(x,y) = exp(-||x-y||²/(2ℓ²))`
- Mapping: `d = 2ℓ²`, so `ℓ = sqrt(d/2)`

### Design Matrices

All design matrices use rows as observations: `X` is `n x m` where `n` is number of points and `m` is dimensionality.

### Isotropic vs Separable

- **Isotropic** (`GP`): Single lengthscale `d::Real` for all dimensions
- **Separable** (`GPsep`): Per-dimension lengthscales `d::Vector` (ARD)

### Concentrated Likelihood

Uses the concentrated (profile) likelihood from R laGP:
```
llik = -0.5 * (n * log(0.5 * phi) + ldetK)
```
where `phi = Z' * K⁻¹ * Z`.

## Testing Patterns

Tests compare against R laGP reference values. Key test files:
- `test_gpsep_reference.jl` - Validates separable GP against R reference
- `test_wingwt_reference.jl` - Real dataset validation against R
- `test_mle.jl` - MLE optimization validation
