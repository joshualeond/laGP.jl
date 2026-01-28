# Getting Started

This tutorial covers the basics of using laGP.jl for Gaussian Process regression.

## Creating a GP Model

### Isotropic GP

An isotropic GP uses a single lengthscale parameter for all dimensions:

```julia
using laGP

# Training data: n observations, m dimensions
X = rand(50, 2)  # 50 points in 2D
Z = sin.(X[:, 1]) .+ cos.(X[:, 2]) + 0.1 * randn(50)

# Initial hyperparameters
d = 0.5  # lengthscale
g = 0.01 # nugget (noise variance)

# Create GP
gp = new_gp(X, Z, d, g)
```

### Separable GP (ARD)

A separable GP uses per-dimension lengthscales:

```julia
# Per-dimension lengthscales
d = [0.5, 0.3]  # different lengthscale for each dimension

# Create separable GP
gp_sep = new_gp_sep(X, Z, d, g)
```

## Hyperparameter Estimation

### Using `darg` and `garg`

The `darg` and `garg` functions compute sensible hyperparameter ranges from your data:

```julia
# Get data-adaptive ranges
d_range = darg(X)
g_range = garg(Z)

println("Lengthscale: start=$(d_range.start), range=[$(d_range.min), $(d_range.max)]")
println("Nugget: start=$(g_range.start), range=[$(g_range.min), $(g_range.max)]")

# Create GP with data-adaptive starting values
gp = new_gp(X, Z, d_range.start, g_range.start)
```

### Maximum Likelihood Estimation

Optimize hyperparameters using MLE:

```julia
# Single parameter optimization
mle_gp!(gp, :d; tmax=d_range.max)  # optimize lengthscale only

# Joint MLE of both parameters
jmle_gp!(gp; drange=(d_range.min, d_range.max), grange=(g_range.min, g_range.max))

println("Optimized d: ", gp.d)
println("Optimized g: ", gp.g)
```

For separable GPs:

```julia
d_range_sep = darg_sep(X)
gp_sep = new_gp_sep(X, Z, [r.start for r in d_range_sep.ranges], g_range.start)

jmle_gp_sep!(gp_sep;
    drange=[(r.min, r.max) for r in d_range_sep.ranges],
    grange=(g_range.min, g_range.max)
)

println("Optimized lengthscales: ", gp_sep.d)
```

## Making Predictions

### Basic Prediction

```julia
# Test points
X_test = rand(10, 2)

# Predict (lite=true returns diagonal variances only)
pred = pred_gp(gp, X_test; lite=true)

println("Mean: ", pred.mean)
println("Variance: ", pred.s2)
println("Degrees of freedom: ", pred.df)
```

### Full Covariance Matrix

For posterior sampling, get the full covariance:

```julia
# Full prediction (lite=false returns full covariance matrix)
pred_full = pred_gp(gp, X_test; lite=false)

println("Covariance matrix size: ", size(pred_full.Sigma))

# Draw posterior samples
using Distributions
mvn = MvNormal(pred_full.mean, Symmetric(pred_full.Sigma))
samples = rand(mvn, 100)  # 100 posterior samples
```

## Local Approximate GP

For large datasets, use local approximate GP:

### Single Point Prediction with `lagp`

```julia
# Large training set
X_train = rand(5000, 2)
Z_train = sin.(2π * X_train[:, 1]) .* cos.(2π * X_train[:, 2])

# Reference point to predict
Xref = [0.5, 0.5]

# Local GP prediction
result = lagp(Xref, 6, 50, X_train, Z_train;
    d=0.1, g=1e-6,
    method=:alc  # :alc, :mspe, or :nn
)

println("Prediction: ", result.mean)
println("Variance: ", result.var)
println("Local design indices: ", result.indices)
```

### Batch Predictions with `agp`

```julia
# Multiple test points
X_test = rand(100, 2)

# Get hyperparameter ranges
d_range = darg(X_train)
g_range = garg(Z_train)

# Approximate GP predictions
result = agp(X_train, Z_train, X_test;
    start=6,   # initial nearest neighbors
    endpt=50,  # final local design size
    d=(start=d_range.start, mle=false),
    g=(start=g_range.start, mle=false),
    method=:alc,
    parallel=true  # use multi-threading
)

println("Predictions shape: ", size(result.mean))
```

## Acquisition Methods

laGP supports three acquisition methods for local design selection:

| Method | Description | Use Case |
|--------|-------------|----------|
| `:alc` | Active Learning Cohn | Best accuracy, slower |
| `:mspe` | Mean Squared Prediction Error | Balance of speed/accuracy |
| `:nn` | Nearest Neighbors | Fastest, less accurate |

```julia
# Compare methods
result_alc = agp(X_train, Z_train, X_test; method=:alc, ...)
result_mspe = agp(X_train, Z_train, X_test; method=:mspe, ...)
result_nn = agp(X_train, Z_train, X_test; method=:nn, ...)
```

## Next Steps

- [Theory](theory.md): Mathematical background on GP kernels and acquisition functions
- [Examples](examples/demo.md): Complete worked examples
- [API Reference](api.md): Full function documentation
