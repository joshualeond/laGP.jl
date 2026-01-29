# Theory

This page covers the mathematical background for laGP.jl.

## Gaussian Process Basics

A Gaussian Process (GP) is a collection of random variables, any finite subset of which have a joint Gaussian distribution. A GP is fully specified by its mean function ``m(x)`` and covariance function ``k(x, x')``:

```math
f(x) \sim \mathcal{GP}(m(x), k(x, x'))
```

Given training data ``\{X, Z\}`` where ``X`` is an ``n \times m`` design matrix and ``Z`` is an ``n``-vector of responses, the GP posterior at test points ``X_*`` is:

```math
\begin{aligned}
\mu_* &= k(X_*, X) K^{-1} Z \\
\Sigma_* &= k(X_*, X_*) - k(X_*, X) K^{-1} k(X, X_*)
\end{aligned}
```

where ``K = k(X, X) + g I`` is the training covariance matrix with nugget ``g``.

## Kernel Parameterization

### Isotropic Squared Exponential

laGP uses the following parameterization:

```math
k(x, y) = \exp\left(-\frac{\|x - y\|^2}{d}\right)
```

where ``d`` is the lengthscale parameter.

!!! note "KernelFunctions.jl Comparison"
    KernelFunctions.jl uses: ``k(x,y) = \exp(-\|x-y\|^2/(2\ell^2))``

    The mapping is: ``d = 2\ell^2``, so ``\ell = \sqrt{d/2}``

### Separable (ARD) Kernel

For separable/anisotropic GPs, each dimension has its own lengthscale:

```math
k(x, y) = \exp\left(-\sum_{j=1}^{m} \frac{(x_j - y_j)^2}{d_j}\right)
```

This allows the model to capture different smoothness in different input directions.

## Concentrated Likelihood

laGP uses the concentrated (profile) log-likelihood:

```math
\ell = -\frac{1}{2}\left(n \log\left(\frac{\phi}{2}\right) + \log|K|\right)
```

where ``\phi = Z^\top K^{-1} Z``.

This formulation profiles out the variance parameter, leaving only ``d`` and ``g`` to optimize.

### Gradients

The gradient with respect to the nugget ``g``:

```math
\frac{\partial \ell}{\partial g} = -\frac{1}{2} \text{tr}(K^{-1}) + \frac{n}{2\phi} (K^{-1}Z)^\top (K^{-1}Z)
```

The gradient with respect to lengthscale ``d``:

```math
\frac{\partial \ell}{\partial d} = -\frac{1}{2} \text{tr}\left(K^{-1} \frac{\partial K}{\partial d}\right) + \frac{n}{2\phi} Z^\top K^{-1} \frac{\partial K}{\partial d} K^{-1} Z
```

where for the isotropic kernel:

```math
\frac{\partial K_{ij}}{\partial d} = K_{ij} \frac{\|x_i - x_j\|^2}{d^2}
```

## Inverse-Gamma Priors

MLE optimization uses Inverse-Gamma priors for regularization:

```math
p(\theta) \propto \theta^{-a-1} \exp\left(-\frac{b}{\theta}\right)
```

where ``a`` is the shape and ``b`` is the scale parameter.

The log-prior and its gradient:

```math
\begin{aligned}
\log p(\theta) &= a \log b - \log\Gamma(a) - (a+1)\log\theta - \frac{b}{\theta} \\
\frac{\partial \log p}{\partial \theta} &= -\frac{a+1}{\theta} + \frac{b}{\theta^2}
\end{aligned}
```

Default values: ``a = 3/2``, with ``b`` computed from the data-adaptive parameter ranges.

## Local Approximate GP

For large datasets, building a full GP is computationally prohibitive (``O(n^3)``). Local approximate GP builds a small local model for each prediction point.

### Algorithm

For each prediction point ``x_*``:

1. **Initialize**: Select ``n_0`` nearest neighbors to ``x_*``
2. **Iterate**: Until local design has ``n_{\text{end}}`` points:
   - Evaluate acquisition function on all remaining candidates
   - Add the point that maximizes (ALC) or minimizes (MSPE) the criterion
3. **Predict**: Make prediction using the local GP

### Active Learning Cohn (ALC)

ALC measures the expected reduction in predictive variance at reference points if a candidate point were added:

```math
\text{ALC}(x) = \frac{1}{n_{\text{ref}}} \sum_{j=1}^{n_{\text{ref}}} \left[\sigma^2(x_{\text{ref},j}) - \sigma^2_{x}(x_{\text{ref},j})\right]
```

where ``\sigma^2_x`` is the variance after adding point ``x`` to the design.

Higher ALC values indicate points that would reduce prediction uncertainty more.

### Mean Squared Prediction Error (MSPE)

MSPE combines current variance with expected variance reduction:

```math
\text{MSPE}(x) = \frac{n+1}{n-1} \bar{\sigma}^2 - \frac{n+1}{n-1} \cdot \frac{n-2}{n} \cdot \text{ALC}(x)
```

where ``\bar{\sigma}^2`` is the average predictive variance at reference points.

Lower MSPE values indicate better candidate points.

### Nearest Neighbor (NN)

The simplest approach: just use the ``n_{\text{end}}`` nearest neighbors. Fast but doesn't account for prediction location.

## Prediction Variance Scaling

laGP scales the posterior variance using the Student-t distribution:

```math
s^2 = \frac{\phi}{n} \left(1 + g - k_*^\top K^{-1} k_*\right)
```

This gives wider intervals than the standard GP formulation, accounting for hyperparameter uncertainty.

The degrees of freedom is ``n`` (number of training points).

## Data-Adaptive Hyperparameter Ranges

### Lengthscale (darg)

The `darg` function computes ranges from pairwise distances:

- **Start**: 10th percentile of pairwise squared distances
- **Min**: Half the minimum non-zero distance (floored at ``\sqrt{\epsilon}``)
- **Max**: Maximum pairwise distance

### Nugget (garg)

The `garg` function computes ranges from response variability:

- **Start**: 2.5th percentile of squared residuals from mean
- **Min**: ``\sqrt{\epsilon}`` (machine epsilon)
- **Max**: Maximum squared residual

## References

1. Gramacy, R. B. (2016). laGP: Large-Scale Spatial Modeling via Local Approximate Gaussian Processes in R. *Journal of Statistical Software*, 72(1), 1-46.

2. Gramacy, R. B. and Apley, D. W. (2015). Local Gaussian Process Approximation for Large Computer Experiments. *Journal of Computational and Graphical Statistics*, 24(2), 561-578.

3. Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
