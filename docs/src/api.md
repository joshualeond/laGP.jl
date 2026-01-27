# API Reference

## Types

```@docs
GP
GPsep
GPPrediction
GPPredictionFull
```

## Core GP Functions (Isotropic)

```@docs
new_gp
pred_gp
llik_gp
dllik_gp
update_gp!
```

## Core GP Functions (Separable)

```@docs
new_gp_sep
pred_gp_sep
llik_gp_sep
dllik_gp_sep
update_gp_sep!
```

## MLE Functions (Isotropic)

```@docs
mle_gp
jmle_gp
darg
garg
```

## MLE Functions (Separable)

```@docs
mle_gp_sep
jmle_gp_sep
darg_sep
```

## AD-based Gradient Functions

```@docs
neg_llik_ad
dllik_ad
```

## Acquisition Functions

```@docs
alc_gp
mspe_gp
```

## Local GP Functions

```@docs
lagp
agp
```

## Plotting Functions

```@docs
plot_gp_surface
plot_gp_variance
plot_local_design
plot_agp_predictions
contour_with_constraints
```

## Index

```@index
```
