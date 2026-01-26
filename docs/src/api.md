# API Reference

## Types

### Legacy Types

```@docs
GP
GPsep
GPPrediction
GPPredictionFull
```

### AbstractGPs-backed Types

```@docs
GPModel
GPModelSep
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

## Core GP Functions (AbstractGPs-backed Isotropic)

```@docs
new_gp_model
pred_gp_model
llik_gp_model
dllik_gp_model
update_gp_model!
```

## Core GP Functions (AbstractGPs-backed Separable)

```@docs
new_gp_model_sep
pred_gp_model_sep
llik_gp_model_sep
dllik_gp_model_sep
update_gp_model_sep!
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

## MLE Functions (AbstractGPs-backed)

```@docs
mle_gp_model
jmle_gp_model
jmle_gp_model_sep
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
alc_gp_model
mspe_gp_model
```

## Local GP Functions

```@docs
lagp
agp
lagp_model
agp_model
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
