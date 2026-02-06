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
d2llik_gp
update_gp!
extend_gp!
```

## Core GP Functions (Separable)

```@docs
new_gp_sep
pred_gp_sep
llik_gp_sep
dllik_gp_sep
d2llik_gp_sep_nug
update_gp_sep!
extend_gp_sep!
```

## MLE Functions (Isotropic)

```@docs
mle_gp!
jmle_gp!
amle_gp!
darg
garg
```

## MLE Functions (Separable)

```@docs
mle_gp_sep!
jmle_gp_sep!
amle_gp_sep!
darg_sep
```

## Acquisition Functions

```@docs
alc_gp
mspe_gp
```

## Local GP Functions (Isotropic)

```@docs
lagp
agp
```

## Local GP Functions (Separable)

```@docs
lagp_sep
agp_sep
```

## Index

```@index
```
