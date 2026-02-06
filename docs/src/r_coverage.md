# R Port Coverage (Temporary Internal Note)

This file tracks the Julia port status against the original `laGP` R/C package.
It is an engineering aid during port validation and can be removed once the
R/C source tree is retired.

## Implemented in Julia (core modeling pipeline)

- `newGP` / `newGPsep` -> `new_gp` / `new_gp_sep`
- `predGP` / `predGPsep` -> `pred_gp` / `pred_gp_sep`
- `llikGP` / `llikGPsep` -> `llik_gp` / `llik_gp_sep`
- `mleGP` / `mleGPsep` -> `mle_gp!` / `mle_gp_sep!`
- `jmleGP` / `jmleGPsep` -> `jmle_gp!` / `jmle_gp_sep!`
- `aGP` / `aGPsep` -> `agp` / `agp_sep`
- `laGP` / `laGPsep` -> `lagp` / `lagp_sep`
- `alcGP` / `mspeGP` -> `alc_gp` / `mspe_gp`
- `darg` / `garg` -> `darg` / `garg`

## Partial or intentionally different

- `updateGP` in R appends data; Julia separates responsibilities:
  - data growth: `extend_gp!` / `extend_gp_sep!`
  - hyperparameter refresh: `update_gp!` / `update_gp_sep!`
- `darg_sep` exists in Julia to support separable workflows (R has single `darg` helper).
- Alternating optimizers `amle_gp!` / `amle_gp_sep!` are Julia additions.

## Not planned in this Julia-only package

- Ray and EFI acquisition variants (`alcray*`, `fish*`)
- Calibration/discrepancy helpers (`discrep.est`, `fcalib`)
- R utility wrappers (`distance`, `optim.auglag`, `optim.efi`)
- R/C object lifecycle APIs (`deleteGP*`) based on integer handles
- R-specific parallel entry points (`aGP.parallel`, `.R` variants)
