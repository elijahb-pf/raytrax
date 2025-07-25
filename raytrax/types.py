from typing import Protocol, runtime_checkable

import jax
import jaxtyping as jt


@runtime_checkable
class WoutLike(Protocol):
    """Protocol for objects that can be used as VmecWOut."""

    rmnc: jt.Float[jax.Array, "n_surfaces n_fourier_coefficients"]
    zmns: jt.Float[jax.Array, "n_surfaces n_fourier_coefficients"]
    xm: jt.Int[jax.Array, "n_fourier_coefficients"]
    xn: jt.Int[jax.Array, "n_fourier_coefficients"]
    bsupumnc: jt.Float[jax.Array, "n_surfaces n_fourier_coefficients_nyquist"]
    bsupvmnc: jt.Float[jax.Array, "n_surfaces n_fourier_coefficients_nyquist"]
    xm_nyq: jt.Int[jax.Array, "n_fourier_coefficients_nyquist"]
    xn_nyq: jt.Int[jax.Array, "n_fourier_coefficients_nyquist"]
    ns: int
    nfp: int
    lasym: bool
