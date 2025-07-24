from enum import Enum
from functools import partial
from typing import Protocol

import jax
import numpy as np
import jax.numpy as jnp
import jaxtyping as jt
from scipy.interpolate import griddata


class WoutLike(Protocol):
    """Protocol for objects that can be used as VmecWOut."""

    rmnc: jt.Float[np.ndarray, "n_surfaces n_fourier_coefficients"]
    zmns: jt.Float[np.ndarray, "n_surfaces n_fourier_coefficients"]
    xm: jt.Int[np.ndarray, "n_fourier_coefficients"]
    xn: jt.Int[np.ndarray, "n_fourier_coefficients"]
    bsupumnc: jt.Float[np.ndarray, "n_surfaces n_fourier_coefficients_nyquist"]
    bsupvmnc: jt.Float[np.ndarray, "n_surfaces n_fourier_coefficients_nyquist"]
    xm_nyq: jt.Int[np.ndarray, "n_fourier_coefficients_nyquist"]
    xn_nyq: jt.Int[np.ndarray, "n_fourier_coefficients_nyquist"]
    lasym: bool


class FourierBasis(Enum):
    COS = "cos"
    SIN = "sin"


class FourierDerivative(Enum):
    NO = "no"
    TOROIDAL = "toroidal"
    POLOIDAL = "poloidal"


@partial(jax.jit, static_argnames=["basis", "derivative"])
def inverse_fourier_transform(
    fourier_coefficients: jt.Float[jax.Array, "n_s n_fourier_coefficients"],
    poloidal_mode_numbers: jt.Int[jax.Array, " n_fourier_coefficients"],
    toroidal_mode_numbers: jt.Int[jax.Array, " n_fourier_coefficients"],
    s_theta_phi: jt.Float[jax.Array, "n_s n_theta n_phi s_theta_phi=3"],
    basis: FourierBasis,
    derivative: FourierDerivative = FourierDerivative.NO,
) -> jt.Float[jax.Array, "n_s n_theta n_phi"]:
    """Transform an array of Fourier coefficients into values on a toroidal grid."""
    m = poloidal_mode_numbers[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
    n = toroidal_mode_numbers[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
    theta = s_theta_phi[jnp.newaxis, :, :, :, 1]
    phi = s_theta_phi[jnp.newaxis, :, :, :, 2]
    angle = m * theta - n * phi
    coefficients = fourier_coefficients[:, :, jnp.newaxis, jnp.newaxis]
    if basis == FourierBasis.COS:
        if derivative == FourierDerivative.POLOIDAL:
            return jnp.sum(-m * coefficients * jnp.sin(angle), axis=0)
        elif derivative == FourierDerivative.TOROIDAL:
            return jnp.sum(n * coefficients * jnp.sin(angle), axis=0)
        else:
            return jnp.sum(coefficients * jnp.cos(angle), axis=0)
    elif basis == FourierBasis.SIN:
        if derivative == FourierDerivative.POLOIDAL:
            return jnp.sum(m * coefficients * jnp.cos(angle), axis=0)
        elif derivative == FourierDerivative.TOROIDAL:
            return jnp.sum(-n * coefficients * jnp.cos(angle), axis=0)
        else:
            return jnp.sum(coefficients * jnp.sin(angle), axis=0)


def evaluate_rphiz_on_toroidal_grid(
    equilibrium: WoutLike,
    s_theta_phi: jt.Float[jax.Array, "n_s n_theta n_phi sthetaphi=3"],
) -> jt.Float[jax.Array, "n_s n_theta n_phi rphiz=3"]:
    """Evaluate the cylindrical coordinates (r, phi, z) on a toroidal grid."""
    if equilibrium.lasym:
        raise NotImplementedError(
            "Non stellarator symmetric equilibria are not supported yet."
        )
    r = inverse_fourier_transform(
        fourier_coefficients=equilibrium.rmnc,
        poloidal_mode_numbers=equilibrium.xm,
        toroidal_mode_numbers=equilibrium.xn,
        s_theta_phi=s_theta_phi,
        basis=FourierBasis.COS,
    )
    z = inverse_fourier_transform(
        fourier_coefficients=equilibrium.zmns,
        poloidal_mode_numbers=equilibrium.xm,
        toroidal_mode_numbers=equilibrium.xn,
        s_theta_phi=s_theta_phi,
        basis=FourierBasis.SIN,
    )
    phi = s_theta_phi[:, :, :, 2]
    return jnp.stack([r, phi, z], axis=-1)


def evaluate_magnetic_field_on_toroidal_grid(
    equilibrium: WoutLike,
    s_theta_phi: jt.Float[jax.Array, "n_s n_theta n_phi sthetaphi=3"],
) -> jt.Float[jax.Array, "n_s n_theta n_phi bxyz=3"]:
    """Evaluate the cartesian components of the magnetic field on a toroidal grid."""
    if equilibrium.lasym:
        raise NotImplementedError(
            "Non stellarator symmetric equilibria are not supported yet."
        )
    b_theta = inverse_fourier_transform(
        fourier_coefficients=equilibrium.bsupumnc,
        poloidal_mode_numbers=equilibrium.xm_nyq,
        toroidal_mode_numbers=equilibrium.xn_nyq,
        s_theta_phi=s_theta_phi,
        basis=FourierBasis.COS,
    )
    b_phi = inverse_fourier_transform(
        fourier_coefficients=equilibrium.bsupvmnc,
        poloidal_mode_numbers=equilibrium.xm_nyq,
        toroidal_mode_numbers=equilibrium.xn_nyq,
        s_theta_phi=s_theta_phi,
        basis=FourierBasis.COS,
    )
    r = inverse_fourier_transform(
        fourier_coefficients=equilibrium.rmnc,
        poloidal_mode_numbers=equilibrium.xm,
        toroidal_mode_numbers=equilibrium.xn,
        s_theta_phi=s_theta_phi,
        basis=FourierBasis.COS,
    )
    dr_dtheta = inverse_fourier_transform(
        fourier_coefficients=equilibrium.rmnc,
        poloidal_mode_numbers=equilibrium.xm,
        toroidal_mode_numbers=equilibrium.xn,
        s_theta_phi=s_theta_phi,
        basis=FourierBasis.COS,
        derivative=FourierDerivative.POLOIDAL,
    )
    dz_dtheta = inverse_fourier_transform(
        fourier_coefficients=equilibrium.zmns,
        poloidal_mode_numbers=equilibrium.xm,
        toroidal_mode_numbers=equilibrium.xn,
        s_theta_phi=s_theta_phi,
        basis=FourierBasis.SIN,
        derivative=FourierDerivative.POLOIDAL,
    )
    dr_dphi = inverse_fourier_transform(
        fourier_coefficients=equilibrium.rmnc,
        poloidal_mode_numbers=equilibrium.xm,
        toroidal_mode_numbers=equilibrium.xn,
        s_theta_phi=s_theta_phi,
        basis=FourierBasis.COS,
        derivative=FourierDerivative.TOROIDAL,
    )
    dz_dphi = inverse_fourier_transform(
        fourier_coefficients=equilibrium.zmns,
        poloidal_mode_numbers=equilibrium.xm,
        toroidal_mode_numbers=equilibrium.xn,
        s_theta_phi=s_theta_phi,
        basis=FourierBasis.SIN,
        derivative=FourierDerivative.TOROIDAL,
    )
    phi = s_theta_phi[:, :, :, 2]
    # See eq. (3.68) of https://arxiv.org/abs/2502.04374
    dxyz_dtheta = jnp.stack(
        [dr_dtheta * jnp.cos(phi), dr_dtheta * jnp.sin(phi), dz_dtheta], axis=-1
    )
    # See eq. (3.69) of https://arxiv.org/abs/2502.04374
    dxyz_dphi = jnp.stack(
        [
            dr_dphi * jnp.cos(phi) - r * jnp.sin(phi),
            dr_dphi * jnp.sin(phi) + r * jnp.cos(phi),
            dz_dphi,
        ],
        axis=-1,
    )
    return b_theta[..., jnp.newaxis] * dxyz_dtheta + b_phi[..., jnp.newaxis] * dxyz_dphi


def interpolate_toroidal_to_cylindrical_grid(
    rphiz_toroidal: jt.Float[np.ndarray, "n_s n_theta n_phi rphiz=3"],
    rz_cylindrical: jt.Float[np.ndarray, "n_r n_z rz=2"],
    value_toroidal: jt.Float[np.ndarray, "n_s n_theta n_phi"],
) -> jt.Float[np.ndarray, "n_r n_phi n_z rhothetaphi=3"]:
    """Interpolate toroidal coordinates to a cylindrical grid."""
    (n_r, n_z, _) = rz_cylindrical.shape
    values = []
    phis = rphiz_toroidal[0, 0, :, 1]
    for i_phi, _ in enumerate(phis):
        value = griddata(
            np.array(rphiz_toroidal[:, :, i_phi, ::2]).reshape(-1, 2),
            np.array(value_toroidal[:, :, i_phi]).reshape(-1, 3),
            np.array(rz_cylindrical).reshape(-1, 2),
            method="linear",
        ).reshape(n_r, n_z, 3)
        values.append(value)
    return np.stack(values, axis=1)
