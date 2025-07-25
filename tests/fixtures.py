from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@dataclass
class TestWout:
    rmnc: jax.Array
    zmns: jax.Array
    xm: jax.Array
    xn: jax.Array
    bsupumnc: jax.Array
    bsupvmnc: jax.Array
    xm_nyq: jax.Array
    xn_nyq: jax.Array
    ns: int
    lasym: bool = False


@pytest.fixture
def torus_wout():
    """Fixture for a torus shaped Wout-like object."""
    n_surfaces = 5
    major_radius = 2.0
    minor_radius = 0.5
    rmnc = np.zeros((n_surfaces, 2))
    rmnc[:, 0] = major_radius
    rmnc[:, 1] = np.sqrt(np.linspace(0, 1, n_surfaces)) * minor_radius

    xm = np.array([0, 1])
    xn = np.array([0, 0])
    zmns = np.zeros((n_surfaces, 2))
    zmns[:, 1] = np.sqrt(np.linspace(0, 1, n_surfaces)) * minor_radius

    xm_nyq = np.array([0, 1])
    xn_nyq = np.array([0, 0])
    bsupumnc = np.zeros((n_surfaces - 1, 2))
    bsupvmnc = np.zeros((n_surfaces - 1, 2))
    bsupvmnc[:, 0] = 0.7
    return TestWout(
        rmnc=jnp.array(rmnc),
        zmns=jnp.array(zmns),
        xm=jnp.array(xm),
        xn=jnp.array(xn),
        bsupumnc=jnp.array(bsupumnc),
        bsupvmnc=jnp.array(bsupvmnc),
        xm_nyq=jnp.array(xm_nyq),
        xn_nyq=jnp.array(xn_nyq),
        ns=5,
        lasym=False,
    )
