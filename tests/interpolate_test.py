import jax.numpy as jnp
import numpy as np

from raytrax.fourier import evaluate_rphiz_on_toroidal_grid
from raytrax.interpolate import interpolate_toroidal_to_cylindrical_grid

from .fixtures import torus_wout


def test_interpolate_toroidal_to_cylindrical_grid(torus_wout):
    rho_theta_phi = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(0, 1, 8),
            jnp.linspace(0, 2 * jnp.pi, 6),
            jnp.linspace(0, 2 * jnp.pi, 7),
            indexing="ij",
        ),
        axis=-1,
    )
    rphiz_toroidal = evaluate_rphiz_on_toroidal_grid(torus_wout, rho_theta_phi)
    rmin = np.min(rphiz_toroidal[..., 0])
    rmax = np.max(rphiz_toroidal[..., 0])
    zmin = np.min(rphiz_toroidal[..., 2])
    zmax = np.max(rphiz_toroidal[..., 2])
    rz_cylindrical = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(rmin, rmax, 4),
            jnp.linspace(zmin, zmax, 5),
            indexing="ij",
        ),
        axis=-1,
    )
    values_cylindrical = interpolate_toroidal_to_cylindrical_grid(
        rphiz_toroidal=rphiz_toroidal,
        rz_cylindrical=rz_cylindrical,
        value_toroidal=jnp.ones((8, 6, 7, 3)),
    )
    assert values_cylindrical.shape == (4, 7, 5, 3)
    # some of them will be NaN, but not all
    assert np.any(np.isfinite(values_cylindrical))
    # all values should be either NaN or 1.0
    np.testing.assert_allclose(
        values_cylindrical[np.isfinite(values_cylindrical)], 1.0, rtol=0, atol=1e-15
    )
