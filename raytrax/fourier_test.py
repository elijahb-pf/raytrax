from dataclasses import dataclass

import numpy as np
import pytest

from raytrax.fourier import evaluate_rphiz_on_toroidal_grid


@dataclass
class TestWout:
    rmnc: np.ndarray
    zmns: np.ndarray
    xm: np.ndarray
    xn: np.ndarray
    lasym: bool = False


@pytest.fixture
def torus_wout():
    """Fixture for a torus shaped Wout-like object."""
    n_surfaces = 10
    major_radius = 2.0
    minor_radius = 0.5
    rmnc = np.zeros((2, n_surfaces))
    rmnc[0] = major_radius
    rmnc[1] = minor_radius

    xm = np.array([0, 1])
    xn = np.array([0, 0])
    zmns = np.zeros((2, n_surfaces))
    zmns[1] = minor_radius
    return TestWout(rmnc=rmnc, zmns=zmns, xm=xm, xn=xn, lasym=False)


def test_evaluate_rphiz_on_toroidal_grid(torus_wout):
    """Test the evaluate_rphiz_on_toroidal_grid function."""
    s_theta_phi = np.random.rand(10, 20, 30, 3)
    rphiz = evaluate_rphiz_on_toroidal_grid(torus_wout, s_theta_phi)

    assert rphiz.shape == (10, 20, 30, 3)
    # some of them will be NaN, but hopefully not all
    assert np.any(np.isfinite(rphiz))
