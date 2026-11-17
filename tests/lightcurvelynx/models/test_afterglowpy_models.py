import numpy as np
import pytest
from astropy.cosmology import Planck18

afterglowpy = pytest.importorskip("afterglowpy")

from lightcurvelynx.models.afterglowpy_models import (  # noqa: E402
    CM_PER_PC,
    SPEED_OF_LIGHT_A_S,
    AfterglowpyModel,
)

JET = {
    "E0": 1e53,
    "theta_core": 0.1,
    "theta_wing": 0.4,
    "n0": 0.1,
    "p": 2.3,
    "epsilon_e": 0.1,
    "epsilon_B": 1e-3,
}


@pytest.mark.parametrize("theta_obs", [0.0, 0.3])
def test_afterglowpy_model_matches_direct(theta_obs):
    """The gridded and interpolated SED matches afterglowpy evaluated directly, and is 0 before t0."""
    model = AfterglowpyModel(
        **JET, theta_obs=theta_obs, ra=0.0, dec=0.0, redshift=0.5, t0=60000.0, cosmology=Planck18
    )
    state = model.sample_parameters(num_samples=1)
    assert not model.apply_redshift
    times = 60000.0 + np.array([-1.0, 0.01, 0.3, 2.0, 15.0, 120.0])
    waves = SPEED_OF_LIGHT_A_S / np.array([30e9, 95e9, 150e9, 280e9])
    sed = model.evaluate_sed(times, waves, state)
    assert np.all(sed[0] == 0.0)
    params = model.get_local_params(state)
    assert params["distance"] * CM_PER_PC == pytest.approx(Planck18.luminosity_distance(0.5).to("cm").value)
    tt, ww = np.meshgrid((times[1:] - 60000.0) * 86400.0, waves, indexing="ij")
    kwargs = model.afterglowpy_params(params)
    direct = afterglowpy.fluxDensity(tt.ravel(), SPEED_OF_LIGHT_A_S / ww.ravel(), **kwargs).reshape(tt.shape)
    np.testing.assert_allclose(sed[1:], direct * 1e6, rtol=0.08)
