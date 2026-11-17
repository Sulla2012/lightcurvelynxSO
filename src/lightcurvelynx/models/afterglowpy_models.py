"""GRB afterglow models computed with afterglowpy.

https://github.com/geoffryan/afterglowpy
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from lightcurvelynx.models.physical_model import SEDModel

try:
    import afterglowpy
except ImportError:  # pragma: no cover
    afterglowpy = None

SPEED_OF_LIGHT_A_S = 2.99792458e18  # Angstrom per second
CM_PER_PC = 3.0856775814913673e18
NJY_PER_MJY = 1e6
SECONDS_PER_DAY = 86400.0
_JET_PARAMS = ("E0", "theta_core", "theta_wing", "theta_obs", "n0", "p", "epsilon_e", "epsilon_B", "xi_N")


class AfterglowpyModel(SEDModel):
    """A structured-jet GRB afterglow (forward shock) computed with afterglowpy.

    afterglowpy returns the observer-frame flux density for a given redshift and luminosity
    distance, so lightcurvelynx's own redshifting is disabled. Evaluating afterglowpy at every
    requested (time, wavelength) is slow, so the model evaluates it on a grid of ``n_times``
    log-spaced times after ``t0`` and ``n_freqs`` log-spaced frequencies spanning the request, and
    interpolates log flux linearly in log time and log frequency. Afterglow spectra and light curves
    are smooth broken power laws, so this is accurate to a few percent with the default grid.
    afterglowpy does not include synchrotron self-absorption, so fluxes below the self-absorption
    frequency (typically a few to tens of GHz at early times) are overestimated.

    Parameterized values include:

    * E0 - Isotropic-equivalent kinetic energy of the jet core, in erg.
    * theta_core - Jet core half-opening angle, in radians.
    * theta_wing - Truncation angle of the jet wings, in radians (>= theta_core).
    * theta_obs - Viewing angle from the jet axis, in radians.
    * n0 - Circumburst number density, in cm^-3.
    * p - Electron power-law index.
    * epsilon_e - Fraction of shock energy in electrons.
    * epsilon_B - Fraction of shock energy in magnetic fields.
    * xi_N - Fraction of electrons accelerated.
    * dec, distance (pc), ra, redshift, t0 (MJD of the burst) [from BasePhysicalModel]

    Parameters
    ----------
    E0, theta_core, theta_wing, theta_obs, n0, p, epsilon_e, epsilon_B : parameter
        The jet parameters described above (values or parameter nodes).
    xi_N : parameter
        Fraction of electrons accelerated. Default 1.
    jet_type : str
        "gaussian" or "tophat".
    n_times, n_freqs : int
        Size of the afterglowpy evaluation grid.
    t_min_days : float
        Earliest time after the burst (observer frame, days) on the grid; earlier times use it.
    afterglowpy_kwargs : dict, optional
        Extra keyword arguments for ``afterglowpy.fluxDensity`` (e.g. ``{"tRes": 300}`` for speed).
    **kwargs
        Other parameters (ra, dec, redshift, t0, distance or cosmology, node_label, ...).
    """

    def __init__(
        self,
        E0,
        theta_core,
        theta_wing,
        theta_obs,
        n0,
        p,
        epsilon_e,
        epsilon_B,
        xi_N=1.0,
        *,
        jet_type="gaussian",
        n_times=40,
        n_freqs=6,
        t_min_days=1e-3,
        afterglowpy_kwargs=None,
        **kwargs,
    ):
        if afterglowpy is None:  # pragma: no cover
            raise ImportError(
                "AfterglowpyModel requires afterglowpy. Install it with `pip install afterglowpy`."
            )
        super().__init__(**kwargs)
        # afterglowpy works in the observer frame given z and d_L, so do not redshift again.
        self.set_apply_redshift(False)
        values = (E0, theta_core, theta_wing, theta_obs, n0, p, epsilon_e, epsilon_B, xi_N)
        for name, value in zip(_JET_PARAMS, values, strict=True):
            self.add_parameter(name, value, description=f"afterglowpy {name}.", **kwargs)
        self.jet_type = {"gaussian": afterglowpy.jet.Gaussian, "tophat": afterglowpy.jet.TopHat}[jet_type]
        self.n_times, self.n_freqs, self.t_min_days = n_times, n_freqs, t_min_days
        self.afterglowpy_kwargs = afterglowpy_kwargs or {}

    def afterglowpy_params(self, params):
        """Return the afterglowpy keyword arguments for one set of local parameters.

        Parameters
        ----------
        params : dict
            Local parameter values (from ``get_local_params``).

        Returns
        -------
        dict
            Keyword arguments for ``afterglowpy.fluxDensity`` (without times and frequencies).
        """
        names = dict(zip(_JET_PARAMS, ("E0", "thetaCore", "thetaWing", "thetaObs", "n0", "p",
                                       "epsilon_e", "epsilon_B", "xi_N"), strict=True))  # fmt: skip
        return {
            "jetType": self.jet_type,
            "specType": afterglowpy.jet.SimpleSpec,
            **{names[key]: params[key] for key in _JET_PARAMS},
            "z": params["redshift"],
            "d_L": params["distance"] * CM_PER_PC,
            **self.afterglowpy_kwargs,
        }

    def compute_sed(self, times, wavelengths, graph_state, **kwargs):
        """Draw effect-free observations for this object.

        Parameters
        ----------
        times : numpy.ndarray
            A length T array of observer frame times (MJD).
        wavelengths : numpy.ndarray
            A length N array of observer frame wavelengths (Angstroms).
        graph_state : GraphState
            An object mapping graph parameters to their values.
        **kwargs : dict, optional
            Any additional keyword arguments.

        Returns
        -------
        flux_density : numpy.ndarray
            A length T x N matrix of observer frame flux densities (nJy); zero before ``t0``.
        """
        params = self.get_local_params(graph_state)
        flux = np.zeros((len(times), len(wavelengths)))
        dt = np.asarray(times) - params["t0"]
        after = dt > 0
        if not np.any(after) or len(wavelengths) == 0:
            return flux
        t_grid = np.geomspace(self.t_min_days, max(dt[after].max(), 2 * self.t_min_days), self.n_times)
        nu = SPEED_OF_LIGHT_A_S / np.asarray(wavelengths, dtype=float)
        nu_grid = np.geomspace(nu.min(), max(nu.max(), nu.min() * 1.001), self.n_freqs)
        tt, nn = np.meshgrid(t_grid * SECONDS_PER_DAY, nu_grid, indexing="ij")
        grid_flux = afterglowpy.fluxDensity(tt.ravel(), nn.ravel(), **self.afterglowpy_params(params))
        log_flux = np.log(np.maximum(grid_flux.reshape(tt.shape), 1e-300))
        interp = RegularGridInterpolator((np.log(t_grid), np.log(nu_grid)), log_flux)
        log_t = np.log(np.clip(dt[after], t_grid[0], t_grid[-1]))
        points = np.stack(np.broadcast_arrays(log_t[:, None], np.log(nu)[None, :]), axis=-1)
        flux[after] = np.exp(interp(points)) * NJY_PER_MJY
        return flux
