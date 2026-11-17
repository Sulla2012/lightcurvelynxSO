"""Simons Observatory (SO) observation tables built from simulated scan time maps.

SO does not take discrete pointings: each optics tube (OT) scans continuously in azimuth at a
fixed elevation while the sky drifts through the scan. We therefore simulate one sidereal day of
scanning in each azimuth range, accumulate per-pixel hit counts and mean observation times for
each tube and range on a pixell CAR map, and materialize those maps into a standard ObsTable with
one row per (tube, day, observed pixel). Because the scan is defined in local sidereal time (LST),
it repeats every sidereal day, so the simulated day is replicated for the full date range.
Pointings are precessed from the mean equator and equinox of a reference epoch to ICRS, and
instrument downtime can be simulated and removed from the table, and visits within 30 deg of the
sun or moon are dropped with their exposure given to the rest of that day's sky (see below).

Idealized scan model
--------------------
* Site at 22d57m31s S, 67d47m15s W; boresight elevation fixed at 40 deg.
* A patch is crossed once while rising, in the east range (az 30-150 deg), and once while
  setting, in the west range (az 210-330 deg), around each meridian transit. SO switches daily
  between the rising and the setting scan of a patch, so successive transits of a patch alternate
  between its rising and its setting crossing.
* Azimuth follows a triangle wave at 1 deg/s. Each 120 s sweep is followed by a turnaround lasting
  5% of a sweep, during which no data are taken.
* Precession is computed once for the reference epoch (it moves the sky by ~0.014 deg per year);
  refraction (~0.01 deg at 40 deg elevation), nutation and aberration (< 0.006 deg) are ignored.
* 13 tubes of 1.25 deg diameter in a hex pattern (1.75 deg spacing, half the outer ring
  missing); 8 MF, 4 UHF, 1 LF (``SO_TUBE_TYPES``). A pixel is hit whenever its center is inside
  a tube's FoV.
* Tubes are dichroic, so each visit gives one row per band of the tube (``SO_TUBE_BANDS``):
  LF -> f030, f040; MF -> f090, f150; UHF -> f220, f280.
* Visits whose pixel is within 30 deg of the sun or moon are dropped (SO scans another patch).
* ASSUMPTION (sun/moon avoidance conserves observing time): SO's scheduler moves scans away from
  the sun and moon rather than idling, so avoidance does not reduce the total observing time. We
  do not model where the displaced scans go. Instead, the exposure removed by the cut on each UTC
  day is redistributed uniformly over that tube's remaining visits that day: their ``exposure_s``
  is multiplied by ``exposure_scale`` = 1 / (1 - f), where f is the fraction of the day's exposure
  that was cut (~14% on average, 4-29% per day). Total daily (and yearly) exposure therefore
  matches a table without the cut, but the extra depth is spread evenly instead of landing on
  the specific patches the real scheduler would choose.
* Each pixel is therefore observed at most once per tube per sidereal day.
"""

import functools

import erfa
import numpy as np
import pandas as pd
from astropy.coordinates import EarthLocation, get_body
from astropy.time import Time

from lightcurvelynx.astro_utils.passbands import Passband, PassbandGroup
from lightcurvelynx.obstable.obs_table import ObsTable

try:
    from pixell import enmap
except ImportError:  # pragma: no cover
    enmap = None

SO_LATITUDE_DEG = -(22.0 + 57.0 / 60.0 + 31.0 / 3600.0)
SO_LONGITUDE_DEG = -(67.0 + 47.0 / 60.0 + 15.0 / 3600.0)
SO_AVOID_DEG = 30.0
# Tube index -> optics tube name and type. Index 0 is the central tube, 1-6 the inner ring and
# 7-12 the outer ring, in the order of ``tube_offsets``. ASSUMPTION: i1-i6 and o1-o6 are numbered
# in the same rotational order (and with the same starting angle) as ``tube_offsets``; the true
# focal-plane orientation of the numbering has not been checked. Types are from the UFM
# assignment (uv = UHF, mv = MF, ln = LF).
SO_TUBE_NAMES = ("c1", "i1", "i2", "i3", "i4", "i5", "i6", "o1", "o2", "o3", "o4", "o5", "o6")
SO_TUBE_TYPES = ("UHF", "MF", "UHF", "MF", "MF", "UHF", "MF", "UHF", "MF", "MF", "MF", "MF", "LF")
# Each tube is dichroic: every visit observes both bands of its type simultaneously.
SO_TUBE_BANDS = {"LF": ("f030", "f040"), "MF": ("f090", "f150"), "UHF": ("f220", "f280")}
SIDEREAL_DAY_S = 86164.0905
SPEED_OF_LIGHT_A_S = 2.99792458e18  # Angstrom per second
NJY_PER_MJY = 1e6
# Expected point-source depth (1 sigma, mJy) of one full day's map per band, combining all tubes
# that observe the band.
SO_DAILY_DEPTH_MJY = {"f030": 37.0, "f040": 25.0, "f090": 6.9, "f150": 8.3, "f220": 26.0, "f280": 49.0}
# LF bands have no measured bandpass: top-hats of unit response, (center, full width) in GHz.
SO_LF_TOPHATS = {"f030": (27.9, 4.7), "f040": (40.2, 16.9)}
SO_DEFAULT_EPOCH_MJD = 61406.0  # 2027-01-01
MAP_FIELDS = ("hits", "time_s", "start_s", "duration_s", "exposure_s", "n_passes")
SIDE_NAMES = ("east", "west")
_SCALAR_SCAN_KEYS = ("elevation_deg", "speed_deg_s", "turnaround_frac", "fov_deg", "tube_spacing_deg")
_HEADER_KEYS = ("res_deg", "dt_s", "lat_deg", "epoch_mjd", *_SCALAR_SCAN_KEYS)
SCAN_DEFAULTS = {
    "elevation_deg": 40.0,
    "az_ranges": ((30.0, 150.0), (210.0, 330.0)),
    "speed_deg_s": 1.0,
    "turnaround_frac": 0.05,
    "fov_deg": 1.25,
    "tube_spacing_deg": 1.75,
}


def main_lobe(response, threshold=0.01):
    """Return a mask of the contiguous region around the peak where the response exceeds a threshold.

    Measured SO bandpasses have low-frequency artifacts (up to ~20% of the peak at a few GHz). Once
    converted to wavelength those span an enormous range and would dominate the passband, so only
    the main lobe around the peak is kept.

    Parameters
    ----------
    response : numpy.ndarray
        Spectral response on an increasing frequency grid.
    threshold : float
        Fraction of the peak response that bounds the main lobe.

    Returns
    -------
    numpy.ndarray
        Boolean mask that is True inside the main lobe.
    """
    above = np.asarray(response) > threshold * np.max(response)
    peak = np.argmax(response)
    # Label contiguous runs of ``above``; keep the run that contains the peak.
    runs = np.cumsum(np.diff(above.astype(int), prepend=0) != 0)
    return above & (runs == runs[peak])


def ghz_passband(freq_ghz, response, name, survey="SO"):
    """Build a bolometer Passband from a response sampled on an increasing frequency grid (GHz).

    SO detectors are bolometers, so the band-averaged flux density is weighted uniformly in
    frequency: F_b = int f_nu S(nu) dnu / int S(nu) dnu. lightcurvelynx instead normalizes a
    transmission table T with the photon-counting convention, F_b = int f_nu T(lam) / lam dlam /
    int T(lam) / lam dlam, which equals weighting f_nu by T / nu. Passing T = S * nu therefore
    gives the bolometer weighting; the stored ``transmission_table`` is S * nu (in GHz), not the
    measured response.

    Parameters
    ----------
    freq_ghz : numpy.ndarray
        Positive, increasing frequencies in GHz.
    response : numpy.ndarray
        Spectral response S at each frequency (flat, bolometer weighting).
    name : str
        Filter name.
    survey : str
        Survey name.

    Returns
    -------
    Passband
        The passband on the native grid converted to wavelengths in Angstroms.
    """
    freq_ghz = np.asarray(freq_ghz, dtype=float)
    table = np.column_stack((SPEED_OF_LIGHT_A_S / (freq_ghz * 1e9), np.asarray(response) * freq_ghz))[::-1]
    return Passband(table, survey, name, delta_wave=None, trim_quantile=None)


def tophat_passbands(tophats=None, *, step_ghz=0.1, survey="SO"):
    """Build unit-response top-hat passbands.

    Parameters
    ----------
    tophats : dict, optional
        Maps filter name to (center, full width) in GHz. Defaults to ``SO_LF_TOPHATS``.
    step_ghz : float
        Frequency sampling of each top-hat in GHz.
    survey : str
        Survey name.

    Returns
    -------
    list of Passband
        One passband per entry, with response 1 from center - width/2 to center + width/2.
    """
    tophats = SO_LF_TOPHATS if tophats is None else tophats
    passbands = []
    for name, (center, width) in tophats.items():
        freq = np.linspace(center - width / 2, center + width / 2, int(np.ceil(width / step_ghz)) + 1)
        passbands.append(ghz_passband(freq, np.ones_like(freq), name, survey))
    return passbands


def so_passbands(files, *, threshold=0.01, min_ghz=40.0, tophats=None, survey="SO"):
    """Load SO bandpasses measured on a frequency grid into a PassbandGroup.

    Each CSV file has a ``frequency`` column (GHz) and one ``median_<band>`` column per band (e.g.
    ``LAT_MF_median_bands.csv`` with ``median_f090`` and ``median_f150``). Negative responses are
    set to zero, only the main lobe of each band is kept (``main_lobe``), and frequencies are
    converted to wavelengths in Angstroms. The native grid is kept (no resampling or trimming).
    The band names (``f090``, ...) match ``SO_TUBE_BANDS``. Bands without a measured bandpass
    (by default LF: ``f030`` and ``f040``) are added as unit-response top-hats from ``tophats``.

    The measured detectors do not respond below ~40 GHz, so any response there (an artifact of
    the measurement) is removed with ``min_ghz``. Band fluxes use flat (bolometer) weighting in
    frequency; see ``ghz_passband``.

    Parameters
    ----------
    files : str, Path or list of them
        CSV bandpass files.
    threshold : float
        Fraction of the peak response that bounds each band's main lobe.
    min_ghz : float
        Measured responses below this frequency (GHz) are set to zero. Not applied to top-hats.
    tophats : dict, optional
        Top-hat bands to add, as name -> (center, full width) in GHz. Defaults to
        ``SO_LF_TOPHATS``; pass an empty dict to add none.
    survey : str
        Survey name of the passbands.

    Returns
    -------
    PassbandGroup
        One passband per ``median_<band>`` column plus one per top-hat.
    """
    passbands = tophat_passbands(tophats, survey=survey)
    for fname in np.atleast_1d(files):
        data = pd.read_csv(fname).sort_values("frequency")
        freq_ghz = data["frequency"].to_numpy()
        for column in (c for c in data.columns if c.startswith("median_")):
            response = np.where(freq_ghz >= min_ghz, np.clip(data[column].to_numpy(), 0.0, None), 0.0)
            keep = main_lobe(response, threshold) & (response > 0)
            name = column.removeprefix("median_")
            passbands.append(ghz_passband(freq_ghz[keep], response[keep], name, survey))
    return PassbandGroup(passbands)


def _require_pixell():
    """Raise an informative error if pixell is not installed."""
    if enmap is None:  # pragma: no cover
        raise ImportError("SO time maps require pixell. Install it with `pip install pixell`.")


def tube_offsets(spacing_deg=1.75):
    """Return the focal-plane offsets of the 13 SO optics tubes.

    The layout is a hex pattern: one central tube, six tubes on the inner ring, and six of the
    twelve second-ring positions (those at 30 + 60k deg), so half of the outer ring is missing.

    Parameters
    ----------
    spacing_deg : float
        Center-to-center separation of neighboring tubes in degrees.

    Returns
    -------
    numpy.ndarray
        Array of shape (13, 2) with the (cross-elevation, elevation) offsets in degrees.
    """
    angles = np.radians(np.arange(6) * 60.0)
    ring = lambda radius, phase: radius * np.column_stack((np.cos(angles + phase), np.sin(angles + phase)))  # noqa: E731
    return np.vstack(([[0.0, 0.0]], ring(spacing_deg, 0.0), ring(spacing_deg * np.sqrt(3.0), np.pi / 6)))


def tube_pointing(az_deg, el_deg, offsets):
    """Apply focal-plane tube offsets to boresight pointings (flat-sky approximation).

    Parameters
    ----------
    az_deg, el_deg : numpy.ndarray
        Boresight azimuth and elevation in degrees, shape (T,).
    offsets : numpy.ndarray
        Tube (cross-elevation, elevation) offsets in degrees, shape (N, 2).

    Returns
    -------
    az, el : numpy.ndarray
        Tube-center azimuth and elevation in degrees, shape (T, N).
    """
    az = az_deg[:, None] + offsets[None, :, 0] / np.cos(np.radians(el_deg))[:, None]
    return az, el_deg[:, None] + offsets[None, :, 1]


def angular_separation_deg(lon1, lat1, lon2, lat2):
    """Return the great-circle separation (haversine) between positions in degrees.

    Parameters
    ----------
    lon1, lat1, lon2, lat2 : numpy.ndarray
        Longitudes (RA or azimuth) and latitudes (Dec or elevation) in degrees.

    Returns
    -------
    numpy.ndarray
        Angular separation in degrees.
    """
    lon1, lat1, lon2, lat2 = (np.radians(v) for v in (lon1, lat1, lon2, lat2))
    h = np.sin(0.5 * (lat2 - lat1)) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(0.5 * (lon2 - lon1)) ** 2
    return np.degrees(2.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0))))


def scan_time_step(res_deg, speed_deg_s, az_ranges, turnaround_frac, dt_s=None, **_):
    """Choose the sample interval so that one azimuth sweep is an integer number of samples.

    Parameters
    ----------
    res_deg : float
        Map resolution in degrees; the default step moves the boresight half a pixel.
    speed_deg_s : float
        Azimuth scan speed in degrees per second.
    az_ranges : tuple
        Scan azimuth ranges; the first range sets the sweep duration.
    turnaround_frac : float
        Turnaround duration as a fraction of the sweep duration.
    dt_s : float, optional
        Requested (maximum) sample interval in seconds.

    Returns
    -------
    dt_s : float
        Sample interval in seconds.
    cycle_s : float
        Duration of one sweep plus its turnaround in seconds.
    """
    sweep_s = (az_ranges[0][1] - az_ranges[0][0]) / speed_deg_s
    requested = res_deg / (2.0 * speed_deg_s) if dt_s is None else dt_s
    return sweep_s / int(np.ceil(sweep_s / requested)), sweep_s * (1.0 + turnaround_frac)


def boresight_azel(t_s, side, *, elevation_deg, az_ranges, speed_deg_s, turnaround_frac, **_):
    """Return the boresight azimuth and elevation while scanning one azimuth range.

    Each constant-speed sweep is followed by a turnaround at the end of the range lasting
    ``turnaround_frac`` of a sweep; samples taken during turnarounds are flagged as invalid.

    Parameters
    ----------
    t_s : numpy.ndarray
        Time in seconds since the start of the scan (LST = 0).
    side : int
        Index of the scanned azimuth range (0 = east/rising, 1 = west/setting).
    elevation_deg : float
        Fixed boresight elevation in degrees.
    az_ranges : tuple
        ((east_min, east_max), (west_min, west_max)) azimuth ranges in degrees.
    speed_deg_s : float
        Azimuth scan speed in degrees per second.
    turnaround_frac : float
        Turnaround duration as a fraction of the sweep duration.

    Returns
    -------
    az, el : numpy.ndarray
        Boresight azimuth and elevation in degrees (held at the range end during turnarounds).
    valid : numpy.ndarray
        True for samples taken during a sweep (not a turnaround).
    """
    low, high = az_ranges[side]
    throw_s = (high - low) / speed_deg_s
    cycle_s = throw_s * (1.0 + turnaround_frac)
    phase = t_s % cycle_s
    forward = (t_s // cycle_s) % 2 == 0
    travel = speed_deg_s * np.minimum(phase, throw_s)
    az = np.where(forward, low + travel, high - travel)
    return az, np.full_like(az, elevation_deg), phase < throw_s


def precession_matrix(epoch_mjd):
    """Return the IAU 2006 bias-precession matrix from ICRS to the mean equator and equinox of date.

    Parameters
    ----------
    epoch_mjd : float
        Epoch (MJD, TT; the UTC-TT difference is negligible here).

    Returns
    -------
    numpy.ndarray
        3x3 rotation matrix R with v_date = R @ v_icrs.
    """
    return erfa.pmat06(2400000.5, epoch_mjd)


def rotate_radec(ra_deg, dec_deg, matrix):
    """Apply a rotation matrix to equatorial coordinates.

    Parameters
    ----------
    ra_deg, dec_deg : numpy.ndarray
        Right ascension and declination in degrees.
    matrix : numpy.ndarray
        3x3 rotation matrix.

    Returns
    -------
    ra, dec : numpy.ndarray
        Rotated right ascension in [0, 360) and declination in degrees.
    """
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    vec = np.stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)), axis=-1) @ matrix.T
    ra_out = np.degrees(np.arctan2(vec[..., 1], vec[..., 0])) % 360.0
    return ra_out, np.degrees(np.arcsin(np.clip(vec[..., 2], -1.0, 1.0)))


def horizon_to_equatorial(az_deg, el_deg, lst_deg, lat_deg=SO_LATITUDE_DEG):
    """Convert horizon coordinates to equatorial coordinates (no refraction or aberration).

    Parameters
    ----------
    az_deg, el_deg : numpy.ndarray
        Azimuth (east of north) and elevation in degrees.
    lst_deg : numpy.ndarray
        Local sidereal time in degrees.
    lat_deg : float
        Site latitude in degrees.

    Returns
    -------
    ra, dec : numpy.ndarray
        Right ascension in [0, 360) and declination in degrees.
    """
    az, el, lat = np.radians(az_deg), np.radians(el_deg), np.radians(lat_deg)
    sin_dec = np.sin(lat) * np.sin(el) + np.cos(lat) * np.cos(el) * np.cos(az)
    hour_angle = np.arctan2(
        -np.sin(az) * np.cos(el), np.cos(lat) * np.sin(el) - np.sin(lat) * np.cos(el) * np.cos(az)
    )
    ra = (lst_deg - np.degrees(hour_angle)) % 360.0
    return ra, np.degrees(np.arcsin(np.clip(sin_dec, -1.0, 1.0)))


def sky_to_pixel(shape, wcs, ra, dec):
    """Return the flat index of the nearest map pixel for each sky position.

    Parameters
    ----------
    shape : tuple
        Map shape; the last two entries are (ny, nx).
    wcs : astropy.wcs.WCS
        Full-sky (in RA) CAR map WCS.
    ra, dec : numpy.ndarray
        Sky coordinates in degrees.

    Returns
    -------
    numpy.ndarray
        Flat pixel index (y * nx + x), or -1 for positions outside the map.
    """
    _require_pixell()
    ny, nx = shape[-2:]
    y, x = np.floor(enmap.sky2pix(shape, wcs, np.radians([np.ravel(dec), np.ravel(ra)])) + 0.5).astype(int)
    x %= nx
    return np.where((y >= 0) & (y < ny), y * nx + x, -1)


def transit_day(time_s, ra_deg):
    """Return the index of the sidereal day in which the meridian transit nearest a crossing occurs.

    The rising crossing of a patch precedes its transit and the setting crossing follows it, so
    both crossings around one transit share an index; the daily rising/setting switch alternates
    on this index.

    Parameters
    ----------
    time_s : numpy.ndarray
        Crossing times in seconds since a reference LST = 0.
    ra_deg : numpy.ndarray
        Right ascension of the patch in degrees.

    Returns
    -------
    numpy.ndarray
        Integer day index counted from the reference LST = 0.
    """
    ra_s = (np.asarray(ra_deg) % 360.0) / 360.0 * SIDEREAL_DAY_S
    return np.floor((time_s - ra_s) / SIDEREAL_DAY_S + 0.5).astype(int)


def simulate_time_maps(
    res_deg=0.5,
    *,
    tubes=None,
    dec_range=(-70.0, 30.0),
    dt_s=None,
    lat_deg=SO_LATITUDE_DEG,
    epoch_mjd=SO_DEFAULT_EPOCH_MJD,
    max_points=4_000_000,
    **scan_kwargs,
):
    """Simulate one sidereal day of SO scanning in each azimuth range and accumulate visit maps.

    At every time sample, each map pixel whose center lies within a tube's circular field of view
    is hit, so all statistics are exact for the pixel-center position up to the sample interval.
    Each range is scanned for a full sidereal day, so a pixel is crossed once while rising (east
    range) and once while setting (west range); both crossings are kept and ``from_time_maps``
    picks one per day.

    Parameters
    ----------
    res_deg : float
        Map resolution in degrees.
    tubes : list of int, optional
        Tube indices to simulate. Defaults to all 13.
    dec_range : tuple
        (min, max) declination of the map in degrees.
    dt_s : float, optional
        Maximum sample interval in seconds. Defaults to res_deg / (2 * speed); it is reduced so
        each azimuth sweep holds an integer number of samples.
    lat_deg : float
        Site latitude in degrees.
    epoch_mjd : float
        Epoch (MJD) whose mean equator and equinox define LST; pointings are precessed to ICRS.
        Use a date near the middle of the survey.
    max_points : int
        Maximum number of (time, tube, pixel) candidates processed per chunk (memory control).
    **scan_kwargs
        Overrides of ``SCAN_DEFAULTS``.

    Returns
    -------
    dict
        ``maps``: enmap of shape (len(MAP_FIELDS), ntube, 2, ny, nx), where the third axis is the
        scan side (0 = east/rising, 1 = west/setting). Per tube, side and pixel it holds the
        number of in-FoV samples (``hits``), the circular mean time of the visit (``time_s``),
        the first in-FoV time (``start_s``), the visit length from first to last in-FoV sample
        (``duration_s``), the total time inside the FoV (``exposure_s``) and the number of azimuth
        sweeps that pass over the pixel (``n_passes``). Times are seconds since LST = 0; unobserved
        pixels are NaN. Map coordinates are ICRS.
        Also contains ``tubes``, ``res_deg``, ``dt_s``, ``lat_deg``, ``epoch_mjd`` and the scan
        parameters.
    """
    _require_pixell()
    scan = {**SCAN_DEFAULTS, **scan_kwargs}
    tubes = np.arange(13) if tubes is None else np.atleast_1d(tubes).astype(int)
    dt_s, cycle_s = scan_time_step(res_deg, dt_s=dt_s, **scan)
    to_icrs = precession_matrix(epoch_mjd).T
    shape, wcs = enmap.band_geometry(np.radians(dec_range), res=np.radians(res_deg))
    ny, nx = shape
    dec_pix, ra_pix = np.degrees(enmap.posmap(shape, wcs))
    dec_pix, ra_pix = dec_pix[:, 0], ra_pix[0]

    # Pixel stencil around each tube center that covers the field of view at all declinations.
    radius = 0.5 * scan["fov_deg"]
    max_dec = min(89.0, max(map(abs, dec_range)) + radius)
    hy = int(np.ceil(radius / res_deg + 0.5))
    hx = int(np.ceil(radius / (res_deg * np.cos(np.radians(max_dec))) + 0.5))
    dy, dx = (g.ravel() for g in np.meshgrid(np.arange(-hy, hy + 1), np.arange(-hx, hx + 1), indexing="ij"))
    centers = tube_offsets(scan["tube_spacing_deg"])[tubes]

    nbins = len(tubes) * 2 * ny * nx
    hits, cos_sum, sin_sum, passes = (np.zeros(nbins) for _ in range(4))
    t_min, t_min_shift = np.full(nbins, np.inf), np.full(nbins, np.inf)
    t_max, t_max_shift = np.full(nbins, -np.inf), np.full(nbins, -np.inf)
    n_samples = int(np.ceil(SIDEREAL_DAY_S / dt_s))
    n_sweeps = int(SIDEREAL_DAY_S // cycle_s) + 1
    chunk = max(1, int(max_points * dt_s / (len(tubes) * len(dy) * cycle_s))) * cycle_s  # whole sweeps
    edges = np.append(np.ceil(np.arange(0.0, SIDEREAL_DAY_S, chunk) / dt_s), n_samples).astype(int)
    for side, (k0, k1) in ((s, e) for s in range(2) for e in zip(edges[:-1], edges[1:], strict=True)):
        k = np.arange(k0, k1)
        t = k * dt_s
        az, el, valid = boresight_azel(t, side, **scan)
        k, t, az, el = k[valid], t[valid], az[valid], el[valid]
        tube_az, tube_el = tube_pointing(az, el, centers)
        lst = (360.0 * t / SIDEREAL_DAY_S)[:, None]
        ra_c, dec_c = rotate_radec(*horizon_to_equatorial(tube_az, tube_el, lst, lat_deg), to_icrs)
        center_pix = sky_to_pixel(shape, wcs, ra_c, dec_c).reshape(ra_c.shape)[..., None]
        y = center_pix // nx + dy
        x = (center_pix % nx + dx) % nx
        inside = (center_pix >= 0) & (y >= 0) & (y < ny)
        y = np.clip(y, 0, ny - 1)
        inside &= angular_separation_deg(ra_pix[x], dec_pix[y], ra_c[..., None], dec_c[..., None]) <= radius
        it, itube, _ = np.nonzero(inside)
        bins = ((itube * 2 + side) * ny + y[inside]) * nx + x[inside]
        tt = t[it]
        phase = 2.0 * np.pi * tt / SIDEREAL_DAY_S
        hits += np.bincount(bins, minlength=nbins)
        cos_sum += np.bincount(bins, weights=np.cos(phase), minlength=nbins)
        sin_sum += np.bincount(bins, weights=np.sin(phase), minlength=nbins)
        # Track first/last times both directly and shifted by half a day, so visits that wrap
        # around LST = 0 still get the correct (shorter) span.
        shifted = (tt + 0.5 * SIDEREAL_DAY_S) % SIDEREAL_DAY_S
        np.minimum.at(t_min, bins, tt)
        np.maximum.at(t_max, bins, tt)
        np.minimum.at(t_min_shift, bins, shifted)
        np.maximum.at(t_max_shift, bins, shifted)
        sweeps = np.unique(bins.astype(np.int64) * n_sweeps + (tt // cycle_s).astype(np.int64))
        passes += np.bincount(sweeps // n_sweeps, minlength=nbins)

    hits, cos_sum, sin_sum, passes, t_min, t_max, t_min_shift, t_max_shift = (
        a.reshape(len(tubes), 2, ny, nx)
        for a in (hits, cos_sum, sin_sum, passes, t_min, t_max, t_min_shift, t_max_shift)
    )
    observed = hits > 0
    span, span_shift = t_max - t_min, t_max_shift - t_min_shift
    use_shift = span_shift < span
    with np.errstate(invalid="ignore"):  # Unobserved pixels have infinite bounds.
        start_s = np.where(use_shift, (t_min_shift - 0.5 * SIDEREAL_DAY_S) % SIDEREAL_DAY_S, t_min)
    mean_s = (np.arctan2(sin_sum, cos_sum) % (2.0 * np.pi)) * SIDEREAL_DAY_S / (2.0 * np.pi)
    nan_if_unobserved = lambda a: np.where(observed, a, np.nan)  # noqa: E731
    fields = {
        "hits": hits,
        "time_s": nan_if_unobserved(mean_s),
        "start_s": nan_if_unobserved(start_s),
        "duration_s": nan_if_unobserved(np.where(use_shift, span_shift, span) + dt_s),
        "exposure_s": hits * dt_s,
        "n_passes": passes,
    }
    return {
        "maps": enmap.enmap(np.stack([fields[name] for name in MAP_FIELDS]), wcs),
        "tubes": tubes,
        "res_deg": float(res_deg),
        "dt_s": float(dt_s),
        "lat_deg": float(lat_deg),
        "epoch_mjd": float(epoch_mjd),
        **scan,
    }


def equatorial_to_horizon(ra_deg, dec_deg, lst_deg, lat_deg=SO_LATITUDE_DEG):
    """Convert equatorial coordinates to horizon coordinates (no refraction or aberration).

    Parameters
    ----------
    ra_deg, dec_deg : float or numpy.ndarray
        Right ascension and declination in degrees.
    lst_deg : numpy.ndarray
        Local sidereal time in degrees.
    lat_deg : float
        Site latitude in degrees.

    Returns
    -------
    az, el : numpy.ndarray
        Azimuth (east of north) in [0, 360) and elevation in degrees.
    """
    hour_angle, dec, lat = np.radians(lst_deg - ra_deg), np.radians(dec_deg), np.radians(lat_deg)
    sin_el = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(hour_angle)
    az = np.arctan2(
        -np.cos(dec) * np.sin(hour_angle),
        np.sin(dec) * np.cos(lat) - np.cos(dec) * np.sin(lat) * np.cos(hour_angle),
    )
    return np.degrees(az) % 360.0, np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))


def point_visits(
    ra,
    dec,
    tube_id,
    *,
    res_deg=0.5,
    dt_s=None,
    lat_deg=SO_LATITUDE_DEG,
    epoch_mjd=SO_DEFAULT_EPOCH_MJD,
    max_gap_s=1800.0,
    **scan_kwargs,
):
    """Trace when one sky position is inside a tube's field of view over one day of each scan side.

    This is an independent, single-point reference for ``simulate_time_maps``: it converts the
    source into horizon coordinates and tests its separation from the tube center there. Pass a
    ``simulate_time_maps`` result as keyword arguments to use the same scan and time step.

    Parameters
    ----------
    ra, dec : float
        Source position (ICRS) in degrees.
    tube_id : int
        Optics-tube index.
    res_deg, dt_s : float, optional
        Used to choose the sample interval as in ``simulate_time_maps``.
    lat_deg : float
        Site latitude in degrees.
    epoch_mjd : float
        Epoch (MJD) of the mean equator and equinox used for LST.
    max_gap_s : float
        In-FoV samples separated by more than this (seconds) belong to different visits.
    **scan_kwargs
        Overrides of ``SCAN_DEFAULTS`` (other keys are ignored).

    Returns
    -------
    list of dict
        One entry per visit with ``side`` (scanned range), ``time_s`` (circular mean), ``start_s``,
        ``duration_s``, ``exposure_s`` and ``n_passes``; times in seconds since LST = 0.
    """
    scan = {key: scan_kwargs.get(key, value) for key, value in SCAN_DEFAULTS.items()}
    dt_s, cycle_s = scan_time_step(res_deg, dt_s=dt_s, **scan)
    k = np.arange(int(np.ceil(SIDEREAL_DAY_S / dt_s)))
    t = k * dt_s
    ra_date, dec_date = rotate_radec(ra, dec, precession_matrix(epoch_mjd))
    src_az, src_el = equatorial_to_horizon(ra_date, dec_date, 360.0 * t / SIDEREAL_DAY_S, lat_deg)
    visits = []
    for side in range(2):
        az, el, valid = boresight_azel(t, side, **scan)
        tube_az, tube_el = tube_pointing(az, el, tube_offsets(scan["tube_spacing_deg"])[[tube_id]])
        separation = angular_separation_deg(tube_az[:, 0], tube_el[:, 0], src_az, src_el)
        inside = np.flatnonzero(valid & (separation <= 0.5 * scan["fov_deg"]))
        if len(inside) == 0:
            continue
        groups = np.split(inside, np.flatnonzero(np.diff(t[inside]) > max_gap_s) + 1)
        unwrapped = [k[g].astype(float) for g in groups]
        wraps = t[groups[0][0]] < max_gap_s and t[groups[-1][-1]] > SIDEREAL_DAY_S - max_gap_s
        if len(groups) > 1 and wraps:
            unwrapped[0] = np.concatenate((unwrapped.pop() - len(k), unwrapped[0]))
            groups[0] = np.concatenate((groups.pop(), groups[0]))
        for g, ku in zip(groups, unwrapped, strict=True):
            times = ku * dt_s
            visits.append(
                {
                    "side": side,
                    "time_s": times.mean() % SIDEREAL_DAY_S,
                    "start_s": times[0] % SIDEREAL_DAY_S,
                    "duration_s": times[-1] - times[0] + dt_s,
                    "exposure_s": len(g) * dt_s,
                    "n_passes": len(np.unique(t[g] // cycle_s)),
                }
            )
    return visits


def write_time_maps(fname, time_maps):
    """Write simulated time maps (from ``simulate_time_maps``) to a FITS file.

    Parameters
    ----------
    fname : str or Path
        Output FITS filename.
    time_maps : dict
        Output of ``simulate_time_maps``.
    """
    _require_pixell()
    extra = {
        "TUBES": ",".join(map(str, time_maps["tubes"])),
        "AZRANGES": ",".join(str(v) for r in time_maps["az_ranges"] for v in r),
    }
    extra.update({key[:8].upper(): time_maps[key] for key in _HEADER_KEYS})
    enmap.write_map(str(fname), time_maps["maps"], extra=extra, allow_modify=True)


def read_time_maps(fname):
    """Read time maps written by ``write_time_maps``.

    Parameters
    ----------
    fname : str or Path
        FITS filename.

    Returns
    -------
    dict
        The same structure returned by ``simulate_time_maps``.
    """
    _require_pixell()
    from astropy.io import fits

    header = fits.getheader(str(fname))
    ranges = [float(v) for v in header["AZRANGES"].split(",")]
    result = {key: float(header[key[:8].upper()]) for key in _HEADER_KEYS}
    result.update(
        maps=enmap.read_map(str(fname)),
        tubes=np.array([int(v) for v in header["TUBES"].split(",")]),
        az_ranges=(tuple(ranges[:2]), tuple(ranges[2:])),
    )
    return result


def survey_epoch(start_date):
    """Return the start MJD and the MJD of the preceding LST = 0 at the SO site.

    Parameters
    ----------
    start_date : str or float
        ISO date string (UTC) or MJD.

    Returns
    -------
    start_mjd, lst0_mjd : float
        Start time and the time of the last LST = 0 at or before it (MJD).
    """
    is_mjd = isinstance(start_date, int | float | np.number)
    start = Time(start_date, format="mjd" if is_mjd else None, scale="utc")
    lst_start = start.sidereal_time("mean", longitude=SO_LONGITUDE_DEG).deg
    return start.mjd, start.mjd - lst_start / 360.0 * SIDEREAL_DAY_S / 86400.0


def merge_intervals(intervals):
    """Merge overlapping [start, end) intervals.

    Parameters
    ----------
    intervals : array-like
        Intervals of shape (N, 2).

    Returns
    -------
    numpy.ndarray
        Sorted, disjoint intervals of shape (M, 2).
    """
    intervals = np.asarray(intervals, dtype=float).reshape(-1, 2)
    intervals = intervals[np.argsort(intervals[:, 0])]
    if len(intervals) == 0:
        return intervals
    # A new group starts where an interval begins after every earlier interval has ended.
    ends = np.maximum.accumulate(intervals[:, 1])
    starts = np.flatnonzero(np.r_[True, intervals[1:, 0] > ends[:-1]])
    return np.column_stack((intervals[starts, 0], np.maximum.reduceat(intervals[:, 1], starts)))


def overlaps_intervals(t_start, t_end, intervals):
    """Return whether each [t_start, t_end] span overlaps any of the merged intervals.

    Parameters
    ----------
    t_start, t_end : numpy.ndarray
        Span boundaries.
    intervals : numpy.ndarray
        Sorted, disjoint intervals of shape (M, 2) (see ``merge_intervals``).

    Returns
    -------
    numpy.ndarray
        Boolean overlap flag per span.
    """
    if len(intervals) == 0:
        return np.zeros(np.shape(t_start), dtype=bool)
    # Only the last interval starting before t_end can overlap, because the intervals are disjoint.
    idx = np.searchsorted(intervals[:, 0], t_end, side="left") - 1
    return (idx >= 0) & (intervals[np.maximum(idx, 0), 1] > t_start)


def simulate_downtime(
    start_date,
    n_days,
    *,
    uptime=0.65,
    chunk_fraction=0.75,
    chunk_days=(3.0, 14.0),
    chunk_months=(6, 7, 8, 9),
    chunk_season_weight=3.0,
    gap_hours=(1.0, 6.0),
    gap_local_hours=(8.0, 20.0),
    seed=None,
):
    """Draw random instrument downtime intervals.

    Downtime has two parts: long outages of days to weeks, preferentially in the austral winter
    (June-September), and scattered interruptions of a few hours that start during local daytime.
    Outages are added until they cover ``chunk_fraction`` of the downtime budget, then
    interruptions until the total downtime reaches ``(1 - uptime) * n_days``.

    Parameters
    ----------
    start_date : str or float
        Survey start as an ISO date string (UTC) or MJD.
    n_days : float
        Survey length in days.
    uptime : float
        Fraction of the time the instrument is observing.
    chunk_fraction : float
        Fraction of the downtime in long outages.
    chunk_days : tuple
        (min, max) outage length in days (uniform).
    chunk_months : tuple
        Months in which outages are preferentially placed.
    chunk_season_weight : float
        Relative rate of outage starts in ``chunk_months``.
    gap_hours : tuple
        (min, max) interruption length in hours (uniform).
    gap_local_hours : tuple
        (min, max) local mean solar time (hours) at which interruptions start (uniform).
    seed : int, optional
        Random seed.

    Returns
    -------
    numpy.ndarray
        Sorted, disjoint downtime intervals (MJD) of shape (M, 2) within the survey.
    """
    rng = np.random.default_rng(seed)
    start_mjd, _ = survey_epoch(start_date)
    end_mjd = start_mjd + n_days
    days = start_mjd + np.arange(int(np.ceil(n_days)))
    months = Time(days, format="mjd").ymdhms.month
    weights = np.where(np.isin(months, chunk_months), chunk_season_weight, 1.0)
    utc_minus_local = -SO_LONGITUDE_DEG / 360.0  # Local mean solar time = UTC + longitude / 15 h.

    intervals = np.empty((0, 2))
    total = lambda: np.sum(np.diff(intervals, axis=1))  # noqa: E731
    budgets = ((1.0 - uptime) * n_days * chunk_fraction, (1.0 - uptime) * n_days)
    for kind, budget in enumerate(budgets):
        while total() < budget - 1e-9:
            if kind == 0:
                begin = rng.choice(days, p=weights / weights.sum()) + rng.uniform(0.0, 1.0)
                length = rng.uniform(*chunk_days)
            else:
                begin = np.floor(rng.choice(days)) + rng.uniform(*gap_local_hours) / 24.0 + utc_minus_local
                length = rng.uniform(*gap_hours) / 24.0
            length = min(length, budget - total())
            new = np.clip([[begin, begin + length]], start_mjd, end_mjd)
            intervals = merge_intervals(np.vstack((intervals, new)))
    return intervals


def sun_moon_cut(time_mjd, ra_deg, dec_deg, avoid_deg=SO_AVOID_DEG, step_min=10.0):
    """Flag observations that point within ``avoid_deg`` of the sun or the moon.

    SO does not observe near the sun or moon; it scans another patch instead, so those visits
    are lost. The topocentric sun and moon positions are computed on a regular time grid and each
    observation uses the nearest grid time (the moon moves about 0.5 deg per hour, so the default
    10 minute grid is accurate to ~0.1 deg).

    Parameters
    ----------
    time_mjd : numpy.ndarray
        Observation times in MJD.
    ra_deg, dec_deg : numpy.ndarray
        Pointing coordinates in degrees (same length as ``time_mjd``).
    avoid_deg : float
        Minimum allowed separation from the sun and the moon in degrees.
    step_min : float
        Time grid step in minutes.

    Returns
    -------
    numpy.ndarray
        Boolean array that is True for observations too close to the sun or moon.
    """
    time_mjd, ra_deg, dec_deg = (np.asarray(v, dtype=float) for v in (time_mjd, ra_deg, dec_deg))
    if len(time_mjd) == 0:
        return np.zeros(0, dtype=bool)
    # The grid spans whole UTC days so repeated calls over the same dates reuse cached positions.
    first_day, last_day = int(np.floor(time_mjd.min())), int(np.floor(time_mjd.max())) + 1
    grid0, step, positions = body_positions(first_day, last_day, step_min)
    index = np.rint((time_mjd - grid0) / step).astype(int)
    too_close = np.zeros(len(time_mjd), dtype=bool)
    for ra_body, dec_body in positions:
        too_close |= angular_separation_deg(ra_deg, dec_deg, ra_body[index], dec_body[index]) < avoid_deg
    return too_close


@functools.lru_cache(maxsize=8)
def body_positions(first_day, last_day, step_min):
    """Return topocentric sun and moon positions at the SO site on a regular time grid (cached).

    Parameters
    ----------
    first_day, last_day : int
        Grid range in whole MJD days, [first_day, last_day].
    step_min : float
        Grid step in minutes.

    Returns
    -------
    grid0 : float
        MJD of the first grid point.
    step : float
        Grid step in days.
    positions : tuple
        ((ra, dec) of the sun, (ra, dec) of the moon) in degrees on the grid.
    """
    step = step_min / 1440.0
    grid = np.arange(first_day, last_day + step, step)
    site = EarthLocation.from_geodetic(SO_LONGITUDE_DEG, SO_LATITUDE_DEG, height=5200.0)
    bodies = (get_body(body, Time(grid, format="mjd"), site) for body in ("sun", "moon"))
    return grid[0], step, tuple((pos.ra.deg, pos.dec.deg) for pos in bodies)


def daily_exposure_scale(table, cut):
    """Return the factor that restores each tube's daily exposure after dropping ``cut`` visits.

    Parameters
    ----------
    table : pandas.DataFrame
        Visit table with ``time`` (MJD), ``tube_id`` and ``exposure_s`` columns.
    cut : numpy.ndarray
        Boolean array, True for visits to drop.

    Returns
    -------
    pandas.Series
        ``total / kept`` exposure indexed by (``tube_id``, UTC day as floor(MJD)); infinite on a
        day where every visit of a tube is cut.
    """
    exposure = table["exposure_s"].astype(float)
    group = [table["tube_id"].rename("tube_id"), np.floor(table["time"]).rename("day")]
    return exposure.groupby(group).sum() / exposure.where(~cut, 0.0).groupby(group).sum()


def conserve_exposure(table, cut, scale=None):
    """Drop cut visits and give their exposure to the remaining visits of the same tube and day.

    This implements the assumption that sun and moon avoidance re-points the telescope rather than
    losing observing time: on each UTC day, the surviving visits of each tube have their
    ``exposure_s`` multiplied by ``1 / (1 - f)``, where ``f`` is the fraction of that tube's
    exposure on that day removed by ``cut``. The factor is stored in ``exposure_scale``. A day on
    which every visit of a tube is cut has no visits left to receive the time, so it is lost.

    Parameters
    ----------
    table : pandas.DataFrame
        Visit table with ``time`` (MJD), ``tube_id`` and ``exposure_s`` columns.
    cut : numpy.ndarray
        Boolean array, True for visits to drop.
    scale : pandas.Series, optional
        Precomputed ``daily_exposure_scale``, e.g. estimated from the full sky when ``table`` only
        holds a few pixels. Defaults to the scale computed from ``table`` itself; (tube, day)
        pairs missing from ``scale`` get 1.

    Returns
    -------
    pandas.DataFrame
        The surviving visits with rescaled ``exposure_s`` and the ``exposure_scale`` column.
    """
    scale = daily_exposure_scale(table, cut) if scale is None else scale
    keys = pd.MultiIndex.from_arrays([table["tube_id"], np.floor(table["time"])])
    factor = scale.reindex(keys).fillna(1.0).to_numpy()
    table = table.assign(exposure_scale=factor.astype(np.float32))[~cut]
    table["exposure_s"] = (table["exposure_s"] * table["exposure_scale"]).astype(np.float32)
    return table.reset_index(drop=True)


def visit_schedule(time_maps, tube, side, start_date, n_days, *, first_side=0, downtime=None, pixels=None):
    """Return the visits of one tube on one scan side over a date range.

    Successive meridian transits of each pixel alternate between its rising (east) and setting
    (west) crossing; this returns the visits on the transit days assigned to ``side``.

    Parameters
    ----------
    time_maps : dict
        Output of ``simulate_time_maps`` or ``read_time_maps``.
    tube : int
        Tube index (must be in ``time_maps``).
    side : int
        Scan side (0 = east/rising, 1 = west/setting).
    start_date : str or float
        Survey start as an ISO date string (UTC) or MJD.
    n_days : float
        Survey length in days; visits outside [start, start + n_days) are dropped.
    first_side : int
        Scan side for transits on the sidereal day (from LST = 0) containing the start.
    downtime : numpy.ndarray, optional
        Merged downtime intervals (MJD); visits overlapping them are dropped.
    pixels : numpy.ndarray, optional
        Flat map pixel indices to restrict the visits to. Defaults to all observed pixels.

    Returns
    -------
    dict
        Arrays ``pixel`` (flat map index), ``transit_day``, ``time``, ``time_start`` and
        ``time_end`` (MJD), one entry per visit.
    """
    maps = time_maps["maps"]
    start_mjd, lst0_mjd = survey_epoch(start_date)
    sub = maps[:, list(time_maps["tubes"]).index(tube), side].reshape(len(MAP_FIELDS), -1)
    fields = dict(zip(MAP_FIELDS, sub, strict=True))
    observed = np.flatnonzero(fields["hits"] > 0)
    pixels = observed if pixels is None else np.intersect1d(observed, pixels)
    ra = np.degrees(np.asarray(maps.posmap())[1]).ravel()[pixels]
    days = np.arange(-1, int(np.ceil(n_days * 86400.0 / SIDEREAL_DAY_S)) + 2)
    time_s = days[:, None] * SIDEREAL_DAY_S + fields["time_s"][pixels][None, :]
    day = transit_day(time_s, ra)
    mjd = lst0_mjd + time_s / 86400.0
    keep = (mjd >= start_mjd) & (mjd < start_mjd + n_days) & ((day + first_side) % 2 == side)
    pix = np.broadcast_to(pixels, mjd.shape)[keep]
    # Start relative to the mean time, wrapped so visits spanning LST = 0 stay contiguous.
    offset_s = (fields["start_s"][pix] - fields["time_s"][pix] + 0.5 * SIDEREAL_DAY_S) % SIDEREAL_DAY_S
    visits = {"pixel": pix, "transit_day": day[keep], "time": mjd[keep]}
    visits["time_start"] = visits["time"] + (offset_s - 0.5 * SIDEREAL_DAY_S) / 86400.0
    visits["time_end"] = visits["time_start"] + fields["duration_s"][pix] / 86400.0
    if downtime is not None:
        up = ~overlaps_intervals(visits["time_start"], visits["time_end"], downtime)
        visits = {key: value[up] for key, value in visits.items()}
    return visits


def daily_reference_exposure(time_maps, band_names=None):
    """Return the typical exposure of one full day's map per band, summed over the band's tubes.

    For each simulated tube, a pixel's daily exposure is the mean of its east and west crossing
    exposures (successive days alternate between them); the tube's typical value is the median over
    observed pixels. The tubes have nearly identical values (~3.8 s), so the band's reference is the
    mean over the simulated tubes times the number of SO tubes observing the band (e.g. 8 for f090).
    Downtime and the sun and moon cut are not included: the depths describe a typical observing day,
    and ``conserve_exposure`` keeps the average daily exposure unchanged.

    Parameters
    ----------
    time_maps : dict
        Output of ``simulate_time_maps`` or ``read_time_maps`` (any subset of tubes).
    band_names : dict, optional
        Maps tube type to its band names. Defaults to ``SO_TUBE_BANDS``.

    Returns
    -------
    dict
        Reference daily exposure in seconds for each band.
    """
    band_names = SO_TUBE_BANDS if band_names is None else band_names
    exposure = np.asarray(time_maps["maps"][MAP_FIELDS.index("exposure_s")]).mean(axis=1)
    per_tube = np.mean([np.median(daily[daily > 0]) for daily in exposure])
    n_tubes = {kind: SO_TUBE_TYPES.count(kind) for kind in band_names}
    return {band: n_tubes[kind] * per_tube for kind, bands in band_names.items() for band in bands}


def add_bandflux_error(table, time_maps, *, depths=None, band_names=None):
    """Add a ``bandflux_error`` column (nJy) from the daily map depths, scaled by each visit's exposure.

    Noise scales as 1 / sqrt(exposure), so a visit with exposure ``exposure_s`` has error
    ``depth * sqrt(T_day / exposure_s)``, where ``depth`` is the band's depth for one full day's
    map and ``T_day`` is that map's typical exposure (``daily_reference_exposure``). Use with
    ``GivenNoiseModel``.

    Parameters
    ----------
    table : SimonsObservatoryObsTable
        The table to modify in place.
    time_maps : dict
        The time maps the table was built from.
    depths : dict, optional
        One-day map depth (1 sigma, mJy) per band. Defaults to ``SO_DAILY_DEPTH_MJY``.
    band_names : dict, optional
        Maps tube type to its band names. Defaults to ``SO_TUBE_BANDS``.
    """
    depths = SO_DAILY_DEPTH_MJY if depths is None else depths
    reference = daily_reference_exposure(time_maps, band_names)
    bands = table["filter"].to_numpy(dtype=str)
    error = np.empty(len(bands))
    for band in np.unique(bands):
        rows = bands == band
        error[rows] = depths[band] * NJY_PER_MJY * np.sqrt(reference[band] / table["exposure_s"][rows])
    table.add_column("bandflux_error", error, overwrite=True)


class SimonsObservatoryObsTable(ObsTable):
    """An ObsTable of Simons Observatory map-pixel observations for one or more optics tubes.

    Each row is one visit of one map pixel by one tube on one day; ``side`` says whether it was the
    rising (east) or setting (west) crossing, which alternate with ``transit_day``. ``ra`` and
    ``dec`` are the pixel center and ``time`` is the mean in-FoV time of the visit (MJD).
    ``time_start``/``time_end`` (MJD) and ``duration_s`` give the span from the first to the last
    in-FoV sample, ``exposure_s`` the total time inside the tube's field of view (s), ``n_passes``
    the number of azimuth sweeps over the pixel, ``hits`` the number of in-FoV samples, and
    ``transit_day`` the sidereal day (from the LST = 0 preceding the start) of the transit the
    visit belongs to. Rows are normally created with ``from_time_maps``.

    Spatial queries map a position to its map pixel instead of using a cone search, so each source
    matches exactly the rows of the pixel that contains it.

    Parameters
    ----------
    table : dict or pandas.DataFrame
        Observation rows; must include ``pixel_index`` (flat map pixel index).
    map_shape : tuple
        Map shape (ny, nx) of the time maps.
    map_wcs : astropy.wcs.WCS
        Map WCS of the time maps.
    **kwargs
        Additional arguments and survey values passed to ``ObsTable``.
    """

    _required_columns = ["ra", "dec", "time", "filter", "pixel_index"]
    _default_survey_values = {
        **ObsTable._default_survey_values,
        "survey_name": "SO",
        "site_latitude_deg": SO_LATITUDE_DEG,
        "site_longitude_deg": SO_LONGITUDE_DEG,
        "map_shape": None,
        "map_wcs": None,
    }

    def __init__(self, table, *, map_shape, map_wcs, **kwargs):
        super().__init__(table, map_shape=tuple(map_shape[-2:]), map_wcs=map_wcs, **kwargs)

    @classmethod
    def from_time_maps(
        cls,
        time_maps,
        start_date,
        n_days,
        *,
        tubes=None,
        band_names=None,
        first_side=0,
        downtime=None,
        avoid_deg=SO_AVOID_DEG,
        pixels=None,
        n_scale_pixels=3000,
        seed=0,
        scale=None,
        **kwargs,
    ):
        """Materialize simulated time maps into an ObsTable over a date range.

        Successive meridian transits of each pixel alternate between its rising (east) and its
        setting (west) crossing, so each pixel is observed by each tube at most once per day.

        Parameters
        ----------
        time_maps : dict
            Output of ``simulate_time_maps`` or ``read_time_maps``.
        start_date : str or float
            Start of the survey as an ISO date string (UTC) or MJD.
        n_days : float
            Survey length in days; rows outside [start, start + n_days) are dropped.
        tubes : list of int, optional
            Tubes to include. Defaults to all tubes in ``time_maps``.
        band_names : dict, optional
            Maps each tube type (``LF``, ``MF``, ``UHF``) to the filter names of the bands it
            observes; each visit gives one row per band, with that band as ``filter``. The names
            must match the passbands used with the table. Defaults to ``SO_TUBE_BANDS``.
        first_side : int
            Scan side (0 = east/rising, 1 = west/setting) for transits on the sidereal day (from
            LST = 0) containing the start; the sides alternate on later days.
        downtime : numpy.ndarray, optional
            Downtime intervals (MJD), e.g. from ``simulate_downtime``; visits overlapping them
            are dropped.
        avoid_deg : float or None
            Visits whose pixel center is within this many degrees of the sun or the moon at the
            mean visit time are dropped (SO scans another patch instead). Their exposure is
            redistributed over the tube's remaining visits that day by ``conserve_exposure``, so
            total observing time is unchanged (an assumption; see the module docstring). None
            disables the cut and the rescaling.
        pixels : array_like, optional
            Flat map pixel indices to build rows for, e.g. the pixels containing simulated sources.
            This keeps long surveys small. The sun/moon rescaling must reflect each tube's whole
            day, so it is then estimated from ``n_scale_pixels`` random observed pixels.
        n_scale_pixels : int
            Number of random pixels used to estimate the rescaling when ``pixels`` is given.
        seed : int
            Random seed for choosing those pixels.
        scale : pandas.Series, optional
            Precomputed ``avoidance_scale`` to reuse across several pixel subsets.
        **kwargs
            Additional survey values passed to the constructor.

        Returns
        -------
        SimonsObservatoryObsTable
            The materialized observation table, sorted by time.
        """
        _require_pixell()
        band_names = SO_TUBE_BANDS if band_names is None else band_names
        maps, map_tubes = time_maps["maps"], list(time_maps["tubes"])
        tubes = map_tubes if tubes is None else np.atleast_1d(tubes)

        pieces = []
        for tube, side in ((tube, side) for tube in tubes for side in range(2)):
            bands = list(band_names[SO_TUBE_TYPES[tube]])
            visits = visit_schedule(
                time_maps,
                tube,
                side,
                start_date,
                n_days,
                first_side=first_side,
                downtime=downtime,
                pixels=pixels,
            )
            # One row per (visit, band): repeat every visit for each band of the tube.
            band_codes = np.repeat(np.arange(len(bands)), len(visits["pixel"]))
            visits = {key: np.tile(value, len(bands)) for key, value in visits.items()}
            pix = visits["pixel"]
            fields = maps[:, map_tubes.index(tube), side].reshape(len(MAP_FIELDS), -1)
            pieces.append(
                pd.DataFrame(
                    {
                        **{key: visits[key] for key in ("time", "time_start", "time_end")},
                        "filter": pd.Categorical.from_codes(band_codes, bands),
                        "tube_id": np.full(len(pix), tube, dtype=np.int16),
                        "side": pd.Categorical.from_codes(np.full(len(pix), side), SIDE_NAMES),
                        "transit_day": visits["transit_day"].astype(np.int16),
                        **{
                            name: fields[MAP_FIELDS.index(name)][pix].astype(np.float32)
                            for name in ("hits", "duration_s", "exposure_s", "n_passes")
                        },
                        "pixel_index": pix.astype(np.int32),
                    }
                )
            )
        # Categorical columns keep large multi-tube tables compact (union_categoricals on concat).
        for column in ("filter", "side"):
            categories = sorted({c for piece in pieces for c in piece[column].cat.categories})
            for piece in pieces:
                piece[column] = piece[column].cat.set_categories(categories)
        table = pd.concat(pieces, ignore_index=True).sort_values("time", ignore_index=True)
        dec, ra = np.degrees(np.asarray(maps.posmap()).reshape(2, -1)[:, table["pixel_index"].to_numpy()])
        table["ra"], table["dec"] = ra % 360.0, dec
        if avoid_deg is not None:
            if scale is None and pixels is not None:
                args = (time_maps, start_date, n_days, tubes, first_side, downtime, avoid_deg)
                scale = cls.avoidance_scale(*args, n_scale_pixels, seed)
            cut = sun_moon_cut(table["time"], table["ra"], table["dec"], avoid_deg)
            table = conserve_exposure(table, cut, scale)
        kwargs.setdefault("radius", time_maps["res_deg"] / np.sqrt(2.0))
        kwargs.setdefault("scan_elevation_deg", time_maps["elevation_deg"])
        kwargs.setdefault("fov_deg", time_maps["fov_deg"])
        return cls(table, map_shape=maps.shape, map_wcs=maps.wcs, **kwargs)

    @classmethod
    def avoidance_scale(
        cls, time_maps, start_date, n_days, tubes, first_side, downtime, avoid_deg, n_pixels, seed
    ):
        """Estimate each tube's daily sun/moon exposure rescaling from random observed pixels.

        Parameters
        ----------
        time_maps : dict
            Output of ``simulate_time_maps`` or ``read_time_maps``.
        start_date : str or float
            Survey start as an ISO date string (UTC) or MJD.
        n_days : float
            Survey length in days.
        tubes : list of int
            Tubes to estimate the rescaling for.
        first_side : int
            Scan side for transits on the sidereal day containing the start.
        downtime : numpy.ndarray or None
            Downtime intervals (MJD).
        avoid_deg : float
            Sun and moon avoidance radius in degrees.
        n_pixels : int
            Number of random observed pixels to use.
        seed : int
            Random seed.

        Returns
        -------
        pandas.Series
            ``daily_exposure_scale`` indexed by (``tube_id``, UTC day).
        """
        observed = np.flatnonzero(np.asarray(time_maps["maps"][MAP_FIELDS.index("hits")]).sum((0, 1)) > 0)
        sample = np.random.default_rng(seed).choice(observed, min(n_pixels, len(observed)), replace=False)
        one_band = {kind: bands[:1] for kind, bands in SO_TUBE_BANDS.items()}
        scales = []
        for tube in tubes:
            reference = cls.from_time_maps(
                time_maps,
                start_date,
                n_days,
                tubes=[tube],
                band_names=one_band,
                first_side=first_side,
                downtime=downtime,
                avoid_deg=None,
                pixels=sample,
            )._table
            cut = sun_moon_cut(reference["time"], reference["ra"], reference["dec"], avoid_deg)
            scales.append(daily_exposure_scale(reference, cut))
        return pd.concat(scales)

    def copy(self):
        """Create a copy of the table that retains its SO-specific spatial lookup."""
        return type(self)(
            self._table,
            colmap=self._colmap.copy(),
            saturation_mags=None if self._saturation_mags is None else self._saturation_mags.copy(),
            **self.survey_values,
        )

    def set_detector_footprint(self, detector_footprint, wcs=None):
        """Reject detector footprints because map pixels define the coverage."""
        if detector_footprint is not None:
            raise NotImplementedError("SimonsObservatoryObsTable uses map pixels, not detector footprints.")
        self._detector_footprint = None

    def _build_spatial_data(self):
        """Build a pixel -> rows lookup (rows sorted by pixel index) in place of a KD-tree."""
        pixels = self._table["pixel_index"].to_numpy()
        self._pixel_order = np.argsort(pixels, kind="stable")
        self._sorted_pixels = pixels[self._pixel_order]
        self._spatial_data = None

    def range_search(self, query_ra, query_dec, *, radius=None, t_min=None, t_max=None):
        """Return the indices of the rows observing the map pixel containing each query point.

        Parameters
        ----------
        query_ra, query_dec : float or numpy.ndarray
            Query coordinates in degrees.
        radius : float, optional
            Ignored; the map pixel defines the match.
        t_min, t_max : float, numpy.ndarray or None, optional
            Inclusive time bounds (MJD) per query.

        Returns
        -------
        numpy.ndarray or list[numpy.ndarray]
            Row indices (sorted by time) for a scalar query, or a list of them for array queries.
        """
        is_scalar = np.isscalar(query_ra) and np.isscalar(query_dec)
        ra, dec = np.atleast_1d(query_ra).astype(float), np.atleast_1d(query_dec).astype(float)
        if ra.shape != dec.shape or ra.ndim != 1 or np.any(np.isnan(ra)) or np.any(np.isnan(dec)):
            raise ValueError("Query RA and Dec must be 1-d, non-NaN arrays of the same length.")
        t_min = np.broadcast_to(-np.inf if t_min is None else t_min, ra.shape)
        t_max = np.broadcast_to(np.inf if t_max is None else t_max, ra.shape)

        pixels = sky_to_pixel(self.survey_values["map_shape"], self.survey_values["map_wcs"], ra, dec)
        lows = np.searchsorted(self._sorted_pixels, pixels, side="left")
        highs = np.searchsorted(self._sorted_pixels, pixels, side="right")
        times = self._table["time"].to_numpy()
        inds = []
        for low, high, tlo, thi in zip(lows, highs, t_min, t_max, strict=True):
            rows = np.sort(self._pixel_order[low:high])
            inds.append(rows[(times[rows] >= tlo) & (times[rows] <= thi)])
        return inds[0] if is_scalar else inds
