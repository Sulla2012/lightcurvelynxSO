"""Simons Observatory (SO) observation tables built from simulated scan time maps.

SO does not take discrete pointings: each optics tube (OT) scans continuously in azimuth at a
fixed elevation while the sky drifts through the scan. We therefore simulate one sidereal day of
an idealized scan strategy, accumulate per-pixel hit counts and mean observation times for each
tube on a pixell CAR map, and materialize those maps into a standard ObsTable with one row per
(day, observed pixel). Because the schedule is defined in local sidereal time (LST), the scan
repeats every sidereal day, so a single simulated day is replicated for the full date range.

Idealized scan model
--------------------
* Site at 22d57m31s S, 67d47m15s W; boresight elevation fixed at 40 deg.
* The telescope alternates between an east (az 30-150 deg) and a west (az 210-330 deg) scan
  range in blocks of ``block_hours``, phased so that the first block starts at LST = 0.
* Azimuth follows a triangle wave at 1 deg/s with instantaneous turnarounds.
* 13 tubes of 1.25 deg diameter in a hex pattern (1.75 deg spacing, half the outer ring
  missing); 8 MF, 4 UHF, 1 LF. A pixel is hit whenever its center is inside a tube's FoV.
* Each pixel is observed at most once per tube per sidereal day: if it is crossed in both the
  east and west ranges only the crossing with more hits is kept.
"""

import numpy as np
import pandas as pd
from astropy.time import Time

from lightcurvelynx.obstable.obs_table import ObsTable

try:
    from pixell import enmap
except ImportError:  # pragma: no cover
    enmap = None

SO_LATITUDE_DEG = -(22.0 + 57.0 / 60.0 + 31.0 / 3600.0)
SO_LONGITUDE_DEG = -(67.0 + 47.0 / 60.0 + 15.0 / 3600.0)
SO_TUBE_FILTERS = ("MF",) * 8 + ("UHF",) * 4 + ("LF",)
SIDEREAL_DAY_S = 86164.0905
MAP_FIELDS = ("hits", "time_s", "side", "start_s", "duration_s", "exposure_s", "n_passes")
SIDE_NAMES = ("east", "west")
_SCALAR_SCAN_KEYS = ("elevation_deg", "speed_deg_s", "block_hours", "fov_deg", "tube_spacing_deg")
_HEADER_KEYS = ("res_deg", "dt_s", "lat_deg", *_SCALAR_SCAN_KEYS)
SCAN_DEFAULTS = {
    "elevation_deg": 40.0,
    "az_ranges": ((30.0, 150.0), (210.0, 330.0)),
    "speed_deg_s": 1.0,
    "block_hours": 1.0,
    "fov_deg": 1.25,
    "tube_spacing_deg": 1.75,
}


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


def scan_time_step(res_deg, speed_deg_s, az_ranges, dt_s=None, **_):
    """Choose the sample interval so that one azimuth sweep is an integer number of samples.

    Parameters
    ----------
    res_deg : float
        Map resolution in degrees; the default step moves the boresight half a pixel.
    speed_deg_s : float
        Azimuth scan speed in degrees per second.
    az_ranges : tuple
        Scan azimuth ranges; the first range sets the sweep duration.
    dt_s : float, optional
        Requested (maximum) sample interval in seconds.

    Returns
    -------
    dt_s : float
        Sample interval in seconds.
    samples_per_sweep : int
        Number of samples in one one-way azimuth sweep.
    """
    sweep_s = (az_ranges[0][1] - az_ranges[0][0]) / speed_deg_s
    requested = res_deg / (2.0 * speed_deg_s) if dt_s is None else dt_s
    samples_per_sweep = int(np.ceil(sweep_s / requested))
    return sweep_s / samples_per_sweep, samples_per_sweep


def boresight_azel(t_s, *, elevation_deg, az_ranges, speed_deg_s, block_hours, **_):
    """Return the boresight azimuth, elevation, and active scan side at the given times.

    Parameters
    ----------
    t_s : numpy.ndarray
        Time in seconds since the start of the schedule (LST = 0).
    elevation_deg : float
        Fixed boresight elevation in degrees.
    az_ranges : tuple
        ((east_min, east_max), (west_min, west_max)) azimuth ranges in degrees.
    speed_deg_s : float
        Azimuth scan speed in degrees per second.
    block_hours : float
        Length of each east/west scheduling block in hours.

    Returns
    -------
    az, el : numpy.ndarray
        Boresight azimuth and elevation in degrees.
    side : numpy.ndarray
        Index of the active scan range (0 = east, 1 = west).
    """
    side = (t_s // (3600.0 * block_hours)).astype(int) % 2
    low, high = np.asarray(az_ranges, dtype=float)[side].T
    throw_s = (high - low) / speed_deg_s
    phase = t_s % (2.0 * throw_s)
    az = low + speed_deg_s * np.where(phase < throw_s, phase, 2.0 * throw_s - phase)
    return az, np.full_like(az, elevation_deg), side


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


def simulate_time_maps(
    res_deg=0.5,
    *,
    tubes=None,
    dec_range=(-70.0, 30.0),
    dt_s=None,
    lat_deg=SO_LATITUDE_DEG,
    max_points=4_000_000,
    **scan_kwargs,
):
    """Simulate one sidereal day of SO scanning and accumulate per-tube visit maps.

    At every time sample, each map pixel whose center lies within a tube's circular field of view
    is hit, so all statistics are exact for the pixel-center position up to the sample interval.
    A pixel is typically crossed once while rising (east range) and once while setting (west
    range); only the crossing with more hits is kept for each tube.

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
    max_points : int
        Maximum number of (time, tube, pixel) candidates processed per chunk (memory control).
    **scan_kwargs
        Overrides of ``SCAN_DEFAULTS``.

    Returns
    -------
    dict
        ``maps``: enmap of shape (len(MAP_FIELDS), ntube, ny, nx). Per tube and pixel it holds the
        number of in-FoV samples (``hits``), the circular mean time of the visit (``time_s``),
        the scan side (``side``: 0 = east, 1 = west, -1 unobserved), the first in-FoV time
        (``start_s``), the visit length from first to last in-FoV sample (``duration_s``), the
        total time inside the FoV (``exposure_s``) and the number of azimuth sweeps that pass over
        the pixel (``n_passes``). Times are seconds since LST = 0; unobserved pixels are NaN.
        Also contains ``tubes``, ``res_deg``, ``dt_s``, ``lat_deg`` and the scan parameters.
    """
    _require_pixell()
    scan = {**SCAN_DEFAULTS, **scan_kwargs}
    tubes = np.arange(13) if tubes is None else np.atleast_1d(tubes).astype(int)
    dt_s, per_sweep = scan_time_step(res_deg, dt_s=dt_s, **scan)
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
    n_sweeps = n_samples // per_sweep + 1
    chunk = max(1, max_points // (len(tubes) * len(dy) * per_sweep)) * per_sweep  # whole sweeps
    for start in range(0, n_samples, chunk):
        k = np.arange(start, min(start + chunk, n_samples))
        t = k * dt_s
        az, el, side = boresight_azel(t, **scan)
        tube_az, tube_el = tube_pointing(az, el, centers)
        ra_c, dec_c = horizon_to_equatorial(tube_az, tube_el, (360.0 * t / SIDEREAL_DAY_S)[:, None], lat_deg)
        center_pix = sky_to_pixel(shape, wcs, ra_c, dec_c).reshape(ra_c.shape)[..., None]
        y = center_pix // nx + dy
        x = (center_pix % nx + dx) % nx
        inside = (center_pix >= 0) & (y >= 0) & (y < ny)
        y = np.clip(y, 0, ny - 1)
        inside &= angular_separation_deg(ra_pix[x], dec_pix[y], ra_c[..., None], dec_c[..., None]) <= radius
        it, itube, _ = np.nonzero(inside)
        bins = ((itube * 2 + side[it]) * ny + y[inside]) * nx + x[inside]
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
        sweeps = np.unique(bins.astype(np.int64) * n_sweeps + k[it] // per_sweep)
        passes += np.bincount(sweeps // n_sweeps, minlength=nbins)

    # Keep the side (crossing) with more hits for each tube and pixel.
    arrays = [a.reshape(len(tubes), 2, ny, nx) for a in (hits, cos_sum, sin_sum, passes)]
    arrays += [a.reshape(len(tubes), 2, ny, nx) for a in (t_min, t_max, t_min_shift, t_max_shift)]
    best = np.argmax(arrays[0], axis=1)[:, None]
    hits, cos_sum, sin_sum, passes, t_min, t_max, t_min_shift, t_max_shift = (
        np.take_along_axis(a, best, axis=1)[:, 0] for a in arrays
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
        "side": np.where(observed, best[:, 0], -1),
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
    ra, dec, tube_id, *, res_deg=0.5, dt_s=None, lat_deg=SO_LATITUDE_DEG, max_gap_s=1800.0, **scan_kwargs
):
    """Trace when one sky position is inside a tube's field of view over one sidereal day.

    This is an independent, single-point reference for ``simulate_time_maps``: it converts the
    source into horizon coordinates and tests its separation from the tube center there. Pass a
    ``simulate_time_maps`` result as keyword arguments to use the same scan and time step.

    Parameters
    ----------
    ra, dec : float
        Source position in degrees.
    tube_id : int
        Optics-tube index.
    res_deg, dt_s : float, optional
        Used to choose the sample interval as in ``simulate_time_maps``.
    lat_deg : float
        Site latitude in degrees.
    max_gap_s : float
        In-FoV samples separated by more than this (seconds) belong to different visits.
    **scan_kwargs
        Overrides of ``SCAN_DEFAULTS`` (other keys are ignored).

    Returns
    -------
    list of dict
        One entry per visit with ``side``, ``time_s`` (circular mean), ``start_s``,
        ``duration_s``, ``exposure_s`` and ``n_passes``; times in seconds since LST = 0.
    """
    scan = {key: scan_kwargs.get(key, value) for key, value in SCAN_DEFAULTS.items()}
    dt_s, per_sweep = scan_time_step(res_deg, dt_s=dt_s, **scan)
    k = np.arange(int(np.ceil(SIDEREAL_DAY_S / dt_s)))
    t = k * dt_s
    az, el, side = boresight_azel(t, **scan)
    tube_az, tube_el = tube_pointing(az, el, tube_offsets(scan["tube_spacing_deg"])[[tube_id]])
    src_az, src_el = equatorial_to_horizon(ra, dec, 360.0 * t / SIDEREAL_DAY_S, lat_deg)
    inside = np.flatnonzero(
        angular_separation_deg(tube_az[:, 0], tube_el[:, 0], src_az, src_el) <= 0.5 * scan["fov_deg"]
    )
    if len(inside) == 0:
        return []
    groups = np.split(inside, np.flatnonzero(np.diff(t[inside]) > max_gap_s) + 1)
    unwrapped = [k[g].astype(float) for g in groups]
    if len(groups) > 1 and t[groups[0][0]] < max_gap_s and t[groups[-1][-1]] > SIDEREAL_DAY_S - max_gap_s:
        unwrapped[0] = np.concatenate((unwrapped.pop() - len(k), unwrapped[0]))
        groups[0] = np.concatenate((groups.pop(), groups[0]))
    visits = []
    for g, ku in zip(groups, unwrapped, strict=True):
        times = ku * dt_s
        visits.append(
            {
                "side": int(np.bincount(side[g]).argmax()),
                "time_s": times.mean() % SIDEREAL_DAY_S,
                "start_s": times[0] % SIDEREAL_DAY_S,
                "duration_s": times[-1] - times[0] + dt_s,
                "exposure_s": len(g) * dt_s,
                "n_passes": len(np.unique(k[g] // per_sweep)),
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


class SimonsObservatoryObsTable(ObsTable):
    """An ObsTable of Simons Observatory map-pixel observations for one or more optics tubes.

    Each row is one visit of one map pixel by one tube on one day. ``ra`` and ``dec`` are the pixel
    center and ``time`` is the mean in-FoV time of the visit (MJD). ``time_start``/``time_end``
    (MJD) and ``duration_s`` give the span from the first to the last in-FoV sample, ``exposure_s``
    the total time inside the tube's field of view (s), ``n_passes`` the number of azimuth sweeps
    over the pixel, and ``hits`` the number of in-FoV samples. Rows are normally created with
    ``from_time_maps``.

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
    def from_time_maps(cls, time_maps, start_date, n_days, *, tubes=None, filter_names=None, **kwargs):
        """Materialize simulated time maps into an ObsTable over a date range.

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
        filter_names : sequence of str, optional
            Filter label for each of the 13 tubes. Defaults to ``SO_TUBE_FILTERS``.
        **kwargs
            Additional survey values passed to the constructor.

        Returns
        -------
        SimonsObservatoryObsTable
            The materialized observation table, sorted by time.
        """
        _require_pixell()
        filter_names = SO_TUBE_FILTERS if filter_names is None else filter_names
        maps, map_tubes = time_maps["maps"], list(time_maps["tubes"])
        tubes = map_tubes if tubes is None else np.atleast_1d(tubes)
        is_mjd = isinstance(start_date, int | float | np.number)
        start = Time(start_date, format="mjd" if is_mjd else None, scale="utc")
        start_mjd = start.mjd
        lst_start = start.sidereal_time("mean", longitude=SO_LONGITUDE_DEG).deg
        lst0_mjd = start_mjd - lst_start / 360.0 * SIDEREAL_DAY_S / 86400.0
        days = np.arange(int(np.ceil(n_days * 86400.0 / SIDEREAL_DAY_S)) + 2)

        pieces = []
        for tube in tubes:
            fields = dict(
                zip(MAP_FIELDS, maps[:, map_tubes.index(tube)].reshape(len(MAP_FIELDS), -1), strict=True)
            )
            pixels = np.flatnonzero(fields["hits"] > 0)
            mjd = lst0_mjd + (days[:, None] * SIDEREAL_DAY_S + fields["time_s"][pixels][None, :]) / 86400.0
            keep = (mjd >= start_mjd) & (mjd < start_mjd + n_days)
            pix = np.broadcast_to(pixels, mjd.shape)[keep]
            # Start relative to the mean time, wrapped so visits spanning LST = 0 stay contiguous.
            start_offset_s = (
                fields["start_s"][pix] - fields["time_s"][pix] + 0.5 * SIDEREAL_DAY_S
            ) % SIDEREAL_DAY_S
            time_start = mjd[keep] + (start_offset_s - 0.5 * SIDEREAL_DAY_S) / 86400.0
            pieces.append(
                pd.DataFrame(
                    {
                        "time": mjd[keep],
                        "time_start": time_start,
                        "time_end": time_start + fields["duration_s"][pix] / 86400.0,
                        "filter": pd.Categorical.from_codes(np.zeros(len(pix), int), [filter_names[tube]]),
                        "tube_id": np.full(len(pix), tube, dtype=np.int16),
                        "side": pd.Categorical.from_codes(fields["side"][pix].astype(int), SIDE_NAMES),
                        **{
                            name: fields[name][pix].astype(np.float32)
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
        kwargs.setdefault("radius", time_maps["res_deg"] / np.sqrt(2.0))
        kwargs.setdefault("scan_elevation_deg", time_maps["elevation_deg"])
        kwargs.setdefault("fov_deg", time_maps["fov_deg"])
        return cls(table, map_shape=maps.shape, map_wcs=maps.wcs, **kwargs)

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
