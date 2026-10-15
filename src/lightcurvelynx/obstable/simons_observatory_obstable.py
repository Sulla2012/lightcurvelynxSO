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
  missing); 8 MF, 4 UHF, 1 LF. Detector positions are sampled on a grid within each tube.
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
MAP_FIELDS = ("hits", "time_s", "side", "exposure_s")
SIDE_NAMES = ("east", "west")
_SCALAR_SCAN_KEYS = ("elevation_deg", "speed_deg_s", "block_hours", "fov_deg", "tube_spacing_deg")
_HEADER_KEYS = ("res_deg", "dt_s", "det_spacing_deg", "lat_deg", *_SCALAR_SCAN_KEYS)
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


def detector_offsets(fov_deg, spacing_deg):
    """Return a square grid of detector positions filling a circular tube field of view.

    Parameters
    ----------
    fov_deg : float
        Diameter of the field of view in degrees.
    spacing_deg : float
        Grid spacing in degrees.

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 2) with the (cross-elevation, elevation) offsets in degrees.
    """
    n = int(np.floor(0.5 * fov_deg / spacing_deg))
    grid = np.arange(-n, n + 1) * spacing_deg
    dx, dy = (g.ravel() for g in np.meshgrid(grid, grid))
    keep = np.hypot(dx, dy) <= 0.5 * fov_deg + 1e-9
    return np.column_stack((dx[keep], dy[keep]))


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
    dec_range=(-80.0, 35.0),
    dt_s=None,
    det_spacing_deg=None,
    lat_deg=SO_LATITUDE_DEG,
    max_points=4_000_000,
    **scan_kwargs,
):
    """Simulate one sidereal day of SO scanning and accumulate per-tube time and hit maps.

    Parameters
    ----------
    res_deg : float
        Map resolution in degrees.
    tubes : list of int, optional
        Tube indices to simulate. Defaults to all 13.
    dec_range : tuple
        (min, max) declination of the map in degrees.
    dt_s : float, optional
        Sample interval in seconds. Defaults to res_deg / (2 * speed) so the scan moves at most
        half a pixel per sample.
    det_spacing_deg : float, optional
        Spacing of the sampled detector grid. Defaults to min(res_deg, fov / 4).
    lat_deg : float
        Site latitude in degrees.
    max_points : int
        Maximum number of pointing samples processed per chunk (memory control).
    **scan_kwargs
        Overrides of ``SCAN_DEFAULTS``.

    Returns
    -------
    dict
        ``maps``: enmap of shape (4, ntube, ny, nx) holding, per tube, the number of detector
        samples (``hits``), the circular mean observation time in seconds since LST = 0
        (``time_s``, NaN where unobserved), the scan side used (``side``, 0 = east, 1 = west,
        -1 unobserved) and the time in seconds the pixel spends inside the tube field of view
        (``exposure_s``). Also contains ``tubes`` and all scan parameters.
    """
    _require_pixell()
    scan = {**SCAN_DEFAULTS, **scan_kwargs}
    tubes = np.arange(13) if tubes is None else np.atleast_1d(tubes).astype(int)
    dt_s = res_deg / (2.0 * scan["speed_deg_s"]) if dt_s is None else dt_s
    det_spacing_deg = min(res_deg, scan["fov_deg"] / 4.0) if det_spacing_deg is None else det_spacing_deg
    shape, wcs = enmap.band_geometry(np.radians(dec_range), res=np.radians(res_deg))
    npix = shape[0] * shape[1]

    # All (tube, detector) pointing offsets in the focal plane.
    dets = detector_offsets(scan["fov_deg"], det_spacing_deg)
    centers = tube_offsets(scan["tube_spacing_deg"])[tubes]
    offsets = (centers[:, None, :] + dets[None, :, :]).reshape(-1, 2)
    tube_of = np.repeat(np.arange(len(tubes)), len(dets))

    nbins = len(tubes) * 2 * npix
    hits, cos_sum, sin_sum = np.zeros(nbins), np.zeros(nbins), np.zeros(nbins)
    times = np.arange(0.0, SIDEREAL_DAY_S, dt_s)
    chunk = max(1, max_points // len(offsets))
    for start in range(0, len(times), chunk):
        t = times[start : start + chunk]
        az, el, side = boresight_azel(t, **scan)
        point_az = az[:, None] + offsets[None, :, 0] / np.cos(np.radians(el))[:, None]
        point_el = el[:, None] + offsets[None, :, 1]
        ra, dec = horizon_to_equatorial(point_az, point_el, (360.0 * t / SIDEREAL_DAY_S)[:, None], lat_deg)
        pix = sky_to_pixel(shape, wcs, ra, dec).reshape(ra.shape)
        bins = ((tube_of[None, :] * 2 + side[:, None]) * npix + pix).ravel()
        phase = np.repeat(2.0 * np.pi * t / SIDEREAL_DAY_S, len(offsets))
        good = pix.ravel() >= 0
        bins, phase = bins[good], phase[good]
        hits += np.bincount(bins, minlength=nbins)
        cos_sum += np.bincount(bins, weights=np.cos(phase), minlength=nbins)
        sin_sum += np.bincount(bins, weights=np.sin(phase), minlength=nbins)

    # Keep only the side (crossing) with more hits so each pixel is observed once per day.
    hits, cos_sum, sin_sum = (a.reshape(len(tubes), 2, *shape) for a in (hits, cos_sum, sin_sum))
    best = np.argmax(hits, axis=1)[:, None]
    pick = lambda a: np.take_along_axis(a, best, axis=1)[:, 0]  # noqa: E731
    best_hits = pick(hits)
    observed = best_hits > 0
    mean_phase = np.arctan2(pick(sin_sum), pick(cos_sum)) % (2.0 * np.pi)
    pixel_area_deg2 = np.degrees(np.degrees(enmap.pixsizemap(shape, wcs)))
    maps = enmap.enmap(
        np.stack(
            [
                best_hits,
                np.where(observed, mean_phase * SIDEREAL_DAY_S / (2.0 * np.pi), np.nan),
                np.where(observed, best[:, 0], -1),
                best_hits * dt_s * det_spacing_deg**2 / pixel_area_deg2,
            ]
        ),
        wcs,
    )
    return {
        "maps": maps,
        "tubes": tubes,
        "res_deg": float(res_deg),
        "dt_s": float(dt_s),
        "det_spacing_deg": float(det_spacing_deg),
        "lat_deg": float(lat_deg),
        **scan,
    }


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

    Each row is one effective observation of one map pixel by one tube on one day: ``ra`` and
    ``dec`` are the pixel center, ``time`` the hit-weighted mean observation time (MJD), ``hits``
    the number of detector samples, and ``exposure_s`` the time (seconds) the pixel spent inside the
    tube's field of view. Rows are normally created with ``from_time_maps``/``create_tube_tables``.

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
            hits, time_s, side, exposure = maps[:, map_tubes.index(tube)].reshape(4, -1)
            pixels = np.flatnonzero(hits > 0)
            mjd = lst0_mjd + (days[:, None] * SIDEREAL_DAY_S + time_s[pixels][None, :]) / 86400.0
            keep = (mjd >= start_mjd) & (mjd < start_mjd + n_days)
            pix = np.broadcast_to(pixels, mjd.shape)[keep]
            pieces.append(
                pd.DataFrame(
                    {
                        "time": mjd[keep],
                        "filter": filter_names[tube],
                        "tube_id": int(tube),
                        "side": np.asarray(SIDE_NAMES)[side[pix].astype(int)],
                        "hits": hits[pix],
                        "exposure_s": exposure[pix],
                        "pixel_index": pix,
                    }
                )
            )
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
