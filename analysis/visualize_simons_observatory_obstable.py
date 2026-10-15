"""Simulate SO scan time maps, build the ObsTable, and write an HTML report with diagnostic plots.

Examples
--------
Quick look (2 deg pixels, three tubes, three days)::

    python analysis/visualize_simons_observatory_obstable.py --debug

Full run (0.5 deg pixels, all 13 tubes, one week)::

    python analysis/visualize_simons_observatory_obstable.py --res 0.5 --days 7

Time maps are cached as FITS files in ``--outdir`` and reused unless ``--overwrite`` is given.
"""

import argparse
import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.time import Time  # noqa: E402

from lightcurvelynx.obstable.simons_observatory_obstable import (  # noqa: E402
    MAP_FIELDS,
    SIDE_NAMES,
    SIDEREAL_DAY_S,
    SO_LONGITUDE_DEG,
    SO_TUBE_FILTERS,
    SimonsObservatoryObsTable,
    point_visits,
    read_time_maps,
    simulate_downtime,
    simulate_time_maps,
    sky_to_pixel,
    survey_epoch,
    tube_offsets,
    visit_schedule,
    write_time_maps,
)

YEAR_DAYS = 365.25

# Field -> (colorbar label, colormap, scale factor applied before plotting).
FIELD_LABELS = {
    "time_s": ("LST of observation (h)", "twilight", 24.0 / SIDEREAL_DAY_S),
    "duration_s": ("Visit duration, first to last in-FoV sample (min)", "viridis", 1.0 / 60.0),
    "exposure_s": ("Time in tube FoV per visit (s)", "magma", 1.0),
    "n_passes": ("Azimuth sweeps over pixel per visit", "cividis", 1.0),
}
BAND_COLORS = {"MF": "tab:blue", "UHF": "tab:red", "LF": "tab:green"}


def parse_args():
    """Parse the command line arguments.

    Returns
    -------
    argparse.Namespace
        The parsed arguments; ``--debug`` overrides resolution, tubes and days.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--res", type=float, default=0.5, help="Map resolution in degrees.")
    parser.add_argument("--tubes", type=int, nargs="+", default=list(range(13)), help="Tube indices.")
    parser.add_argument("--days", type=float, default=7.0, help="Survey length in days for the ObsTable.")
    parser.add_argument("--start", default="2026-10-14", help="Survey start date (UTC).")
    parser.add_argument(
        "--first-side", type=int, default=0, help="Side of the first day (0 rising, 1 setting)."
    )
    parser.add_argument("--uptime", type=float, default=0.65, help="Instrument uptime fraction.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the downtime realization.")
    parser.add_argument(
        "--epoch", type=float, help="Precession epoch (MJD); default is mid-way through the year."
    )
    parser.add_argument("--outdir", type=Path, default=Path("analysis/so_report"), help="Report directory.")
    parser.add_argument("--overwrite", action="store_true", help="Re-simulate even if cached maps exist.")
    parser.add_argument("--debug", action="store_true", help="Quick run: 2 deg pixels, tubes 0 1 12, 3 days.")
    args = parser.parse_args()
    if args.debug:
        args.res, args.tubes, args.days = 2.0, [0, 1, 12], 3.0
    if args.epoch is None:
        args.epoch = survey_epoch(args.start)[0] + YEAR_DAYS / 2.0
    return args


def load_or_simulate(args):
    """Load cached time maps from ``args.outdir`` or simulate and cache them.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.

    Returns
    -------
    dict
        Time maps as returned by ``simulate_time_maps``.
    """
    tag = f"res{args.res:g}_ep{args.epoch:.0f}_tubes{'-'.join(map(str, args.tubes))}"
    fname = args.outdir / f"time_maps_{tag}.fits"
    if fname.exists() and not args.overwrite:
        print(f"Reading cached time maps from {fname}")
        return read_time_maps(fname)
    print(f"Simulating one sidereal day for tubes {args.tubes} at {args.res} deg...", flush=True)
    time_maps = simulate_time_maps(args.res, tubes=args.tubes, epoch_mjd=args.epoch)
    write_time_maps(fname, time_maps)
    return time_maps


def sky_imshow(ax, image, maps, **kwargs):
    """Show a CAR map with RA increasing to the left and RA/Dec axes in degrees.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    image : numpy.ndarray
        2-d map (ny, nx) on the geometry of ``maps``.
    maps : pixell.enmap.ndmap
        Map defining the geometry.
    **kwargs
        Passed to ``imshow``.

    Returns
    -------
    matplotlib.image.AxesImage
        The image artist.
    """
    dec, ra = np.degrees(maps.posmap())
    res = abs(ra[0, 1] - ra[0, 0])
    ra_left, ra_right = ra[0, 0] + res / 2, ra[0, -1] - res / 2
    extent = [ra_left, ra_right, dec[0, 0] - res / 2, dec[-1, 0] + res / 2]
    artist = ax.imshow(np.ma.masked_invalid(image), origin="lower", extent=extent, aspect="equal", **kwargs)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.xaxis.set_major_formatter(lambda value, _: f"{value % 360:.0f}")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    return artist


def plot_tube_grid(time_maps, field, side, fname):
    """Plot one map field of one scan side for every simulated tube in a grid.

    Parameters
    ----------
    time_maps : dict
        Output of ``simulate_time_maps``.
    field : str
        One of ``FIELD_LABELS``.
    side : int
        Scan side (0 = east/rising, 1 = west/setting).
    fname : Path
        Output image filename.
    """
    maps, tubes = time_maps["maps"][:, :, side], time_maps["tubes"]
    label, cmap, scale = FIELD_LABELS[field]
    data = np.where(maps[0] > 0, maps[MAP_FIELDS.index(field)] * scale, np.nan)
    ncol = min(len(tubes), 3)
    nrow = int(np.ceil(len(tubes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.5 * ncol, 2.4 * nrow + 0.6), squeeze=False)
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    for ax, tube, image in zip(axes.flat, tubes, data, strict=False):
        artist = sky_imshow(ax, image, maps, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Tube {tube} ({SO_TUBE_FILTERS[tube]})", fontsize=10)
    for ax in axes.flat[len(tubes) :]:
        ax.set_visible(False)
    fig.colorbar(artist, ax=axes, label=label, shrink=0.8)
    fig.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_coverage(time_maps, fname):
    """Plot the number of tubes observing each pixel on a rising day and on a setting day.

    Parameters
    ----------
    time_maps : dict
        Output of ``simulate_time_maps``.
    fname : Path
        Output image filename.
    """
    maps = time_maps["maps"]
    count = (maps[0] > 0).sum(axis=0).astype(float)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    for ax, side, name in zip(axes, range(2), ("rising (east)", "setting (west)"), strict=True):
        artist = sky_imshow(
            ax, np.where(count[side] > 0, count[side], np.nan), maps, cmap="viridis", vmin=1, vmax=count.max()
        )
        ax.set_title(f"{name} day: coverage union of {len(time_maps['tubes'])} tubes")
    fig.colorbar(artist, ax=axes, label="Tubes observing pixel per day")
    fig.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_layout(time_maps, fname):
    """Plot the focal-plane tube layout with the sampled detector positions.

    Parameters
    ----------
    time_maps : dict
        Output of ``simulate_time_maps``.
    fname : Path
        Output image filename.
    """
    offsets = tube_offsets(time_maps["tube_spacing_deg"])
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for tube, (x, y) in enumerate(offsets):
        filt = SO_TUBE_FILTERS[tube]
        alpha = 0.6 if tube in time_maps["tubes"] else 0.15
        ax.add_patch(plt.Circle((x, y), time_maps["fov_deg"] / 2, color=BAND_COLORS[filt], alpha=alpha))
        ax.text(x, y, f"{tube}\n{filt}", ha="center", va="center", fontsize=8)
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Cross-elevation offset (deg)")
    ax.set_ylabel("Elevation offset (deg)")
    ax.set_title("Optics-tube layout (faded = not simulated)")
    fig.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_table_checks(table, start_mjd, fname):
    """Plot ObsTable diagnostics: observations per (tube, pixel, day) and per-source cadence.

    Parameters
    ----------
    table : SimonsObservatoryObsTable
        The materialized observation table.
    start_mjd : float
        Survey start (MJD).
    fname : Path
        Output image filename.
    """
    df = table._table
    per_day = df.groupby(["tube_id", "pixel_index", "transit_day"]).size()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    ax1.hist(per_day.to_numpy(), bins=np.arange(0.5, 4.5), rwidth=0.8)
    ax1.set_xlabel("Observations per (tube, pixel, transit day)")
    ax1.set_ylabel("Count")
    ax1.set_yscale("log")
    ax1.set_title("Each point is hit once per day")

    rng = np.random.default_rng(0)
    for dec in (-55.0, -30.0, -5.0, 15.0):
        ra = rng.uniform(0, 360)
        inds = table.range_search(ra, dec)
        times = table["time"][inds]
        tubes = table["tube_id"][inds]
        utc_hour = (times % 1.0) * 24.0
        ax2.scatter(times - start_mjd, utc_hour, s=12, label=f"RA {ra:.0f}, Dec {dec:.0f} ({len(inds)} obs)")
        for t, h, tube in zip(times - start_mjd, utc_hour, tubes, strict=True):
            ax2.annotate(str(tube), (t, h), fontsize=5, alpha=0.6)
    ax2.set_xlabel("Days since start")
    ax2.set_ylabel("UTC hour of observation")
    ax2.set_title("Observation times for random sources (labels = tube)")
    ax2.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close(fig)


def weighted_percentile(values, weights, q):
    """Return weighted percentiles (inverse of the weighted empirical CDF).

    Parameters
    ----------
    values, weights : numpy.ndarray
        Samples and their weights.
    q : sequence of float
        Percentiles in [0, 100].

    Returns
    -------
    numpy.ndarray
        The percentiles.
    """
    order = np.argsort(values)
    cdf = np.cumsum(weights[order]) / np.sum(weights)
    return values[order][np.minimum(np.searchsorted(cdf, np.asarray(q) / 100.0), len(values) - 1)]


def year_visits(time_maps, start, first_side, downtime, block_days=30.0):
    """Yield the visits of every tube and side over one year, in blocks to bound memory.

    Parameters
    ----------
    time_maps : dict
        Output of ``simulate_time_maps``.
    start : str
        Start date (UTC) of the year.
    first_side : int
        Scan side of the first transit day.
    downtime : numpy.ndarray or None
        Merged downtime intervals (MJD).
    block_days : float
        Length of each block in days.

    Yields
    ------
    band : str
        Band of the tube.
    pixel, transit_day, utc_day : numpy.ndarray
        Flat pixel index, transit day (counted from the start's LST = 0) and UTC day since start.
    """
    start_mjd, lst0_mjd = survey_epoch(start)
    for begin in np.arange(0.0, YEAR_DAYS, block_days):
        block_lst0 = survey_epoch(start_mjd + begin)[1]
        # Transit days of a block count from its own LST = 0; shift them (and the parity) to the year's.
        offset = int(round((block_lst0 - lst0_mjd) * 86400.0 / SIDEREAL_DAY_S))
        for tube in time_maps["tubes"]:
            for side in range(2):
                visits = visit_schedule(
                    time_maps,
                    tube,
                    side,
                    start_mjd + begin,
                    min(block_days, YEAR_DAYS - begin),
                    first_side=(first_side + offset) % 2,
                    downtime=downtime,
                )
                utc_day = np.floor(visits["time"] - start_mjd).astype(int)
                yield SO_TUBE_FILTERS[tube], visits["pixel"], visits["transit_day"] + offset, utc_day


def year_coverage(time_maps, start, first_side, downtime, area, footprint=None):
    """Summarize one year of observations per band: daily sky fraction, days observed and revisit gaps.

    Parameters
    ----------
    time_maps : dict
        Output of ``simulate_time_maps``.
    start : str
        Start date (UTC) of the year.
    first_side : int
        Scan side of the first transit day.
    downtime : numpy.ndarray or None
        Merged downtime intervals (MJD).
    area : numpy.ndarray
        Flat per-pixel area as a fraction of the full sky.
    footprint : dict, optional
        Band -> boolean pixel mask over which days per year are summarized. Defaults to the
        pixels observed at least once.

    Returns
    -------
    dict
        Band -> ``daily`` (sky fraction per full UTC day), ``days`` (days observed per pixel),
        ``footprint`` (pixel mask), and ``gaps``/``gap_weights`` (days between successive observed
        transit days, weighted by pixel area).
    """
    npix = area.size
    bands = [*sorted({SO_TUBE_FILTERS[tube] for tube in time_maps["tubes"]}), "all"]
    n_transit, n_utc = int(YEAR_DAYS * 86400.0 / SIDEREAL_DAY_S) + 3, int(np.ceil(YEAR_DAYS))
    flags = {band: (np.zeros((npix, n_transit), bool), np.zeros((npix, n_utc), bool)) for band in bands}
    for band, pixel, transit, utc_day in year_visits(time_maps, start, first_side, downtime):
        for key in (band, "all"):
            flags[key][0][pixel, transit + 1] = True
            flags[key][1][pixel, utc_day] = True
    stats = {}
    for band, (transit, utc) in flags.items():
        rows, cols = np.nonzero(transit)
        same = rows[1:] == rows[:-1]
        days = transit.sum(axis=1)
        stats[band] = {
            "daily": area @ utc[:, : int(YEAR_DAYS)],
            "days": days,
            "footprint": days > 0 if footprint is None else footprint[band],
            "gaps": np.diff(cols)[same] * SIDEREAL_DAY_S / 86400.0,
            "gap_weights": area[rows[1:][same]],
        }
    return stats


def plot_year_coverage(time_maps, ideal, actual, fname):
    """Plot one year of sky coverage with and without downtime.

    Parameters
    ----------
    time_maps : dict
        Output of ``simulate_time_maps``.
    ideal, actual : dict
        ``year_coverage`` results without and with downtime.
    fname : Path
        Output image filename.
    """
    maps = time_maps["maps"]
    area = np.asarray(maps.pixsizemap(broadcastable=False)).ravel() / (4.0 * np.pi)
    colors = {**BAND_COLORS, "all": "k"}
    fig, axes = plt.subplots(1, 4, figsize=(20, 3.8), gridspec_kw={"width_ratios": [1.3, 1, 1, 1.6]})
    axes[0].plot(ideal["all"]["daily"], color="0.6", lw=1, label="all, no downtime")
    for band, values in actual.items():
        axes[0].plot(values["daily"], color=colors[band], lw=1 if band == "all" else 0.6, label=band)
        seen = values["footprint"]
        bins = np.arange(-5, 380, 10)
        axes[1].hist(
            values["days"][seen], bins, weights=area[seen], histtype="step", color=colors[band], label=band
        )
        axes[1].hist(
            ideal[band]["days"][seen], bins, weights=area[seen], histtype="step", ls=":", color=colors[band]
        )
        gaps = np.clip(values["gaps"], 0, 30)
        axes[2].hist(
            gaps,
            np.arange(0, 31, 0.5),
            weights=values["gap_weights"],
            density=True,
            histtype="step",
            color=colors[band],
            label=band,
        )
    axes[0].set_xlabel("UTC day of year since start")
    axes[0].set_ylabel("Fraction of full sky observed")
    axes[0].legend(fontsize=7, ncol=2, loc="lower left")
    axes[1].set_xlabel("Days per year a pixel is observed (dotted: no downtime)")
    axes[1].set_ylabel("Sky fraction")
    axes[2].set_xlabel("Revisit gap (days, last bin = 30+)")
    axes[2].set_ylabel("Area-weighted density")
    axes[2].set_yscale("log")
    image = actual["all"]["days"].reshape(maps.shape[-2:]).astype(float)
    image[~actual["all"]["footprint"].reshape(image.shape)] = np.nan
    artist = sky_imshow(axes[3], image, maps, cmap="viridis")
    fig.colorbar(artist, ax=axes[3], label="Days per year observed (any band)")
    fig.tight_layout()
    fig.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_downtime(downtime, start, fname):
    """Plot the downtime realization as a date versus local-time raster with monthly uptime.

    Parameters
    ----------
    downtime : numpy.ndarray
        Merged downtime intervals (MJD).
    start : str
        Start date (UTC) of the year.
    fname : Path
        Output image filename.
    """
    start_mjd = survey_epoch(start)[0]
    step = 1.0 / 96.0  # 15 minutes.
    times = start_mjd + np.arange(0.0, YEAR_DAYS, step)
    down = np.zeros(times.size, bool)
    for begin, end in downtime:
        down[(times >= begin) & (times < end)] = True
    local = times + SO_LONGITUDE_DEG / 360.0
    local_day = np.floor(local - np.floor(local[0])).astype(int)
    raster = np.full((local_day.max() + 1, 96), np.nan)
    raster[local_day, ((local % 1.0) * 96).astype(int)] = down
    months = Time(times, format="mjd").ymdhms.month
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    ax1.imshow(
        raster.T,
        origin="lower",
        aspect="auto",
        cmap="Greys",
        extent=[0, raster.shape[0], 0, 24],
        vmin=0,
        vmax=1.5,
    )
    ax1.set_xlabel(f"Local day since {start}")
    ax1.set_ylabel("Local mean solar time (h)")
    ax1.set_title(f"Downtime (dark), uptime {1 - down.mean():.3f}")
    ax2.bar(np.arange(1, 13), [1 - down[months == m].mean() for m in range(1, 13)])
    ax2.set_xticks(np.arange(1, 13))
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Uptime")
    fig.tight_layout()
    fig.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close(fig)


def random_sources(n, rng, dec_range=(-60.0, 20.0)):
    """Draw sky positions uniformly on the sphere within a declination band.

    Parameters
    ----------
    n : int
        Number of positions.
    rng : numpy.random.Generator
        Random number generator.
    dec_range : tuple
        (min, max) declination in degrees.

    Returns
    -------
    ra, dec : numpy.ndarray
        Positions in degrees.
    """
    sin_dec = rng.uniform(*np.sin(np.radians(dec_range)), n)
    return rng.uniform(0.0, 360.0, n), np.degrees(np.arcsin(sin_dec))


def telescope_visits(rows, merge_gap_days=1.0 / 24.0):
    """Merge per-tube rows of one band into telescope visits (tube visits within the gap).

    Parameters
    ----------
    rows : pandas.DataFrame
        ObsTable rows for one source and band, sorted by time.
    merge_gap_days : float
        Tube visits whose mean times differ by less than this are one telescope visit.

    Returns
    -------
    pandas.DataFrame
        One row per visit with mean ``time`` (MJD), ``span_s`` (first start to last end) and
        summed tube ``exposure_s``.
    """
    times = rows["time"].to_numpy()
    group = np.cumsum(np.diff(times, prepend=-np.inf) > merge_gap_days)
    merged = rows.groupby(group).agg(
        time=("time", "mean"),
        start=("time_start", "min"),
        end=("time_end", "max"),
        exposure_s=("exposure_s", "sum"),
    )
    merged["span_s"] = (merged["end"] - merged["start"]) * 86400.0
    return merged


def plot_cadence(table, n_days, fname, n_sources=1000):
    """Plot per-band revisit intervals, visits per day, visit spans and exposure for random sources.

    Parameters
    ----------
    table : SimonsObservatoryObsTable
        The materialized observation table.
    n_days : float
        Survey length in days.
    fname : Path
        Output image filename.
    n_sources : int
        Number of random source positions.

    Returns
    -------
    dict
        Median statistics per band for the report summary.
    """
    df = table._table
    ra, dec = random_sources(n_sources, np.random.default_rng(0))
    stats = {band: {"gaps": [], "per_day": [], "span": [], "exposure": []} for band in table.filters}
    for inds in table.range_search(ra, dec):
        rows = df.iloc[inds]
        for band in table.filters:
            visits = telescope_visits(rows[rows["filter"] == band])
            stats[band]["per_day"].append(len(visits) / n_days)
            stats[band]["gaps"].extend(np.diff(visits["time"]) * 24.0)
            stats[band]["span"].extend(visits["span_s"] / 60.0)
            stats[band]["exposure"].extend(visits["exposure_s"])

    panels = [
        ("gaps", "Revisit interval (h)", np.arange(0, 49, 0.5)),
        ("per_day", "Telescope visits per day", np.arange(-0.125, 2.4, 0.25)),
        ("span", "Visit span over all band tubes (min)", np.arange(0, 41, 1)),
        ("exposure", "Exposure per visit, summed over band tubes (s)", 40),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))
    for ax, (key, label, bins) in zip(axes, panels, strict=True):
        for band, values in stats.items():
            ax.hist(
                values[key],
                bins=bins,
                histtype="step",
                lw=1.5,
                color=BAND_COLORS.get(band),
                label=band,
                density=True,
            )
        ax.set_xlabel(label)
    axes[0].set_ylabel("Density")
    axes[0].legend()
    fig.suptitle(f"Cadence for {n_sources} random sources (Dec -60 to +20), tube visits merged within 1 h")
    fig.tight_layout()
    fig.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return {
        band: {key: float(np.median(values[key])) if len(values[key]) else np.nan for key in values}
        for band, values in stats.items()
    }


def plot_validation(time_maps, fname, n_sources=200):
    """Compare map-pixel visit statistics with the exact single-point trace at random positions.

    Parameters
    ----------
    time_maps : dict
        Output of ``simulate_time_maps``.
    fname : Path
        Output image filename.
    n_sources : int
        Number of random source positions (traced with the first simulated tube).

    Returns
    -------
    dict
        Median and 90th-percentile absolute errors for the report summary.
    """
    maps, tube = time_maps["maps"], time_maps["tubes"][0]
    fields = dict(zip(MAP_FIELDS, maps[:, 0].reshape(len(MAP_FIELDS), 2, -1), strict=True))
    ra, dec = random_sources(n_sources, np.random.default_rng(1))
    pixels = sky_to_pixel(maps.shape, maps.wcs, ra, dec)
    pairs = []
    for r, d, pix in zip(ra, dec, pixels, strict=True):
        for visit in point_visits(r, d, tube, **time_maps) if pix >= 0 else []:
            side = visit["side"]
            if fields["hits"][side, pix] > 0:
                keys = ("duration_s", "exposure_s", "n_passes", "time_s")
                pairs.append([(fields[k][side, pix], visit[k]) for k in keys])
    pairs = np.array(pairs)
    time_err = (
        pairs[:, 3, 0] - pairs[:, 3, 1] + 0.5 * SIDEREAL_DAY_S
    ) % SIDEREAL_DAY_S - 0.5 * SIDEREAL_DAY_S

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    for ax, i, label in zip(axes, range(3), ("Duration (s)", "Exposure (s)", "Passes"), strict=False):
        ax.scatter(pairs[:, i, 1], pairs[:, i, 0], s=8, alpha=0.6)
        lim = [0, pairs[:, i].max() * 1.05]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlabel(f"{label}: exact point trace")
        ax.set_ylabel(f"{label}: map pixel")
    axes[3].hist(time_err, bins=40)
    axes[3].set_xlabel("Mean visit time error, pixel - point (s)")
    fig.suptitle(
        f"Pixelization error at {time_maps['res_deg']} deg for {len(pairs)} random sources (tube {tube})"
    )
    fig.tight_layout()
    fig.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close(fig)
    errors = {
        "duration (s)": pairs[:, 0, 0] - pairs[:, 0, 1],
        "exposure (s)": pairs[:, 1, 0] - pairs[:, 1, 1],
    }
    errors.update({"passes": pairs[:, 2, 0] - pairs[:, 2, 1], "mean time (s)": time_err})
    return {key: np.percentile(np.abs(v), [50, 90]) for key, v in errors.items()}


def write_report(outdir, figures, summary):
    """Write the HTML report referencing the figures.

    Parameters
    ----------
    outdir : Path
        Report directory.
    figures : list of (str, str, str)
        (filename, title, caption) for each figure, relative to ``outdir``.
    summary : dict
        Key run parameters and statistics to tabulate.
    """
    rows = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>" for k, v in summary.items()
    )
    figs = "".join(
        f"<figure><h3>{html.escape(title)}</h3><img src='{name}'><figcaption>{caption}</figcaption></figure>"
        for name, title, caption in figures
    )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>SO Scan ObsTable</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1150px;margin:2em auto;padding:0 16px;
line-height:1.45}} img{{max-width:100%;border:1px solid #ddd}} table{{border-collapse:collapse}}
th,td{{padding:3px 10px;border-bottom:1px solid #eee;text-align:left}} figure{{margin:2em 0}}
figcaption{{color:#444}} code{{background:#f4f4f4}}
</style></head><body>
<h1>Simons Observatory scan-based ObsTable</h1>
<p>SO scans continuously in azimuth at fixed elevation, so instead of discrete pointings we simulate one
sidereal day of the idealized scan, accumulate per-pixel hit and mean-time maps for every optics tube with
<code>pixell</code>, and materialize them into a standard <code>ObsTable</code> with one row per
(tube, observed pixel, day). Source queries map the source to its pixel, so the normal
<code>simulate_lightcurves</code> path is used unchanged.</p>
<h2>Model assumptions</h2>
<ul>
<li>Site 22&deg;57'31"S 67&deg;47'15"W; boresight elevation 40&deg;; azimuth triangle wave at 1&deg;/s.
Each 120&nbsp;s sweep is followed by a 6&nbsp;s (5%) turnaround at the end of the range with no data.</li>
<li>Each patch is crossed once while rising, in the east range (az 30&ndash;150&deg;), and once while
setting, in the west range (az 210&ndash;330&deg;). SO switches daily between the two for each patch:
the rising and setting crossings around one meridian transit share a <code>transit_day</code>, and
successive transit days use opposite crossings. Each range is simulated for a full sidereal day and
<code>from_time_maps</code> picks one crossing per day.</li>
<li>Each pixel is therefore observed at most once per tube per day. Row time is the hit-weighted (circular)
mean of that crossing.</li>
<li>13 tubes, 1.25&deg; field of view, 1.75&deg; hex spacing with half the outer ring missing; 8 MF,
4 UHF, 1 LF.</li>
<li>Pointings are precessed (IAU 2006) from the mean equator and equinox of one epoch, mid-way through
the year, to ICRS; over a multi-year survey this drifts by ~0.014&deg; per year. Refraction (~0.01&deg;
at 40&deg; elevation), nutation and aberration (&lt;0.006&deg;) are ignored.</li>
<li>Downtime: 65% uptime by default, of which 60% of the lost time is in ~week-long outages
(5&ndash;9 days, three times more likely to start in June&ndash;September) and the rest in 1&ndash;6&nbsp;h
interruptions starting between 08:00 and 20:00 local time. Visits overlapping downtime are dropped. The
7-day ObsTable diagnostics above the yearly plots use no downtime. No sun/moon avoidance.</li>
<li>The az 30&ndash;150&deg; / 210&ndash;330&deg; throws limit coverage to roughly Dec &minus;59&deg; to
+21&deg; (plus tube offsets), narrower than the full 40&deg;-elevation ring (&minus;73&deg; to +27&deg;).</li>
</ul>
<h2>Run summary</h2><table>{rows}</table>
{figs}
</body></html>"""
    (outdir / "report.html").write_text(page)


def main():
    """Simulate (or load) time maps, build the ObsTable, and write the report."""
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    time_maps = load_or_simulate(args)
    start_mjd = Time(args.start, scale="utc").mjd
    table = SimonsObservatoryObsTable.from_time_maps(
        time_maps, args.start, args.days, first_side=args.first_side
    )
    print(f"Built ObsTable with {len(table):,} rows")

    prefix = "debug_" if args.debug else ""
    entries = []

    def add(key, title, caption, plot, *plot_args):
        """Make one figure and register it for the report."""
        name = f"{prefix}{key}.png"
        result = plot(*plot_args, args.outdir / name)
        entries.append((name, title, caption))
        return result

    cadence = add(
        "cadence",
        "Cadence and visit length (per band)",
        "Each source's successive transits alternate between its rising and setting crossing, so revisit"
        " intervals alternate between one sidereal day plus and minus the source's rising-to-setting offset"
        " (which depends on declination), giving two peaks symmetric about 24 h. Edge pixels seen on only one"
        " crossing are revisited every two days. A visit is a train of short passes (one per 2-minute"
        " azimuth sweep) over its span.",
        plot_cadence,
        table,
        args.days,
    )
    errors = add(
        "validation",
        "Validation against an exact single-point trace",
        "Map statistics are exact at pixel centers; the scatter here is pixelization error for arbitrary"
        " source positions. Duration is quantized by the sweep period, so off-by-one-pass errors move it by"
        " ~2 minutes.",
        plot_validation,
        time_maps,
    )
    for field, title in (
        ("duration_s", "Visit duration"),
        ("exposure_s", "Exposure"),
        ("n_passes", "Passes"),
    ):
        caption = FIELD_LABELS[field][0] + ", rising crossing (setting is the mirror image)."
        add(field, f"{title} maps", caption, plot_tube_grid, time_maps, field, 0)
    for side, name in enumerate(SIDE_NAMES):
        crossing = ("rising", "setting")[side]
        caption = f"Mean LST of each pixel's visit on {crossing} days ({name} azimuth range)."
        add(
            f"time_s_{name}",
            f"Observation-time maps: {crossing}",
            caption,
            plot_tube_grid,
            time_maps,
            "time_s",
            side,
        )
    downtime = simulate_downtime(args.start, YEAR_DAYS, uptime=args.uptime, seed=args.seed)
    add(
        "downtime",
        "Downtime realization",
        f"One random year of downtime for {args.uptime:.0%} uptime: about week-long outages, preferentially"
        " June&ndash;September, plus interruptions of a few hours starting during local daytime.",
        plot_downtime,
        downtime,
        args.start,
    )
    print("Accumulating one year of visits...", flush=True)
    area = np.asarray(time_maps["maps"].pixsizemap(broadcastable=False)).ravel() / (4.0 * np.pi)
    ideal = year_coverage(time_maps, args.start, args.first_side, None, area)
    footprint = {band: values["footprint"] for band, values in ideal.items()}
    actual = year_coverage(time_maps, args.start, args.first_side, downtime, area, footprint)
    add(
        "year_coverage",
        "One year of coverage with downtime",
        "From left: fraction of the full sky observed on each UTC day (grey: no downtime); area-weighted"
        " days per year each footprint pixel is observed (dotted: no downtime); area-weighted revisit"
        " gaps between observed transit days (one sidereal day when nothing is lost; longer gaps come from"
        " downtime and edge pixels seen on one crossing only); days observed per year, any band. Sun/moon"
        " avoidance is not modeled.",
        plot_year_coverage,
        time_maps,
        ideal,
        actual,
    )
    add(
        "coverage",
        "Coverage union",
        "Number of tubes that observe each pixel on a rising day and on a setting day.",
        plot_coverage,
        time_maps,
    )
    add(
        "layout",
        "Tube layout",
        "Hex focal plane; circles are 1.25&deg; fields of view.",
        plot_layout,
        time_maps,
    )
    add(
        "table_checks",
        "ObsTable checks",
        "Rows per (tube, pixel, day), and example cadence.",
        plot_table_checks,
        table,
        start_mjd,
    )
    if (args.outdir / "before.png").exists():
        entries.append(("before.png", "Previous implementation (for comparison)", "Analytic model output."))

    fields = dict(zip(MAP_FIELDS, time_maps["maps"], strict=True))
    observed = fields["hits"] > 0
    dec = np.degrees(time_maps["maps"].posmap()[0])[observed.any((0, 1))]
    summary = {
        "Mode": "debug" if args.debug else "full",
        "Resolution (deg) / sample interval (s)": f"{time_maps['res_deg']} / {time_maps['dt_s']:.3f}",
        "First day scan side": SIDE_NAMES[args.first_side],
        "Precession epoch (MJD) / uptime / downtime seed": (
            f"{time_maps['epoch_mjd']:.1f} / {args.uptime} / {args.seed}"
        ),
        "Tubes": ", ".join(map(str, time_maps["tubes"])),
        "Start / days": f"{args.start} / {args.days:g}",
        "ObsTable rows": f"{len(table):,}",
        "Observed Dec range (deg)": f"{dec.min():.1f} to {dec.max():.1f}",
        "Per tube visit: median duration (min) / exposure (s) / passes": " / ".join(
            f"{np.median(fields[key][observed]) * scale:.1f}"
            for key, scale in (("duration_s", 1 / 60), ("exposure_s", 1), ("n_passes", 1))
        ),
        "Fraction of observed (tube, pixel) seen on both crossings": (
            f"{np.mean(observed.all(1)[observed.any(1)]):.3f}"
        ),
    }
    for band in actual:
        row = {}
        for label, values in (("no downtime", ideal[band]), ("downtime", actual[band])):
            seen, daily = values["footprint"], values["daily"]
            p10, p50, p90 = weighted_percentile(values["days"][seen], area[seen], [10, 50, 90])
            row[label] = f"{daily.mean():.3f} ({np.percentile(daily, 10):.3f}-{np.percentile(daily, 90):.3f})"
            row[label + " days"] = f"{p10:.0f} / {p50:.0f} / {p90:.0f}"
        summary[f"{band}: daily sky fraction mean (10-90%), no downtime / downtime"] = (
            f"{row['no downtime']} / {row['downtime']}"
        )
        summary[f"{band}: days per year observed (10/50/90% of area), no downtime / downtime"] = (
            f"{row['no downtime days']} / {row['downtime days']}"
        )
        gaps = weighted_percentile(actual[band]["gaps"], actual[band]["gap_weights"], [50, 90, 99])
        summary[f"{band}: revisit gap with downtime, median / 90% / 99% / max (days)"] = (
            " / ".join(f"{g:.1f}" for g in gaps) + f" / {actual[band]['gaps'].max():.1f}"
        )
    for band, values in cadence.items():
        summary[f"{band}: median revisit (h) / visits per day / span (min) / exposure (s)"] = " / ".join(
            f"{values[key]:.2f}" for key in ("gaps", "per_day", "span", "exposure")
        )
    for key, (p50, p90) in errors.items():
        summary[f"Pixelization |error| in {key}: median / 90%"] = f"{p50:.2f} / {p90:.2f}"
    write_report(args.outdir, entries, summary)
    print(f"Wrote {args.outdir / 'report.html'}")


if __name__ == "__main__":
    main()
