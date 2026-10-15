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
    SO_TUBE_FILTERS,
    SimonsObservatoryObsTable,
    point_visits,
    read_time_maps,
    simulate_time_maps,
    sky_to_pixel,
    tube_offsets,
    write_time_maps,
)

# Field -> (colorbar label, colormap, scale factor applied before plotting).
FIELD_LABELS = {
    "time_s": ("LST of observation (h)", "twilight", 24.0 / SIDEREAL_DAY_S),
    "duration_s": ("Visit duration, first to last in-FoV sample (min)", "viridis", 1.0 / 60.0),
    "exposure_s": ("Time in tube FoV per visit (s)", "magma", 1.0),
    "n_passes": ("Azimuth sweeps over pixel per visit", "cividis", 1.0),
    "side": ("Scan side (0 = east, 1 = west)", "coolwarm", 1.0),
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
    parser.add_argument("--block-hours", type=float, default=1.0, help="East/west schedule block length.")
    parser.add_argument("--outdir", type=Path, default=Path("analysis/so_report"), help="Report directory.")
    parser.add_argument("--overwrite", action="store_true", help="Re-simulate even if cached maps exist.")
    parser.add_argument("--debug", action="store_true", help="Quick run: 2 deg pixels, tubes 0 1 12, 3 days.")
    args = parser.parse_args()
    if args.debug:
        args.res, args.tubes, args.days = 2.0, [0, 1, 12], 3.0
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
    tag = f"res{args.res:g}_block{args.block_hours:g}_tubes{'-'.join(map(str, args.tubes))}"
    fname = args.outdir / f"time_maps_{tag}.fits"
    if fname.exists() and not args.overwrite:
        print(f"Reading cached time maps from {fname}")
        return read_time_maps(fname)
    print(f"Simulating one sidereal day for tubes {args.tubes} at {args.res} deg...", flush=True)
    time_maps = simulate_time_maps(args.res, tubes=args.tubes, block_hours=args.block_hours)
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


def plot_tube_grid(time_maps, field, fname):
    """Plot one map field for every simulated tube in a grid.

    Parameters
    ----------
    time_maps : dict
        Output of ``simulate_time_maps``.
    field : str
        One of ``FIELD_LABELS``.
    fname : Path
        Output image filename.
    """
    maps, tubes = time_maps["maps"], time_maps["tubes"]
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
    """Plot the number of tubes observing each pixel in one sidereal day.

    Parameters
    ----------
    time_maps : dict
        Output of ``simulate_time_maps``.
    fname : Path
        Output image filename.
    """
    maps = time_maps["maps"]
    count = (maps[0] > 0).sum(axis=0).astype(float)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    artist = sky_imshow(ax, np.where(count > 0, count, np.nan), maps, cmap="viridis")
    fig.colorbar(artist, ax=ax, label="Tubes observing pixel per day")
    ax.set_title(f"Coverage union of {len(time_maps['tubes'])} tubes")
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
    day = np.floor((df["time"] - start_mjd) * 86400.0 / SIDEREAL_DAY_S).astype(int)
    per_day = df.groupby([df["tube_id"], df["pixel_index"], day]).size()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    ax1.hist(per_day.to_numpy(), bins=np.arange(0.5, 4.5), rwidth=0.8)
    ax1.set_xlabel("Observations per (tube, pixel, sidereal day)")
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
    fields = dict(zip(MAP_FIELDS, maps[:, 0].reshape(len(MAP_FIELDS), -1), strict=True))
    ra, dec = random_sources(n_sources, np.random.default_rng(1))
    pixels = sky_to_pixel(maps.shape, maps.wcs, ra, dec)
    pairs = []
    for r, d, pix in zip(ra, dec, pixels, strict=True):
        visits = [v for v in point_visits(r, d, tube, **time_maps) if v["side"] == fields["side"][pix]]
        if pix >= 0 and fields["hits"][pix] > 0 and visits:
            truth = max(visits, key=lambda v: v["exposure_s"])
            pairs.append(
                [(fields[k][pix], truth[k]) for k in ("duration_s", "exposure_s", "n_passes", "time_s")]
            )
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
<li>Site 22&deg;57'31"S 67&deg;47'15"W; boresight elevation 40&deg;; azimuth triangle wave at 1&deg;/s with
instantaneous turnarounds.</li>
<li>Schedule alternates east (az 30&ndash;150&deg;) and west (az 210&ndash;330&deg;) blocks, phased in LST,
so the pattern repeats every sidereal day. <b>The real SO block schedule is still to be decided.</b></li>
<li>Each pixel is observed at most once per tube per day: if crossed in both ranges, only the crossing with
more hits is kept. Row time is the hit-weighted (circular) mean of that crossing.</li>
<li>13 tubes, 1.25&deg; field of view, 1.75&deg; hex spacing with half the outer ring missing; generic
filter labels (8 MF, 4 UHF, 1 LF) to be replaced by the real mapping.</li>
<li>No refraction, precession, sun/moon avoidance, weather, or turnaround overhead.</li>
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
    table = SimonsObservatoryObsTable.from_time_maps(time_maps, args.start, args.days)
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
        "Revisit intervals cluster at one sidereal day (23.93 h); sources seen on both the rising and setting"
        " crossings (by different tubes) also get shorter intervals. A visit is a train of short passes"
        " (one per 2-minute azimuth sweep) over its span.",
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
        add(field, f"{title} maps", FIELD_LABELS[field][0] + ".", plot_tube_grid, time_maps, field)
    add(
        "time_s",
        "Observation-time maps",
        "Mean LST of each pixel's daily visit.",
        plot_tube_grid,
        time_maps,
        "time_s",
    )
    add(
        "side", "Scan side", "Azimuth range of each visit (east = rising).", plot_tube_grid, time_maps, "side"
    )
    add(
        "coverage",
        "Coverage union",
        "Number of tubes that observe each pixel in a day.",
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
    dec = np.degrees(time_maps["maps"].posmap()[0])
    summary = {
        "Mode": "debug" if args.debug else "full",
        "Resolution (deg) / sample interval (s)": f"{time_maps['res_deg']} / {time_maps['dt_s']:.3f}",
        "Block length (h)": time_maps["block_hours"],
        "Tubes": ", ".join(map(str, time_maps["tubes"])),
        "Start / days": f"{args.start} / {args.days:g}",
        "ObsTable rows": f"{len(table):,}",
        "Observed Dec range (deg)": f"{dec[observed.any(0)].min():.1f} to {dec[observed.any(0)].max():.1f}",
        "Per tube visit: median duration (min) / exposure (s) / passes": " / ".join(
            f"{np.median(fields[key][observed]) * scale:.1f}"
            for key, scale in (("duration_s", 1 / 60), ("exposure_s", 1), ("n_passes", 1))
        ),
        "East / west fraction": "/".join(
            f"{np.mean(fields['side'][observed] == i):.2f}" for i in range(len(SIDE_NAMES))
        ),
    }
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
