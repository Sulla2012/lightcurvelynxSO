"""Simulate one year of long-GRB afterglows observed by Simons Observatory and count detections.

The population includes every viewing angle (on-axis and orphan afterglows) of Gaussian-jet
forward-shock afterglows (afterglowpy). Each burst is screened with a cheap peak-flux estimate;
bursts that could plausibly be detected are simulated with lightcurvelynx through the SO
ObsTable (downtime, sun/moon avoidance, per-visit noise). A burst is detected when, in any band,
its visits within one transit day (one depth-1 map) combine to at least ``--snr`` sigma.

Examples
--------
Quick look (2 deg maps, 2% of the population)::

    python analysis/simulate_grbs.py --debug

Full run (0.5 deg maps, one year, near-on-axis bursts oversampled 10x)::

    python analysis/simulate_grbs.py
"""

import argparse
import html
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import afterglowpy  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from astropy.cosmology import Planck18  # noqa: E402

from lightcurvelynx.math_nodes.given_sampler import TableSampler  # noqa: E402
from lightcurvelynx.models.afterglowpy_models import AfterglowpyModel  # noqa: E402
from lightcurvelynx.noise_models.base_noise_models import GivenNoiseModel  # noqa: E402
from lightcurvelynx.obstable.simons_observatory_obstable import (  # noqa: E402
    MAP_FIELDS,
    SO_AVOID_DEG,
    SO_DAILY_DEPTH_MJY,
    SimonsObservatoryObsTable,
    add_bandflux_error,
    read_time_maps,
    simulate_downtime,
    simulate_time_maps,
    sky_to_pixel,
    so_passbands,
    survey_epoch,
    write_time_maps,
)
from lightcurvelynx.simulate import simulate_lightcurves  # noqa: E402
from lightcurvelynx.survey_info import SurveyInfo  # noqa: E402

# Wanderman & Piran (2010): local rate of long GRBs beamed toward us (L_iso > 1e50 erg/s), and its
# redshift evolution (1+z)^2.1 up to z = 3, then (1+z)^-1.4.
RATE_GPC3_YR = 1.3
SCREEN_NU_HZ = 94.1e9  # f090 flat-weighted center, the most sensitive band relative to its depth
SCREEN_DAYS = np.geomspace(1e-3, 100.0, 10)
PARAM_COLUMNS = ["E0", "theta_core", "theta_wing", "theta_obs", "n0", "p", "epsilon_e", "epsilon_B"]


def parse_args():
    """Parse the command line.

    Returns
    -------
    argparse.Namespace
        The parsed arguments; ``--debug`` overrides resolution and population fraction.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--start", default="2027-01-01", help="Survey and burst window start (UTC).")
    parser.add_argument("--days", type=float, default=365.25, help="Days over which bursts occur.")
    parser.add_argument("--tail", type=float, default=60.0, help="Extra survey days after the last burst.")
    parser.add_argument("--res", type=float, default=0.5, help="Time-map resolution in degrees.")
    parser.add_argument("--zmax", type=float, default=6.0, help="Maximum burst redshift.")
    parser.add_argument(
        "--oversample", type=int, default=10, help="Oversampling of near-on-axis/nearby bursts."
    )
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of the population to simulate.")
    parser.add_argument("--snr", type=float, default=5.0, help="Detection threshold in one daily map.")
    parser.add_argument(
        "--screen-factor", type=float, default=0.05, help="Screen keeps peak > factor * threshold."
    )
    parser.add_argument(
        "--n-leak", type=int, default=300, help="Screened-out bursts simulated to check the screen."
    )
    parser.add_argument("--uptime", type=float, default=0.65, help="Instrument uptime fraction.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--nproc", type=int, default=8, help="Worker processes.")
    parser.add_argument("--chunk", type=int, default=1000, help="Bursts simulated per ObsTable chunk.")
    parser.add_argument(
        "--passbands",
        type=Path,
        nargs="+",
        default=[Path("LAT_MF_median_bands.csv"), Path("LAT_UHF_median_bands.csv")],
        help="Measured bandpass CSV files.",
    )
    parser.add_argument("--outdir", type=Path, default=Path("analysis/grb_report"), help="Report directory.")
    parser.add_argument("--debug", action="store_true", help="Quick run: 2 deg maps, 2%% of the population.")
    args = parser.parse_args()
    if args.debug:
        args.res, args.fraction, args.oversample, args.n_leak = 2.0, 0.02, 2, 50
    return args


def load_time_maps(res, epoch, outdir):
    """Load cached all-tube time maps or simulate and cache them.

    Parameters
    ----------
    res : float
        Map resolution in degrees.
    epoch : float
        Precession epoch (MJD).
    outdir : Path
        Cache directory.

    Returns
    -------
    dict
        Time maps from ``simulate_time_maps``.
    """
    fname = outdir / f"time_maps_res{res:g}_ep{epoch:.0f}.fits"
    if fname.exists():
        return read_time_maps(fname)
    print(f"Simulating time maps at {res} deg...", flush=True)
    time_maps = simulate_time_maps(res, epoch_mjd=epoch)
    write_time_maps(fname, time_maps)
    return time_maps


def redshift_rate(z):
    """Return the all-sky rate of bursts beamed toward us per unit redshift (per year).

    Parameters
    ----------
    z : numpy.ndarray
        Redshifts.

    Returns
    -------
    numpy.ndarray
        dN/dz/dt (observer-frame years), including time dilation 1/(1+z).
    """
    evolution = np.where(z <= 3.0, (1 + z) ** 2.1, 4.0**2.1 * ((1 + z) / 4.0) ** -1.4)
    dvdz = Planck18.differential_comoving_volume(z).to("Gpc3/sr").value * 4 * np.pi
    return RATE_GPC3_YR * evolution / (1 + z) * dvdz


def sample_jets(n, rng):
    """Draw jet and microphysics parameters.

    Parameters
    ----------
    n : int
        Number of bursts.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    dict
        Parameter arrays (afterglowpy names in ``PARAM_COLUMNS``).
    """
    theta_core = np.clip(10 ** rng.normal(-1.1, 0.25, n), 0.02, 0.3)
    return {
        "E0": 10 ** rng.normal(53.0, 0.75, n),
        "theta_core": theta_core,
        "theta_wing": np.minimum(4.0 * theta_core, np.pi / 2),
        "theta_obs": np.arccos(rng.uniform(0.0, 1.0, n)),
        "n0": np.clip(10 ** rng.normal(-1.0, 1.0, n), 1e-4, 1e2),
        "p": np.clip(rng.normal(2.3, 0.15, n), 2.05, 3.0),
        "epsilon_e": np.full(n, 0.1),
        "epsilon_B": np.clip(10 ** rng.normal(-3.0, 1.0, n), 1e-6, 0.3),
    }


def mean_beaming(rng, n=200_000):
    """Return the mean on-axis solid-angle fraction 1 - cos(theta_core) of the jet population.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator.
    n : int
        Number of draws.

    Returns
    -------
    float
        Mean beaming fraction, which converts the beamed rate to the all-orientation rate.
    """
    return np.mean(1.0 - np.cos(sample_jets(n, rng)["theta_core"]))


def sample_population(args, footprint, maps, rng):
    """Draw the burst population in the SO footprint with stratified oversampling.

    Bursts are drawn from ``oversample`` times the expected all-orientation population. "Near"
    bursts (theta_obs < 2 theta_core or z < 0.25), which dominate detections, are all kept with
    weight 1/oversample; the others are thinned to probability 1/oversample with weight 1, so every
    weighted sum is an unbiased estimate for one survey.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    footprint : numpy.ndarray
        Boolean flat-pixel mask of observed pixels.
    maps : enmap
        Map whose geometry defines the pixels.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    population : pandas.DataFrame
        One row per burst in the footprint with parameters, position, t0, pixel and weight.
    info : dict
        Rate bookkeeping for the report.
    """
    z_grid = np.linspace(1e-4, args.zmax, 4000)
    dndz = redshift_rate(z_grid)
    beamed = np.trapezoid(dndz, z_grid) * args.days / 365.25
    beaming = mean_beaming(rng)
    expected = beamed / beaming * args.fraction
    n = rng.poisson(expected * args.oversample)
    cdf = np.cumsum(dndz) / np.sum(dndz)
    data = sample_jets(n, rng)
    data["z"] = np.interp(rng.uniform(0, 1, n), cdf, z_grid)
    data["ra"] = rng.uniform(0.0, 360.0, n)
    data["dec"] = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, n)))
    data["t0"] = survey_epoch(args.start)[0] + rng.uniform(0.0, args.days, n)
    df = pd.DataFrame(data)
    df["near"] = (df["theta_obs"] < 2 * df["theta_core"]) | (df["z"] < 0.25)
    keep = df["near"] | (rng.uniform(0, 1, n) < 1.0 / args.oversample)
    df = df[keep].copy()
    df["weight"] = np.where(df["near"], 1.0 / args.oversample, 1.0) / args.fraction
    df["pixel"] = sky_to_pixel(maps.shape, maps.wcs, df["ra"].to_numpy(), df["dec"].to_numpy())
    df = df[(df["pixel"] >= 0) & footprint[np.maximum(df["pixel"], 0)]].reset_index(drop=True)
    df["d_L_cm"] = Planck18.luminosity_distance(df["z"].to_numpy()).to("cm").value
    info = {
        "beamed_per_year_allsky": beamed * 365.25 / args.days,
        "mean_beaming": beaming,
        "all_orientation_per_year_allsky": beamed / beaming * 365.25 / args.days,
        "in_footprint_weighted": df["weight"].sum(),
    }
    return df, info


def screen_chunk(rows):
    """Return the maximum f090 flux (mJy) on a coarse time grid for a chunk of bursts.

    Parameters
    ----------
    rows : pandas.DataFrame
        Bursts with afterglowpy parameters, ``z`` and ``d_L_cm``.

    Returns
    -------
    numpy.ndarray
        Peak flux per burst; NaN if afterglowpy failed (such bursts are kept).
    """
    t = SCREEN_DAYS * 86400.0
    nu = np.full_like(t, SCREEN_NU_HZ)
    peaks = np.full(len(rows), np.nan)
    for i, row in enumerate(rows.itertuples()):
        try:
            flux = afterglowpy.fluxDensity(t, nu, **jet_kwargs(row), tRes=100, latRes=3)
            peaks[i] = np.max(flux)
        except Exception:  # noqa: BLE001
            pass
    return peaks


def jet_kwargs(row):
    """Return afterglowpy keyword arguments for one burst row.

    Parameters
    ----------
    row : namedtuple
        A row of the population table.

    Returns
    -------
    dict
        Keyword arguments for ``afterglowpy.fluxDensity`` (without times and frequencies).
    """
    return {
        "jetType": afterglowpy.jet.Gaussian,
        "specType": afterglowpy.jet.SimpleSpec,
        "E0": row.E0,
        "thetaCore": row.theta_core,
        "thetaWing": row.theta_wing,
        "thetaObs": row.theta_obs,
        "n0": row.n0,
        "p": row.p,
        "epsilon_e": row.epsilon_e,
        "epsilon_B": row.epsilon_B,
        "xi_N": 1.0,
        "z": row.z,
        "d_L": row.d_L_cm,
    }


def screen(population, nproc, chunk=500):
    """Compute the screening peak flux of every burst in parallel.

    Parameters
    ----------
    population : pandas.DataFrame
        The burst population.
    nproc : int
        Number of worker processes.
    chunk : int
        Bursts per task.

    Returns
    -------
    numpy.ndarray
        Peak f090 flux on the screening grid (mJy).
    """
    chunks = [population.iloc[i : i + chunk] for i in range(0, len(population), chunk)]
    with ProcessPoolExecutor(nproc) as pool:
        return np.concatenate(list(pool.map(screen_chunk, chunks)))


def simulate_bursts(bursts, time_maps, args, downtime, passbands):
    """Simulate SO light curves of the given bursts with lightcurvelynx, in chunks.

    The sun/moon exposure rescaling is estimated once from random sky pixels and reused, and each
    chunk builds an ObsTable only for its bursts' pixels, which keeps memory per worker small.

    Parameters
    ----------
    bursts : pandas.DataFrame
        Bursts to simulate.
    time_maps : dict
        Time maps.
    args : argparse.Namespace
        Parsed arguments.
    downtime : numpy.ndarray
        Downtime intervals (MJD).
    passbands : PassbandGroup
        SO passbands.

    Returns
    -------
    pandas.DataFrame
        One row per visit of a burst: ``burst`` (row in ``bursts``), time, filter, flux,
        fluxerr and ``transit_day``.
    """
    clock = time.time()
    n_days = args.days + args.tail
    tubes = list(time_maps["tubes"])
    scale = SimonsObservatoryObsTable.avoidance_scale(
        time_maps, args.start, n_days, tubes, 0, downtime, SO_AVOID_DEG, 3000, args.seed
    )
    print(f"  sun/moon rescaling estimated ({time.time() - clock:.0f} s)", flush=True)
    pieces = []
    for first in range(0, len(bursts), args.chunk):
        chunk = bursts.iloc[first : first + args.chunk]
        visits = simulate_chunk(chunk, time_maps, args, downtime, passbands, scale)
        pieces.append(visits.assign(burst=visits["burst"] + first))
        print(f"  simulated bursts {first:,}-{first + len(chunk):,} of {len(bursts):,}", flush=True)
    return pd.concat(pieces, ignore_index=True)


def simulate_chunk(bursts, time_maps, args, downtime, passbands, scale):
    """Simulate SO light curves of one chunk of bursts with lightcurvelynx.

    Parameters
    ----------
    bursts : pandas.DataFrame
        Bursts to simulate.
    time_maps : dict
        Time maps.
    args : argparse.Namespace
        Parsed arguments.
    downtime : numpy.ndarray
        Downtime intervals (MJD).
    passbands : PassbandGroup
        SO passbands.
    scale : pandas.Series
        Sun/moon exposure rescaling from ``SimonsObservatoryObsTable.avoidance_scale``.

    Returns
    -------
    pandas.DataFrame
        One row per visit of a burst: ``burst`` (row in this chunk), time, filter, flux,
        fluxerr and ``transit_day``.
    """
    clock = time.time()
    table = SimonsObservatoryObsTable.from_time_maps(
        time_maps,
        args.start,
        args.days + args.tail,
        downtime=downtime,
        pixels=np.unique(bursts["pixel"]),
        scale=scale,
    )
    add_bandflux_error(table, time_maps)
    print(
        f"  ObsTable for {len(bursts):,} bursts: {len(table):,} rows ({time.time() - clock:.0f} s)",
        flush=True,
    )
    survey = SurveyInfo(table, passbands=passbands, noise_model=GivenNoiseModel())
    sampler = TableSampler(
        bursts[["ra", "dec", "z", "t0", *PARAM_COLUMNS]].reset_index(drop=True), in_order=True
    )
    model = AfterglowpyModel(
        **{name: getattr(sampler, name) for name in PARAM_COLUMNS},
        ra=sampler.ra,
        dec=sampler.dec,
        redshift=sampler.z,
        t0=sampler.t0,
        cosmology=Planck18,
        node_label="grb",
    )
    results = simulate_lightcurves(
        model,
        len(bursts),
        survey,
        obstable_save_cols=["transit_day", "tube_id"],
        param_cols=["grb.E0", "grb.theta_obs"],
        obs_time_window_offset=(0.0, args.tail),
        rng=np.random.default_rng(args.seed),
        num_jobs=args.nproc,
        batch_size=max(1, int(np.ceil(len(bursts) / args.nproc))),
        progress_bar=False,
    )
    print(f"  lightcurvelynx simulation ({time.time() - clock:.0f} s)", flush=True)
    # The parameters must come back in the order of ``bursts`` (in-order TableSampler).
    assert np.allclose(results["grb_E0"].to_numpy(), bursts["E0"].to_numpy(), rtol=1e-9)
    pieces = []
    for burst, lc in enumerate(results["lightcurve"]):
        if lc is None or len(lc) == 0:
            continue
        pieces.append(lc[["mjd", "filter", "flux", "fluxerr", "transit_day"]].assign(burst=burst))
    columns = ["mjd", "filter", "flux", "fluxerr", "transit_day", "burst"]
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=columns)


def daily_snr(visits):
    """Combine each burst's visits per band and transit day (one depth-1 map) and return the S/N.

    Parameters
    ----------
    visits : pandas.DataFrame
        Output of ``simulate_bursts``.

    Returns
    -------
    pandas.DataFrame
        One row per (burst, filter, transit_day) with combined flux, error (nJy), S/N and mean time.
    """
    ivar = 1.0 / visits["fluxerr"] ** 2
    grouped = visits.assign(ivar=ivar, wflux=visits["flux"] * ivar, wtime=visits["mjd"] * ivar).groupby(
        ["burst", "filter", "transit_day"], observed=True
    )
    sums = grouped[["ivar", "wflux", "wtime"]].sum()
    out = pd.DataFrame(
        {
            "flux": sums["wflux"] / sums["ivar"],
            "fluxerr": 1.0 / np.sqrt(sums["ivar"]),
            "mjd": sums["wtime"] / sums["ivar"],
        }
    )
    out["snr"] = out["flux"] / out["fluxerr"]
    return out.reset_index()


def coadd_snr(daily):
    """Co-add each burst's daily maps across bands and return the S/N of two co-adds per transit day.

    ``snr_sum`` is the straight sum, sum(F_b) / sqrt(sum(sigma_b^2)). ``snr_ivar`` weights each band
    by 1/sigma_b^2, sum(F_b / sigma_b^2) / sqrt(sum(1 / sigma_b^2)), which is the optimal (matched
    filter) co-add for a flat spectrum in F_nu and is not swamped by the noisy LF and f280 bands.

    Parameters
    ----------
    daily : pandas.DataFrame
        Output of ``daily_snr``.

    Returns
    -------
    pandas.DataFrame
        One row per (burst, transit_day) with ``snr_band`` (best single band), ``snr_ivar``,
        ``snr_sum`` and ``mjd``.
    """
    ivar = 1.0 / daily["fluxerr"] ** 2
    terms = daily.assign(var=daily["fluxerr"] ** 2, ivar=ivar, wflux=daily["flux"] * ivar)
    sums = terms.groupby(["burst", "transit_day"])[["flux", "var", "ivar", "wflux", "snr", "mjd"]].agg(
        {"flux": "sum", "var": "sum", "ivar": "sum", "wflux": "sum", "snr": "max", "mjd": "mean"}
    )
    return pd.DataFrame(
        {
            "snr_band": sums["snr"],
            "snr_ivar": sums["wflux"] / np.sqrt(sums["ivar"]),
            "snr_sum": sums["flux"] / np.sqrt(sums["var"]),
            "mjd": sums["mjd"],
        }
    ).reset_index()


def classify(df):
    """Label bursts by viewing angle.

    Parameters
    ----------
    df : pandas.DataFrame
        Bursts with ``theta_obs`` and ``theta_core``.

    Returns
    -------
    pandas.Series
        "on-axis" (theta_obs < theta_core), "near off-axis" (< 2 theta_core) or "orphan".
    """
    ratio = df["theta_obs"] / df["theta_core"]
    return pd.Series(
        np.select([ratio < 1, ratio < 2], ["on-axis", "near off-axis"], "orphan"), index=df.index
    )


def plot_population(population, detected, fname):
    """Plot the weighted population and detections versus redshift and viewing angle.

    Parameters
    ----------
    population : pandas.DataFrame
        All bursts in the footprint.
    detected : pandas.DataFrame
        Detected bursts.
    fname : Path
        Output figure file.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, column, bins, label in (
        (axes[0], "z", np.linspace(0, 6, 31), "Redshift"),
        (axes[1], "ratio", np.geomspace(0.01, 30, 31), "theta_obs / theta_core"),
        (axes[2], "log_E0", np.linspace(50, 56, 31), "log10 E0 (erg)"),
    ):
        for df, kwargs in ((population, {"label": "in footprint"}), (detected, {"label": "detected"})):
            values = {"ratio": df["theta_obs"] / df["theta_core"], "log_E0": np.log10(df["E0"])}.get(
                column, df.get(column)
            )
            ax.hist(values, bins, weights=df["weight"], histtype="step", **kwargs)
        ax.set(xlabel=label, ylabel="Bursts per year", yscale="log")
        if column == "ratio":
            ax.set_xscale("log")
    axes[0].legend()
    fig.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_lightcurves(detected, daily, fname, n=6):
    """Plot daily-map light curves of the brightest detected bursts.

    Parameters
    ----------
    detected : pandas.DataFrame
        Detected bursts (with ``burst`` id and ``peak_snr``).
    daily : pandas.DataFrame
        Output of ``daily_snr``.
    fname : Path
        Output figure file.
    n : int
        Number of bursts to plot.
    """
    chosen = detected.sort_values("peak_snr", ascending=False).head(n)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), squeeze=False)
    for ax, (_, burst) in zip(axes.ravel(), chosen.iterrows(), strict=False):
        rows = daily[daily["burst"] == burst["burst"]]
        for band, group in rows.groupby("filter", observed=True):
            ax.errorbar(
                group["mjd"] - burst["t0"],
                group["flux"] / 1e6,
                group["fluxerr"] / 1e6,
                fmt="o",
                ms=3,
                label=band,
            )
        ax.axhline(0, color="k", lw=0.5)
        ax.set(
            xscale="symlog",
            xlabel="Days since burst",
            ylabel="Daily-map flux (mJy)",
            title=f"z={burst['z']:.2f}, theta_obs/theta_c={burst['theta_obs'] / burst['theta_core']:.1f}, "
            f"peak S/N {burst['peak_snr']:.0f}",
        )
        ax.legend(fontsize=7)
    for ax in axes.ravel()[len(chosen) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_screen(population, threshold, fname):
    """Plot the screening peak flux of the population against the screen and detection thresholds.

    Parameters
    ----------
    population : pandas.DataFrame
        Bursts with ``screen_peak`` (mJy), ``weight`` and ``detected``.
    threshold : float
        Daily detection threshold: 5 sigma of the inverse-variance co-add (mJy).
    fname : Path
        Output figure file.
    """
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bins = np.geomspace(1e-8, 1e5, 60)
    peaks = np.clip(population["screen_peak"].fillna(1e5), 1e-8, 1e5)
    ax.hist(peaks, bins, weights=population["weight"], histtype="step", label="all in footprint")
    det = population["detected"]
    ax.hist(peaks[det], bins, weights=population["weight"][det], histtype="step", label="detected")
    for value, style, label in (
        (threshold, "-", "5 sigma daily co-add"),
        (threshold * population.attrs["factor"], ":", "screen"),
    ):
        ax.axvline(value, color="k", ls=style, label=label)
    ax.set(
        xscale="log", yscale="log", xlabel="Peak f090 flux on screening grid (mJy)", ylabel="Bursts per year"
    )
    ax.legend(fontsize=8)
    fig.savefig(fname, dpi=110, bbox_inches="tight")
    plt.close(fig)


def write_report(outdir, figures, summary, assumptions):
    """Write the HTML report.

    Parameters
    ----------
    outdir : Path
        Report directory.
    figures : list of (str, str, str)
        (filename, title, caption) per figure.
    summary : dict
        Key results to tabulate.
    assumptions : list of str
        Model assumptions (HTML).
    """
    rows = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>" for k, v in summary.items()
    )
    items = "".join(f"<li>{a}</li>" for a in assumptions)
    figs = "".join(
        f"<figure><h3>{html.escape(t)}</h3><img src='{n}'><figcaption>{c}</figcaption></figure>"
        for n, t, c in figures
    )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>SO GRB Yield</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1150px;margin:2em auto;padding:0 16px;
line-height:1.45}}
img{{max-width:100%;border:1px solid #ddd}} table{{border-collapse:collapse}}
th,td{{border:1px solid #ccc;padding:4px 8px;text-align:left;vertical-align:top}}</style></head><body>
<h1>Simons Observatory: expected GRB afterglow detections</h1>
<h2>Results</h2><table>{rows}</table>
<h2>Assumptions</h2><ul>{items}</ul>{figs}</body></html>"""
    (outdir / "report.html").write_text(page)


def main():
    """Run the population, screen, lightcurvelynx simulation, detection and report."""
    # The two bands of a dichroic tube share visit times; lightcurvelynx warns about duplicates.
    warnings.filterwarnings("ignore", message="Found duplicate observation times")
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    prefix = "debug_" if args.debug else ""
    rng = np.random.default_rng(args.seed)
    clock = time.time()

    start_mjd = survey_epoch(args.start)[0]
    time_maps = load_time_maps(args.res, start_mjd + args.days / 2, args.outdir)
    maps = time_maps["maps"]
    footprint = np.asarray(maps[MAP_FIELDS.index("hits")]).sum((0, 1)).ravel() > 0
    population, info = sample_population(args, footprint, maps, rng)
    print(
        f"{len(population):,} bursts in footprint to screen ({info['in_footprint_weighted']:.0f} per year)",
        flush=True,
    )

    # Screen against the most sensitive statistic: the inverse-variance co-add of the daily depths.
    coadd_depth = 1.0 / np.sqrt(np.sum([1.0 / d**2 for d in SO_DAILY_DEPTH_MJY.values()]))
    threshold = args.snr * coadd_depth
    population["screen_peak"] = screen(population, args.nproc)
    passed = population["screen_peak"].isna() | (population["screen_peak"] >= args.screen_factor * threshold)
    print(f"{passed.sum():,} passed the screen ({time.time() - clock:.0f} s)", flush=True)

    # Also simulate a random sample of screened-out bursts to check that none would be detected.
    leak_ids = rng.choice(np.flatnonzero(~passed), min(args.n_leak, int((~passed).sum())), replace=False)
    run = pd.concat([population[passed], population.iloc[leak_ids]])
    run["leak_check"] = np.r_[np.zeros(passed.sum(), bool), np.ones(len(leak_ids), bool)]
    run = run.reset_index(names="pop_index")

    downtime = simulate_downtime(args.start, args.days + args.tail, uptime=args.uptime, seed=args.seed)
    passbands = so_passbands([f for f in args.passbands if f.exists()])
    visits = simulate_bursts(run, time_maps, args, downtime, passbands)
    print(f"Simulated {len(run):,} bursts, {len(visits):,} visits ({time.time() - clock:.0f} s)", flush=True)

    daily = daily_snr(visits)
    days = coadd_snr(daily)
    # Headline criterion: a single band or the inverse-variance co-add reaches the threshold.
    days["snr"] = days[["snr_band", "snr_ivar"]].max(axis=1)
    per_burst = days.groupby("burst")[["snr", "snr_band", "snr_ivar", "snr_sum"]].max()
    n_days = days[days["snr"] >= args.snr].groupby("burst")["transit_day"].nunique()
    for column in per_burst:
        run[f"peak_{column}"] = per_burst[column].reindex(run.index).fillna(0.0).to_numpy()
    run["days_detected"] = n_days.reindex(run.index).fillna(0).astype(int).to_numpy()
    run["burst"] = run.index
    run["class"] = classify(run)
    population["detected"] = False
    sim = run[~run["leak_check"]]
    population.loc[sim["pop_index"], "detected"] = (sim["peak_snr"] >= args.snr).to_numpy()
    population.attrs["factor"] = args.screen_factor
    population["class"] = classify(population)
    detected = sim[sim["peak_snr"] >= args.snr]
    leaked = run[run["leak_check"] & (run["peak_snr"] >= args.snr)]
    run.to_csv(args.outdir / f"{prefix}simulated_bursts.csv", index=False)

    figures = []

    def add(key, title, caption, plot, *plot_args):
        """Make one figure and register it for the report."""
        name = f"{prefix}{key}.png"
        plot(*plot_args, args.outdir / name)
        figures.append((name, title, caption))

    add(
        "population",
        "Population and detections",
        "Weighted bursts per year in the SO footprint (all viewing angles) and those detected, versus"
        " redshift, viewing angle relative to the jet core, and core isotropic-equivalent energy.",
        plot_population,
        population,
        detected,
    )
    add(
        "screen",
        "Screening",
        "Peak f090 flux of every burst on a coarse 10-point time grid. Bursts below the screen (dotted) are"
        " not simulated; a random sample of them is simulated to check that none would be detected.",
        plot_screen,
        population,
        threshold,
    )
    if len(detected):
        add(
            "lightcurves",
            "Brightest detections",
            "Daily-map (one transit day, all tubes of a band combined) flux and 1 sigma error per band.",
            plot_lightcurves,
            detected,
            daily,
        )

    weights = detected["weight"]
    band_hits = daily[daily["snr"] >= args.snr]
    per_band = {
        band: detected.loc[
            detected["burst"].isin(band_hits.loc[band_hits["filter"] == band, "burst"]), "weight"
        ]
        for band in SO_DAILY_DEPTH_MJY
    }
    by_class = detected.groupby("class")["weight"].agg(["sum", lambda w: np.sqrt(np.sum(w**2))])
    summary = {
        "Mode": "debug" if args.debug else "full",
        "Expected detections per year (MC error)": (
            f"{weights.sum():.2f} +/- {np.sqrt(np.sum(weights**2)):.2f}"
        ),
        **{f"  {cls}": f"{row['sum']:.2f} +/- {row.iloc[1]:.2f}" for cls, row in by_class.iterrows()},
        "  detected on 2+ days": f"{weights[detected['days_detected'] >= 2].sum():.2f}",
        "Simulated detections (unweighted)": len(detected),
        "Detection criterion": (
            f">= {args.snr:g} sigma in one transit day, in a single band or the inverse-variance co-add"
        ),
        **{
            f"Expected per year with {label}": f"{sim.loc[sim[column] >= args.snr, 'weight'].sum():.2f}"
            for label, column in (
                ("best single band only", "peak_snr_band"),
                ("inverse-variance co-add only", "peak_snr_ivar"),
                ("straight-sum co-add only, sum F / sqrt(sum sigma^2)", "peak_snr_sum"),
            )
        },
        "Daily depth: f090 / inverse-variance co-add / straight sum (mJy)": (
            f"{SO_DAILY_DEPTH_MJY['f090']:g} / {coadd_depth:.1f} / "
            f"{np.sqrt(np.sum([d**2 for d in SO_DAILY_DEPTH_MJY.values()])):.0f}"
        ),
        "Long GRBs per year, beamed toward us, all sky (z < zmax)": f"{info['beamed_per_year_allsky']:.0f}",
        "Mean beaming fraction 1 - cos(theta_core)": f"{info['mean_beaming']:.4f}",
        "Long GRBs per year, all orientations, all sky": f"{info['all_orientation_per_year_allsky']:.3g}",
        "In SO footprint per year (weighted)": f"{info['in_footprint_weighted']:.3g}",
        "Bursts screened / passed / simulated": f"{len(population):,} / {passed.sum():,} / {len(run):,}",
        "Screen leak check: detected among screened-out sample": f"{len(leaked)} of {len(leak_ids)}",
        "Detections per band (weighted; a burst may count in several)": ", ".join(
            f"{band}: {w.sum():.2f}" for band, w in per_band.items()
        ),
        "Resolution (deg) / oversample / population fraction / seed": (
            f"{args.res} / {args.oversample} / {args.fraction} / {args.seed}"
        ),
        "Runtime (s)": f"{time.time() - clock:.0f}",
    }
    assumptions = [
        "Long GRBs only, as forward-shock afterglows from Gaussian structured jets (afterglowpy, no reverse"
        " shock, no synchrotron self-absorption, no counter-jet).",
        f"Rate: Wanderman &amp; Piran (2010), {RATE_GPC3_YR} Gpc<sup>-3</sup> yr<sup>-1</sup> locally for"
        " bursts beamed toward us, evolving as (1+z)<sup>2.1</sup> to z = 3 and (1+z)<sup>-1.4</sup> beyond,"
        f" up to z = {args.zmax:g}. The all-orientation rate divides this by the mean beaming fraction.",
        "Viewing angles isotropic (cos theta_obs uniform). theta_core log-normal (median 0.08 rad, 0.25 dex,"
        " 0.02-0.3), theta_wing = 4 theta_core.",
        "E0 (core isotropic-equivalent kinetic energy) log-normal, median 10<sup>53</sup> erg, 0.75 dex; n0"
        " log-normal, median 0.1 cm<sup>-3</sup>, 1 dex; epsilon_B log-normal, median 10<sup>-3</sup>, 1 dex;"
        " epsilon_e = 0.1; p normal 2.3 &plusmn; 0.15. The energy distribution is not tied to the luminosity"
        " cut of the rate.",
        "Bursts occur uniformly over the year; each is observed for up to the tail period after the burst.",
        f"SO survey: one year from {args.start}, {args.uptime:.0%} uptime, sun/moon avoidance with conserved"
        " exposure, per-visit noise from the one-day depths.",
        "Detection: a daily map is all visits of a band on one transit day combined; any band at the"
        " threshold counts. Real candidates would also need follow-up/vetting.",
    ]
    write_report(args.outdir, figures, summary, assumptions)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Wrote {args.outdir / 'report.html'}")


if __name__ == "__main__":
    main()
