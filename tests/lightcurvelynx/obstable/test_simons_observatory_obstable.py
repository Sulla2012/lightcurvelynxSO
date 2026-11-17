import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body
from astropy.time import Time

pytest.importorskip("pixell")

from lightcurvelynx.astro_utils.passbands import PassbandGroup  # noqa: E402
from lightcurvelynx.models.basic_models import ConstantSEDModel  # noqa: E402
from lightcurvelynx.noise_models.base_noise_models import GivenNoiseModel  # noqa: E402
from lightcurvelynx.obstable.simons_observatory_obstable import (  # noqa: E402
    MAP_FIELDS,
    SIDE_NAMES,
    SIDEREAL_DAY_S,
    SO_DAILY_DEPTH_MJY,
    SO_LATITUDE_DEG,
    SO_LF_TOPHATS,
    SO_LONGITUDE_DEG,
    SimonsObservatoryObsTable,
    add_bandflux_error,
    boresight_azel,
    conserve_exposure,
    daily_reference_exposure,
    equatorial_to_horizon,
    horizon_to_equatorial,
    main_lobe,
    merge_intervals,
    overlaps_intervals,
    point_visits,
    precession_matrix,
    read_time_maps,
    rotate_radec,
    simulate_downtime,
    simulate_time_maps,
    so_passbands,
    sun_moon_cut,
    tophat_passbands,
    tube_offsets,
    write_time_maps,
)
from lightcurvelynx.simulate import simulate_lightcurves  # noqa: E402
from lightcurvelynx.survey_info import SurveyInfo  # noqa: E402

START = "2026-10-14"
SO_SITE = EarthLocation(lat=SO_LATITUDE_DEG * u.deg, lon=SO_LONGITUDE_DEG * u.deg, height=5200 * u.m)


@pytest.fixture(scope="module")
def time_maps():
    """Coarse time maps for two tubes (central UHF tube c1 and outer MF tube o3)."""
    return simulate_time_maps(2.0, tubes=[0, 9])


@pytest.fixture(scope="module")
def table(time_maps):
    """A three-day table built from the coarse time maps."""
    return SimonsObservatoryObsTable.from_time_maps(time_maps, START, 3)


def test_tube_layout():
    """There are 13 tubes; each non-central tube has a neighbor at the tube spacing."""
    offsets = tube_offsets(1.75)
    assert offsets.shape == (13, 2)
    dist = np.hypot(*(offsets[:, None, :] - offsets[None, :, :]).T)
    np.fill_diagonal(dist, np.inf)
    assert np.allclose(dist.min(axis=0), 1.75)


def test_boresight_schedule():
    """The boresight sweeps each azimuth range at the scan speed, with 5% turnarounds at the ends."""
    t = np.arange(0.0, 4 * 3600.0, 0.5)
    scan = {"elevation_deg": 40.0, "az_ranges": ((30, 150), (210, 330)), "speed_deg_s": 1.0}
    for side, (low, high) in enumerate(scan["az_ranges"]):
        az, el, valid = boresight_azel(t, side, turnaround_frac=0.05, **scan)
        assert np.all(el == 40.0)
        assert np.all((az >= low) & (az <= high)) and az.min() == low and az.max() == high
        assert np.mean(valid) == pytest.approx(1.0 / 1.05, abs=1e-3)
        assert np.all(np.isin(az[~valid], [low, high]))
        moving = valid[1:] & valid[:-1]
        assert np.allclose(np.abs(np.diff(az))[moving], 0.5)
        # Turnarounds last 6 s (12 samples) per 120 s sweep.
        runs = np.diff(np.flatnonzero(np.diff(valid.astype(int)) != 0))
        assert set(runs[::2]) | set(runs[1::2]) == {12, 240}


def test_horizon_to_equatorial_matches_astropy():
    """With precession, our horizon -> ICRS conversion matches astropy (refraction off) to < 0.01 deg."""
    rng = np.random.default_rng(1)
    az, el = rng.uniform(0, 360, 50), rng.uniform(20, 80, 50)
    obstime = Time(61327.3 + rng.uniform(0, 1, 50), format="mjd")
    frame = AltAz(obstime=obstime, location=SO_SITE, pressure=0)
    icrs = SkyCoord(az=az * u.deg, alt=el * u.deg, frame=frame).icrs
    lst = obstime.sidereal_time("mean", longitude=SO_LONGITUDE_DEG).deg
    ra_date, dec_date = horizon_to_equatorial(az, el, lst)
    sep = SkyCoord(ra_date * u.deg, dec_date * u.deg).separation(icrs).deg
    assert np.max(sep) > 0.3  # Precession since J2000 moves positions by up to ~0.37 deg in 2026.
    ra, dec = rotate_radec(ra_date, dec_date, precession_matrix(61327.8).T)
    # What remains is nutation and annual aberration (each < 0.006 deg).
    assert np.max(SkyCoord(ra * u.deg, dec * u.deg).separation(icrs).deg) < 0.01


def test_precession_round_trip():
    """Precessing to the epoch of date and back returns the input, and J2000 is (almost) a no-op."""
    rng = np.random.default_rng(2)
    ra, dec = rng.uniform(0, 360, 100), rng.uniform(-80, 80, 100)
    matrix = precession_matrix(61406.0)
    ra2, dec2 = rotate_radec(*rotate_radec(ra, dec, matrix), matrix.T)
    assert np.allclose(np.cos(np.radians(ra - ra2)), 1.0) and np.allclose(dec, dec2)
    ra_j2000, dec_j2000 = rotate_radec(ra, dec, precession_matrix(51544.5))
    assert np.allclose(dec_j2000, dec, atol=1e-4)


def test_time_map_coverage(time_maps):
    """Observed declinations match the analytic range of the az 30-150 / 210-330 scan."""
    maps = time_maps["maps"]
    assert maps.shape[:3] == (len(MAP_FIELDS), 2, 2)
    fields = dict(zip(MAP_FIELDS, maps, strict=True))
    observed = fields["hits"] > 0
    assert np.array_equal(observed, np.isfinite(fields["time_s"]))
    # Rising and setting crossings of a pixel happen at different times.
    both = observed[:, 0] & observed[:, 1]
    assert np.all(fields["time_s"][:, 0][both] != fields["time_s"][:, 1][both])
    assert np.allclose(fields["exposure_s"], fields["hits"] * time_maps["dt_s"])
    assert np.all(fields["duration_s"][observed] >= fields["exposure_s"][observed])
    assert np.all(fields["n_passes"][observed] >= 1)
    time_s = fields["time_s"]
    assert np.all((time_s[observed] >= 0) & (time_s[observed] < SIDEREAL_DAY_S))
    dec = np.degrees(maps.posmap()[0])
    for obs in observed.reshape(-1, *observed.shape[2:]):
        assert -63.0 < dec[obs].min() < -55.0
        assert 17.0 < dec[obs].max() < 25.0


def test_horizon_round_trip():
    """Equatorial -> horizon -> equatorial returns the input position."""
    rng = np.random.default_rng(3)
    ra, dec, lst = rng.uniform(0, 360, 100), rng.uniform(-80, 20, 100), rng.uniform(0, 360, 100)
    az, el = equatorial_to_horizon(ra, dec, lst)
    ra2, dec2 = horizon_to_equatorial(az, el, lst)
    assert np.allclose(np.cos(np.radians(ra - ra2)), 1.0) and np.allclose(dec, dec2)


def test_map_visits_match_point_trace(time_maps):
    """Visit statistics at pixel centers agree with the independent single-point trace."""
    fields = dict(zip(MAP_FIELDS, time_maps["maps"], strict=True))
    dec, ra = np.degrees(time_maps["maps"].posmap())
    rng = np.random.default_rng(4)
    for index, tube in enumerate(time_maps["tubes"]):
        observed = np.argwhere(fields["hits"][index] > 0)
        for side, y, x in observed[rng.choice(len(observed), 10, replace=False)]:
            visits = [v for v in point_visits(ra[y, x], dec[y, x], tube, **time_maps) if v["side"] == side]
            assert len(visits) == 1
            for key in ("duration_s", "exposure_s", "n_passes", "start_s"):
                assert fields[key][index, side, y, x] == pytest.approx(visits[0][key], abs=0.01)
            dt = (
                fields["time_s"][index, side, y, x] - visits[0]["time_s"] + 0.5 * SIDEREAL_DAY_S
            ) % SIDEREAL_DAY_S
            assert dt == pytest.approx(0.5 * SIDEREAL_DAY_S, abs=0.01)


def test_table_pointing_matches_astropy(table):
    """At each row's time, the pixel center is within the tube's field of view and scan range."""
    rows = table._table.sample(300, random_state=2)
    frame = AltAz(obstime=Time(rows["time"].to_numpy(), format="mjd"), location=SO_SITE, pressure=0)
    altaz = SkyCoord(rows["ra"].to_numpy() * u.deg, rows["dec"].to_numpy() * u.deg).transform_to(frame)
    offsets = tube_offsets()[rows["tube_id"].to_numpy()]
    # Field of view radius + pixel half diagonal + nutation/aberration slop.
    assert np.all(np.abs(altaz.alt.deg - 40.0 - offsets[:, 1]) < 0.625 + np.sqrt(2) + 0.02)
    east = rows["side"].to_numpy() == "east"
    assert np.all((altaz.az.deg[east] > 20) & (altaz.az.deg[east] < 160))
    assert np.all((altaz.az.deg[~east] > 200) & (altaz.az.deg[~east] < 340))


def test_table_alternates_rising_and_setting(time_maps, table):
    """Each (tube, pixel) is seen once per sidereal day, alternating rising and setting crossings."""
    t0 = Time(START, scale="utc").mjd
    times = table["time"]
    assert times.min() >= t0 and times.max() < t0 + 3
    assert np.all(np.diff(times) >= 0)
    df = table._table
    assert np.all((df["side"] == "east") == (df["transit_day"] % 2 == 0))
    # Where a tube sees both crossings, revisits are one sidereal day plus or minus the
    # rising-to-setting offset (< 12 h); same-side revisits are exactly two sidereal days.
    sidereal_day = SIDEREAL_DAY_S / 86400.0
    for _, group in df.groupby(["tube_id", "filter", "pixel_index"], observed=True):
        sides = group["side"].to_numpy()
        if len(set(sides)) < 2:
            continue
        assert np.all(np.abs(np.diff(group["time"]) - sidereal_day) < 0.5)
        assert np.all(np.diff(group["transit_day"]) == 1) and np.all(sides[1:] != sides[:-1])
    for _, group in df.groupby(["tube_id", "filter", "pixel_index", "side"], observed=True):
        assert np.allclose(np.diff(group["time"]), 2.0 * sidereal_day)
    # Rising (east) crossings precede the pixel's meridian transit; setting crossings follow it.
    lst = Time(df["time"].to_numpy(), format="mjd").sidereal_time("mean", longitude=SO_LONGITUDE_DEG).deg
    hour_angle = (lst - df["ra"].to_numpy() + 180.0) % 360.0 - 180.0
    assert np.all((hour_angle < 0) == (df["side"] == "east"))
    west_first = SimonsObservatoryObsTable.from_time_maps(time_maps, START, 3, first_side=1)._table
    assert np.all((west_first["side"] == "west") == (west_first["transit_day"] % 2 == 0))
    assert set(table.filters) == {"f090", "f150", "f220", "f280"}
    assert np.all((df["time_start"] <= df["time"]) & (df["time"] <= df["time_end"]))
    assert np.allclose((df["time_end"] - df["time_start"]) * 86400.0, df["duration_s"])
    assert set(df["side"]) <= set(SIDE_NAMES)


def test_range_search(table):
    """Range search returns exactly the rows of the containing pixel, respecting time bounds."""
    row = table._table.iloc[len(table) // 2]
    ra, dec = row["ra"] + 0.3, row["dec"] - 0.3
    inds = table.range_search(ra, dec)
    assert len(inds) > 0
    assert np.all(table._table["pixel_index"].to_numpy()[inds] == row["pixel_index"])
    expected = np.flatnonzero(table._table["pixel_index"].to_numpy() == row["pixel_index"])
    assert np.array_equal(inds, expected)

    t_min = row["time"] - 0.1
    bounded = table.range_search(ra, dec, t_min=t_min, t_max=t_min + 0.2)
    assert row.name in bounded and np.all(np.abs(table["time"][bounded] - row["time"]) < 0.2)

    # Array queries; a dec outside the map matches nothing.
    results = table.range_search(np.array([ra, 10.0]), np.array([dec, 80.0]))
    assert np.array_equal(results[0], inds) and len(results[1]) == 0

    # Copies and row filters keep the pixel lookup consistent.
    filtered = table.copy().filter_rows(table["tube_id"] == 0)
    assert isinstance(filtered, SimonsObservatoryObsTable)
    assert np.all(filtered["tube_id"][filtered.range_search(ra, dec)] == 0)


def test_time_map_io(time_maps, tmp_path):
    """Time maps round-trip through FITS."""
    fname = tmp_path / "so_time_maps.fits"
    write_time_maps(fname, time_maps)
    loaded = read_time_maps(fname)
    assert np.allclose(loaded["maps"], time_maps["maps"], equal_nan=True)
    assert np.array_equal(loaded["tubes"], time_maps["tubes"])
    for key in ("res_deg", "dt_s", "epoch_mjd", "turnaround_frac", "elevation_deg", "az_ranges"):
        assert np.allclose(loaded[key], time_maps[key])


def test_simulate_lightcurves(time_maps, passbands_dir):
    """The standard simulation path works on an SO table with per-row flux errors."""
    table = SimonsObservatoryObsTable.from_time_maps(
        time_maps, START, 3, tubes=[0], band_names={"UHF": ("g",)}
    )
    table.add_column("bandflux_error", 10.0 / np.sqrt(table["exposure_s"]))
    row = table._table.iloc[0]
    passbands = PassbandGroup.from_preset("LSST", table_dir=passbands_dir, filters=["g"])
    survey = SurveyInfo(table, passbands=passbands, noise_model=GivenNoiseModel())
    source = ConstantSEDModel(
        brightness=100.0, t0=row["time"], ra=row["ra"], dec=row["dec"], redshift=0.0, node_label="source"
    )
    results = simulate_lightcurves(
        source, 1, survey, obstable_save_cols=["tube_id", "exposure_s", "pixel_index"], progress_bar=False
    )
    lightcurve = results["lightcurve"].iloc[0]
    # The first crossing is at the start time; 3 days hold 4 crossings, alternating rising/setting.
    assert len(lightcurve) == np.sum(table["pixel_index"] == row["pixel_index"]) >= 2
    assert np.all(lightcurve["pixel_index"] == row["pixel_index"])
    assert np.allclose(lightcurve["fluxerr"], 10.0 / np.sqrt(lightcurve["exposure_s"]))


def test_intervals():
    """Intervals merge, and spans overlapping them are flagged."""
    merged = merge_intervals([[3, 4], [0, 1], [0.5, 2], [1.5, 1.8], [5, 6]])
    assert np.array_equal(merged, [[0, 2], [3, 4], [5, 6]])
    flags = overlaps_intervals(
        np.array([2.1, 2.5, 3.9, -1.0, 6.0]), np.array([2.9, 3.1, 4.5, -0.5, 7.0]), merged
    )
    assert np.array_equal(flags, [False, True, True, False, False])


def test_simulate_downtime():
    """Downtime meets the uptime target, with week-long winter outages and daytime interruptions."""
    downtime = simulate_downtime("2027-01-01", 365.25, seed=3)
    lengths = np.diff(downtime, axis=1)[:, 0]
    assert 1.0 - lengths.sum() / 365.25 == pytest.approx(0.65, abs=1e-6)
    assert np.all(downtime[1:, 0] > downtime[:-1, 1])
    long = lengths > 2.0
    assert lengths[long].sum() / 365.25 == pytest.approx(0.35 * 0.75, abs=0.05)
    winter = np.isin(Time(downtime[long, 0], format="mjd").ymdhms.month, (6, 7, 8, 9))
    assert winter.mean() > 4 / 12
    local_hour = ((downtime[~long, 0] + SO_LONGITUDE_DEG / 360.0) % 1.0) * 24.0
    assert np.mean((local_hour >= 8.0) & (local_hour <= 20.0)) > 0.9


def test_table_downtime(time_maps, table):
    """Visits overlapping downtime are removed and all others are kept."""
    start = Time(START).mjd
    downtime = merge_intervals([[start + 0.5, start + 1.5], [start + 2.2, start + 2.3]])
    reduced = SimonsObservatoryObsTable.from_time_maps(time_maps, START, 3, downtime=downtime)._table
    df = table._table
    hit = overlaps_intervals(df["time_start"].to_numpy(), df["time_end"].to_numpy(), downtime)
    assert 0 < hit.sum() < len(df) and len(reduced) == np.sum(~hit)
    assert not np.any(
        overlaps_intervals(reduced["time_start"].to_numpy(), reduced["time_end"].to_numpy(), downtime)
    )


def test_sun_moon_cut(time_maps, table):
    """The cut matches astropy separations and the table keeps nothing within 30 deg."""
    full = SimonsObservatoryObsTable.from_time_maps(time_maps, START, 3, avoid_deg=None)._table
    df = table._table
    assert len(full) > len(df)
    for data in (full.iloc[::500], df.iloc[::500]):
        t = Time(data["time"].to_numpy(), format="mjd")
        pos = SkyCoord(data["ra"].to_numpy() * u.deg, data["dec"].to_numpy() * u.deg)
        # The GCRS body direction (barycentric ICRS would be wrong for the sun and moon).
        bodies = (get_body(b, t, SO_SITE) for b in ("sun", "moon"))
        sep = np.minimum(*(pos.separation(SkyCoord(g.ra, g.dec)).deg for g in bodies))
        flagged = sun_moon_cut(data["time"], data["ra"], data["dec"])
        assert np.all(flagged[sep < 29.5]) and not np.any(flagged[sep > 30.5])
    assert not np.any(sun_moon_cut(df["time"], df["ra"], df["dec"]))


def test_conserve_exposure(time_maps, table):
    """Sun and moon avoidance keeps each tube's daily exposure unchanged."""
    full = SimonsObservatoryObsTable.from_time_maps(time_maps, START, 3, avoid_deg=None)._table
    df = table._table
    daily = lambda d: d.groupby([d["tube_id"], np.floor(d["time"])])["exposure_s"].sum()  # noqa: E731
    np.testing.assert_allclose(daily(df), daily(full).loc[daily(df).index], rtol=1e-5)
    assert np.all(df["exposure_scale"] >= 1.0) and np.any(df["exposure_scale"] > 1.0)
    small = full.iloc[:4].assign(tube_id=0, time=np.floor(full["time"].iloc[0]) + 0.1, exposure_s=1.0)
    out = conserve_exposure(small, np.array([True, False, False, False]))
    np.testing.assert_allclose(out["exposure_s"], 4.0 / 3.0)


def test_bands_per_tube(table):
    """Each visit gives one identical row per band of its tube."""
    df = table._table
    keys = ["tube_id", "pixel_index", "time"]
    for tube, bands in ((0, {"f220", "f280"}), (9, {"f090", "f150"})):
        rows = df[df["tube_id"] == tube]
        assert set(rows["filter"]) == bands
        per_visit = rows.groupby(keys, observed=True)["filter"].agg(lambda f: frozenset(f))
        assert np.all(per_visit == frozenset(bands))
        first, second = (rows[rows["filter"] == b].sort_values(keys) for b in sorted(bands))
        assert np.array_equal(first["exposure_s"].to_numpy(), second["exposure_s"].to_numpy())


def test_main_lobe():
    """Only the contiguous region around the peak above the threshold is kept."""
    response = np.array([0.0, 0.2, 0.0, 0.0, 0.5, 1.0, 0.6, 0.005, 0.3])
    assert np.array_equal(main_lobe(response), [0, 0, 0, 0, 1, 1, 1, 0, 0])


def test_so_passbands(tmp_path):
    """Bandpasses on a GHz grid load as passbands without the low-frequency artifact."""
    freq = np.linspace(0.0, 200.0, 201)
    band = lambda center: np.exp(-0.5 * ((freq - center) / 10.0) ** 2)  # noqa: E731
    artifact = 0.2 * (freq < 10.0)
    fname = tmp_path / "bands.csv"
    df = {"frequency": freq, "median_f090": band(90.0) + artifact - 0.01, "median_f150": band(150.0)}
    pd.DataFrame(df).to_csv(fname, index=False)
    assert set(so_passbands(fname, tophats={}).filters) == {"f090", "f150"}
    group = so_passbands(fname)
    assert set(group.filters) == {"f030", "f040", "f090", "f150"}
    for name, (center, width) in SO_LF_TOPHATS.items():
        ghz = 2.99792458e18 / group[name].waves / 1e9
        np.testing.assert_allclose([ghz.min(), ghz.max()], [center - width / 2, center + width / 2])
    for name, center in (("f090", 90.0), ("f150", 150.0)):
        ghz = 2.99792458e18 / group[name].waves / 1e9
        assert np.all(np.diff(group[name].waves) > 0)
        assert 50.0 < ghz.min() < center < ghz.max() < 200.0


def test_so_passband_bolometer_weighting():
    """Band fluxes weight f_nu uniformly in frequency (bolometer), not by 1/frequency."""
    for passband in tophat_passbands({"wide": (40.0, 40.0)}):
        nu = 2.99792458e18 / passband.waves / 1e9
        flux = passband.fluxes_to_bandflux(nu[None, :] ** 2)
        # Flat weighting over 20-60 GHz: mean of nu^2 = (60^3 - 20^3) / (3 * 40).
        np.testing.assert_allclose(flux, (60.0**3 - 20.0**3) / 120.0, rtol=1e-3)


def test_bandflux_error(time_maps, table):
    """A day's visits of a band, combined over all its tubes, reach the daily depth."""
    reference = daily_reference_exposure(time_maps)
    assert reference["f090"] == pytest.approx(8 * reference["f220"] / 4)
    assert 2.0 < reference["f220"] / 4 < 6.0  # per-tube daily exposure of a few seconds
    table = table.copy()
    add_bandflux_error(table, time_maps)
    df = table._table
    for band in ("f090", "f220"):
        rows = df[df["filter"] == band]
        # A visit with the full daily exposure would reach the daily depth (nJy); noise ~ 1/sqrt(t).
        depth = rows["bandflux_error"] * np.sqrt(rows["exposure_s"] / reference[band])
        np.testing.assert_allclose(depth, SO_DAILY_DEPTH_MJY[band] * 1e6, rtol=1e-5)


def test_pixel_subset(time_maps, table):
    """A pixel subset keeps exactly those rows, with the full-sky sun/moon rescaling."""
    df = table._table
    pixels = np.unique(df["pixel_index"])[::7]
    sub = SimonsObservatoryObsTable.from_time_maps(
        time_maps, START, 3, pixels=pixels, n_scale_pixels=100000
    )._table
    full = df[df["pixel_index"].isin(pixels)]
    keys = ["tube_id", "filter", "pixel_index", "time"]
    merged = full.merge(sub, on=keys, suffixes=("_full", "_sub"), validate="one_to_one")
    assert len(merged) == len(full) == len(sub)
    np.testing.assert_allclose(merged["exposure_s_full"], merged["exposure_s_sub"], rtol=1e-5)
