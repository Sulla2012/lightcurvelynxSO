import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

pytest.importorskip("pixell")

from lightcurvelynx.astro_utils.passbands import PassbandGroup  # noqa: E402
from lightcurvelynx.models.basic_models import ConstantSEDModel  # noqa: E402
from lightcurvelynx.noise_models.base_noise_models import GivenNoiseModel  # noqa: E402
from lightcurvelynx.obstable.simons_observatory_obstable import (  # noqa: E402
    MAP_FIELDS,
    SIDE_NAMES,
    SIDEREAL_DAY_S,
    SO_LATITUDE_DEG,
    SO_LONGITUDE_DEG,
    SimonsObservatoryObsTable,
    boresight_azel,
    equatorial_to_horizon,
    horizon_to_equatorial,
    point_visits,
    read_time_maps,
    simulate_time_maps,
    tube_offsets,
    write_time_maps,
)
from lightcurvelynx.simulate import simulate_lightcurves  # noqa: E402
from lightcurvelynx.survey_info import SurveyInfo  # noqa: E402

START = "2026-10-14"
SO_SITE = EarthLocation(lat=SO_LATITUDE_DEG * u.deg, lon=SO_LONGITUDE_DEG * u.deg, height=5200 * u.m)


@pytest.fixture(scope="module")
def time_maps():
    """Coarse time maps for two tubes (center and an outer UHF tube)."""
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
    """The boresight scans each azimuth range at the scan speed and alternates by block."""
    t = np.arange(0.0, 4 * 3600.0, 0.5)
    az, el, side = boresight_azel(
        t, elevation_deg=40.0, az_ranges=((30, 150), (210, 330)), speed_deg_s=1.0, block_hours=1.0
    )
    assert np.all(el == 40.0)
    assert np.array_equal(side, (t // 3600).astype(int) % 2)
    assert np.all((az[side == 0] >= 30) & (az[side == 0] <= 150))
    assert np.all((az[side == 1] >= 210) & (az[side == 1] <= 330))
    steps = np.abs(np.diff(az))[np.diff(side) == 0]
    assert np.allclose(steps, 0.5)


def test_horizon_to_equatorial_matches_astropy():
    """Our spherical-trig conversion agrees with astropy (refraction off) to < 0.05 deg."""
    rng = np.random.default_rng(1)
    az, el = rng.uniform(0, 360, 50), rng.uniform(20, 80, 50)
    obstime = Time(61327.3 + rng.uniform(0, 1, 50), format="mjd")
    frame = AltAz(obstime=obstime, location=SO_SITE, pressure=0)
    icrs = SkyCoord(az=az * u.deg, alt=el * u.deg, frame=frame).icrs
    lst = obstime.sidereal_time("apparent", longitude=SO_LONGITUDE_DEG).deg
    ra, dec = horizon_to_equatorial(az, el, lst)
    sep = SkyCoord(ra * u.deg, dec * u.deg).separation(icrs).deg
    # Precession between J2000 and the epoch of date is ~0.36 deg in 2026.
    assert np.max(sep) < 0.5


def test_time_map_coverage(time_maps):
    """Observed declinations match the analytic range of the az 30-150 / 210-330 scan."""
    maps = time_maps["maps"]
    assert maps.shape[:2] == (len(MAP_FIELDS), 2)
    fields = dict(zip(MAP_FIELDS, maps, strict=True))
    observed = fields["hits"] > 0
    assert np.array_equal(observed, np.isfinite(fields["time_s"]))
    assert np.all(fields["side"][observed] >= 0) and np.all(fields["side"][~observed] == -1)
    assert np.allclose(fields["exposure_s"], fields["hits"] * time_maps["dt_s"])
    assert np.all(fields["duration_s"][observed] >= fields["exposure_s"][observed])
    assert np.all(fields["n_passes"][observed] >= 1)
    time_s = fields["time_s"]
    assert np.all((time_s[observed] >= 0) & (time_s[observed] < SIDEREAL_DAY_S))
    dec = np.degrees(maps.posmap()[0])
    for obs in observed:
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
        for y, x in observed[rng.choice(len(observed), 10, replace=False)]:
            side = fields["side"][index, y, x]
            visits = [v for v in point_visits(ra[y, x], dec[y, x], tube, **time_maps) if v["side"] == side]
            visit = max(visits, key=lambda v: v["exposure_s"])
            for key in ("duration_s", "exposure_s", "n_passes", "start_s"):
                assert fields[key][index, y, x] == pytest.approx(visit[key], abs=0.01)
            dt = (fields["time_s"][index, y, x] - visit["time_s"] + 0.5 * SIDEREAL_DAY_S) % SIDEREAL_DAY_S
            assert dt == pytest.approx(0.5 * SIDEREAL_DAY_S, abs=0.01)


def test_table_pointing_matches_astropy(table):
    """At each row's time, the pixel center is within the tube's field of view and scan range."""
    rows = table._table.sample(300, random_state=2)
    frame = AltAz(obstime=Time(rows["time"].to_numpy(), format="mjd"), location=SO_SITE, pressure=0)
    altaz = SkyCoord(rows["ra"].to_numpy() * u.deg, rows["dec"].to_numpy() * u.deg).transform_to(frame)
    offsets = tube_offsets()[rows["tube_id"].to_numpy()]
    # Field of view radius + pixel half diagonal + precession slop.
    assert np.all(np.abs(altaz.alt.deg - 40.0 - offsets[:, 1]) < 0.625 + np.sqrt(2) + 0.4)
    east = rows["side"].to_numpy() == "east"
    assert np.all((altaz.az.deg[east] > 20) & (altaz.az.deg[east] < 160))
    assert np.all((altaz.az.deg[~east] > 200) & (altaz.az.deg[~east] < 340))


def test_table_one_observation_per_sidereal_day(table):
    """Each (tube, pixel) is observed once per sidereal day, repeating exactly."""
    t0 = Time(START, scale="utc").mjd
    times = table["time"]
    assert times.min() >= t0 and times.max() < t0 + 3
    assert np.all(np.diff(times) >= 0)
    for _, group in table._table.groupby(["tube_id", "pixel_index"]):
        assert np.allclose(np.diff(group["time"]), SIDEREAL_DAY_S / 86400.0)
    assert set(table.filters) == {"MF", "UHF"}
    df = table._table
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
    for key in ("res_deg", "dt_s", "block_hours", "elevation_deg", "az_ranges"):
        assert np.allclose(loaded[key], time_maps[key])


def test_simulate_lightcurves(time_maps, passbands_dir):
    """The standard simulation path works on an SO table with per-row flux errors."""
    table = SimonsObservatoryObsTable.from_time_maps(time_maps, START, 3, tubes=[0], filter_names=["g"] * 13)
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
    # The first crossing is at the start time, so 3 solar days hold 4 (sidereal) crossings.
    assert len(lightcurve) == np.sum(table["pixel_index"] == row["pixel_index"]) == 4
    assert np.all(lightcurve["pixel_index"] == row["pixel_index"])
    assert np.allclose(lightcurve["fluxerr"], 10.0 / np.sqrt(lightcurve["exposure_s"]))
