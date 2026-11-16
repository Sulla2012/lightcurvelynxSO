Survey Data
========================================================================================

In order to generate a simulation, the simulator needs information on whether the telescope is pointing and the noise characteristics at the time. As you can imagine, the available data and its format varies significantly from survey to survey (and even between data releases of the same survey). `LightCurveLynx` uses an object-based model to encapsulate the survey information and provide a common interface.

`LightCurveLynx` includes built-in support for a range of instruments and surveys. The characteristics of these
instruments and surveys are generally derived from published values and data releases (see comments in the code
for specific details). However, **users should always carefully validate the science outputs for their use case.**


SurveyInfo
-------------------------------------------------------------------------------

The ``SurveyInfo`` class is a data wrapper that holds all of the information needed to simulate a survey, including the pointing information (``ObsTable``), passbands (``PassbandGroup``), and noise model for each survey.


ObsTable
-------------------------------------------------------------------------------

The ``ObsTable`` parent class is top level interface for storing and manipulating survey data. At a minimum, it requires the pointing information (RA, dec, and time for image). In addition it can store a wide range of per-visit information (e.g. airmass, exposure time, filter, etc.) and or survey-specific constants (e.g. pixel scale and saturation magnitudes).

In addition to a common API, ``ObsTable`` includes common processing that is used across many different types of surveys, including:

    * Column name mapping (for users trying to load data from custom tables),
    * Filtering rows by index or mask (``filter_rows``),
    * Addition and querying of pixel-level detector footprints (e.g., ``set_detector_footprint``),
    * Construction and use of spatial data structures for fast search including built-in cone search (``range_search``)
    * Application of saturation thresholds,
    * Plotting the survey's footprint on the sky (``plot_footprint``),
    * Construction of a `multi-order coverage map (MOC) <https://www.ivoa.net/documents/MOC/>`__ of the survey (``build_moc``), 
    * Resampling the table (``make_resampled_table``), and
    * Writing the observation information to disk.

Users do not work with the ``ObsTable`` class directly, but instead use one of the child classes that are designed for specific surveys. We discuss some of these below.


LSST Data (LSSTObsTable)
-------------------------------------------------------------------------------

The ``LSSTObsTable`` class stores actual pointing and noise information from the Rubin Observatory data releases. It includes functions to read from the currently defined data formats, including:

    * `DP1 <https://sdm-schemas.lsst.io/dp1.html#CcdVisit>`__ and `DP2+ <https://sdm-schemas.lsst.io/lsstcam.html#CcdVisit>`__ as CCDVisit tables, using the ``from_ccdvisit_table`` method, and
    * `Science Validation <https://survey-strategy.lsst.io/progress/sv_status/sv_20250930.html>`__ as the released DB file, using the ``from_sv_visits_table`` method.

The survey-specific constants are set to `published values <https://lsstcam.lsst.io/index.html>`__.

See the (:doc:`Rubin CCDVisit Notebook <notebooks/pre_executed/rubin_ccdvisit>`) for an example of how to load Rubin data using the ``LSSTObsTable`` class.


OpSim Data (OpSim)
-------------------------------------------------------------------------------

The ``OpSim`` class stores pointing and noise information from the Rubin Observatory's simulated survey data (OpSim).  The survey-specific constants are set to `the values defined in the simulator <https://smtn-002.lsst.io/v/OPSIM-1171/index.html>`__.

See the (:doc:`Rubin OpSim Notebook <notebooks/opsim_notebook>`) for an example of how to load simulated Rubin data using the ``OpSim`` class. This notebook also provides an overview of the functionality of the ``ObsTable`` class.


Roman Simulated Data (RomanObsTable)
-------------------------------------------------------------------------------
The ``RomanObsTable`` class stores pointing and noise information from the `Nancy Grace Roman Space Telescope <https://roman.gsfc.nasa.gov>`__'s simulated survey data. The survey-specific constants are set to the values defined in the `Roman documentation <https://roman-docs.stsci.edu/>`__ and the `Roman GitHub repository <https://github.com/RomanSpaceTelescope>`__. 

Special preprocessing is needed to handle the spectra observations (spectra simulation within LightCurveLynx is in early testing). Please contact the LightCurveLynx team if you need help with this.


SkyMapper Data (SkyMapperObsTable)
-------------------------------------------------------------------------------
The ``SkyMapperObsTable`` class stores pointing and noise information from the `SkyMapper Southern Survey <https://skymapper.anu.edu.au/>`__.

See the (:doc:`SkyMapper Example Notebook <notebooks/pre_executed/skymapper_example>`) for an example of how to load SkyMapper data using the ``SkyMapperObsTable`` class.


ZTF Data (ZTFObsTable)
-------------------------------------------------------------------------------
The ``ZTFObsTable`` class stores pointing and noise information from the `Zwicky Transient Facility (ZTF) <https://www.ztf.caltech.edu/>`__ visit information.


Argus Data (ArgusObsTable)
-------------------------------------------------------------------------------
The ``ArgusObsTable`` class stores pointing and noise information from simulations of the upcoming `Argus Array <https://www.argusarray.org/>`__. This class is still being validated and may change. Please contact the LightCurveLynx team if you want to use this class or have suggestions for improvement.

Simons Observatory (SimonsObservatoryObsTable)
-------------------------------------------------------------------------------
The Simons Observatory (SO) scans continuously in azimuth at a fixed elevation instead of taking
discrete pointings. ``simulate_time_maps`` simulates one sidereal day of an idealized scan with
`pixell <https://github.com/simonsobs/pixell>`__. At every time sample it finds the map pixels whose
centers lie inside each of the 13 optics tubes' fields of view. From those hits it records, per tube
and pixel, the visit's mean time, start, duration (first to last in-FoV sample), exposure (total
seconds in the FoV) and the number of azimuth sweeps over the pixel. These values are exact for the
pixel center, up to the sample interval; ``point_visits`` traces a single position independently and
is used to validate them. ``SimonsObservatoryObsTable.from_time_maps`` turns the maps into a standard
``ObsTable`` with one row per (tube, observed pixel, day). It has the columns ``time`` (mean, MJD),
``time_start``, ``time_end``, ``duration_s``, ``exposure_s``, ``exposure_scale``, ``n_passes``,
``hits``, ``side`` (``east``/``west``), ``transit_day``, ``tube_id`` and ``pixel_index``. A source matches exactly the
rows of the map pixel that contains it, so the standard ``simulate_lightcurves`` path is used.

The idealized scan model:

* The site is at 22d57m31s S, 67d47m15s W, and the boresight is fixed at 40 deg elevation.
* Azimuth follows a triangle wave at 1 deg/s with 120 deg throws. Each sweep is followed by a
  turnaround lasting 5% of a sweep, during which no data are taken.
* A patch is crossed once while rising, in the east range (az 30-150 deg), and once while
  setting, in the west range (az 210-330 deg). SO switches daily between the two, so successive
  transits of a patch alternate between the rising and the setting crossing. The pattern is
  fixed in local sidereal time and repeats every sidereal day, so only one day is simulated.
* Each pixel is observed at most once per tube per day.
* There are 13 tubes in a hex pattern: 1.25 deg fields of view, 1.75 deg spacing, and half the
  outer ring missing. ``SO_TUBE_NAMES`` and ``SO_TUBE_TYPES`` give each tube index its name and
  type: c1 (center), i1-i6 (inner ring) and o1-o6 (outer ring), with 8 MF, 4 UHF and 1 LF tubes
  (UHF: c1, i2, i5, o1; LF: o6; MF: the rest).
* Tubes are dichroic, so every visit observes two bands at once and gives one row per band:
  LF -> ``f030``, ``f040``; MF -> ``f090``, ``f150``; UHF -> ``f220``, ``f280``
  (``SO_TUBE_BANDS``). The ``filter`` column holds the band, which must match a passband name in
  the ``PassbandGroup``; pass ``band_names`` to use other names. The two rows of a visit are
  identical apart from ``filter``.

.. important::

   **Assumption: tube numbering.** Tube index 0 is c1, 1-6 are i1-i6 and 7-12 are o1-o6, in the
   rotational order of ``tube_offsets``. The real orientation of the i/o numbering on the focal
   plane has not been checked. This only affects which few-degree offset each band's tubes have,
   not the band coverage as a whole.
* Pointings are precessed from the mean equator and equinox of ``epoch_mjd`` (default 2027-01-01)
  to ICRS. Choose an epoch near the middle of the survey; precession moves the sky by about
  0.014 deg per year. Refraction, nutation and aberration (all below 0.01 deg) are ignored.
* Visits are dropped when the pixel is within 30 deg of the sun or moon (``avoid_deg``; ``None``
  disables the cut). SO scans another patch instead, so those visits are lost. The topocentric
  sun and moon positions come from astropy on a 10 minute grid.
* Weather and maintenance are modeled statistically as downtime (see below).

.. important::

   **Assumption: sun and moon avoidance does not reduce observing time.** SO's scheduler moves
   scans away from the sun and moon instead of idling, so in reality the total observing time is
   unchanged and only *where* the telescope scans changes. The table does not model where the
   displaced scans go. Instead, on each UTC day the exposure removed by the cut is redistributed
   uniformly over the same tube's remaining visits: their ``exposure_s`` is multiplied by
   ``exposure_scale`` = 1 / (1 - f), where f is the fraction of that tube's exposure cut that day.
   Daily and yearly exposure therefore match a table without the cut. Over 2027 the cut removes
   about 14% of the exposure (4-29% on a given day), so the remaining sky is on average about
   8% deeper (1/sqrt(1 - 0.14)). The extra depth is spread evenly rather than placed on the
   patches the real scheduler would pick. Divide ``exposure_s`` by ``exposure_scale`` to recover
   the raw scan exposure. Pass ``avoid_deg=None`` to disable both the cut and the rescaling.

Because of the azimuth throws, coverage spans roughly Dec -59 to +21 deg, plus the tube offsets.

Downtime
^^^^^^^^
``simulate_downtime`` draws random downtime intervals, which ``from_time_maps`` uses to drop any
visit that overlaps them. The total downtime is ``(1 - uptime) * n_days`` (default uptime 0.65), split
into two kinds:

* Long outages (default 75% of the downtime) lasting 3-14 days, placed preferentially in the
  austral winter (June-September), standing in for weather and maintenance.
* Short interruptions (the remaining 25%) lasting 1-6 hours, starting in local daytime (8-20 h
  mean solar time).

The intervals are random draws that reproduce the right amount and seasonality of lost time, not
a simulation of the actual weather. The sun and moon cut is separate and geometric, so it comes
on top of the downtime.

Passbands
^^^^^^^^^
``so_passbands`` loads measured bandpasses from CSV files with a ``frequency`` column (GHz) and
``median_<band>`` columns (e.g. ``LAT_MF_median_bands.csv`` with ``median_f090`` and
``median_f150``) into a ``PassbandGroup`` whose names match the ``filter`` column:

* The detectors do not respond below ~40 GHz, so measured response below ``min_ghz`` (40 GHz)
  is a measurement artifact (up to ~20% of the peak at a few GHz) and is set to zero, as are
  negative values. Only the main lobe of each band is then kept: the contiguous region around
  the peak where the response exceeds 1% of the peak (``threshold``). Left in, the artifact
  would span an enormous wavelength range and dominate the passband.
* Frequencies are converted to wavelengths in Angstroms, and the native grid is kept.
* **Assumption: LF bands.** There are no measured LF bandpasses, so ``f030`` and ``f040`` are
  unit-response top-hats with centers 27.9 and 40.2 GHz and full widths 4.7 and 16.9 GHz
  (``SO_LF_TOPHATS``).
* **Bolometer weighting.** SO detectors are bolometers, so a band flux is the flux density
  weighted uniformly in frequency, F_b = int f_nu S dnu / int S dnu. lightcurvelynx's
  ``Passband`` uses the photon-counting convention (weight S / nu), so ``so_passbands`` stores
  S * nu as the transmission table, which gives exactly the flat weighting. The stored
  ``transmission_table`` is therefore not the measured response.

.. code-block:: python

    passbands = so_passbands(["LAT_MF_median_bands.csv", "LAT_UHF_median_bands.csv"])

Noise
^^^^^
SO detectors are bolometers and the noise is white, so a visit's flux error follows from the
expected point-source depth of one full day's map in its band (``SO_DAILY_DEPTH_MJY``, 1 sigma):

========== ===== ===== ===== ===== ===== =====
Band (GHz) 30    40    90    150   220   280
========== ===== ===== ===== ===== ===== =====
Depth, mJy 37    25    6.9   8.3   26    49
========== ===== ===== ===== ===== ===== =====

A day's map of a band combines all tubes observing it (1 LF, 8 MF or 4 UHF tubes).
``daily_reference_exposure`` gives the typical exposure of that map per pixel: each tube
contributes ~3.8 s per day (the median over pixels of the mean of its rising and setting crossing),
so the reference is ~3.7 s for LF, ~30 s for MF and ~15 s for UHF bands. ``add_bandflux_error``
then sets, in nJy,

    bandflux_error = depth * sqrt(T_day / exposure_s)

for use with ``GivenNoiseModel``. A single MF visit therefore has about sqrt(8) times the daily
depth (e.g. ~19 mJy at f090). Combining a pixel's visits over one transit day reproduces the daily
depths to within 1-3%. With the sun and moon cut on, ``exposure_s`` includes ``exposure_scale``,
so the sky that is observed is on average ~8% deeper than these numbers: the time not spent
near the sun and moon goes to the rest of the sky.

.. code-block:: python

    from lightcurvelynx.obstable.simons_observatory_obstable import (
        SimonsObservatoryObsTable,
        add_bandflux_error,
        simulate_downtime,
        simulate_time_maps,
        so_passbands,
        write_time_maps,
    )

    time_maps = simulate_time_maps(res_deg=0.5)  # all 13 tubes
    write_time_maps("so_time_maps.fits", time_maps)  # reload with read_time_maps
    downtime = simulate_downtime("2026-01-01", 30, seed=1)
    table = SimonsObservatoryObsTable.from_time_maps(
        time_maps, "2026-01-01", n_days=30, tubes=[0], downtime=downtime
    )
    add_bandflux_error(table, time_maps)
    survey = SurveyInfo(table, passbands=passbands, noise_model=GivenNoiseModel())

For tubes on the same band, visits a few minutes apart are effectively one telescope visit. Each
tube keeps its own crossing (rising or setting), so a source can be visited twice per day by
different tubes. Tables are large: all 13 tubes at 0.5 deg have about 2.4M rows per day (two bands per
visit) and need
about 0.3 kB per row, so build one table per band (or per tube) for long surveys. At 0.5 deg, the
pixelization error for arbitrary source positions is about 30 s (median) in visit time and up to one
sweep (about 2 minutes) in duration, which is negligible for transients.
``analysis/visualize_simons_observatory_obstable.py`` (use ``--debug`` for a quick run)
simulates the maps, builds the table, and writes an HTML report of diagnostic plots to
``analysis/so_report/``.


All Sky Simulations
-------------------------------------------------------------------------------
The ``LocationFreeObsTable`` class allows users to simulate objects regardless of where they are in the sky. It only requires time and filter columns. It does not perform any spatial filtering. All objects are assumed to be viewed at each time. See the (:doc:`All-Sky ObsTable Notebook <notebooks/location_free_obstable>`) for an example of how to use the ``LocationFreeObsTable`` class.
