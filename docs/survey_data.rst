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
``time_start``, ``time_end``, ``duration_s``, ``exposure_s``, ``n_passes``, ``hits``, ``side``
(``east``/``west``), ``tube_id`` and ``pixel_index``. A source matches exactly the rows of the map
pixel that contains it, so the standard ``simulate_lightcurves`` path is used.

The idealized scan model:

* The site is at 22d57m31s S, 67d47m15s W, and the boresight is fixed at 40 deg elevation.
* Azimuth sweeps at 1 deg/s, with instantaneous turnarounds.
* The scan alternates between an east range (az 30-150 deg) and a west range (az 210-330 deg) in
  blocks of ``block_hours``, phased in LST. The pattern therefore repeats every sidereal day, and
  only one day needs to be simulated.
* Each pixel is observed at most once per tube per day. If a pixel is crossed in both ranges,
  only the crossing with more hits is kept.
* There are 13 tubes in a hex pattern: 1.25 deg fields of view, 1.75 deg spacing, and half the
  outer ring missing. Filter labels are generic (8 ``MF``, 4 ``UHF``, 1 ``LF``); pass
  ``filter_names`` once the real mapping is known.
* Refraction, precession, sun/moon avoidance, weather and turnaround overheads are ignored.

Because of the azimuth throws, coverage spans roughly Dec -59 to +21 deg, plus the tube offsets.

.. code-block:: python

    from lightcurvelynx.obstable.simons_observatory_obstable import (
        SimonsObservatoryObsTable,
        simulate_time_maps,
        write_time_maps,
    )

    time_maps = simulate_time_maps(res_deg=0.5, block_hours=1.0)  # all 13 tubes
    write_time_maps("so_time_maps.fits", time_maps)  # reload with read_time_maps
    table = SimonsObservatoryObsTable.from_time_maps(time_maps, "2026-01-01", n_days=30, tubes=[0])
    table.add_column("bandflux_error", sigma_1s / np.sqrt(table["exposure_s"]))
    survey = SurveyInfo(table, passbands=passbands, noise_model=GivenNoiseModel())

For tubes on the same band, visits a few minutes apart are effectively one telescope visit. Each
tube keeps its own crossing (rising or setting), so a source can be visited twice per day by
different tubes. Tables are large: all 13 tubes at 0.5 deg have about 1.2M rows per day and need
about 0.3 kB per row, so build one table per band (or per tube) for long surveys. At 0.5 deg, the
pixelization error for arbitrary source positions is about 30 s (median) in visit time and up to one
sweep (about 2 minutes) in duration. ``analysis/visualize_simons_observatory_obstable.py`` (use ``--debug`` for a quick run)
simulates the maps, builds the table, and writes an HTML report of diagnostic plots to
``analysis/so_report/``.


All Sky Simulations
-------------------------------------------------------------------------------
The ``LocationFreeObsTable`` class allows users to simulate objects regardless of where they are in the sky. It only requires time and filter columns. It does not perform any spatial filtering. All objects are assumed to be viewed at each time. See the (:doc:`All-Sky ObsTable Notebook <notebooks/location_free_obstable>`) for an example of how to use the ``LocationFreeObsTable`` class.
