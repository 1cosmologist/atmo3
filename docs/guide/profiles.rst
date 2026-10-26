Weather profiles
================

:class:`~atmo3.atmosphere.Atmosphere` does not know where its weather data come
from. It is initialised from a **profile dictionary** of 1-D arrays defined on
a common altitude axis. Any source that can produce this dictionary works.

The standard
------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Key
     - Unit
     - Description
   * - ``'z'``
     - m
     - Altitude above sea level, strictly increasing
   * - ``'temperature'``
     - K
     - Air temperature
   * - ``'pressure'``
     - Pa
     - Total air pressure
   * - ``'specific_humidity'``
     - kg kg\ :sup:`-1`
     - Water-vapour mass fraction of moist air
   * - ``'temperature_std'``
     - K
     - *Optional.* Horizontal RMS temperature fluctuation at each height

:func:`atmo3.profiles.validate_profiles` checks a dictionary against this
standard and is applied to every input.

Passing profiles to the atmosphere
----------------------------------

``profiles`` accepts either a dictionary or the name of an AFGL reference
atmosphere:

.. code-block:: python

   atmo = a3.Atmosphere(profiles='us_standard', ...)                       # a name
   atmo = a3.Atmosphere(profiles=my_profiles, ...)                         # a dictionary

The profiles are interpolated onto the altitude axis of the grid by
:class:`~atmo3.calibration.AtmosphereCalibrator`: linearly for the temperature,
humidity and temperature fluctuation, and in log space for the pressure.

AFGL reference atmospheres
--------------------------

:func:`atmo3.profiles.afgl_profiles` loads the six AFGL (1986) reference
atmospheres [#afgl]_ shipped with the package and converts them to the standard
units.

.. list-table::
   :header-rows: 1

   * - Name
     - Number
     - Table
   * - ``'tropical'``
     - 1
     - ``'1a'``
   * - ``'midlatitude_summer'``
     - 2
     - ``'1b'``
   * - ``'midlatitude_winter'``
     - 3
     - ``'1c'``
   * - ``'subarctic_summer'``
     - 4
     - ``'1d'``
   * - ``'subarctic_winter'``
     - 5
     - ``'1e'``
   * - ``'us_standard'``
     - 6
     - ``'1f'``

.. code-block:: python

   p = a3.afgl_profiles('midlatitude_summer')
   p['z'][:3]            # array([   0., 1000., 2000.])  metres
   p['pressure'][0]      # 101300.0  Pa

The tabulated quantities are converted as follows:

* altitude: km → m;
* pressure: mb (hPa) → Pa, a factor of 100;
* temperature: already in K;
* water vapour: the volume mixing ratio in ppmv is a mole fraction of total
  (moist) air, and is converted to a specific humidity with
  :func:`atmo3.atm_utils.volume_mixing_ratio_to_specific_humidity`.

AFGL profiles carry no horizontal variability, so they have no
``'temperature_std'``. Provide ``temperature_std`` to the atmosphere to set the
temperature fluctuations (see :doc:`calibration`).

The tables are looked up in the installed package, then in ``data/afgl_data`` of
a source checkout. Pass ``table_dir`` to use another location.

ERA5 reanalysis
---------------

:class:`atmo3.era5_utils.ERA5Handler` reads ERA5 pressure-level files at a site
and epoch and returns the same dictionaries:

.. code-block:: python

   from datetime import datetime, timezone

   era5 = a3.ERA5Handler(geopotential_file, datetime(2023, 9, 16, tzinfo=timezone.utc),
                         site_lonlat=[-67.78, -22.95])
   profiles = era5.profiles(temperature_file, specific_humidity_file)
   northwind, eastwind = era5.wind_profiles(northwind_file, eastwind_file)

The geopotential file defines the pressure levels, the horizontal grid and the
height of each level. The other files must share them. ``profiles()`` includes
the horizontal standard deviation of the temperature as ``'temperature_std'``.

Your own measurements
---------------------

Build the dictionary yourself, for example from radiosonde data:

.. code-block:: python

   my_profiles = {
       'z': z_metres,
       'temperature': temperature_kelvin,
       'pressure': pressure_pascal,
       'specific_humidity': q_kg_per_kg,
   }
   atmo = a3.Atmosphere(profiles=my_profiles, ...)

.. rubric:: References

.. [#afgl] G. P. Anderson, S. A. Clough, F. X. Kneizys, J. H. Chetwynd and
   E. P. Shettle, *AFGL Atmospheric Constituent Profiles (0-120 km)*,
   AFGL-TR-86-0110, 1986.
