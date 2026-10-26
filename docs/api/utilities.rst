Atmospheric utilities and constants
===================================

Humidity, pressure and temperature relations
--------------------------------------------

.. automodule:: atmo3.atm_utils
   :undoc-members:

Constants
---------

:mod:`atmo3.constants` defines (SI units unless noted):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Meaning
   * - ``g``
     - Standard gravity, m s\ :sup:`-2`
   * - ``molar_mass_dry_air``, ``molar_mass_water_vapor``
     - Molar masses, kg mol\ :sup:`-1`
   * - ``R_dry_air``, ``R_water_vapor``
     - Specific gas constants, J kg\ :sup:`-1` K\ :sup:`-1`
   * - ``pressure_at_sea_level``, ``temperature_at_sea_level``, ``temperature_lapse_rate``
     - Standard-atmosphere reference values (Pa, K, K m\ :sup:`-1`)
   * - ``earth_radius``
     - Mean Earth radius, m
   * - ``km_to_m``, ``mbar_to_Pa``, ``ppmv``
     - Unit conversions: km to m, mb (hPa) to Pa, ppmv to a mole fraction
   * - ``O2_DATA``, ``H2O_DATA``
     - Oxygen and water-vapour line parameters of ITU-R P.676 (tables 1 and 2)
