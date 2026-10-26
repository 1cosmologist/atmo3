Observing
=========

:class:`~atmo3.observation.Observer` computes the lines of sight of a scan
through the box and samples the fields along them.

Scan
----

.. code-block:: python

   obs = a3.Observer(
       grid_wsp=atmo.grid_wsp,
       boresight=jnp.array([x0, y0]),      # position of the telescope in the box (m)
       passband={'nu': ..., 'g(nu)': ...},
       fwhm_arcmin=10.,
   )
   obs.compute_los_for_scan(times, azimuth_deg, elevation_deg)

The boresight must lie in the central half of the box in both horizontal
directions. ``obs.los_obj`` has shape ``(n_samples, n_los, 5)`` with
``[x, y, altitude, path length, inside-box mask]`` for every point of every line
of sight, one per altitude cell. ``obs.scan_component(field)`` interpolates a
``[Nx, Ny, Nz]`` field onto these points (zero outside the box).

Wind
----

The wind moves the frozen cube across the lines of sight. The positions of the
points at time :math:`t` after the first sample are shifted by
:math:`(-u\,t,\, -v\,t)` with :math:`u` the east and :math:`v` the north wind at
the altitude of the point.

``northwind`` and ``eastwind`` accept:

a float
   A wind in m s\ :sup:`-1` that is independent of height.

a dictionary
   A height-dependent profile ``{'z': altitude (m), 'northwind' or 'eastwind':
   wind (m/s)}``, interpolated onto the altitude axis. Outside its range the
   wind is held at its end values.

``None``
   No wind. If only one component is given, the other is zero.

.. code-block:: python

   obs = a3.Observer(grid_wsp=atmo.grid_wsp, northwind=2.0, eastwind=-3.0, ...)

   northwind, eastwind = era5.wind_profiles(northwind_file, eastwind_file)
   obs = a3.Observer(grid_wsp=atmo.grid_wsp, northwind=northwind, eastwind=eastwind, ...)

The winds are stored as arrays on the altitude axis in ``obs.north_wind`` and
``obs.east_wind``.

Evolving a cube with the wind
-----------------------------

:func:`atmo3.obs_utils.wind_evolved_field` instead rolls the cube by whole cells
with the wind, which is convenient for animations. It expects winds in cells
per second:

.. code-block:: python

   ew = obs.east_wind / atmo.grid_wsp.grid_spacing[0]
   nw = obs.north_wind / atmo.grid_wsp.grid_spacing[1]
   evolved = a3.obs_utils.wind_evolved_field(field, nw, ew, 30.0)   # 30 s later

Emission
--------

The temperature, pressure and water-vapour density along the lines of sight are
passed to :func:`~atmo3.emission.get_emission`, which integrates the radiative
transfer with the millimetre-wave absorption models for oxygen and water vapour
and returns the specific intensity at each frequency.
:func:`~atmo3.emission_utils.intensitySI_to_Tb` converts it to a brightness
temperature. See :doc:`../quickstart` for a complete example.
