Quickstart
==========

This walks through a complete simulation: build an atmosphere, generate a
realization, scan through it and compute the brightness temperature. It uses a
bundled AFGL reference atmosphere, so no external data are needed.

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)
   import jax.numpy as jnp
   import numpy as np
   import atmo3 as a3

1. Build the atmosphere
-----------------------

``profiles`` is either the name of an AFGL reference atmosphere or a dictionary
of your own vertical profiles (see :doc:`guide/profiles`). The optional site
statistics recalibrate the profiles (see :doc:`guide/calibration`).

.. code-block:: python

   nside_grid = [64, 64, 32]                # Nx, Ny, Nz
   box_length = [12000., 12000., 3000.]     # Lx, Ly, Lz in metres

   atmo = a3.Atmosphere(
       nside_grid=nside_grid,
       box_length_in_m=box_length,
       site_altitude=5100.,                 # metres above sea level
       profiles='midlatitude_summer',
       temperature_mean=272.2,              # K, at the site
       temperature_std=0.3,                 # K
       pwv_mean=1.15,                       # mm
       pwv_std=0.03,                        # mm
   )

2. Add turbulent components
---------------------------

Each component is a Gaussian random field with a power spectrum that you
provide, scaled in height by the profiles. A von Kármán spectrum with an
injection scale of 500 m is used here.

.. code-block:: python

   k = np.arange(4 * nside_grid[0]) * jnp.min(atmo.grid_wsp.dk)
   k0 = 2 * np.pi / 500.
   pofk = (k0**2 + k**2) ** (-11 / 6)
   pofk /= pofk.max()
   pspec = {'k': k, 'pofk': pofk}

   atmo.add_temperature(power_spec=pspec)
   atmo.add_watervapor(power_spec=pspec)

   atmo.generate_realization(time_step=0)

The fluctuations are stored in ``atmo.components[name].field`` (shape
``[Nx, Ny, Nz]``) and the mean profiles in ``atmo.component_mean[name]['f']``.
The profiles live on the altitude axis ``atmo.z`` (see :doc:`guide/grid`).
The observer adds each mean profile after sampling the fluctuation field, so
there is no need to construct a full 3D total field.

3. Scan through the cube
------------------------

The winds move the cube across the lines of sight. Pass floats for a wind that
does not depend on height, or profile dictionaries (see :doc:`guide/observing`).

.. code-block:: python

   obs = a3.Observer(
       grid_wsp=atmo.grid_wsp,
       northwind=2.0,                                                # m/s
       eastwind={'z': [0., 20000.], 'eastwind': [0., 10.]},          # m/s
       boresight=jnp.array([6000., 6000.]),                          # x, y in the box (m)
       passband={'nu': jnp.array([150.]), 'g(nu)': jnp.array([1.])},
       fwhm_arcmin=10.,
   )

   times = [np.datetime64('2023-09-16T00:00:00') + np.timedelta64(200 * i, 'ms') for i in range(5)]
   azimuth = [0.0, 0.2, 0.4, 0.6, 0.8]      # degrees
   elevation = [45.] * 5                    # degrees
   obs.compute_los_for_scan(times, azimuth, elevation)

   water_vapor_los = obs.scan_atmosphere(atmo, 'water vapor')  # kg m^-3; shape (n_samples, n_los)
   temperature_los = obs.scan_atmosphere(atmo, 'temperature')  # K

``obs.los_obj`` holds, for every sample and point along the line of sight,
``[x, y, altitude, path length, inside-box mask]``.

4. Compute the emission
-----------------------

.. code-block:: python

   altitude_los = obs.los_obj[:, :, 2]
   pressure_los = jnp.interp(altitude_los, atmo.z, atmo.atm_calibrator.pressure)
   ds = jnp.diff(obs.los_obj[:, :, 3], axis=1, prepend=0)    # path-length steps (m)

   freqs_GHz = jnp.array([150., 185.])
   intensity = a3.get_emission(temperature_los, pressure_los, water_vapor_los, ds, freqs_GHz)
   brightness_temperature = a3.intensitySI_to_Tb(intensity[:, :, None], freqs_GHz[None, None, :])[:, :, 0]

For the observer's passband, ``obs.observe_atmosphere(atmo)`` performs these
steps and returns one band-weighted brightness temperature (K_RJ) per sample.
The passband response ``g(nu)`` contains discrete weights that sum to one.

.. warning::

   Points of a line of sight that leave the box horizontally are filled with
   zeros by :meth:`~atmo3.observation.Observer.scan_atmosphere`, which makes the
   emission undefined. Choose a box and a boresight so that the whole scan,
   including the wind drift, stays inside it. At 45° elevation the horizontal
   extent of the path equals the box height.
