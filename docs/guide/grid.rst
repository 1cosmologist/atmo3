Grid and altitude axis
======================

The simulation box is described by a :class:`~atmo3.grid_utils.GridWorkspace`,
available as ``atmo.grid_wsp``. For ``N = [Nx, Ny, Nz]`` cells over a box of
length ``Lbox = [Lx, Ly, Lz]``, the grid spacing is ``Lbox / N`` and each axis
is ``arange(N) * spacing``.

The altitude axis
-----------------

The third axis is shifted by the site altitude to give altitudes above sea level:

.. code-block:: python

   z = atmo.grid_wsp.grid_axis(axis=2, altitude_axis=True)   # site_altitude + arange(Nz) * Lz / Nz

``atmo.z`` is exactly this array, and it is the only altitude axis used
throughout atmo3:

* the calibrator evaluates every profile on it;
* the ``'h'`` entries of the mean and scaling dictionaries refer to it;
* :class:`~atmo3.observation.Observer` evaluates the winds and the lines of
  sight on it;
* PWV integrals are taken over it.

.. note::

   The axis starts at the site altitude and stops one cell *short* of
   ``site_altitude + Lz``, as the axes of a periodic box do. Do not rebuild it
   with ``linspace(site_altitude, site_altitude + Lz, Nz)``, whose spacing
   differs and would shift the mean profiles by up to one cell at the top.

Index-aligned arrays
--------------------

Because the profiles are defined on this axis, the mean profile of a component
can be added index by index to its field:

.. code-block:: python

   total = atmo.components['temperature'].field + atmo.component_mean['temperature']['f'].reshape(1, 1, -1)

Pressure, and any profile you need along a line of sight, is obtained by
interpolating over ``atmo.z``:

.. code-block:: python

   pressure_los = jnp.interp(altitude_los, atmo.z, atmo.atm_calibrator.pressure)

Wavenumbers
-----------

``atmo.grid_wsp.dk`` is the wavenumber spacing (rad m\ :sup:`-1`) used to sample
the power spectrum you provide to
:meth:`~atmo3.atmosphere.Atmosphere.add_temperature` and
:meth:`~atmo3.atmosphere.Atmosphere.add_watervapor`.
