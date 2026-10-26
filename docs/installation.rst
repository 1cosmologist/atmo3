Installation
============

atmo3 requires Python 3.12 or newer. It is built with
`meson-python <https://mesonbuild.com/meson-python/>`_, which is installed
automatically as a build dependency.

.. code-block:: bash

   git clone git@github.com:1cosmologist/atmo3.git
   cd atmo3
   pip install .

The install also ships the AFGL reference-atmosphere tables used by
:func:`atmo3.profiles.afgl_profiles` (``atmo3/data/afgl_data`` inside the
installed package).

Dependencies
------------

``jax`` and ``numpy`` are the core dependencies. ``xarray`` (with ``scipy``) is
needed to read ERA5 files through :class:`atmo3.era5_utils.ERA5Handler`.

.. note::

   For double-precision results, which are recommended for the FFTs and the
   numerical integrals, enable 64-bit mode in JAX *before* creating any array:

   .. code-block:: python

      import jax
      jax.config.update("jax_enable_x64", True)

Building this documentation
---------------------------

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-build -b html docs docs/_build/html
