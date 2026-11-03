Quickstart
==========

This quickstart follows the full example in ``examples/test_box.py``. It
builds an atmospheric cube, generates temperature and water-vapour
fluctuations, scans the cube, and calculates precipitable water vapour (PWV)
and observed brightness temperature. See :doc:`installation` to install
``atmo3``. The example also uses Matplotlib to save its plots:

.. code-block:: console

   python -m pip install matplotlib
   python examples/test_box.py

The example uses a ``512 x 512 x 256`` grid in a ``20 x 20 x 10 km`` box and
a 500 m injection scale. It scans 360 samples at one-second intervals, with
the azimuth changing by one degree per sample at a fixed elevation of 55
degrees. This full-size simulation can require substantial memory and compute
time; reduce ``nside_grid`` in the script for a smaller test.

The scans return water-vapour density in kg m\ :sup:`-3` and temperature in K,
with shape ``(n_scans, n_los)``. PWV is the line-of-sight integral of the
water-vapour density (kg m\ :sup:`-2`, numerically equivalent to mm of liquid
water). The observed brightness temperature is in K\ :sub:`RJ`.

The complete example is included below:

.. literalinclude:: ../examples/test_box.py
   :language: python
   :linenos:
   :caption: examples/test_box.py

The script saves these four plots beside ``test_box.py``. The documentation
uses the corresponding images in ``docs/images``:

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item::

      .. figure:: images/rho_wv_scan.png
         :alt: Log-normalized water-vapour density along the line of sight over scan time.
         :width: 100%

         Water-vapour density along the scan.

   .. grid-item::

      .. figure:: images/T_atm_scan.png
         :alt: Log-normalized atmospheric temperature along the line of sight over scan time.
         :width: 100%

         Atmospheric temperature along the scan.

   .. grid-item::

      .. figure:: images/pwv_scan.png
         :alt: Precipitable water vapour versus time sample.
         :width: 100%

         PWV integrated along each scan.

   .. grid-item::

      .. figure:: images/T_b_obs.png
         :alt: Observed brightness temperature in K_RJ versus time sample.
         :width: 100%

         Observed brightness temperature at 150 GHz.
