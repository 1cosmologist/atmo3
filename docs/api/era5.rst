ERA5 data handling
==================

.. automodule:: atmo3.era5_utils

Downloading ERA5 data
---------------------

Install the optional downloader with ``pip install 'atmo3[era5]'``. Download
one UTC day for a preset or custom region:

.. code-block:: python

   from datetime import date
   from atmo3 import fetch_era5_data

   data = fetch_era5_data("cerro_toco", date(2023, 9, 16),
                          ["temperature", "specific_humidity"])
   temperature = data[0]  # xarray Dataset dictionary, including NumPy arrays

   region = {"region_name": "my_site", "bbox_WSEN": (-69, -24, -67, -22)}
   fetch_era5_data(region, "2023-09-16", ["temperature"], output_path="data")

Omitting ``variables`` fetches, in order, ``fraction_of_cloud_cover``,
``geopotential``, ``specific_cloud_ice_water_content``,
``specific_cloud_liquid_water_content``, ``specific_humidity``, ``temperature``,
``u_component_of_wind``, and ``v_component_of_wind``.

The bounding box is west, south, east, north in degrees. The downloader does
not support boxes crossing 0 degrees longitude. All 24 hours and the original
script's 37 pressure levels (in hPa) are requested. Returned dictionaries follow
``xarray.Dataset.to_dict(data="array")`` and retain the dataset coordinates and
attributes; variable units remain those of ERA5. The downloader uses a temporary
disk cache even when data are returned in memory.

With an output directory, one NetCDF file per variable is written under
``data/era5-{region_name}/{year}/{variable}/``, and the function returns
``None``. Existing files are reused by the downloader.

.. automodule:: atmo3.era5_data_fetcher
