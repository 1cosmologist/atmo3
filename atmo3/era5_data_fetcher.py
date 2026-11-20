"""Download ERA5 data for a region and a single UTC day."""

from datetime import date as Date, datetime, timedelta
from math import isfinite
from numbers import Real
from pathlib import Path
import re

__all__ = ["REGION_PRESETS", "fetch_era5_data"]

REGION_PRESETS = {
    "cerro_toco": {"region_name": "cerro_toco", "bbox_WSEN": (-69, -24, -67, -22)},
    "teide": {"region_name": "teide", "bbox_WSEN": (-17, 27, -15, 29)},
}

_PRESSURE_LEVELS = (
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225,
    250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800,
    825, 850, 875, 900, 925, 950, 975, 1000,
)

_DEFAULT_VARIABLES = (
    "fraction_of_cloud_cover",
    "geopotential",
    "specific_cloud_ice_water_content",
    "specific_cloud_liquid_water_content",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
)

# ERA5 data is fetched from Google cloud public data storage: https://console.cloud.google.com/storage/browser/gcp-public-data-arco-era5/raw/date-variable-pressure_level/

def fetch_era5_data(region, date, variables=None, output_path=None):
    """Fetch ERA5 data for one UTC day.

    Parameters
    ----------
    region : str or dict
        Preset name, or a dictionary with ``region_name`` and ``bbox_WSEN``
        (west, south, east, north in degrees). Bounding boxes crossing 0 degrees
        longitude are not supported by the downloader.
    date : datetime.date or str
        UTC day, as a date or ISO-format date string.
    variables : sequence of str, optional
        ERA5 variable names, in the desired return order. Defaults to cloud
        cover, geopotential, cloud ice and liquid water, specific humidity,
        temperature, and the u and v wind components.
    output_path : str or pathlib.Path, optional
        Local output directory. Each variable is saved as a NetCDF file under
        ``era5-{region_name}/{year}/{variable}/``.

    Returns
    -------
    list of dict or None
        Without an output directory, one xarray Dataset dictionary per variable,
        with NumPy arrays, coordinates and attributes. Otherwise, returns None.
        The downloader may use temporary disk space in either mode.
    """
    if isinstance(region, str):
        if region not in REGION_PRESETS:
            raise ValueError(f"Unknown ERA5 region preset: {region!r}")
        region = REGION_PRESETS[region]
    if not isinstance(region, dict) or not {"region_name", "bbox_WSEN"} <= region.keys():
        raise ValueError("region must contain region_name and bbox_WSEN")

    name = region["region_name"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("region_name must contain only letters, digits, underscores or hyphens")
    bbox = region["bbox_WSEN"]
    if (
        not isinstance(bbox, (tuple, list))
        or len(bbox) != 4
        or any(not isinstance(value, Real) or not isfinite(value) for value in bbox)
    ):
        raise ValueError("bbox_WSEN must contain four finite coordinates in degrees")
    west, south, east, north = bbox
    if not (-180 <= west < east <= 360 and -90 <= south < north <= 90):
        raise ValueError("bbox_WSEN must be ordered west, south, east, north within geographic bounds")
    if west < 0 and east >= 0:
        raise ValueError("Bounding boxes crossing 0 degrees longitude are not supported")

    if isinstance(date, str):
        date = datetime.strptime(date, "%Y-%m-%d").date()
    if not isinstance(date, Date) or isinstance(date, datetime):
        raise TypeError("date must be a datetime.date or YYYY-MM-DD string")
    if variables is None:
        variables = _DEFAULT_VARIABLES
    if isinstance(variables, str):
        raise TypeError("variables must be a sequence of ERA5 variable names, not a string")
    variables = list(variables)
    if not variables or any(
        not isinstance(var, str) or not re.fullmatch(r"[A-Za-z0-9_]+", var)
        for var in variables
    ):
        raise ValueError("variables must be a non-empty sequence of valid ERA5 variable names")

    import era5_downloader as era5down

    results = []
    for var in variables:
        kwargs = {}
        if output_path is not None:
            root = str(Path(output_path)).replace("{", "{{").replace("}", "}}")
            kwargs["dest_path"] = (
                f"{root}/era5-{{region_name}}/{{time:%Y}}/{var}/"
                f"{var}_era5-{{region_name}}-{{time:%Y%m%d}}.nc"
            )
        downloader = era5down.ERA5Downloader(
            bbox_WSEN=tuple(bbox),
            region_name=name,
            levels=list(_PRESSURE_LEVELS),
            variables=[var],
            time_steps=tuple(range(24)),
            **kwargs,
        )
        if output_path is None:
            with downloader.get_raw(date.isoformat()) as dataset:
                results.append(dataset.to_dict(data="array"))
        else:
            downloader.download(date.isoformat())
    return results if output_path is None else None