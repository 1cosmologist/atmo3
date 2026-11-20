"""
ERA5 data handling utilities.

ERA5 pressure-level reanalysis files or fetched dictionaries are read at a given epoch and turned
into the standardized profile dictionaries expected by
:class:`atmosphere.Atmosphere` and :class:`observation.Observer` (see
:mod:`profiles`), so that the rest of the code stays independent of the
data source.

The files are expected to hold a single variable on the dimensions
``(time, level, latitude, longitude)``, with ``level`` in hPa:
geopotential (m² s⁻²), temperature (K), specific humidity (kg kg⁻¹) and
the wind components (m s⁻¹).
"""

import numpy as np
import xarray as xr

from . import constants as con
from .profiles import validate_profiles


def _interp_extrapolate(x, xp, fp):
    """Linear interpolation of ``fp(xp)`` at ``x``, extrapolating linearly outside ``xp``."""
    y = np.interp(x, xp, fp)
    low, high = x < xp[0], x > xp[-1]
    y = np.where(low, fp[0] + (x - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0]), y)
    y = np.where(high, fp[-1] + (x - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]), y)
    return y


class ERA5Handler:
    """Read ERA5 pressure-level data at a site and epoch as standardized profiles.

    Parameters
    ----------
    geopotential_file : str or pathlib.Path, optional
        ERA5 geopotential NetCDF file. If omitted, geopotential is selected
        from ``era5_data``. It defines the pressure levels, horizontal grid,
        and height of each level; all other data must share its grid.
    time_utc : datetime
        UTC epoch, which must be present in the data.
    site_lonlat : list of float
        Site [longitude, latitude] in degrees.
    era5_data : list of dict, optional
        In-memory result of :func:`atmo3.era5_data_fetcher.fetch_era5_data`.
        Variables are identified by their ERA5 ``name_requested`` attribute.

    Attributes
    ----------
    lat, lon : np.ndarray
        Latitude and longitude of the ERA5 grid (degrees).
    pressure : np.ndarray
        Pressure of the levels (Pa), decreasing with index.
    z : np.ndarray
        Geopotential height of the levels above sea level at the site
        (m), increasing with index.
    z_geo : np.ndarray
        Geopotential height of the levels over the horizontal grid (m),
        shape ``(n_levels, n_lat, n_lon)``.
    """

    def __init__(self, geopotential_file=None, time_utc=None, site_lonlat=None, *, era5_data=None):
        if time_utc is None or site_lonlat is None:
            raise ValueError("time_utc and site_lonlat are required")
        self.time_utc = time_utc
        self.site_coordinates = site_lonlat
        self._era5_data = era5_data
        self._levels = None

        geopotential = self._load(self._source(geopotential_file, "geopotential"), "geopotential")
        self._levels = geopotential.level.values

        self.lat = geopotential.latitude.values
        self.lon = geopotential.longitude.values
        self._site_lon = self._site_longitude()

        self.pressure = self._levels * con.mbar_to_Pa
        self.z_geo = geopotential.values / con.g
        self.z = (geopotential.interp(latitude=self.site_coordinates[1], longitude=self._site_lon).values) / con.g

        if not np.all(np.diff(self.z) > 0):
            raise ValueError("ERA5 geopotential height does not increase monotonically with decreasing pressure at the site.")

    def _site_longitude(self):
        # ERA5 longitudes are either in [0, 360) or in [-180, 180).
        lon = self.site_coordinates[0]
        return lon % 360 if self.lon.min() >= 0 else (lon + 180) % 360 - 180

    def _source(self, filepath, variable):
        if filepath is not None:
            return filepath
        if self._era5_data is None:
            raise ValueError(f"No file or fetched data provided for {variable}")
        matches = [
            item for item in self._era5_data
            if any(
                var.get("attrs", {}).get("name_requested") == variable
                and "level" in var.get("dims", ())
                for var in item.get("data_vars", {}).values()
            )
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one pressure-level dataset for {variable}, found {len(matches)}")
        return matches[0]

    def _load(self, source, variable=None):
        """Read a file or fetched dictionary at the epoch, in decreasing pressure."""
        from_dict = isinstance(source, dict)
        dataset = xr.Dataset.from_dict(source) if from_dict else xr.open_dataset(source)
        with dataset:
            candidates = [
                data for data in dataset.data_vars.values()
                if "level" in data.dims
            ]
            if variable is not None and (from_dict or len(candidates) > 1):
                candidates = [data for data in candidates if data.attrs.get("name_requested") == variable]
            if len(candidates) != 1:
                raise ValueError(f"Expected one pressure-level variable for {variable}, found {len(candidates)}")
            data = candidates[0].sel(time=f'{self.time_utc:%Y-%m-%dT%H:%M:%S}').sortby('level', ascending=False).astype(float).load()

        if self._levels is not None and (data.level.size != self._levels.size or not np.allclose(data.level.values, self._levels)):
            raise ValueError("Pressure levels differ from the geopotential data.")
        if self._levels is not None and (
            data.latitude.size != self.lat.size or data.longitude.size != self.lon.size
            or not np.allclose(data.latitude.values, self.lat)
            or not np.allclose(data.longitude.values, self.lon)
        ):
            raise ValueError("Horizontal grid differs from the geopotential data.")
        return data

    def interp2site(self, filepath, *, variable=None):
        """Profile of an ERA5 variable at the site, on the ERA5 levels.

        Returns
        -------
        np.ndarray
            Values at the heights :attr:`z`, in the units of the input.
        """
        data = self._load(filepath, variable)
        return data.interp(latitude=self.site_coordinates[1], longitude=self._site_lon).values

    def horizontal_std(self, filepath, *, variable=None):
        """Standard deviation of an ERA5 variable over the horizontal grid at fixed heights.

        Each column is interpolated to the heights :attr:`z` of the site
        levels before taking the standard deviation over latitude and
        longitude.

        Returns
        -------
        np.ndarray
            Standard deviation at the heights :attr:`z`, in the units of the input.
        """
        data = self._load(filepath, variable).values
        if data.shape != self.z_geo.shape:
            raise ValueError(f"Horizontal grid of '{filepath}' differs from the geopotential file.")

        nlev, nlat, nlon = data.shape
        columns = np.empty((nlat, nlon, nlev))
        for i in range(nlat):
            for j in range(nlon):
                columns[i, j] = _interp_extrapolate(self.z, self.z_geo[:, i, j], data[:, i, j])

        return np.std(columns, axis=(0, 1))

    def profiles(self, temperature_file=None, spec_humidity_file=None, temperature_std=True):
        """Standardized profile dictionary for :class:`atmosphere.Atmosphere`.

        Parameters
        ----------
        temperature_file, spec_humidity_file : str or pathlib.Path, optional
            ERA5 temperature and specific-humidity files. Omitted inputs are
            selected from the fetched dictionaries supplied to the constructor.
        temperature_std : bool, optional
            Include the horizontal standard deviation of the temperature
            as ``'temperature_std'``.  Defaults to True.

        Returns
        -------
        dict
            Profiles with keys ``'z'`` (m), ``'temperature'`` (K),
            ``'pressure'`` (Pa), ``'specific_humidity'`` (kg kg⁻¹) and,
            optionally, ``'temperature_std'`` (K).
        """
        temperature = self._source(temperature_file, "temperature")
        humidity = self._source(spec_humidity_file, "specific_humidity")
        profiles = {
            'z':                 self.z,
            'temperature':       self.interp2site(temperature, variable="temperature"),
            'pressure':          self.pressure,
            'specific_humidity': self.interp2site(humidity, variable="specific_humidity"),
        }
        if temperature_std:
            profiles['temperature_std'] = self.horizontal_std(temperature, variable="temperature")

        return validate_profiles(profiles)

    def wind_profiles(self, northwind_file=None, eastwind_file=None):
        """Wind profiles for :class:`observation.Observer`.

        Parameters
        ----------
        northwind_file, eastwind_file : str or pathlib.Path, optional
            ERA5 northward (v) and eastward (u) wind-component files. Omitted
            inputs are selected from the fetched dictionaries.

        Returns
        -------
        northwind, eastwind : dict
            ``{'z': heights (m), 'northwind' or 'eastwind': wind (m/s)}``.
        """
        north = self._source(northwind_file, "v_component_of_wind")
        east = self._source(eastwind_file, "u_component_of_wind")
        return (
            {'z': self.z, 'northwind': self.interp2site(north, variable="v_component_of_wind")},
            {'z': self.z, 'eastwind': self.interp2site(east, variable="u_component_of_wind")},
        )
