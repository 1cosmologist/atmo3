"""
ERA5 data handling utilities.

ERA5 pressure-level reanalysis files are read at a given epoch and turned
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
    geopotential_file : str
        ERA5 geopotential NetCDF file.  It defines the pressure levels,
        the horizontal grid and the height of each level; all other
        files must share its levels and horizontal grid.
    time_utc : datetime
        UTC epoch, which must be present in the files.
    site_lonlat : list of float
        Site [longitude, latitude] in degrees.

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

    def __init__(self, geopotential_file, time_utc, site_lonlat):
        self.time_utc = time_utc
        self.site_coordinates = site_lonlat
        self._levels = None

        geopotential = self._load(geopotential_file)
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

    def _load(self, filepath):
        """Open a file at the epoch, with levels ordered by decreasing pressure."""
        with xr.open_dataarray(filepath) as dataset:
            data = dataset.sel(time=f'{self.time_utc:%Y-%m-%dT%H:%M:%S}').sortby('level', ascending=False).astype(float).load()

        if self._levels is not None and (data.level.size != self._levels.size or not np.allclose(data.level.values, self._levels)):
            raise ValueError(f"Pressure levels of '{filepath}' differ from the geopotential file.")
        return data

    def interp2site(self, filepath):
        """Profile of an ERA5 variable at the site, on the ERA5 levels.

        Returns
        -------
        np.ndarray
            Values at the heights :attr:`z`, in the units of the file.
        """
        data = self._load(filepath)
        return data.interp(latitude=self.site_coordinates[1], longitude=self._site_lon).values

    def horizontal_std(self, filepath):
        """Standard deviation of an ERA5 variable over the horizontal grid at fixed heights.

        Each column is interpolated to the heights :attr:`z` of the site
        levels before taking the standard deviation over latitude and
        longitude.

        Returns
        -------
        np.ndarray
            Standard deviation at the heights :attr:`z`, in the units of the file.
        """
        data = self._load(filepath).values
        if data.shape != self.z_geo.shape:
            raise ValueError(f"Horizontal grid of '{filepath}' differs from the geopotential file.")

        nlev, nlat, nlon = data.shape
        columns = np.empty((nlat, nlon, nlev))
        for i in range(nlat):
            for j in range(nlon):
                columns[i, j] = _interp_extrapolate(self.z, self.z_geo[:, i, j], data[:, i, j])

        return np.std(columns, axis=(0, 1))

    def profiles(self, temperature_file, spec_humidity_file, temperature_std=True):
        """Standardized profile dictionary for :class:`atmosphere.Atmosphere`.

        Parameters
        ----------
        temperature_file, spec_humidity_file : str
            ERA5 temperature and specific-humidity files.
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
        profiles = {
            'z':                 self.z,
            'temperature':       self.interp2site(temperature_file),
            'pressure':          self.pressure,
            'specific_humidity': self.interp2site(spec_humidity_file),
        }
        if temperature_std:
            profiles['temperature_std'] = self.horizontal_std(temperature_file)

        return validate_profiles(profiles)

    def wind_profiles(self, northwind_file, eastwind_file):
        """Wind profiles for :class:`observation.Observer`.

        Parameters
        ----------
        northwind_file, eastwind_file : str
            ERA5 northward (v) and eastward (u) wind-component files.

        Returns
        -------
        northwind, eastwind : dict
            ``{'z': heights (m), 'northwind' or 'eastwind': wind (m/s)}``.
        """
        return (
            {'z': self.z, 'northwind': self.interp2site(northwind_file)},
            {'z': self.z, 'eastwind':  self.interp2site(eastwind_file)},
        )
