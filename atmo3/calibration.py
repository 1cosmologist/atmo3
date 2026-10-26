import jax.numpy as jnp
import pandas as pd

from . import constants as const
from . import atm_utils as au 
from . import box
from .profiles import validate_profiles

class AtmosphereCalibrator:
    """
    Interpolate standardized atmospheric profiles onto the simulation
    z-grid and calibrate them against on-site meteorological
    observations from the APEX weather station.

    On construction the calibrator interpolates the temperature,
    pressure and specific-humidity profiles (see :mod:`profiles`) onto
    the z-grid, normalises them so that the ground-level temperature and
    column-integrated precipitable water vapour (PWV) match the mean
    values recorded by APEX in a one-hour window centred on the
    simulation epoch, and pre-computes the RMS fluctuation amplitudes
    used as height-dependent scaling profiles by the :class:`box.Box`
    field generator.

    The APEX weather-data file is expected to be a headerless CSV with
    columns: ``UT, PWV, Temperature, Humidity, Wind_Dir, Wind_Speed``.
    """

    def __init__(self, 
            z_grid: jnp.ndarray = None,
            profiles: dict = None,
            time_utc = None,
            apexdatafile: str = None
    ) -> None :
        """
        Build the calibrated atmospheric profiles.

        Parameters
        ----------
        z_grid : jnp.ndarray
            Altitude axis of the simulation grid above sea level (m).
        profiles : dict
            Standardized profile dictionary (see :mod:`profiles`) with
            keys ``'z'`` (m), ``'temperature'`` (K), ``'pressure'`` (Pa)
            and ``'specific_humidity'`` (kg kg⁻¹), plus optionally
            ``'temperature_std'`` (K), the RMS temperature fluctuation.
            If ``'temperature_std'`` is absent, a height-independent
            fluctuation equal to the APEX temperature standard
            deviation is used.
        time_utc : datetime
            UTC timestamp of the simulation epoch.
        apexdatafile : str, optional
            Path to the APEX weather-station CSV file.  Rows within
            ±30 minutes of ``time_utc`` are averaged to obtain
            reference PWV and temperature values.

        Attributes
        ----------
        pressure : jnp.ndarray
            Pressure profile on ``z_grid`` (Pa).
        temperature_profile : jnp.ndarray
            Temperature profile normalised to match the APEX
            ground-level mean temperature (K).
        spec_humidity_profile : jnp.ndarray
            Specific-humidity profile normalised so that the
            column-integrated PWV matches the APEX mean PWV
            (kg kg⁻¹).
        vir_temperature : jnp.ndarray
            Virtual temperature derived from the calibrated temperature
            and specific-humidity profiles (K).
        q2rho_h2o : jnp.ndarray
            Conversion factor from specific humidity to water-vapour
            mass density: ``p / (R_dry · T_virtual)`` (kg m⁻³ / (kg kg⁻¹)).
        temp_fluctuation_profile : jnp.ndarray
            Height-dependent RMS temperature-fluctuation amplitude (K),
            from ``profiles['temperature_std']`` if provided.
        spec_humidity_fluctuation_profile : jnp.ndarray
            Height-dependent RMS specific-humidity fluctuation amplitude,
            set to 0.1 % of the calibrated mean specific-humidity
            profile (kg kg⁻¹).
        apex_pwv_mean : float
            Mean PWV from the APEX CSV window (mm).
        apex_pwv_std : float
            Standard deviation of PWV from the APEX CSV window (mm).
        apex_temperature_mean : float
            Mean temperature from the APEX CSV window (K or °C,
            as stored in the file).
        apex_temperature_std : float
            Standard deviation of temperature from the APEX CSV window.
        """
        profiles = validate_profiles(profiles)
        z_grid = jnp.asarray(z_grid)

        # Pressure falls roughly exponentially with height, so interpolate its logarithm.
        self.pressure = jnp.exp(jnp.interp(z_grid, profiles['z'], jnp.log(profiles['pressure']), left='extrapolate', right='extrapolate'))
        self.temperature_profile = jnp.interp(z_grid, profiles['z'], profiles['temperature'], left='extrapolate', right='extrapolate')
        self.spec_humidity_profile = jnp.clip(
            jnp.interp(z_grid, profiles['z'], profiles['specific_humidity'], left='extrapolate', right='extrapolate'),
            0.0
        )

        self.vir_temperature = au.virtual_temperature(self.temperature_profile, self.spec_humidity_profile)
        self.q2rho_h2o = self.pressure / (const.R_dry_air * self.vir_temperature)

        t0 = pd.Timestamp(time_utc).replace(tzinfo=None)
        t_start = t0 - pd.Timedelta(minutes=30)
        t_end   = t0 + pd.Timedelta(minutes=30)

        chunks = []
        for chunk in pd.read_csv(
            apexdatafile,
            header=None,
            names=['UT', 'PWV', 'Temperature', 'Humidity', 'Wind_Dir', 'Wind_Speed'],
            parse_dates=['UT'],
            chunksize=1024
        ):
            chunk = chunk[(chunk['UT'] >= t_start) & (chunk['UT'] <= t_end)]
            if not chunk.empty:
                chunks.append(chunk)
                
        apexdata = pd.concat(chunks) if chunks else pd.DataFrame(
            columns=['UT', 'PWV', 'Temperature', 'Humidity', 'Wind_Dir', 'Wind_Speed']
        )

        apexdata['Temperature'] += 273.15  # APEX is in Celsius
        self.apex_pwv_mean = apexdata['PWV'].mean()
        self.apex_pwv_std  = apexdata['PWV'].std()
        self.apex_temperature_mean = apexdata['Temperature'].mean()
        self.apex_temperature_std  = apexdata['Temperature'].std()

        profile_pwv = jnp.trapezoid(self.q2rho_h2o*self.spec_humidity_profile, x=z_grid)

        self._mean_spec_humidity_normalization = self.apex_pwv_mean / profile_pwv
        self._mean_temperature_normalization = self.apex_temperature_mean / self.temperature_profile[0]

        if 'temperature_std' in profiles:
            self.temp_fluctuation_profile = jnp.interp(z_grid, profiles['z'], profiles['temperature_std'], left='extrapolate', right='extrapolate')
        else:
            self.temp_fluctuation_profile = jnp.full_like(z_grid, self.apex_temperature_std, dtype=self.temperature_profile.dtype)

        self._temperature_fluctuation_norm = self.apex_temperature_std / self.temp_fluctuation_profile[0]

        self.temperature_profile      *= self._mean_temperature_normalization
        # self.temp_fluctuation_profile *= self._temperature_fluctuation_norm

        self.spec_humidity_profile    *= self._mean_spec_humidity_normalization
        
        self.vir_temperature = au.virtual_temperature(self.temperature_profile, self.spec_humidity_profile)
        self.q2rho_h2o = self.pressure / (const.R_dry_air * self.vir_temperature)
        
        self.spec_humidity_fluctuation_profile = 1e-3 * jnp.copy(self.spec_humidity_profile)
        
        
        print(self.apex_pwv_mean, self.apex_pwv_std)
        
        
    def calibrate_pwv(
        self,
        z_axis: jnp.ndarray,
        water_box: type[box.Box] = None
    ) -> None :
        """
        Rescale a water-vapour fluctuation field to match the APEX PWV
        standard deviation.

        The column-integrated PWV is computed for each horizontal pixel
        of ``water_box.field`` by integrating along the vertical axis.
        The resulting sky-plane PWV map has a standard deviation
        ``sigma_pwv``; the field is then multiplied by
        ``apex_pwv_std / sigma_pwv`` so that the simulated PWV
        fluctuations have the same amplitude as observed by APEX.

        The normalisation factor is stored as
        ``self._spec_hum_fluctuation_norm`` for later inspection.

        Parameters
        ----------
        z_axis : jnp.ndarray
            1-D altitude array corresponding to the third axis of
            ``water_box.field`` (m).  Passed as the ``x`` argument to
            ``jnp.trapezoid``.
        water_box : box.Box
            Water-vapour :class:`box.Box` instance whose ``field``
            attribute (shape ``[Nx, Ny, Nz]``) is rescaled in-place.
        """
        pwv_plane = jnp.trapezoid(water_box.field, x=z_axis, axis=2)
        sigma_pwv = jnp.std(pwv_plane)
        
        self._spec_hum_fluctuation_norm = self.apex_pwv_std / sigma_pwv 
        
        water_box.field = water_box.field * self._spec_hum_fluctuation_norm
        
        

        
        
        