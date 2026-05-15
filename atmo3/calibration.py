import jax.numpy as jnp
import pandas as pd

from . import constants as const
from . import super_grid as sg 
from . import atm_utils as au 
from . import box

class AtmosphereCalibrator:
    """
    Calibrate ERA5-derived atmospheric profiles against on-site
    meteorological observations from the APEX weather station.

    On construction the calibrator reads ERA5 temperature and specific-
    humidity profiles, normalises them so that the ground-level
    temperature and column-integrated precipitable water vapour (PWV)
    match the mean values recorded by APEX in a one-hour window centred
    on the simulation epoch, and pre-computes the RMS fluctuation
    amplitudes used as height-dependent scaling profiles by the
    :class:`box.Box` field generator.

    The APEX weather-data file is expected to be a headerless CSV with
    columns: ``UT, PWV, Temperature, Humidity, Wind_Dir, Wind_Speed``.
    """

    def __init__(self, 
            super_grid: type[sg.SuperGrid] = None, 
            temperature_file: str = None, 
            sp_humidity_file: str = None,
            cc_file: str = None,           # <-- NEW
            ciwc_file: str = None,         # <-- NEW
            clwc_file: str = None,         # <-- NEW 
            apexdatafile: str = None
    ) -> None :
        """
        Build the calibrated atmospheric profiles.

        Parameters
        ----------
        super_grid : sg.SuperGrid
            Vertical super-grid that provides the pressure levels,
            altitude axis, and ERA5 interpolation helpers.
        temperature_file : str, optional
            Path to an ERA5 temperature NetCDF file.  Used to compute
            both the mean temperature profile and the RMS temperature-
            fluctuation profile.
        cc_file : str, optional
            Path to an ERA5 cloud cover NetCDF file.
        ciwc_file : str, optional
            Path to an ERA5 cloud ice water content NetCDF file.
        clwc_file : str, optional
            Path to an ERA5 cloud liquid water content NetCDF file.
        sp_humidity_file : str, optional
            Path to an ERA5 specific-humidity NetCDF file.  Used to
            compute the mean specific-humidity profile.
        apexdatafile : str, optional
            Path to the APEX weather-station CSV file.  Rows within
            ±30 minutes of ``super_grid.time_utc`` are averaged to
            obtain reference PWV and temperature values.

        Attributes
        ----------
        pressure : jnp.ndarray
            Pressure levels from ``super_grid`` (Pa).
        temperature_profile : jnp.ndarray
            ERA5 temperature profile normalised to match the APEX
            ground-level mean temperature (K).
        spec_humidity_profile : jnp.ndarray
            ERA5 specific-humidity profile normalised so that the
            column-integrated PWV matches the APEX mean PWV
            (kg kg⁻¹).
        vir_temperature : jnp.ndarray
            Virtual temperature derived from the calibrated temperature
            and specific-humidity profiles (K).
        q2rho_h2o : jnp.ndarray
            Conversion factor from specific humidity to water-vapour
            mass density: ``p / (R_dry · T_virtual)`` (kg m⁻³ / (kg kg⁻¹)).
        temp_fluctuation_profile : jnp.ndarray
            Height-dependent RMS temperature-fluctuation amplitude,
            derived from the horizontal variance in the ERA5 temperature
            grid and rescaled to match the APEX temperature standard
            deviation (K).
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
        self.pressure = super_grid.pressure
        self.temperature_profile = super_grid.era5_interp2site(temperature_file)
        self.spec_humidity_profile = super_grid.era5_interp2site(sp_humidity_file)

        self.cc_profile = super_grid.era5_interp2site(cc_file) if cc_file is not None else None
        self.ciwc_profile = super_grid.era5_interp2site(ciwc_file) if ciwc_file is not None else None
        self.clwc_profile = super_grid.era5_interp2site(clwc_file) if clwc_file is not None else None

        self.vir_temperature = au.virtual_temperature(self.temperature_profile, self.spec_humidity_profile)
        self.q2rho_h2o = self.pressure / (const.R_dry_air * self.vir_temperature)

        t0 = pd.Timestamp(super_grid.time_utc[0]).replace(tzinfo=None)
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

        profile_pwv = jnp.trapezoid(self.q2rho_h2o*self.spec_humidity_profile, x=super_grid.z)

        self._mean_spec_humidity_normalization = self.apex_pwv_mean / profile_pwv
        self._mean_temperature_normalization = self.apex_temperature_mean / self.temperature_profile[0]


        temperature_grid = super_grid.property_from_era5(temperature_file)
        
        self.temp_fluctuation_profile = jnp.std(temperature_grid, axis=(0, 1))

        # print(self.temp_fluctuation_profile)
        self._temperature_fluctuation_norm = self.apex_temperature_std / self.temp_fluctuation_profile[0]
        # print(self._temperature_fluctuation_norm)
        del temperature_grid

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
        
        

        
        
        