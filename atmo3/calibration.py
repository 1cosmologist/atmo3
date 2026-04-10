import jax.numpy as jnp
import pandas as pd

from . import constants as const
from . import super_grid as sg 
from . import atm_utils as au 
from . import box

class AtmosphereCalibrator:
    def __init__(self, 
            super_grid: type[sg.SuperGrid] = None, 
            temperature_file: str = None, 
            sp_humidity_file: str = None, 
            apexdatafile: str = None
    ) -> None :
        self.pressure = super_grid.pressure
        self.temperature_profile = super_grid.era5_interp2site(temperature_file)
        self.spec_humidity_profile = super_grid.era5_interp2site(sp_humidity_file)

        self.vir_temperature = au.virtual_temperature(self.temperature_profile, self.spec_humidity_profile)
        self.q2rho_h2o = self.pressure / (const.R_dry_air * self.vir_temperature)

        t0 = pd.Timestamp(super_grid.time_utc)
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

        self.apex_pwv_mean = apexdata['PWV'].mean()
        self.apex_pwv_std  = apexdata['PWV'].std()
        self.apex_temperature_mean = apexdata['Temperature'].mean()
        self.apex_temperature_std  = apexdata['Temperature'].std()

        profile_pwv = jnp.trapezoid(self.q2rho_h2o*self.spec_humidity_profile, x=super_grid.z)

        self._mean_spec_humidity_normalization = self.apex_pwv_mean / profile_pwv
        self._mean_temperature_normalization = self.apex_temperature_mean / self.temperature_profile[0]


        temperature_grid = super_grid.property_from_era5(temperature_file)
        self.temp_fluctuation_profile = jnp.std(temperature_grid, axis=(0, 1))

        self._temperature_fluctuation_norm = self.apex_temperature_std / self.temp_fluctuation_profile[0]

        del temperature_grid

        self.temperature_profile      *= self._mean_temperature_normalization
        self.temp_fluctuation_profile *= self._temperature_fluctuation_norm

        self.spec_humidity_profile    *= self._mean_spec_humidity_normalization
        
        self.vir_temperature = au.virtual_temperature(self.temperature_profile, self.spec_humidity_profile)
        self.q2rho_h2o = self.pressure / (const.R_dry_air * self.vir_temperature)
        
        self.spec_humidity_fluctuation_profile = 1e-3 * jnp.copy(self.spec_humidity_profile)
        
        
    def calibrate_pwv(
        self,
        z_axis: jnp.ndarray,
        water_box: type[box.Box] = None
    ) -> None :
        pwv_plane = jnp.trapezoid(water_box.field, x=z_axis, axis=2)
        sigma_pwv = jnp.std(pwv_plane)
        
        self._spec_hum_fluctuation_norm = self.apex_pwv_std / sigma_pwv 
        
        water_box.field = water_box.field * self._spec_hum_fluctuation_norm
        
        

        
        
        