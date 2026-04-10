from . import box
from . import super_grid
from . import calibration 
from . import grid_utils as gutl
from . import constants  as const
import jax
import jax.numpy as jnp
import numpy as np
from datetime import datetime, timezone
import gc

class Atmosphere:
    
    def __init__(self, 
                 nside_grid: list = [256, 256, 128], 
                 box_length_in_m: list = [20000.0, 20000.0, 10000.0],
                 site_altitude: float = 0.0,
                 site_coordinates: list = [0.0, 0.0],
                 time_utc: type[datetime] = datetime(2023, 9, 1, 0, 0, tzinfo=timezone.utc),
                 geopotential_file_era5: str = None,
                 temperature_file_era5: str = None,
                 spec_humidity_file_era5: str = None,
                 apex_datafile: str = None
        ) -> None:
        
        """
        Initialize the atmosphere object.

        Parameters
        ----------
        nside_grid : int, optional
            Number of cells per side of the grid. Defaults to 128.
        box_length_in_m : float, optional
            Box length in meters. Defaults to 10000.0.

        Attributes
        ----------
        N : int
            Number of cells per side of the grid.
        Lbox : float
            Box length in meters.
        grid_wsp : gutl.GridWorkspace
            Grid workspace object.
        component_names : list
            List of component names.
        components : dict
            Dictionary of component objects.
        """
        self.N                = jnp.array(nside_grid)
        self.Lbox             = jnp.array(box_length_in_m)
        self.site_altitude    = site_altitude
        self.session_time     = time_utc,
        self.site_coordinates = jnp.array(site_coordinates)
        self.grid_wsp         = gutl.GridWorkspace(
                                                N=self.N, 
                                                Lbox=self.Lbox, 
                                                site_altitude=self.site_altitude
                                                )
        
        self.super_grid       = super_grid.SuperGrid(
                                                geopotential_file=geopotential_file_era5,
                                                z_max=self.Lbox[2],
                                                Nz=self.N[2],
                                                time_utc=self.session_time,
                                                site_coordinates=self.site_coordinates,
                                                site_altitude=self.site_altitude
                                                )
        self.atm_calibrator   = calibration.AtmosphereCalibrator(
                                                super_grid=self.super_grid,
                                                temperature_file=temperature_file_era5,
                                                sp_humidity_file=spec_humidity_file_era5,
                                                apexdatafile=apex_datafile,
                                                )
        
        self.component_names  = []
        self.components       = {}
        self.component_mean   = {}

    def _add_component(
        self,
        field_name: str,
        field_unit: str,
        pspec: dict,
        zscale: dict,
        seed: int,
        mean: dict = None,
        nsub: int = 1024**3
    ) -> None:
        
        """
        Add a component to the atmosphere.

        Parameters
        ----------
        field_name : str
            Name of the component.
        field_unit : str
            Unit of the component.
        pspec : dict
            Power spectrum of the component.
        rescale : dict
            Rescaling factors as a function of height.
        seed : int
            Random seed.
        nsub : int, optional
            Number of subsamples for random number generation. Defaults to 1024**3.
        """
        
        self.component_names.append(field_name)
        self.components[field_name] = box.Box(
                                        grid_wsp=self.grid_wsp,
                                        field_name=field_name,
                                        field_unit=field_unit,
                                        spectrum=pspec,
                                        zscaling=zscale,
                                        seed=seed,
                                        nsub=nsub,                               
                                    )
        self.component_mean[field_name] = mean

    
    def add_temperature(self,
                        power_spec: dict, 
                        seed: int = 13579, 
                        ) -> None :
        self._add_component(
            field_name='temperature',
            field_unit='K',
            pspec=power_spec,
            zscale={'h': self.super_grid.z, 'f':self.atm_calibrator.temp_fluctuation_profile},
            seed=seed,
            mean= {'h': self.super_grid.z, 'f': self.atm_calibrator.temperature_profile},
        )
        
    def add_watervapor(self,
                       power_spec:dict,
                       seed: int = 24680,
                       ) -> None :
        self._add_component(
            field_name='water vapor',
            field_unit='kg / m^3',
            pspec=power_spec,
            zscale={'h': self.super_grid.z, 'f':self.atm_calibrator.spec_humidity_fluctuation_profile*self.atm_calibrator.q2rho_h2o},
            seed=seed,
            mean= {'h': self.super_grid.z, 'f': self.atm_calibrator.spec_humidity_profile*self.atm_calibrator.q2rho_h2o},
        )

    def generate_realization(
        self,
        time_step: int = 0,
        component_name: str = None
    ) -> None:
        
        """
        Generate a realization of the atmospheric component(s).

        Parameters
        ----------
        time_step : int, optional
            Time step of the realization. Defaults to 0.
        component_name : str, optional
            Name of the component to generate. If not given, all components are generated.
        """
        if component_name in self.component_names:
            self.components[component_name].generate_field_realization(time_step=time_step)
            if component_name == 'water vapor': self.atm_calibrator.calibrate_pwv(self.grid_wsp.grid_axis(axis=2, altitude_axis=True), self.components[component_name].field)
        else:
            for component in self.components.values():
                component.generate_field_realization(time_step=time_step)
                if component.field_name == 'water vapor': self.atm_calibrator.calibrate_pwv(self.grid_wsp.grid_axis(axis=2, altitude_axis=True), component.field)
                