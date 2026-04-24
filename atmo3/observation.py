import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from functools import partial
from .grid_utils import GridWorkspace
from .super_grid import SuperGrid
from . import obs_utils

class Observer:
    
    def __init__(
        self,
        grid_wsp: type[GridWorkspace] = None,
        super_grid: type[SuperGrid] = None,
        northwind_era5_file: str = None,
        eastwind_era5_file: str = None,
        boresight: jnp.ndarray = None,       # in grid coordinates
        passband: dict = None, 
        fwhm_arcmin: float = None
    ):
        self.grid_wsp = grid_wsp
        self.super_grid = super_grid
        
        self.north_wind = None; self.east_wind = None
        
        if (not northwind_era5_file == None) and (not eastwind_era5_file == None):
            self.north_wind = self.super_grid.era5_interp2site(northwind_era5_file)
            self.east_wind  = self.super_grid.era5_interp2site(eastwind_era5_file)
            
        
        if not isinstance(boresight, jnp.ndarray):
            raise TypeError("Boresight has to provided.")
        
        if not (self.grid_wsp.N[0]//2 < boresight[0] < 3*self.grid_wsp.N[0]//2):
            raise ValueError("Boresight x-coord is incorrectly set.")    
        
        if not (self.grid_wsp.N[1]//2 < boresight[1] < 3*self.grid_wsp.N[1]//2):
            raise ValueError("Boresight y-coord is incorrectly set.")  
        
        self.boresight = boresight 
        
        if not isinstance(passband, dict):
            raise ValueError("Passband required! Format: dict{'freq_GHz', 'g_nu'}.")

        self.passband = passband
        ### TODO: Normalization of the passband is TBD. Assume it is normalized and in units of K_RJ
        
        if not isinstance(fwhm_arcmin, float):
            raise ValueError("FWHM in arcmin required!")        
        
        self.fwhm_arcmin = fwhm_arcmin
        
        self.axes = (self.grid_wsp.grid_axis(axis=0), self.grid_wsp.grid_axis(axis=1), self.grid_wsp.grid_axis(axis=2))
        
        
        ### TODO: Calculate max time before refreshing simulation
        
    def compute_los_for_scan(
        self,
        timelist: list,
        azimuth_deg: list,
        elevation_deg: list
    ):
        timearray = np.array(timelist, dtype='datetime64[ns]')
        delta_t_in_s   = (timearray - timearray[0]) / np.timedelta64(1, 's')
        
        azimuth   = jnp.deg2rad(jnp.array(azimuth_deg))
        elevation = jnp.deg2rad(jnp.array(elevation_deg))
        
        x = jnp.cos(elevation) * jnp.cos(azimuth)
        y = jnp.cos(elevation) * jnp.sin(azimuth)
        z = jnp.sin(elevation)
        
        pos_vec = jnp.array([x, y, z]) / jnp.sqrt(x**2. + y**2. + z**2.)
        alt_arr = self.grid_wsp.grid_axis(axis=2, altitude_axis=True)
        
        self.los_obj = []
        for sample in range(len(delta_t_in_s)):
            self.los_obj.append(obs_utils.los_points_coords_radius(
                        self.grid_wsp.site_altitude,
                        self.grid_wsp.Lbox,
                        alt_arr, 
                        pos_vec[:,sample], 
                        self.boresight,
                        north_wind = self.north_wind,
                        east_wind = self.east_wind,
                        delta_t = delta_t_in_s, 
                        max_radius = True
                    ))
            
        self.los_obj
        
    def scan_component(
        self,
        component_field: jnp.ndarray
    ):
        interpol = jsp.intepolate.RegularGridInterpolator(self.axes, component_field, fill_value=0.)
        return interpol(self.los_obj[:,:,0:3])
        
        
        