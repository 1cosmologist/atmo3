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
        
        if not (self.grid_wsp.Lbox[0]/4 < boresight[0] < 3*self.grid_wsp.Lbox[0]/4):
            raise ValueError("Boresight x-coord is incorrectly set.")    
        
        if not (self.grid_wsp.Lbox[1]/4 < boresight[1] < 3*self.grid_wsp.Lbox[1]/4):
            raise ValueError("Boresight y-coord is incorrectly set.")  
        
        self.boresight = boresight 
        
        if not isinstance(passband, dict):
            raise ValueError("Passband required! Format: dict{'freq_GHz', 'g_nu'}.")

        self.passband = passband
        ### TODO: Normalization of the passband is TBD. Assume it is normalized and in units of K_RJ
        
        if not isinstance(fwhm_arcmin, float):
            raise ValueError("FWHM in arcmin required!")        
        
        self.fwhm_arcmin = fwhm_arcmin
        
        self.axes = (self.grid_wsp.grid_axis(axis=0), self.grid_wsp.grid_axis(axis=1), self.grid_wsp.grid_axis(axis=2, altitude_axis=True))
        
        
        ### TODO: Calculate max time before refreshing simulation
        
    def compute_los_for_scan(
        self,
        timelist: list,
        azimuth_deg: list,
        elevation_deg: list
    ):
        """
        Computes the line-of-sight coordinates and properties for a sequence of observations.
        
        Parameters
        ----------
        timelist : list
            List of datetime-like objects representing the observation times.
        azimuth_deg : list
            List of azimuth angles in degrees, length matches timelist representing n_scans.
        elevation_deg : list
            List of elevation angles in degrees, length matches timelist representing n_scans.
            
        Returns
        -------
        None
            Sets `self.los_obj`, a JAX array of shape (n_scans, n_los, 5).
            The last dimension contains: [x_los, y_los, altitude_slice, radius, max_radius_mask].
        """
        timearray = np.array(timelist, dtype='datetime64[ns]')
        delta_t_in_s   = (timearray - timearray[0]) / np.timedelta64(1, 's')
        
        azimuth   = jnp.deg2rad(jnp.array(azimuth_deg))
        elevation = jnp.deg2rad(jnp.array(elevation_deg))
        
        x = jnp.cos(elevation) * jnp.cos(azimuth)
        y = jnp.cos(elevation) * jnp.sin(azimuth)
        z = jnp.sin(elevation)
        
        pos_vec = jnp.array([x, y, z]) / jnp.sqrt(x**2. + y**2. + z**2.)
        alt_arr = self.grid_wsp.grid_axis(axis=2, altitude_axis=True)
        
        self.los_obj = jax.vmap(
            lambda uv, dt: obs_utils.los_points_coords_radius(
                self.grid_wsp.site_altitude,
                self.grid_wsp.Lbox,
                alt_arr, 
                uv, 
                self.boresight,
                north_wind=self.north_wind,
                east_wind=self.east_wind,
                delta_t=dt, 
                max_radius=True
            ),
            in_axes=(1, 0)
        )(pos_vec, jnp.array(delta_t_in_s))
        
    def scan_component(
        self,
        component_field: jnp.ndarray
    ):
        interpol = jsp.interpolate.RegularGridInterpolator(self.axes, component_field, fill_value=0., method='linear')
        shape = self.los_obj.shape
        return interpol(self.los_obj[:,:,0:3].reshape(shape[0]*shape[1], 3)).reshape(shape[0], shape[1])
        
        
        