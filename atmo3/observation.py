import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from functools import partial

from .grid_utils import GridWorkspace
from .atmosphere import Atmosphere
from . import obs_utils
from .profiles import wind_on_grid
from .emission import get_emission
from .emission_utils import intensitySI_to_Tb

class Observer:
    
    def __init__(
        self,
        grid_wsp: type[GridWorkspace] = None,
        northwind: float | dict = None,
        eastwind: float | dict = None,
        boresight: jnp.ndarray = None,       # in grid coordinates
        passband: dict = None, 
        fwhm_arcmin: float = None
    ):
        """
        Parameters
        ----------
        grid_wsp : GridWorkspace
            Grid workspace of the simulated atmosphere.
        northwind, eastwind : float or dict, optional
            North and east wind components (m/s).  A float is a wind
            independent of height; a dict is a height-dependent profile
            ``{'z': altitude above sea level (m), 'northwind' or
            'eastwind': wind (m/s)}``, interpolated onto the grid
            altitude axis.  If only one component is given the other is
            taken to be zero; if both are ``None`` no wind is applied.
        boresight : jnp.ndarray
            Boresight position in grid coordinates (m).
        passband : dict
            Passband with the frequencies and response.
        fwhm_arcmin : float
            Beam FWHM in arcmin.
        """
        self.grid_wsp = grid_wsp
        
        altitude_axis = self.grid_wsp.grid_axis(axis=2, altitude_axis=True)
        
        self.north_wind = None; self.east_wind = None
        
        if northwind is not None or eastwind is not None:
            self.north_wind = wind_on_grid(northwind, altitude_axis, 'northwind')
            self.east_wind  = wind_on_grid(eastwind,  altitude_axis, 'eastwind')
            
        
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


    def scan_atmosphere_fluctuation(
        self,
        atmosphere: type[Atmosphere],
        component_name: str,
    ):
        """Sample a component's 3D fluctuation field along the lines of sight.

        Returns an array of shape ``(n_scans, n_los)`` in the component's
        physical units, or zero if its field has not been generated. Samples
        outside the grid are zero.
        """
        if component_name not in atmosphere.component_names:
            raise KeyError(f"{component_name} is not a component")

        component = atmosphere.components[component_name]
        if not hasattr(component, "field"):
            return jnp.zeros(self.los_obj.shape[:2])

        interpol = jsp.interpolate.RegularGridInterpolator(
            self.axes, component.field, fill_value=0., method='linear'
        )
        shape = self.los_obj.shape
        return interpol(self.los_obj[:, :, :3].reshape(shape[0] * shape[1], 3)).reshape(shape[0], shape[1])

    def scan_atmosphere(
        self,
        atmosphere: type[Atmosphere],
        component_name: str,
    ):
        """Sample the mean plus fluctuations along each line of sight.

        The mean profile is evaluated at each point's altitude (m), including
        for angled scans. Returns ``(n_scans, n_los)`` in the component's
        physical units; samples outside the grid are zero.
        """
        if component_name not in atmosphere.component_names:
            raise KeyError(f"{component_name} is not a component")

        coords = self.los_obj[:, :, :3]
        mean = jnp.interp(coords[:, :, 2], self.axes[2], atmosphere.component_mean[component_name]['f'])
        fluctuation = self.scan_atmosphere_fluctuation(atmosphere, component_name)
        lower = jnp.array([axis[0] for axis in self.axes])
        upper = jnp.array([axis[-1] for axis in self.axes])
        inside = jnp.all((coords >= lower) & (coords <= upper), axis=-1)
        return jnp.where(inside, mean + fluctuation, 0.)

    def observe_atmosphere(
        self,
        atmosphere: type[Atmosphere]
    ):
        """Return band-weighted atmospheric brightness temperatures (K_RJ).

        Samples temperature (K), pressure (Pa), and water-vapor density
        (kg m^-3) along each line of sight, then integrates emission over path
        length (m). ``passband['nu']`` is in GHz; ``passband['g(nu)']`` contains
        discrete weights that sum to one. Returns one value per scan.
        """
        pressure    = jnp.interp(self.los_obj[:,:,2], self.axes[2], atmosphere.atm_calibrator.pressure)
        temperature = self.scan_atmosphere(atmosphere, 'temperature')
        water_vapor = self.scan_atmosphere(atmosphere, 'water vapor')
        
        I_atm = get_emission(temperature, pressure, water_vapor, jnp.diff(self.los_obj[:,:,3], axis=1, prepend=0), self.passband['nu'])
        T_atm = intensitySI_to_Tb(I_atm[:,:,None], self.passband['nu'][None, None,:])[:,:,0]
        
        return jnp.sum(T_atm * self.passband['g(nu)'][None,:], axis=1)
