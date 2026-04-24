import jax.scipy as jsp
import jax.numpy as jnp
import numpy as np
from . import constants as con
import xarray as xr

class SuperGrid:
    '''A class to compute the average of a field over the horizontal planes.'''
    
    def __init__(self, geopotential_file, z_max, Nz, time_utc, site_lonlat, site_altitude=0.0):
        dataset = xr.open_dataarray(geopotential_file)
        
        self.lat  = jnp.asarray(dataset.latitude.values)
        self.lon  = jnp.asarray(dataset.longitude.values)
        self.plev = jnp.asarray(dataset.level.values).sort(descending=True) * 100. # convert from hPa to Pa
        
        self.site_coordinates = site_lonlat
        self.time_utc = time_utc
        
        self.geopotential = dataset.sel(time=f'{self.time_utc[0]:%Y-%m-%dT%H:%M:%S}')
        self.zgeo_atsite = jnp.asarray(self.geopotential.interp(latitude=self.site_coordinates[1], longitude=self.site_coordinates[0] + 360) / con.g)[::-1]
        self.z_geo = jnp.asarray(self.geopotential / con.g)[::-1]
        self.site_altitude = site_altitude
        
        self.x = 2 * con.earth_radius * jnp.cos(jnp.deg2rad(self.site_coordinates[1])) * jnp.sin(jnp.deg2rad(self.lon - self.site_coordinates[0])/2.)
        self.y = 2 * con.earth_radius * jnp.sin(jnp.deg2rad(self.lat - self.site_coordinates[1])/2.)
        self.z = site_altitude + jnp.linspace(0, z_max, Nz)
        
        # print(self.z_geo.shape, self.zgeo_atsite.shape)
        
        # for i in range(len(self.plev)):
        #     print(i, self.plev[i], self.zgeo_atsite[i])
        
        self.pressure = jnp.interp(self.z, self.zgeo_atsite, self.plev, left='extrapolate', right='extrapolate') # in Pa
        # print(self.pressure)
        
    def era5_interp2site(self, filepath):
        dataset = xr.open_dataarray(filepath)
        interp_prof = jnp.asarray(dataset.sel(time=f'{self.time_utc[0]:%Y-%m-%dT%H:%M:%S}').interp(latitude=self.site_coordinates[1], longitude=self.site_coordinates[0] + 360))[::-1]
        
        return jnp.interp(self.z, self.zgeo_atsite, interp_prof, left='extrapolate', right='extrapolate')
        
    def property_from_era5(self, filepath):
        '''Compute the average of a field over the horizontal planes.'''
        # Interpolate the field to the supergrid
        dataset = xr.open_dataarray(filepath)
        data = jnp.asarray(dataset.sel(time=f'{self.time_utc[0]:%Y-%m-%dT%H:%M:%S}'))[::-1]
            
        property = np.zeros((len(self.x), len(self.y), len(self.z)))
        
        for i in range(len(self.lat)):
            for j in range(len(self.lon)):
                property[i, j] = jnp.interp(self.z, self.z_geo[:,i,j], data[:,i, j], left='extrapolate', right='extrapolate')
    
        
        return jnp.asarray(property)

    def interpolate_property(self, property):
        '''Interpolate the property to the given points.'''
        return jsp.interpolate.RegularGridInterpolator((self.x, self.y, self.z), property.values, fill_value=0.)
        


        