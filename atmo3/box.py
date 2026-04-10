import jax 
from functools import partial
import jax.numpy as jnp
import gc 

from . import parallel_rng as rng
from .grid_utils import GridWorkspace

class Box:
    '''A class to generate a 3D field realization of a physical variable given a power spectrum.'''
    def __init__(
        self, 
        grid_wsp: type[GridWorkspace] = None,
        field_name: str = 'specific humidity',
        field_unit: str = 'g kg^-1',
        spectrum: dict = None,
        zscaling: dict = None,
        seed: int = 123456789,
        nsub: int = 1024**3,
    ) -> None:
        self.grid_wsp   = grid_wsp
        self.field_name = field_name
        self.field_unit = field_unit
        self.spectrum   = jnp.array([spectrum['k'], spectrum['pofk']])
        self.zscaling   = zscaling
        self.seed       = seed
        self.nsub       = nsub
        
        self.rng_stream = rng.Parallel_rng(seedkey=self.seed, nsub=self.nsub)

    @partial(jax.jit, static_argnames=['self'])
    def _random_delta_k(self):
        delta_k = self.rng_stream.generate(start=0, size=self.grid_wsp.cshape[0]*self.grid_wsp.cshape[1]*self.grid_wsp.cshape[2]) \
                    + 1j * self.rng_stream.generate(start=0, size=self.grid_wsp.cshape[0]*self.grid_wsp.cshape[1]*self.grid_wsp.cshape[2])
        delta_k = jnp.reshape(delta_k, self.grid_wsp.cshape)
        delta_k[0,0,0] = 0. + 0.j  # Set the DC mode to zero
        
        return delta_k
    
    @partial(jax.jit, static_argnames=['self'])
    def _apply_grid_transfer_function(self, delta_k):
        spectrum_white = (2. * jnp.pi)**3 / (self.grid_wsp.N[0]*self.grid_wsp.N[1] * self.grid_wsp.N[2] * self.grid_wsp.d3k)  # Power spectrum of white noise
        transfer_function = jnp.sqrt(self.spectrum[1] / spectrum_white)
        transfer_interp = self.grid_wsp.interp2kgrid(transfer_function, self.spectrum[0])
        del transfer_function ; gc.collect()

        return delta_k * transfer_interp
    
    @partial(jax.jit, static_argnames=['self'])
    def _rescale_field(self, field):
        
        z = self.grid_wsp.grid_axis(2, altitude_axis=True)
        scaling_interp = self.grid_wsp.interp2grid(z, self.zscaling['z'], self.zscaling['f'])
        del z ; gc.collect()
        return field * scaling_interp
    
    @partial(jax.jit, static_argnames=['self'])
    def generate_field_fluctuations(self, time_step=0):
        self.rng_stream.set_seedkey(time_step)
        
        delta_k = self._random_delta_k()
        delta_k = self._apply_grid_transfer_function(delta_k)
        
        delta_r = jnp.fft.irfftn(delta_k, s=self.grid_wsp.rshape)
        del delta_k ; gc.collect()
        
        self.field = delta_r / jnp.std(delta_r)  # Normalize to unit variance
        del delta_r ; gc.collect()
        
        self.field = self._rescale_field(self.field)
        