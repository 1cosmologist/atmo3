import jax
import sys
import os
import gc

from . import parallel_rng as rng
from . import multihost_fft as mfft
from .grid_utils import GridWorkspace

import jax.numpy as jnp 
import jax.random as rnd

class Cube:
    '''A class to generate a 3D field realization of a physical variable given a power spectrum.'''
    def __init__(
        self, 
        N: int = 128,
        Lbox: float = 10000.0, # in m
        partype: str = 'jaxshard',
        grid_wsp: type[GridWorkspace] = None,
        field_name: str = 'pwv',
        field_unit: str = 'mm',
        pspec: dict = {},
        rescale: dict = {},
        seed: int = 123456789,
        nsub: int = 1024**3,
    ) -> None:

        self.N          = N
        self.Lbox       = Lbox # in m
        self.partype    = partype
        self.grid_wsp   = grid_wsp
        self.field_name = field_name
        self.field_unit = field_unit
        self.pspec      = pspec
        self.rescale    = rescale
        self.seed       = seed
        self.nsub       = nsub
        
        self.rng_stream = rng.Parallel_rng(seedkey=self.seed,nsub=self.nsub)

    def _generate_sharded_noise(self, N):           
        start   = self.start
        end     = self.end

        noise = self.rng_stream.generate(start=start*N**2,size=(end-start)*N**2).astype(jnp.float32)
        noise = jnp.reshape(noise,(end-start,N,N))
        return jnp.transpose(noise,(1,0,2)) 

    def _generate_serial_noise(self, N):
        
        noise = self.rng_stream.generate(start=0,size=N**3).astype(jnp.float32)
        noise = jnp.reshape(noise,(N,N,N))
        return jnp.transpose(noise,(1,0,2))

    def _apply_grid_transfer_function(self, field, transfer_data):
        transfer_cdm = self.grid_wsp.interp2kgrid(transfer_data[0], transfer_data[1])
        del transfer_data ; gc.collect()

        return field*transfer_cdm

    def _generate_noise(self):
        N = self.N

        if self.partype is None:
            self.field = self._generate_serial_noise(N)
        elif self.partype == 'jaxshard':
            self.field = self._generate_sharded_noise(N)

    def _noise2field(self):
        import numpy as np
        power    = np.asarray([self.pspec['k'],self.pspec['pofk']])
        transfer = power
        p_whitenoise = (2*np.pi)**3/(self.d3k*self.N**3) # white noise power spectrum
        transfer[1] = (power[1] / p_whitenoise)**0.5 # transfer(k) = sqrt[P(k)/P_whitenoise]
        transfer = jnp.asarray(transfer)

        self.field = mfft.fft(
                    self._apply_grid_transfer_function(mfft.fft(self.field), transfer),
                    direction='c2r')
        
    def _rescale_field(self):
        rescale_interp = self.interp2grid(self.rescale['h'], self.rescale['f'])
        self.field *= rescale_interp

    def generate_field_realization(self, time_step=0):
        self.rng_stream.set_seedkey(time_step)
        self._generate_noise()
        self._noise2field()
        self._rescale_field()
        
    def compute_emission(self):
        pass
        
        