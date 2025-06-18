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
        partition: str = 'jaxshard',
        grid_wsp: type[GridWorkspace] = None,
        field_name: str = 'water mass density',
        field_unit: str = 'g m^-3',
        pspec: dict = {},
        rescale: dict = {},
        seed: int = 123456789,
        nsub: int = 1024**3,
    ) -> None:

        """
        Initialize the cube object.

        Parameters
        ----------
        N : int, optional
            Number of cells per side of the grid. Defaults to 128.
        Lbox : float, optional
            Box length in meters. Defaults to 10000.0.
        partition : str, optional
            Partition type. Defaults to 'jaxshard'.
        grid_wsp : GridWorkspace, optional
            Grid workspace object. Defaults to None.
        field_name : str, optional
            Name of the physical variable. Defaults to 'water mass density'.
        field_unit : str, optional
            Unit of the physical variable. Defaults to 'g m^-3'.
        pspec : dict, optional
            Power spectrum of the physical variable. Defaults to an empty dictionary.
        rescale : dict, optional
            Rescaling factors as a function of height. Defaults to an empty dictionary.
        seed : int, optional
            Random seed. Defaults to 123456789.
        nsub : int, optional
            Number of subsamples for random number generation. Defaults to 1024**3.
        """
        self.N          = N
        self.Lbox       = Lbox # in m
        self.grid_wsp   = grid_wsp
        self.field_name = field_name
        self.field_unit = field_unit
        self.pspec      = pspec
        self.rescale    = rescale
        self.seed       = seed
        self.nsub       = nsub
        
        self.rng_stream = rng.Parallel_rng(seedkey=self.seed,nsub=self.nsub)

    def _generate_sharded_noise(self, N):           
        start   = self.grid_wsp.start
        end     = self.grid_wsp.end

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

        if self.grid_wsp.parttype is None:
            self.field = self._generate_serial_noise(N)
        elif self.grid_wsp.parttype == 'jaxshard':
            self.field = self._generate_sharded_noise(N)

    def _noise2field(self):
        import numpy as np
        power    = np.asarray([self.pspec['k'],self.pspec['pofk']])
        transfer = power
        p_whitenoise = (2*np.pi)**3/(self.grid_wsp.d3k*self.N**3) # white noise power spectrum
        transfer[1] = (power[1] / p_whitenoise)**0.5 # transfer(k) = sqrt[P(k)/P_whitenoise]
        transfer = jnp.asarray(transfer)

        self.field = mfft.fft(
                    self._apply_grid_transfer_function(mfft.fft(self.field), transfer),
                    direction='c2r')
        
    def _rescale_field(self):
        rescale_interp = self.grid_wsp.interp2grid(self.rescale['h'], self.rescale['f'])
        self.field *= rescale_interp

    def generate_field_realization(self, time_step=0):
        """
        Generate a field realization for the Cube instance at a specified time step.

        This method updates the random number generator seed based on the provided
        time step and generates a 3D field realization of a physical variable using
        the power spectrum, noise generation, and rescaling factors defined for the
        Cube instance.

        Parameters
        ----------
        time_step : int, optional
            The time step for which the field realization is generated. Defaults to 0.
        """

        self.rng_stream.set_seedkey(time_step)
        self._generate_noise()
        self._noise2field()
        self._rescale_field()       