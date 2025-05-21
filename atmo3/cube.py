import jax
import sys
import os
import gc

from . import parallel_rng as rng
from . import multihost_fft as mfft

import jax.numpy as jnp 
import jax.random as rnd

class Cube:
    '''Cube'''
    def __init__(self, **kwargs):

        self.N          = kwargs.get('N', 128)
        self.Lbox       = kwargs.get('Lbox',10000.0) # in m
        self.partype    = kwargs.get('partype','jaxshard')
        self.field_name = kwargs.get('field_name','pwv')
        self.field_unit = kwargs.get('field_unit','mm')
        self.pspec      = kwargs.get('pspec',{})
        
        self.seed            = kwargs.get('seed', 123456789)
        self.nsub            = kwargs.get('nsub', 1024**3)

        self.dk  = 2*jnp.pi/self.Lbox
        self.d3k = self.dk * self.dk * self.dk

        self.rshape       = (self.N,self.N,self.N)
        self.cshape       = (self.N,self.N,self.N//2+1)
        self.rshape_local = (self.N,self.N,self.N)
        self.cshape_local = (self.N,self.N,self.N//2+1)

        self.start = 0
        self.end   = self.N
        
        self.field = None

        # needed for running on CPU with a signle process
        self.ngpus   = 1        
        self.host_id = 0

        if self.partype == 'jaxshard':
            self.ngpus   = jax.device_count()
            self.host_id = jax.process_index()
            self.start   = self.host_id * self.N // self.ngpus
            self.end     = (self.host_id + 1) * self.N // self.ngpus
            self.rshape_local = (self.N, self.N // self.ngpus, self.N)
            self.cshape_local = (self.N, self.N // self.ngpus, self.N // 2 + 1)
            
        self.rng_stream = rng.Parallel_rng(seedkey=self.seed,nsub=self.nsub)

    def k_axis(self, r=False, slab_axis=False):
        if r: 
            k_i = (jnp.fft.rfftfreq(self.N) * self.dk * self.N).astype(jnp.float32)
        else:
            k_i = (jnp.fft.fftfreq(self.N) * self.dk * self.N).astype(jnp.float32)
        if slab_axis: return (k_i[self.start:self.end]).astype(jnp.float32)
        return k_i
    
    def k_square(self, kx, ky, kz):
        kxa,kya,kza = jnp.meshgrid(kx,ky,kz,indexing='ij')
        del kx, ky, kz ; gc.collect()

        k2 = (kxa**2+kya**2+kza**2).astype(jnp.float32)
        del kxa, kya, kza ; gc.collect()

        return k2
    
    def interp2kgrid(self, k_1d, f_1d):
        kx = self.k_axis()
        ky = self.k_axis(slab_axis=True)
        kz = self.k_axis(r=True)

        interp_fcn = jnp.sqrt(self.k_square(kx, ky, kz)).ravel()
        del kx, ky, kz ; gc.collect()

        interp_fcn = jnp.interp(interp_fcn, k_1d, f_1d, left='extrapolate', right='extrapolate')
        return jnp.reshape(interp_fcn, self.cshape_local).astype(jnp.float32)

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
        transfer_cdm = self.interp2kgrid(transfer_data[0], transfer_data[1])
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

    def generate_field_realization(self, time_step = 0):
        self.rng_stream.set_seedkey(time_step)
        self._generate_noise()
        self._noise2field()