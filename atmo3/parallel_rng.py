import jax.numpy as jnp 
import jax.random as rnd
import jax
import os
from functools import partial

class Parallel_rng:
    '''
    Parallel random number generator. This class is used to generate random numbers in parallel on multiple processes.
    '''
    def __init__(self, **kwargs):

        self.force_no_gpu = kwargs.get('force_no_gpu',False)
        self._seedkey     = kwargs.get('seedkey', rnd.PRNGKey(123456789))
        if isinstance(self._seedkey, int):
            self._seedkey = rnd.PRNGKey(self._seedkey)
        self._PRNGkey      = self._seedkey
        self.nsub         = kwargs.get('nsub',1024**3)
        self.dtype        = kwargs.get('dtype', jnp.float32)

    def set_seedkey(self, mc):
        if isinstance(mc, int):
            self._PRNGkey = rnd.fold_in(self._seedkey, mc)
        else: 
            print('ERROR: mc must be an integer')
            exit()

    @staticmethod
    @partial(jax.jit, static_argnames=('num_seqs', 'size', 'nsub', 'dtype', 'dist'))
    def _jit_generate(prng_key, start_seqID, offset, num_seqs, size, nsub, dtype, dist):
        # seqIDs built from dynamic start_seqID + static-length arange
        seqIDs = start_seqID + jnp.arange(num_seqs)

        keys = jax.vmap(rnd.fold_in, in_axes=(None, 0), out_axes=0)(prng_key, seqIDs)

        if dist == 'normal':
            all_blocks = jax.vmap(
                lambda key: rnd.normal(key, dtype=dtype, shape=(nsub,))
            )(keys)

        # dynamic_slice: start_indices dynamic, slice_sizes static
        flat = all_blocks.reshape(-1)
        return jax.lax.dynamic_slice(flat, (offset,), (size,))

    def generate(self, **kwargs):

        if self.force_no_gpu:
            _JAX_PLATFORM_NAME = jax.default_backend()
            jax.default_device("cpu")

        start = kwargs.get('start', 0)
        size  = kwargs.get('size' , 1)
        dist  = kwargs.get('dist' , 'normal')

        _JAX_X64_INITIAL_STATE = jax.config.read('jax_enable_x64')
        jax.config.update('jax_enable_x64', True)

        end         = start + size - 1
        start_seqID = start // self.nsub
        end_seqID   = end   // self.nsub
        num_seqs    = int(end_seqID - start_seqID + 1)
        offset      = start - start_seqID * self.nsub

        seq = self._jit_generate(
            self._PRNGkey,
            jnp.int32(start_seqID),
            jnp.int32(offset),
            num_seqs,
            int(size),
            int(self.nsub),
            self.dtype,
            dist,
        )

        if self.force_no_gpu:
            jax.default_device(_JAX_PLATFORM_NAME)

        jax.config.update('jax_enable_x64', _JAX_X64_INITIAL_STATE)

        return seq






