import jax.numpy as jnp 
import jax.random as rnd
import jax
import os

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

    def generate(self,**kwargs):

        if self.force_no_gpu:
            _JAX_PLATFORM_NAME = jax.default_backend()
            jax.default_device("cpu")

        start = kwargs.get('start',0)
        size  = kwargs.get('size' ,1)
        dist  = kwargs.get('dist','normal')

        # if self.dtype in [jnp.float64, jnp.complex128]:
        _JAX_X64_INITIAL_STATE = jax.config.read('jax_enable_x64')
        jax.config.update('jax_enable_x64', True)

        end = start + size - 1

        start_seqID = start // self.nsub
        end_seqID   = end   // self.nsub

        seqIDs = jnp.arange(start_seqID, end_seqID+1)

        keys = jax.vmap(rnd.fold_in, in_axes=(None, 0), out_axes=0)(self._PRNGkey, seqIDs)

        seq = jnp.zeros(0,dtype=self.dtype)

        for seqID in seqIDs:

            subseq_start = max(seqID * self.nsub, start) - seqID * self.nsub
            subseq_end   = min((seqID+1) * self.nsub -1,end) - seqID * self.nsub

            if dist == 'normal':
                subseq = rnd.normal(keys[seqID], dtype=self.dtype, shape=(self.nsub,))

            seq = jnp.concatenate((seq,subseq[subseq_start:subseq_end+1]))
        
        if self.force_no_gpu:
            jax.default_device(_JAX_PLATFORM_NAME)

        # if self.dtype in [jnp.float64, jnp.complex128, jnp.int64]:
        jax.config.update('jax_enable_x64', _JAX_X64_INITIAL_STATE)

        return seq






