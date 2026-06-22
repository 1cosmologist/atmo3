import jax
from functools import partial
import jax.numpy as jnp
import gc


class GridWorkspace:
   
    def __init__(self, N, Lbox, site_altitude=0.0):
        """
        Initialize the grid workspace object.

        Parameters
        ----------
        N : int
            Number of cells per side of the grid.
        Lbox : float
            Box length in meters.
        partition : str, optional
            Type of partitioning. Defaults to 'jaxshard'.

        Attributes
        ----------
        N : int
            Number of cells per side of the grid.
        Lbox : float
            Box length in meters.
        dk : float
            Grid spacing in k-space.
        d3k : float
            Grid spacing in k-space cubed.
        rshape : tuple
            Shape of a real-space array.
        cshape : tuple
            Shape of a complex-space array.
        rshape_local : tuple
            Shape of a local real-space array.
        cshape_local : tuple
            Shape of a local complex-space array.
        start : int
            Starting index of the local array.
        end : int
            Ending index of the local array.
        grid_spacing : float
            Grid spacing in real-space.
        field : None
            Placeholder for the field array.
        ngpus : int
            Number of GPUs.
        host_id : int
            Host ID.
        parttype : str
            Type of partitioning.
        partaxis : int
            Axis to partition the array.
        """
        self.N     = N
        
        self.Lbox   = Lbox
        
        self.site_altitude = site_altitude  # Default altitude, can be set later if needed
        
        self.dk   = 2 * jnp.pi / self.Lbox
        
        if isinstance(self.dk, jnp.ndarray):
            self.d3k = self.dk[0] * self.dk[1] * self.dk[2]
        else:
            self.d3k     = self.dk**3.

        if isinstance(self.N, jnp.ndarray):
            self.rshape = (self.N[0], self.N[1], self.N[2])
            self.cshape = (self.N[0], self.N[1], self.N[2]//2+1)
        else:
            self.rshape = (self.N, self.N, self.N)
            self.cshape = (self.N, self.N, self.N//2+1)
        
        self.grid_spacing = self.Lbox / self.N
        
        
    @staticmethod
    @partial(jax.jit, static_argnames=('N_i', 'r'))
    def _jit_k_axis(dk_i, N_i, r=False):
        if r:
            return jnp.fft.rfftfreq(N_i) * dk_i * N_i
        return jnp.fft.fftfreq(N_i) * dk_i * N_i

    def k_axis(self, axis, r=False):
        N_tup  = tuple(int(x) for x in jnp.atleast_1d(self.N).tolist())
        dk_arr = jnp.broadcast_to(jnp.asarray(self.dk), (3,))
        N_i    = N_tup[0] if len(N_tup) == 1 else N_tup[axis]
        return self._jit_k_axis(dk_arr[axis], N_i, r)
    
    def k_square(self, kx, ky, kz):
        kxa,kya,kza = jnp.meshgrid(kx,ky,kz,indexing='ij')
        
        del kx, ky, kz ; gc.collect()

        k2 = (kxa**2+kya**2+kza**2)#.astype(jnp.float32)
        del kxa, kya, kza ; gc.collect()

        return k2
    
    @staticmethod
    @partial(jax.jit, static_argnames=('N', 'cshape'))
    def _jit_interp2kgrid(dk, k_1d, f_1d, N, cshape):
        kx = GridWorkspace._jit_k_axis(dk[0], N[0])
        ky = GridWorkspace._jit_k_axis(dk[1], N[1])
        kz = GridWorkspace._jit_k_axis(dk[2], N[2], r=True)
        kxa, kya, kza = jnp.meshgrid(kx, ky, kz, indexing='ij')
        interp_fcn = jnp.sqrt(kxa**2 + kya**2 + kza**2).ravel()
        interp_fcn = jnp.interp(interp_fcn, k_1d, f_1d, left=0., right='extrapolate')
        return jnp.reshape(interp_fcn, cshape)

    def interp2kgrid(self, k_1d, f_1d):
        N      = tuple(int(x) for x in jnp.atleast_1d(self.N).tolist())
        dk     = jnp.broadcast_to(jnp.asarray(self.dk), (3,))
        cshape = tuple(int(x) for x in self.cshape)
        return self._jit_interp2kgrid(dk, k_1d, f_1d, N, cshape)
    
    @staticmethod
    @partial(jax.jit, static_argnames=('N_i', 'altitude_axis'))
    def _jit_grid_axis(grid_spacing_i, site_altitude, N_i, altitude_axis=False):
        ax = jnp.arange(N_i) * grid_spacing_i
        if altitude_axis:
            return site_altitude + ax
        return ax

    def grid_axis(self, axis, altitude_axis=False):
        N_tup = tuple(int(x) for x in jnp.atleast_1d(self.N).tolist())
        gs    = jnp.broadcast_to(jnp.asarray(self.grid_spacing), (3,))
        N_i   = N_tup[0] if len(N_tup) == 1 else N_tup[axis]
        return self._jit_grid_axis(gs[axis], jnp.asarray(self.site_altitude), N_i, altitude_axis)

    @staticmethod
    @partial(jax.jit, static_argnames=('N', 'rshape'))
    def _jit_interp2grid(grid_spacing, site_altitude, x_1d, f_1d, N, rshape):
        x  = GridWorkspace._jit_grid_axis(grid_spacing[0], site_altitude, N[0])
        y  = GridWorkspace._jit_grid_axis(grid_spacing[1], site_altitude, N[1])
        z  = GridWorkspace._jit_grid_axis(grid_spacing[2], site_altitude, N[2], altitude_axis=True)
        _xx, _yy, zz = jnp.meshgrid(x, y, z, indexing='ij')
        interp_fcn = jnp.interp(zz.ravel(), x_1d, f_1d, left='extrapolate', right='extrapolate')
        return jnp.reshape(interp_fcn, rshape)

    def interp2grid(self, x_1d, f_1d):
        N      = tuple(int(x) for x in jnp.atleast_1d(self.N).tolist())
        N      = N * 3 if len(N) == 1 else N
        gs     = jnp.broadcast_to(jnp.asarray(self.grid_spacing), (3,))
        rshape = tuple(int(x) for x in self.rshape)
        return self._jit_interp2grid(gs, jnp.asarray(self.site_altitude), x_1d, f_1d, N, rshape)
    

