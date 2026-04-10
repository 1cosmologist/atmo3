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
        
    
    @partial(jax.jit, static_argnames=['self'])      
    def k_axis(self, axis, r=False):
        return jnp.where(r, (jnp.fft.rfftfreq(self.N[axis]) * self.dk[axis] * self.N[axis]), (jnp.fft.fftfreq(self.N[axis]) * self.dk[axis] * self.N[axis]))#.astype(jnp.float32)
    
    def k_square(self, kx, ky, kz):
        kxa,kya,kza = jnp.meshgrid(kx,ky,kz,indexing='ij')
        
        del kx, ky, kz ; gc.collect()

        k2 = (kxa**2+kya**2+kza**2)#.astype(jnp.float32)
        del kxa, kya, kza ; gc.collect()

        return k2
    
    @partial(jax.jit, static_argnames=['self'])
    def interp2kgrid(self, k_1d, f_1d):
        kx = self.k_axis(0)
        ky = self.k_axis(1)
        kz = self.k_axis(2, r=True)

        interp_fcn = jnp.sqrt(self.k_square(kx, ky, kz)).ravel()
        del kx, ky, kz ; gc.collect()

        interp_fcn = jnp.interp(interp_fcn, k_1d, f_1d, left=0., right='extrapolate')
        return jnp.reshape(interp_fcn, self.cshape_local)#.astype(jnp.float32)
    
    @partial(jax.jit, static_argnames=['self'])
    def grid_axis(self, axis, altitude_axis=False):
        
        return jnp.where(altitude_axis, self.site_altitude + (jnp.arange(self.N[axis]) * self.grid_spacing[axis]), (jnp.arange(self.N[axis]) * self.grid_spacing[axis]))#.astype(jnp.float32)
        
    @partial(jax.jit, static_argnames=['self'])
    def interp2grid(self, x_1d, f_1d):
        x = self.grid_axis(0)
        y = self.grid_axis(1)
        z = self.grid_axis(2, altitude_axis=True)

        xx, yy, zz = jnp.meshgrid(x,y,z, indexing='ij')

        del x, y, z, xx, yy; gc.collect()

        zz = zz.ravel()
        interp_fcn = jnp.interp(zz, x_1d, f_1d, left='extrapolate', right='extrapolate')
        return jnp.reshape(interp_fcn, self.rshape)#.astype(jnp.float32)
    
    