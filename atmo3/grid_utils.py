import jax
import jax.numpy as jnp
import gc


class GridWorkspace:
   
    def __init__(self, N, Lbox):
        self.N      = N
        self.Lbox   = Lbox

        self.dk      = 2*jnp.pi/self.Lbox
        self.d3k     = self.dk * self.dk * self.dk

        self.rshape       = (self.N,self.N,self.N)
        self.cshape       = (self.N,self.N,self.N//2+1)
        self.rshape_local = (self.N,self.N,self.N)
        self.cshape_local = (self.N,self.N,self.N//2+1)

        self.start = 0
        self.end   = self.N
        
        self.grid_spacing = self.Lbox / self.N
        
        self.field = None

        # needed for running on CPU with a single process
        self.ngpus   = 1        
        self.host_id = 0

        if self.partype == 'jaxshard':
            self.ngpus   = jax.device_count()
            self.host_id = jax.process_index()
            self.start   = self.host_id * self.N // self.ngpus
            self.end     = (self.host_id + 1) * self.N // self.ngpus
            
            self.partaxis     = 1
            self.rshape_local = (self.N, self.N // self.ngpus, self.N)
            self.cshape_local = (self.N, self.N // self.ngpus, self.N // 2 + 1)
            
            
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

        interp_fcn = jnp.interp(interp_fcn, k_1d, f_1d, left=0., right='extrapolate')
        return jnp.reshape(interp_fcn, self.cshape_local).astype(jnp.float32)
    
    def grid_axis(self, slab_axis=False):
        x_i = (jnp.arange(self.N) * self.grid_spacing).astype(jnp.float32)
        
        if slab_axis: return x_i[self.start:self.end]
        return x_i
    
    def interp2grid(self, x_1d, f_1d):
        x = self.grid_axis()
        y = self.grid_axis(slab_axis=True)
        z = self.grid_axis()
        
        xx, yy, zz = jnp.meshgrid(x,y,z, indexing='ij')

        del x, y, z, xx, yy; gc.collect()

        zz = zz.ravel()
        interp_fcn = jnp.interp(zz, x_1d, f_1d, left='extrapolate', right='extrapolate')
        return jnp.reshape(interp_fcn, self.rshape_local).astype(jnp.float32)