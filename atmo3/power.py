import jax 
import jax.numpy as jnp
from .grid_utils import GridWorkspace
from .cube import Cube
from . import multihost_fft as mfft
import gc

def compute_power_spectrum(
    cube: Cube, nbins=20
) -> jnp.ndarray:
    kx = cube.grid_wsp.k_axis()
    ky = cube.grid_wsp.k_axis(slab_axis=True)
    kz = cube.grid_wsp.k_axis(r=True)

    k = jnp.sqrt(cube.grid_wsp.k_square(kx, ky, kz))
    del kx, ky, kz ; gc.collect()
    
    num_gpus = jax.device_count()
    global_shape = (k.shape[0],k.shape[1]*num_gpus,k.shape[2])
    
    device_mesh = jax.sharding.Mesh(jax.devices(), axis_names=('gpus',))
    sharding    = jax.sharding.NamedSharding(device_mesh, jax.sharding.PartitionSpec(None,'gpus', None))
    
    
    kshard = jax.make_array_from_single_device_arrays(
            global_shape,
            sharding,
            [k])

    del k; gc.collect()

    # jax.debug.visualize_array_sharding(kshard[0])

    kbin_edges = jnp.histogram_bin_edges(
            kshard[kshard > 0.], 
            bins=nbins, 
        )
    
    # print(f"kbin_edges: {kbin_edges}")
    
    kmodes = jnp.histogram(
        kshard[kshard > 0.], 
        bins=kbin_edges,
        density=False,
    )[0]
    
    # print(f"kmodes: {kmodes}, {jnp.sum(kmodes)}")
    
    
    # kbins = kbin_edges[:-1] + 0.5 * (kbin_edges[1:] - kbin_edges[:-1])
    # kbins = kbins#.astype(jnp.float32)
    
    # kbin_widths = kbin_edges[1:] - kbin_edges[:-1]
    
    kbins = jnp.histogram(
        kshard[kshard > 0.], 
        bins=kbin_edges,
        weights=kshard[kshard > 0.],
    )[0] / kmodes
    
    # print(f"kbins: {kbins}")
    # print(f"kbins_hist: {kbins_hist}")
    # N_i = 4 * jnp.pi * kbins**2 * kbin_widths / cube.grid_wsp.dk**3
    
    # print(f"Number of modes in each bin: {kmodes}")
    # print(f"Number of modes in each bin: {N_i}")
    
    # jax.debug.visualize_array_sharding(kbins)
    
    delta_k2 = mfft.fft(cube.field, direction='r2c') 
    delta_k2 = jnp.abs(delta_k2)**2  # convert to power spectrum

    global_shape = (delta_k2.shape[0],delta_k2.shape[1]*num_gpus,delta_k2.shape[2])

    delta_k2_shard = jax.make_array_from_single_device_arrays(
            global_shape,
            sharding,
            [delta_k2])
    del delta_k2 ; gc.collect()

    # jax.debug.visualize_array_sharding(delta_k2_shard[0])
    
    # P_k = jnp.zeros_like(kbins, dtype=jnp.float32)
    
    # for i in range(nbins):
    #     mask = (kshard >= kbin_edges[i]) & (kshard < kbin_edges[i+1])
    #     delta_k2_shard_i = jnp.where(mask, delta_k2_shard, 0.0)
        
    #     # Sum over the modes in the bin
    #     P_k = P_k.at[i].set(
    #         jnp.sum(delta_k2_shard_i) * volume / N_i[i]
    #     )

    P_k = jnp.histogram(
        kshard[kshard > 0.],
        bins=kbin_edges,
        weights=delta_k2_shard[kshard > 0.],
    )[0] * (cube.grid_wsp.Lbox / cube.grid_wsp.N**2)**3 / kmodes

    # print(volume / N_i)

    del delta_k2_shard, kmodes ; gc.collect()
    
    # jax.debug.visualize_array_sharding(P_k)

    return kbins, P_k

    
    
    
    
    
    
    
    
    
    
    

