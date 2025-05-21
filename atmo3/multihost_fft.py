# Compute real to complex FFTs on GPUs with Jax on multiple GPUs.
# Adapted from example developed by Lukas Winkler:
#   https://gist.github.com/Findus23/eb5ecb9f65ccf13152cda7c7e521cbdd
# and jax custom partitioning documentation:
#   https://jax.readthedocs.io/en/latest/jax.experimental.custom_partitioning.html

from typing import Callable

import jax
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import gc

def fft_partitioner(fft_func: Callable[[jax.Array], jax.Array], partition_spec: P):
    @custom_partitioning
    def func(x):
        return fft_func(x)

    def partition(mesh, arg_shapes, result_shape):
        mesh = jax.tree.map(lambda x: x.sharding, arg_shapes)[0].mesh
        namedsharding = NamedSharding(mesh, partition_spec)
        return mesh, fft_func, namedsharding, (namedsharding,)

    def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
        mesh = jax.tree.map(lambda x: x.sharding, arg_shapes)[0].mesh
        return NamedSharding(mesh, partition_spec)

    func.def_partition(
        partition=partition,
        infer_sharding_from_operands=infer_sharding_from_operands
    )
    return func

def _fft_XY(x):
    return jax.numpy.fft.fftn(x, axes=[0, 1])

def _fft_Z(x):
    return jax.numpy.fft.rfft(x, axis=2)

def _ifft_XY(x):
    return jax.numpy.fft.ifftn(x, axes=[0, 1])

def _ifft_Z(x):
    return jax.numpy.fft.irfft(x, axis=2)

fft_XY = fft_partitioner(_fft_XY, P(None, None, "gpus"))
fft_Z = fft_partitioner(_fft_Z, P(None, "gpus"))
ifft_XY = fft_partitioner(_ifft_XY, P(None, None, "gpus"))
ifft_Z = fft_partitioner(_ifft_Z, P(None, "gpus"))

def rfftn(x):
    x = fft_Z(x)
    x = fft_XY(x)
    return x

def irfftn(x):
    x = ifft_XY(x)
    x = ifft_Z(x)
    return x

def fft(x_np,direction='r2c'):

    from jax import jit
    from jax.experimental import mesh_utils
    from jax.experimental.multihost_utils import sync_global_devices

    num_gpus = jax.device_count()

    global_shape = (x_np.shape[0],x_np.shape[1]*num_gpus,x_np.shape[2])

    devices = mesh_utils.create_device_mesh((num_gpus,))
    mesh = Mesh(devices, axis_names=('gpus',))
    with mesh:
        x_single = jax.device_put(x_np).block_until_ready()
        del x_np ; gc.collect()
        xshard = jax.make_array_from_single_device_arrays(
            global_shape,
            NamedSharding(mesh, P(None, "gpus")),
            [x_single]).block_until_ready()
        del x_single ; gc.collect()
        if direction=='r2c':
            rfftn_jit = jit(
                rfftn,
                in_shardings=(NamedSharding(mesh, P(None, "gpus"))),
                out_shardings=(NamedSharding(mesh, P(None, "gpus")))
            )
        else:
            irfftn_jit = jit(
                irfftn,
                in_shardings=(NamedSharding(mesh, P(None, "gpus"))),
                out_shardings=(NamedSharding(mesh, P(None, "gpus")))
            )
        sync_global_devices("wait for compiler output")

        with jax.spmd_mode('allow_all'):

            if direction=='r2c':
                out_jit: jax.Array = rfftn_jit(xshard).block_until_ready()
            else:
                out_jit: jax.Array = irfftn_jit(xshard).block_until_ready()
            sync_global_devices("loop")
            local_out_subset = out_jit.addressable_data(0)
    return local_out_subset
