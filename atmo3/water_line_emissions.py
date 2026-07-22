import jax
import jax.numpy as jnp
from jax import jit
import h5py
from pathlib import Path

from . import constants as con
from . import line_utils as lu

_linedata_H2O  = Path(__file__).parent.parent / 'data' / 'HITRAN_water_vapor_0-1750GHz.hdf5'
_partition_H2O = Path(__file__).parent.parent / 'data' / 'partition_function_water_vapor.hdf5'

# P_0 is standard atmospheric pressure in hPa
P_0_hPa = con.pressure_at_sea_level / 100.

# Reference temperature for H2O 
T_ref_water = 296. # K

# --- H2O Line Parameters from HITRAN HDF5 ---
# Keys differ from the old npz short-names; descriptions are read from HDF5 attrs.
with h5py.File(_linedata_H2O, 'r') as _hf:
    f0      = jnp.asarray(_hf['nu'][:])         [None, :]  # Transition wavenumber [cm^-1]
    S_ref   = jnp.asarray(_hf['sw'][:])         [None, :]  # Line intensity at 296 K [cm^-1/(molecule cm^-2)]
    E_lower = jnp.asarray(_hf['elower'][:])     [None, :]  # Lower-state energy [cm^-1]
    ga      = jnp.asarray(_hf['gamma_air'][:])  [None, :]  # Air-broadened HWHM at 1 atm, 296 K [cm^-1/atm]
    gs      = jnp.asarray(_hf['gamma_self'][:]) [None, :]  # Self-broadened HWHM at 1 atm, 296 K [cm^-1/atm]
    n_temp  = jnp.asarray(_hf['n_air'][:])      [None, :]  # Temperature exponent for air-broadened HWHM [dimensionless]
    delta   = jnp.asarray(_hf['delta_air'][:])  [None, :]  # Pressure shift at 1 atm [cm^-1/atm]

with h5py.File(_partition_H2O, 'r') as _hf:
    T_grid_Q = jnp.asarray(_hf['T_grid'][:])  # Temperature grid [K]
    Q_grid    = jnp.asarray(_hf['Q'][:])       # Partition function Q(T)

@jit 
def _calculate_h2o_absorption_kernel(freq_grid_GHz, T, T_ref, P_hPa, P_water_hPa, Q_ratio):
    nu = (freq_grid_GHz / con.ghz_to_cm_inv)[:, None]
    
    P_atm = P_hPa / P_0_hPa
    P_water_atm = P_water_hPa / P_0_hPa
    P_dry_atm = P_atm - P_water_atm
    
    S_T     = lu.compute_line_strength(T, T_ref, Q_ratio, f0, S_ref, E_lower)
    gamma   = lu.compute_line_width_cm(T, T_ref_water, n_temp, ga, P_dry_atm, gs, P_water_atm)
    nu_star = lu.compute_line_shift_cm(f0, delta, P_atm)
    
    F_VVH = lu.vvh_750_shape(nu, nu_star, gamma, T)
    return jnp.sum(S_T * F_VVH, axis=1)


def water_line_absorption(freq_grid_GHz, T, P_hPa, P_water_hPa):
    
    Q_ratio = lu.interp_partition_sum(T_ref_water, T_grid_Q, Q_grid) / lu.interp_partition_sum(T, T_grid_Q, Q_grid)
    return _calculate_h2o_absorption_kernel(jnp.asarray(freq_grid_GHz), T, T_ref_water, P_hPa, P_water_hPa, Q_ratio)