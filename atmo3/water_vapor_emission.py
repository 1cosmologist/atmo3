import jax
import jax.numpy as jnp
from jax import jit
import h5py
from pathlib import Path

from . import constants as con
from . import line_utils as lu

jax.config.update("jax_enable_x64", True)

_linedata_H2O    = Path(__file__).parent.parent / 'data' / 'AER_water_vapor_0-1750GHz.hdf5'
_partition_H2O   = Path(__file__).parent.parent / 'data' / 'partition_function_water_vapor.hdf5'
_CONTINUUM_FILE  = Path(__file__).parent.parent / 'data' / 'mt_ckd_continuum.hdf5'

# =====================================================================
# A. Line Absorption - Global Initialization
# =====================================================================

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

# =====================================================================
# B. Continuum Absorption - Global Initialization
# =====================================================================
with h5py.File(_CONTINUUM_FILE, 'r') as _hf:
    _nu_ckd    = jnp.asarray(_hf['wavenumbers'][:])      # Wavenumber grid [cm⁻¹]
    _Cs_296    = jnp.asarray(_hf['self_absco_ref'][:])   # Self-continuum ref coefficient at ref_temp [cm²/molecule]
    _Cf_296    = jnp.asarray(_hf['for_absco_ref'][:])    # Foreign-continuum ref coefficient at ref_temp [cm²/molecule]
    _T_exp     = jnp.asarray(_hf['self_texp'][:])        # Self-continuum temperature scaling exponent (power law), [dimensionless]
    _ref_press = float(_hf['ref_press'][()])             # Reference pressure for the table [mbar]
    _ref_temp  = float(_hf['ref_temp'][()])              # Reference temperature for the table [K]

# Reference number density at the table's reference pressure/temperature
# [molecules / cm^3], matching am's h2o_continuum.c: rho0 = N_STP * (ref_press / P_STP) * (T_STP / ref_temp)
_n_ref_mt_ckd = (_ref_press * 100.0) / (con.k_B * _ref_temp) * con.centi**3

# =====================================================================
# C. Line Absorption - Physics Functions
# =====================================================================
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


# =====================================================================
# D. Continuum Absorption - Physics Function
# =====================================================================
@jit
def water_continuum_absorption(freq_GHz, T):
    """
    Computes the MT_CKD self and air continuum coefficients in cm^5.
    
    freq_GHz : 1D array of frequencies [GHz]
    T        : Ambient temperature [K]
    """
    nu_cm = freq_GHz / con.ghz_to_cm_inv

    # 2. Interpolate the MT_CKD tables to our frequency grid
    Cs_296_interp = jnp.interp(nu_cm, _nu_ckd, _Cs_296)
    Cf_296_interp = jnp.interp(nu_cm, _nu_ckd, _Cf_296)
    Texp_interp   = jnp.interp(nu_cm, _nu_ckd, _T_exp)
    
    # 3. Apply Temperature Scaling
    # Matches am's h2o_continuum.c: kb_tab *= (ref_temp / T)^texp (power law, not exponential-in-1/T)
    Cs_T = Cs_296_interp * jnp.power(_ref_temp / T, Texp_interp)
    Cf_T = Cf_296_interp
    
    # 4. Compute Detailed Balance Radiation Term
    rad_term = nu_cm * jnp.tanh((con.hcbyk_in_cmK * nu_cm) / (2.0 * T))
    
    # 5. Final coefficients in cm^5
    k_self_cm5 = (rad_term * Cs_T) / _n_ref_mt_ckd
    k_air_cm5  = (rad_term * Cf_T) / _n_ref_mt_ckd
    
    return k_self_cm5, k_air_cm5
