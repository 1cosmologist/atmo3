from . import constants as con
import jax
import jax.numpy as jnp
from jax import jit
import h5py
from pathlib import Path

jax.config.update("jax_enable_x64", True)

# =====================================================================
# A. Global Initialization
# =====================================================================
_CONTINUUM_FILE = Path(__file__).parent.parent / 'data' / 'mt_ckd_continuum.hdf5'

with h5py.File(_CONTINUUM_FILE, 'r') as _hf:
    _nu_ckd  = jnp.asarray(_hf['wavenumbers'][:])      # Wavenumber grid [cm⁻¹]
    _Cs_296  = jnp.asarray(_hf['self_absco_ref'][:])   # Self-continuum ref coefficient at 296 K [cm²/molecule]
    _Cf_296  = jnp.asarray(_hf['for_absco_ref'][:])    # Foreign-continuum ref coefficient at 296 K [cm²/molecule]
    _T_exp   = jnp.asarray(_hf['self_texp'][:])        # Self-continuum temperature exponent [dimensionless]

# =====================================================================
# B. JIT-Compiled Physics Function
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
    Cs_T = Cs_296_interp * jnp.exp(Texp_interp * ((1.0 / T) - (1.0 / 296.0)))
    Cf_T = Cf_296_interp
    
    # 4. Compute Detailed Balance Radiation Term
    rad_term = nu_cm * jnp.tanh((con.hcbyk_in_cmK * nu_cm) / (2.0 * T))
    
    # 5. Final coefficients in cm^5
    k_self_cm5 = (rad_term * Cs_T) / con.n_ref_mt_ckd
    k_air_cm5  = (rad_term * Cf_T) / con.n_ref_mt_ckd
    
    return k_self_cm5, k_air_cm5