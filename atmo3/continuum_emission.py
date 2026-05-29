from . import constants as con
import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from pathlib import Path



jax.config.update("jax_enable_x64", True)

# =====================================================================
# A. Global Initialization
# =====================================================================
CURRENT_DIR = Path(__file__).parent

# Now we construct the path to the data folder dynamically
DATA_DIR = CURRENT_DIR / 'data'

data_mt_ckd = np.load(DATA_DIR / 'mt_ckd_continuum.npz')

# Load our precomputed CIA arrays (without the radiation term)
data_cia = np.load(DATA_DIR / 'cia_tables_no_radiation_1750GHz.npz')
_k_N2N2_table = jnp.asarray(data_cia['k_N2N2'])  # Shape: (301, 1751)
_k_O2O2_table = jnp.asarray(data_cia['k_O2O2'])  # Shape: (301, 1751)

# We define the static grid properties used during generation
# This avoids needing to pass the grid arrays themselves into the math!
CIA_T_MIN = 200.0
CIA_T_STEP = 0.5
CIA_F_MIN = 0.0
CIA_F_STEP = 1.0

# =====================================================================
# B. JIT-Compiled Physics Function
# =====================================================================
@jit
def compute_h2o_continuum_jax(freq_GHz, T):
    """
    Computes the MT_CKD self and air continuum coefficients in cm^5.
    
    freq_GHz : 1D array of frequencies [GHz]
    T        : Ambient temperature [K]
    """
    # 1. Extract static arrays into the JAX tracer 
    nu_ckd  = jnp.asarray(data_mt_ckd['nu'])    # MT_CKD Wavenumber grid [cm^-1]
    C_s_296 = jnp.asarray(data_mt_ckd['Cs'])    # MT_CKD Self-broadening reference [cm^2 / molecule]
    C_f_296 = jnp.asarray(data_mt_ckd['Cf'])    # MT_CKD Foreign-broadening reference [cm^2 / molecule]
    T_exp   = jnp.asarray(data_mt_ckd['Texp'])  # MT_CKD Self-broadening temperature exponent [dimensionless]
    
    # USE THE CONSTANTS MODULE HERE
    nu_cm = freq_GHz / con.ghz_to_cm_inv
    
    # 2. Interpolate the MT_CKD tables to our frequency grid
    Cs_296_interp = jnp.interp(nu_cm, nu_ckd, C_s_296)
    Cf_296_interp = jnp.interp(nu_cm, nu_ckd, C_f_296)
    Texp_interp   = jnp.interp(nu_cm, nu_ckd, T_exp)
    
    # 3. Apply Temperature Scaling
    Cs_T = Cs_296_interp * jnp.exp(Texp_interp * ((1.0 / T) - (1.0 / 296.0)))
    Cf_T = Cf_296_interp
    
    # 4. Compute Detailed Balance Radiation Term (USE con.c2)
    rad_term = nu_cm * jnp.tanh((con.c2 * nu_cm) / (2.0 * T))
    
    # 5. Final coefficients in cm^5 (USE con.n_ref_mt_ckd)
    k_self_cm5 = (rad_term * Cs_T) / con.n_ref_mt_ckd
    k_air_cm5  = (rad_term * Cf_T) / con.n_ref_mt_ckd
    
    return k_self_cm5, k_air_cm5


@jit
def compute_cia_continuum_jax(freq_grid_GHz, T):
    """
    Fast Bilinear Interpolation of the precomputed N2-N2 and O2-O2 CIA tables.
    """
    T_idx = (T - CIA_T_MIN) / CIA_T_STEP
    freq_idx = (freq_grid_GHz - CIA_F_MIN) / CIA_F_STEP
    
    T_idx_array = jnp.full_like(freq_idx, T_idx)
    coords = jnp.stack([T_idx_array, freq_idx])
    
    # 1. Interpolate the SMOOTH core function (virtually 0% error)
    k_N2N2_base = jax.scipy.ndimage.map_coordinates(_k_N2N2_table, coords, order=1, mode='constant', cval=0.0)
    k_O2O2_base = jax.scipy.ndimage.map_coordinates(_k_O2O2_table, coords, order=1, mode='constant', cval=0.0)
    
    # 2. Analytically apply the Detailed Balance Radiation Term in JAX
    nu_Hz = freq_grid_GHz * 1e9
    radiation_term = freq_grid_GHz * (1.0 - jnp.exp(-con.h * nu_Hz / (con.k_B * T)))
    
    # 3. Multiply to get the true, highly accurate cross-sections
    k_N2N2 = k_N2N2_base * radiation_term
    k_O2O2 = k_O2O2_base * radiation_term

    k_N2O2 = 1.143 * k_N2N2
    k_O2N2 = 0.822 * k_O2O2
    
    return k_N2N2, k_O2O2, k_N2O2, k_O2N2