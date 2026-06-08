import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from pathlib import Path

from . import constants as con

# CRITICAL: Atmospheric physics requires high precision. 
# JAX defaults to 32-bit floats, which will destroy your exponential terms.
jax.config.update("jax_enable_x64", True)

# =====================================================================
# A. Global Initialization & Data Loading
# =====================================================================
CURRENT_DIR = Path(__file__).parent
DATA_DIR = CURRENT_DIR / 'data'

# Load the NPZ files into memory once at the top of the script.
data_H2O = np.load(DATA_DIR / 'h2o_lines_1750GHz.npz')
data_o2_coupled = np.load(DATA_DIR / 'o2_coupled_lines_1750GHz.npz')
data_o2_uncoupled = np.load(DATA_DIR / 'o2_uncoupled_lines_1750GHz.npz')
data_O3 = np.load(DATA_DIR / 'o3_uncoupled_lines_1000GHz.npz') 
data_partition_functions = np.load(DATA_DIR / 'partition_functions.npz')

# Extract the static temperature grid for partition functions
_T_grid_Q = jnp.asarray(data_partition_functions['T_grid'])


# =====================================================================
# B. Shared Universal Physics (Used by ALL molecules)
# =====================================================================

@jit
def interp_partition_sum_jax(T, Q_grid):
    """ Fast, JIT-compiled partition sum using linear interpolation. """
    return jnp.interp(T, _T_grid_Q, Q_grid)

@jit
def compute_line_strength_jax(T, T_ref, Q_ratio, f0_cm, S_ref, E_lower):
    """ Step A: Temperature-Adjusted Line Strength S(T) """
    boltz_factor = jnp.exp(-con.c2 * E_lower * ((1.0 / T) - (1.0 / T_ref)))
    
    stim_T = 1.0 - jnp.exp(-con.c2 * f0_cm / T)
    stim_ref = 1.0 - jnp.exp(-con.c2 * f0_cm / T_ref)
    
    return S_ref * Q_ratio * boltz_factor * (stim_T / stim_ref)

@jit
def compute_line_width_cm_jax(T, T_ref, n_temp, gamma_air, P_dry_atm, gamma_self=0.0, P_self_atm=0.0):
    """ Step B: Generalized Pressure and Temperature Broadened Line Width """
    temp_scaling = (T_ref / T) ** n_temp
    broadening = (gamma_air * P_dry_atm) + (gamma_self * P_self_atm)
    return temp_scaling * broadening

@jit
def compute_line_shift_cm_jax(f0_cm, delta_air, P_atm):
    """ Step C: Pressure-Induced Line Shift (nu*) """
    return f0_cm + (delta_air * P_atm)


# =====================================================================
# C. Distinct Line Shape Profiles
# =====================================================================

@jit
def vvh_750_shape_jax(nu, nu_star, gamma, T):
    """ Van Vleck-Huber Line Shape with 750 GHz Cutoff (for H2O) """
    cutoff_cm = 750.0 / con.ghz_to_cm_inv
    
    rad_factor = (nu / nu_star) * (jnp.tanh(con.c2 * nu / (2.0 * T)) / jnp.tanh(con.c2 * nu_star / (2.0 * T)))
    
    delta_nu_minus = nu - nu_star
    delta_nu_plus  = nu + nu_star
    
    F_L_minus = (1.0 / jnp.pi) * (gamma / (delta_nu_minus**2 + gamma**2))
    F_L_plus  = (1.0 / jnp.pi) * (gamma / (delta_nu_plus**2 + gamma**2))
    
    F_L_minus = jnp.where(jnp.abs(delta_nu_minus) <= cutoff_cm, F_L_minus, 0.0)
    F_L_plus  = jnp.where(jnp.abs(delta_nu_plus) <= cutoff_cm, F_L_plus, 0.0)
    
    return rad_factor * (F_L_minus + F_L_plus)

@jit
def gross_shape_jax(nu, nu0, gamma):
    """ Gross line shape (for Uncoupled O2) """
    numerator = 4.0 * (nu**2) * gamma
    denominator = ((nu**2) - (nu0**2))**2 + 4.0 * (nu**2) * (gamma**2)
    return (1.0 / jnp.pi) * (numerator / denominator)

@jit
def vvw_coupled_shape_jax(nu, nu0, gamma, P, T, Y0, Y1, g0, g1, dnu0, dnu1):
    """ VVW_coupled line shape including Makarov mixing (for Coupled O2) """
    T_ref = 300.0  
    theta = T_ref / T
    theta_minus_1 = theta - 1.0
    
    Y = P * (Y0 + Y1 * theta_minus_1) * (theta**0.8)
    g = (g0 + g1 * theta_minus_1) * (theta**1.6)
    delta_nu = (dnu0 + dnu1 * theta_minus_1) * (theta**1.6)
    
    P2 = P**2
    g_term = 1.0 + g * P2
    dnu_term = delta_nu * P2
    
    prefactor = (1.0 / jnp.pi) * (nu / nu0)**2
    
    delta_pos = nu - nu0 - dnu_term
    delta_neg = nu + nu0 + dnu_term
    
    term1_num = gamma * g_term + Y * delta_pos
    term1_den = gamma**2 + delta_pos**2
    term2_num = gamma * g_term - Y * delta_neg
    term2_den = gamma**2 + delta_neg**2
    
    return prefactor * (term1_num / term1_den + term2_num / term2_den)


# =====================================================================
# D. The Master JAX Cores (Data Extraction & Pipeline Assembly)
# =====================================================================
# P_0 is standard atmospheric pressure in hPa
P_0_hPa = con.pressure_at_sea_level / 100.0

@jit 
def _calculate_h2o_absorption_jax_core(freq_grid_GHz, T, T_ref, P_hPa, P_water_hPa, Q_ratio):
    nu = (freq_grid_GHz / con.ghz_to_cm_inv)[:, None]
    
    # --- Extracted H2O Parameters ---
    f0 = jnp.asarray(data_H2O['f0'])[None, :]          # Center Frequency [cm^-1]
    S_ref = jnp.asarray(data_H2O['S'])[None, :]        # Reference Line Strength [cm^-1 / (molecule * cm^-2)]
    E_lower = jnp.asarray(data_H2O['E'])[None, :]      # Lower-state energy [cm^-1]
    ga = jnp.asarray(data_H2O['ga'])[None, :]          # Air-broadened half-width [cm^-1 / atm]
    gs = jnp.asarray(data_H2O['gs'])[None, :]          # Self-broadened half-width [cm^-1 / atm]
    n_temp = jnp.asarray(data_H2O['n'])[None, :]       # Temperature dependence coefficient [dimensionless]
    delta = jnp.asarray(data_H2O['d'])[None, :]        # Pressure shift coefficient [cm^-1 / atm]
    
    P_atm = P_hPa / P_0_hPa
    P_water_atm = P_water_hPa / P_0_hPa
    P_dry_atm = P_atm - P_water_atm
    
    S_T = compute_line_strength_jax(T, T_ref, Q_ratio, f0, S_ref, E_lower)
    gamma = compute_line_width_cm_jax(T, 296.0, n_temp, ga, P_dry_atm, gs, P_water_atm)
    nu_star = compute_line_shift_cm_jax(f0, delta, P_atm)
    
    F_VVH = vvh_750_shape_jax(nu, nu_star, gamma, T)
    return jnp.sum(S_T * F_VVH, axis=1)

@jit
def _calculate_o2_coupled_core(freq_grid_GHz, T, T_ref, P_hPa, Q_ratio):
    nu = (freq_grid_GHz / con.ghz_to_cm_inv)[:, None]
    
    # --- Extracted Coupled O2 Parameters ---
    f0 = jnp.asarray(data_o2_coupled['f0'])[None, :]       # Center Frequency [cm^-1]
    S_ref = jnp.asarray(data_o2_coupled['S'])[None, :]     # Reference Line Strength [cm^2 * cm^-1]
    E_lower = jnp.asarray(data_o2_coupled['E'])[None, :]   # Lower-state energy [cm^-1]
    ga = jnp.asarray(data_o2_coupled['gamma'])[None, :]    # Dry air broadening coefficient [cm^-1 / atm]
    n_temp = jnp.asarray(data_o2_coupled['n'])[None, :]    # Temp dependence coefficient for width [dimensionless]
    
    y0 = jnp.asarray(data_o2_coupled['y0'])[None, :]       # 1st-order line mixing parameter [1/atm]
    y1 = jnp.asarray(data_o2_coupled['y1'])[None, :]       # 1st-order line mixing temp parameter [1/atm]
    g0 = jnp.asarray(data_o2_coupled['g0'])[None, :]       # 2nd-order intensity mixing parameter [1/atm^2]
    g1 = jnp.asarray(data_o2_coupled['g1'])[None, :]       # 2nd-order intensity mixing temp parameter [1/atm^2]
    d0 = jnp.asarray(data_o2_coupled['d0'])[None, :]       # 2nd-order frequency shift parameter [cm^-1 / atm^2]
    d1 = jnp.asarray(data_o2_coupled['d1'])[None, :]       # 2nd-order frequency shift temp parameter [cm^-1 / atm^2]
    
    P_dry_atm = P_hPa / P_0_hPa
    
    S_T = compute_line_strength_jax(T, T_ref, Q_ratio, f0, S_ref, E_lower)
    # O2 has no self-broadening component mapped here, just dry air
    gamma_width = compute_line_width_cm_jax(T, 296.0, n_temp, ga, P_dry_atm)
    nu_star = compute_line_shift_cm_jax(f0, 0.0, P_dry_atm) # delta=0 for coupled
    
    F_VVW = vvw_coupled_shape_jax(nu, nu_star, gamma_width, P_dry_atm, T, y0, y1, g0, g1, d0, d1)
    return jnp.sum(S_T * F_VVW, axis=1)

@jit
def _calculate_o2_uncoupled_core(freq_grid_GHz, T, T_ref, P_hPa, Q_ratio):
    nu = (freq_grid_GHz / con.ghz_to_cm_inv)[:, None]
    
    # --- Extracted Uncoupled O2 Parameters ---
    f0 = jnp.asarray(data_o2_uncoupled['f0'])[None, :]       # Center Frequency [cm^-1]
    S_ref = jnp.asarray(data_o2_uncoupled['S'])[None, :]     # Reference Line Strength [cm^2 * cm^-1]
    E_lower = jnp.asarray(data_o2_uncoupled['E'])[None, :]   # Lower-state energy [cm^-1]
    ga = jnp.asarray(data_o2_uncoupled['gamma'])[None, :]    # Dry air broadening coefficient [cm^-1 / atm]
    n_temp = jnp.asarray(data_o2_uncoupled['n'])[None, :]    # Temp dependence coefficient for width [dimensionless]
    delta = jnp.asarray(data_o2_uncoupled['delta'])[None, :] # Pressure shift coefficient [cm^-1 / atm]
    
    P_dry_atm = P_hPa / P_0_hPa
    
    S_T = compute_line_strength_jax(T, T_ref, Q_ratio, f0, S_ref, E_lower)
    gamma_width = compute_line_width_cm_jax(T, 296.0, n_temp, ga, P_dry_atm)
    nu_star = compute_line_shift_cm_jax(f0, delta, P_dry_atm)
    
    F_G = gross_shape_jax(nu, nu_star, gamma_width)
    return jnp.sum(S_T * F_G, axis=1)


@jit
def _calculate_o3_absorption_jax_core(freq_grid_GHz, T, T_ref, P_hPa, P_o3_hPa, Q_ratio):
    nu = (freq_grid_GHz / con.ghz_to_cm_inv)[:, None]
    
    # --- Extracted O3 Parameters ---
    f0 = jnp.asarray(data_O3['f0'])[None, :]          # Center Frequency [cm^-1]
    S_ref = jnp.asarray(data_O3['S'])[None, :]        # Reference Line Strength [cm^-1 / (molecule * cm^-2)]
    E_lower = jnp.asarray(data_O3['E'])[None, :]      # Lower-state energy [cm^-1]
    ga = jnp.asarray(data_O3['ga'])[None, :]          # Air-broadened half-width [cm^-1 / atm]
    gs = jnp.asarray(data_O3['gs'])[None, :]          # Self-broadened half-width [cm^-1 / atm]
    n_temp = jnp.asarray(data_O3['n'])[None, :]       # Temperature dependence coefficient [dimensionless]
    delta = jnp.asarray(data_O3['d'])[None, :]        # Pressure shift coefficient [cm^-1 / atm]
    
    # Pressure conversions
    P_atm = P_hPa / P_0_hPa
    P_o3_atm = P_o3_hPa / P_0_hPa
    P_dry_atm = P_atm - P_o3_atm
    
    # 1. Dynamic Line Strength
    S_T = compute_line_strength_jax(T, T_ref, Q_ratio, f0, S_ref, E_lower)
    
    # 2. Line Width and Shift (uses the generalized trace gas function)
    gamma = compute_line_width_cm_jax(T, 296.0, n_temp, ga, P_dry_atm, gs, P_o3_atm)
    nu_star = compute_line_shift_cm_jax(f0, delta, P_atm)
    
    # 3. Gross Line Shape (reusing the function from uncoupled O2)
    F_G = gross_shape_jax(nu, nu_star, gamma)
    
    # 4. Final integration across all lines
    return jnp.sum(S_T * F_G, axis=1)

# =====================================================================
# E. The Python APIs (Safe Entry Points)
# =====================================================================

def calculate_h2o_absorption_jax(freq_grid_GHz, T, P_hPa, P_water_hPa):
    Q_grid = jnp.asarray(data_partition_functions['H2O'])
    Q_ratio = interp_partition_sum_jax(296.0, Q_grid) / interp_partition_sum_jax(T, Q_grid)
    return _calculate_h2o_absorption_jax_core(jnp.asarray(freq_grid_GHz), T, 296.0, P_hPa, P_water_hPa, Q_ratio)

def calculate_o2_coupled_absorption_jax(freq_grid_GHz, T, P_hPa):
    Q_grid = jnp.asarray(data_partition_functions['O2'])
    Q_ratio = interp_partition_sum_jax(296.0, Q_grid) / interp_partition_sum_jax(T, Q_grid)
    return _calculate_o2_coupled_core(jnp.asarray(freq_grid_GHz), T, 296.0, P_hPa, Q_ratio)

def calculate_o2_uncoupled_absorption_jax(freq_grid_GHz, T, P_hPa):
    Q_grid = jnp.asarray(data_partition_functions['O2'])
    Q_ratio = interp_partition_sum_jax(296.0, Q_grid) / interp_partition_sum_jax(T, Q_grid)
    return _calculate_o2_uncoupled_core(jnp.asarray(freq_grid_GHz), T, 296.0, P_hPa, Q_ratio)

def calculate_o3_absorption_jax(freq_grid_GHz, T, P_hPa, P_o3_hPa):
    Q_grid = jnp.asarray(data_partition_functions['O3'])
    Q_ratio = interp_partition_sum_jax(296.0, Q_grid) / interp_partition_sum_jax(T, Q_grid)
    return _calculate_o3_absorption_jax_core(jnp.asarray(freq_grid_GHz), T, 296.0, P_hPa, P_o3_hPa, Q_ratio)