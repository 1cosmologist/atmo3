import jax
import jax.numpy as jnp
from jax import jit

from . import constants as con

# =====================================================================
# Module-level constants (computed once at import, not per-call)
# =====================================================================
_INV_PI    = 1.0 / jnp.pi
_CUTOFF_CM = 750.0 / con.ghz_to_cm_inv   # VVH 750 GHz cutoff in cm⁻¹
_C2_HALF   = con.hcbyk_in_cmK / 2.0                # hc/2k, used in tanh radiation factor

# =====================================================================
# A. Shared Universal Physics (Used by ALL molecules)
# =====================================================================

@jit
def interp_partition_sum(T, T_grid_Q, Q_grid):
    """ Fast, JIT-compiled partition sum using linear interpolation. """
    return jnp.interp(T, T_grid_Q, Q_grid)

@jit
def compute_line_strength(T, T_ref, Q_ratio, f0_cm, S_ref, E_lower):
    """ Step A: Temperature-Adjusted Line Strength S(T) """
    # Single division instead of two reciprocals
    inv_T     = 1.0 / T
    inv_T_ref = 1.0 / T_ref
    boltz_factor = jnp.exp(-con.hcbyk_in_cmK * E_lower * (inv_T - inv_T_ref))

    # expm1(-x) = -(1 - exp(-x)): numerically stable for small x and avoids
    # catastrophic cancellation near 0; also one fewer intermediate value
    c2_f0 = con.hcbyk_in_cmK * f0_cm
    stim_T   = -jnp.expm1(-c2_f0 * inv_T)
    stim_ref = -jnp.expm1(-c2_f0 * inv_T_ref)

    return S_ref * Q_ratio * boltz_factor * (stim_T / stim_ref)

@jit
def compute_line_width_cm(T, T_ref, n_temp, gamma_air, P_dry_atm, gamma_self=0.0, P_self_atm=0.0):
    """ Step B: Generalized Pressure and Temperature Broadened Line Width """
    temp_scaling = (T_ref / T) ** n_temp
    broadening = (gamma_air * P_dry_atm) + (gamma_self * P_self_atm)
    return temp_scaling * broadening

@jit
def compute_line_shift_cm(f0_cm, delta_air, P_atm):
    """ Step C: Pressure-Induced Line Shift (nu*) """
    return f0_cm + (delta_air * P_atm)


# =====================================================================
# B. Distinct Line Shape Profiles
# =====================================================================

@jit
def vvh_750_shape(nu, nu_star, gamma, T):
    """ Van Vleck-Huber Line Shape with 750 GHz Cutoff (for H2O) """
    # Precompute c2/(2T) once; reuse for both tanh calls
    c2_over_2T = _C2_HALF / T
    rad_factor = (nu / nu_star) * (
        jnp.tanh(c2_over_2T * nu) / jnp.tanh(c2_over_2T * nu_star)
    )

    delta_nu_minus = nu - nu_star
    delta_nu_plus  = nu + nu_star

    # Precompute gamma² once
    gamma2 = gamma * gamma
    F_L_minus = _INV_PI * (gamma / (delta_nu_minus * delta_nu_minus + gamma2))
    F_L_plus  = _INV_PI * (gamma / (delta_nu_plus  * delta_nu_plus  + gamma2))

    F_L_minus = jnp.where(jnp.abs(delta_nu_minus) <= _CUTOFF_CM, F_L_minus, 0.0)
    F_L_plus  = jnp.where(jnp.abs(delta_nu_plus)  <= _CUTOFF_CM, F_L_plus,  0.0)

    return rad_factor * (F_L_minus + F_L_plus)

@jit
def gross_shape(nu, nu0, gamma):
    """ Gross line shape (for Uncoupled O2) """
    # Precompute nu² and nu0² once each
    nu2  = nu  * nu
    nu02 = nu0 * nu0
    diff2 = (nu2 - nu02) ** 2
    numerator   = 4.0 * nu2 * gamma
    denominator = diff2 + 4.0 * nu2 * (gamma * gamma)
    return _INV_PI * (numerator / denominator)

@jit
def vvw_coupled_shape(nu, nu0, gamma, P, T, Y0, Y1, g0, g1, dnu0, dnu1):
    """ VVW_coupled line shape including Makarov mixing (for Coupled O2) """
    T_ref = 300.0
    theta         = T_ref / T
    theta_minus_1 = theta - 1.0

    # theta**1.6 = (theta**0.8)²  — one pow call instead of two
    theta_08 = theta ** 0.8
    theta_16 = theta_08 * theta_08

    Y        = P * (Y0 + Y1 * theta_minus_1) * theta_08
    g        = (g0 + g1 * theta_minus_1) * theta_16
    delta_nu = (dnu0 + dnu1 * theta_minus_1) * theta_16

    P2       = P * P
    g_term   = 1.0 + g * P2
    dnu_term = delta_nu * P2

    prefactor = _INV_PI * (nu / nu0) ** 2

    delta_pos = nu - nu0 - dnu_term
    delta_neg = nu + nu0 + dnu_term

    # Precompute gamma² once
    gamma2 = gamma * gamma

    term1_num = gamma * g_term + Y * delta_pos
    term1_den = gamma2 + delta_pos * delta_pos
    term2_num = gamma * g_term - Y * delta_neg
    term2_den = gamma2 + delta_neg * delta_neg

    return prefactor * (term1_num / term1_den + term2_num / term2_den)
