import jax
import jax.numpy as jnp
from functools import partial

from . import constants

# Import the pre-loaded ITU constants we generated from the CSVs
#from .itu_constants import O2_DATA, H2O_DATA

# ==============================================================================
# 1. Point-wise Physics Core (ITU-R P.676-13 Annex 1)
# ==============================================================================

def compute_attenuation_point(T, P, rho_w, freqs_GHz):
    """
    Computes the specific attenuation for a single spatial point over multiple frequencies.
    Input scalars: T (Kelvin), P (Pa, dry air pressure), e (hPa, water vapor pressure).
    Input vector: freqs_GHz (shape: Nf,)
    Returns: gamma_dry, gamma_wet (both shape: Nf,)
    """
    
    # convert water vapor density to water vapor pressure in hPa
    e = rho_w * constants.R_water_vapor * T / 100.   
    P = P / 100.
    
    # Expand freqs for matrix broadcasting against spectral lines
    # f shape: (Nf, 1)
    f = freqs_GHz[:, None] 
    
    # Temperature parameter
    theta = 300.0 / T
    
    # --- Oxygen Calculation ---
    f_o2 = constants.O2_DATA['f0'][None, :] # Shape: (1, N_o2_lines), frequency of each oxygen line
    
    # Line strength
    S_o2 = constants.O2_DATA['a1'] * 1e-7 * P * (theta**3) * jnp.exp(constants.O2_DATA['a2'] * (1 - theta)) # Eq (3) from ITU-R P.676-13 Annex 1
    
    # Line width and Zeeman broadening
    df_o2 = constants.O2_DATA['a3'] * 1e-4 * (P * (theta**(0.8 - constants.O2_DATA['a4'])) + 1.1 * e * theta) # Eq (6a) from ITU-R P.676-13 Annex 1
    df_o2 = jnp.sqrt(df_o2**2 + 2.25e-6) # Eq (6b) from ITU-R P.676-13 Annex 1
    
    # Interference factor
    delta_o2 = (constants.O2_DATA['a5'] + constants.O2_DATA['a6'] * theta) * 1e-4 * (P + e) * (theta**0.8) # Eq (7) from ITU-R P.676-13 Annex 1
    
    # Line shape (Van Vleck-Weisskopf)
    num1 = df_o2 - delta_o2 * (f_o2 - f)
    den1 = (f_o2 - f)**2 + df_o2**2
    num2 = df_o2 - delta_o2 * (f_o2 + f)
    den2 = (f_o2 + f)**2 + df_o2**2
    F_o2 = (f / f_o2) * ((num1 / den1) + (num2 / den2)) # Eq (5) from ITU-R P.676-13 Annex 1
    
    # Sum over all lines (axis 1 collapses the lines, leaving shape Nf)
    N_pp_o2_lines = jnp.sum(S_o2 * F_o2, axis=1) # You sum over the lines
    
    # Dry Continuum (Debye spectrum + nitrogen attenuation)
    d = 5.6e-4 * (P + e) * (theta**0.8) #eq (9) from ITU-R P.676-13 Annex 1
    term1 = 6.14e-5 / (d * (1 + (freqs_GHz / d)**2))
    term2 = (1.4e-12 * P * (theta**1.5)) / (1 + 1.9e-5 * (freqs_GHz**1.5))
    N_pp_D = freqs_GHz * P * (theta**2) * (term1 + term2) # Eq (8) from ITU-R P.676-13 Annex 1
    
    gamma_dry = 0.1820 * freqs_GHz * (N_pp_o2_lines + N_pp_D) # Eq (1) and (2a) from ITU-R P.676-13 Annex 1


    # --- Water Vapor Calculation ---
    f_h2o = constants.H2O_DATA['f0'][None, :] #frequency of each water vapor line, shape: (1, N_h2o_lines)
    
    # Line strength
    S_h2o = constants.H2O_DATA['b1'] * 1e-1 * e * (theta**3.5) * jnp.exp(constants.H2O_DATA['b2'] * (1 - theta)) # Eq (3) from ITU-R P.676-13 Annex 1
    
    # Line width and Doppler broadening
    df_h2o = constants.H2O_DATA['b3'] * 1e-4 * (P * (theta**constants.H2O_DATA['b4']) + constants.H2O_DATA['b5'] * e * (theta**constants.H2O_DATA['b6'])) # Eq (6a) from ITU-R P.676-13 Annex 1
    df_h2o = 0.535 * df_h2o + jnp.sqrt(0.217 * (df_h2o**2) + (2.1316e-12 * (f_h2o**2)) / theta) # Eq (6b) from ITU-R P.676-13 Annex 1
    
    # Line shape (interference factor delta is 0 for H2O)
    num1_w = df_h2o
    den1_w = (f_h2o - f)**2 + df_h2o**2
    num2_w = df_h2o
    den2_w = (f_h2o + f)**2 + df_h2o**2
    F_h2o = (f / f_h2o) * ((num1_w / den1_w) + (num2_w / den2_w)) # Eq (5) from ITU-R P.676-13 Annex 1, but with delt_H2O=0 for water vapor lines
    
    # Sum over all water vapor lines
    N_pp_h2o = jnp.sum(S_h2o * F_h2o, axis=1)
    
    gamma_wet = 0.1820 * freqs_GHz * N_pp_h2o # Eq (1) and (2b) from ITU-R P.676-13 Annex 1
    
    return gamma_dry, gamma_wet


def attenuation_to_transmission(gamma_nu, ds):
    """
    gamma_nu shape = (n_nu, n_los)
    ds shape = (n_los)
    """
    
    ds = ds[None, :] / 1e3
    
    return jnp.exp( -jnp.log(10.)/10. * jnp.sum(gamma_nu * ds, axis=1))



def B_nu_T(nu_in_GHz, T_bb):
    """
    B_nu_T is the Planck distribution function, defined as: 
    .. math::
        B(\\nu, T) = \\frac{2 h \\nu^3}{c^2} \\frac{1}{e^{x} - 1},
    with 
    .. math::
        x = \\frac{h \\nu}{k_B T}.
    
    It returns the Planck function values for nu (can be vectorized) 
    for a given blackbody temperature.

    Parameters
    ----------
    nu_in_GHz : float or numpy ndarray
        Frequency in GHz at which we want the value of the Planck function.
    T_planck : float, optional
        Temperature of the Planck distribution in Kelvin.
        Default value set to the CMB monopole temperature of 2.7255 K.

    Returns
    -------
    float or numpy ndarray
        A float value (or ndarray) for the value(s) of the Planck distribution.

    """

    nu_in_Hz = nu_in_GHz * constants.giga
    x = constants.h * nu_in_Hz / constants.k_B / T_bb
    prefactor = 2. * constants.h * nu_in_Hz**3. / constants.c**2.

    return prefactor / (jnp.exp(x) - 1.)


@jax.jit
def compute_attenuation_los(T_los, P_los, rho_los, freqs_in_GHz):
    # T_los, P_los, rho_los have shape (n_scans, n_los)
    # We vmap over n_scans (axis 0) and then n_los (axis 0 in the inner fn)
    # Inner vmap: outputs (Nf, n_los) by setting out_axes=-1
    # Outer vmap: outputs (n_scans, Nf, n_los) by setting out_axes=0
    gamma_d, gamma_w = jax.vmap(
        jax.vmap(
            lambda t, p, rho: compute_attenuation_point(t, p, rho, freqs_in_GHz),
            in_axes=(0, 0, 0),
            out_axes=-1
        ),
        in_axes=(0, 0, 0),
        out_axes=0
    )(T_los, P_los, rho_los)
    
    return gamma_w, gamma_d


@jax.jit
def get_emission(T_los, P_los, rho_los, ds, freqs_in_GHz):
    gamma_H2O, gamma_O2 = compute_attenuation_los(T_los, P_los, rho_los, freqs_in_GHz)
    
    # ds needs a dummy frequency dimension to broadcast against (n_scans, Nf, n_los)
    A_nu = (gamma_H2O + gamma_O2) * ds[:, None, :] / 1e3     # Assuming ds is provided in m
    
    # Accumulate attenuation looking inward from the first element
    # Shifted to avoid applying the current shell's attenuation to its own lower boundary
    Alower_nu = jnp.concatenate([jnp.zeros_like(A_nu[:, :, :1]), jnp.cumsum(A_nu[:, :, :-1], axis=2)], axis=2)
    
    # B_nu_T inputs broadcast to (n_scans, Nf, n_los)
    B_nu = B_nu_T(freqs_in_GHz[None, :, None], T_los[:, None, :])
    
    return jnp.sum(B_nu * (1. - 10.**(-A_nu / 10.)) * 10.**(-Alower_nu / 10.), axis=2)
    
    
def intensitySI_to_Tb(I_nu, freq_in_GHz):
    return constants.c**2 / 2. / constants.k_B / (freq_in_GHz * constants.giga)**2. * I_nu