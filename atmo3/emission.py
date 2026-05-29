import jax
import jax.numpy as jnp
from functools import partial
from . import lines_emission
from . import continuum_emission

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


# Here we do the am implementation of the H2O emission

def compute_attenuation_water_vapor_am(T, P_Pa, rho_w, freqs_GHz):
    """
    Computes the specific attenuation for a single spatial point over multiple frequencies.
    Input scalars: T (Kelvin), P (Pa, Total air pressure), rho_water (kg/m^3, water vapor density).
    Input vector: freqs_GHz (shape: Nf,)
    Returns: gamma_dry, gamma_wet (both shape: Nf,) in dB/km
    """

    # convert water vapor density to water vapor pressure in hPa
    e_Pa = rho_w * constants.R_water_vapor * T  

    e_hPa = e_Pa / 100.0
    P_hPa = P_Pa / 100.0

    # 3. Calculate particle densities (molecules / cm^3)
    # n = P / (k_B * T) gives m^-3. Multiply by 1e-6 to get cm^-3.
    n_air_density = ((P_Pa - e_Pa) / (constants.k_B * T)) * 1e-6
    n_water_density = (e_Pa / (constants.k_B * T)) * 1e-6

    # Let's calculate the absorption coefficient k in cm^2/molecule for the H2O lines

    k_lines = lines_emission.calculate_h2o_absorption_jax(freqs_GHz, T, P_hPa, e_hPa) # in cm^2/molecule

    # Now we calculate the continuum absorption coefficient k_continuum in cm^5/molecule^2
    k_continuum_self, k_continuum_air = continuum_emission.compute_h2o_continuum_jax(freqs_GHz, T) # in cm^5/molecule^2

    # Total absorption coefficient for water vapor is the sum of line and continuum contributions
    gamma_wet_tot = (n_water_density * k_lines + n_water_density**2 * k_continuum_self + n_air_density * n_water_density * k_continuum_air) *1e5 * (10.0/jnp.log(10.)) # Convert from cm^2/molecule to dB/km

    return gamma_wet_tot

def compute_attenuation_dry_air_am(T, P_Pa, rho_w, freqs_GHz):
    """
    Computes the specific attenuation for a single spatial point over multiple frequencies.
    Input scalars: T (Kelvin), P (Pa, Total air pressure), rho_water (kg/m^3, water vapor density).
    Input vector: freqs_GHz (shape: Nf,)
    Returns: gamma_dry, gamma_wet (both shape: Nf,) in dB/km
    """

    # convert water vapor density to water vapor pressure in hPa
    e_Pa = rho_w * constants.R_water_vapor * T 
    P_dry_Pa = (P_Pa - e_Pa)

    e_hPa = e_Pa / 100.0
    P_hPa = P_Pa / 100.0
    #P_dry_hPa = P_hPa - e_hPa
    # 3. Calculate particle densities (molecules / cm^3)
    # n = P / (k_B * T) gives m^-3. Multiply by 1e-6 to get cm^-3.
    n_air_density = (P_dry_Pa / (constants.k_B * T)) * 1e-6
    n_O2_density = n_air_density * constants.O2_VOLUME_MIXING_RATIO
    n_N2_density = n_air_density * constants.N2_VOLUME_MIXING_RATIO

    k_lines_coupled = lines_emission.calculate_o2_coupled_absorption_jax(freqs_GHz, T, P_hPa)
    k_lines_uncoupled = lines_emission.calculate_o2_uncoupled_absorption_jax(freqs_GHz, T, P_hPa)

    gamma_dry_lines = (n_O2_density * k_lines_coupled + n_O2_density * k_lines_uncoupled) * 1e5 * (10.0/jnp.log(10.))  # in dB/km
    # Let's calculate the continuum absorption coefficient k_continuum in cm^5/molecule^2
    k_continuum_N2N2, k_continuum_O2O2, k_continuum_N2O2, k_continuum_O2N2 = continuum_emission.compute_cia_continuum_jax(freqs_GHz, T) # in cm^5/molecule^2
    gamma_dry_continuum = (n_N2_density**2 * k_continuum_N2N2 + n_O2_density**2 * k_continuum_O2O2 + n_N2_density * n_O2_density * (k_continuum_N2O2 +k_continuum_O2N2)) * 1e5 * (10.0/jnp.log(10.)) # Convert from cm^5/molecule^2 to dB/km

    return gamma_dry_lines + gamma_dry_continuum



def compute_attenuation_point_am(T, P, rho_w, freqs_GHz):
    """
    Computes the specific attenuation for a single spatial point over multiple frequencies.
    Input scalars: T (Kelvin), P (Pa, Total air pressure), rho_water (kg/m^3, water vapor density).
    Input vector: freqs_GHz (shape: Nf,)
    Returns: gamma_dry, gamma_wet (both shape: Nf,) in dB/km
    """

    gamma_wet = compute_attenuation_water_vapor_am(T, P, rho_w, freqs_GHz)
    gamma_dry = compute_attenuation_dry_air_am(T, P, rho_w, freqs_GHz)
    
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
def compute_tau_zenith(T, P, rho_w, dz, freqs_in_GHz):
    """
    Computes the cumulative zenith optical depth (tau_zenith) from the ground up.
    
    Parameters
    ----------
    T : jnp.ndarray
        1D array of shape (Nz,) - Temperature profile in K.
    P : jnp.ndarray
        1D array of shape (Nz,) - Pressure profile in Pa.
    rho_w : jnp.ndarray
        1D array of shape (Nz,) - Water vapor density profile in kg/m^3.
    dz : jnp.ndarray
        1D array of shape (Nz,) - Vertical thickness of each layer in meters.
    freqs_in_GHz : jnp.ndarray
        1D array of shape (Nf,) - Observation frequencies.
        
    Returns
    -------
    tau_zenith : jnp.ndarray
        2D array of shape (Nz, Nf) representing the cumulative zenith optical depth 
        from the ground up to the top of each cell.
    """
    
    # 1. Vectorize point-wise attenuation over the vertical altitude axis (axis 0)
    # The inner compute_attenuation_point outputs gamma_dry and gamma_wet of shape (Nf,)
    # By mapping over (Nz,), the vmap outputs arrays of shape (Nz, Nf)
    gamma_dry, gamma_wet = jax.vmap(
        lambda t, p, rho: compute_attenuation_point(t, p, rho, freqs_in_GHz),
        in_axes=(0, 0, 0)
    )(T, P, rho_w)
    
    # Total specific attenuation in dB/km (Shape: Nz, Nf)
    gamma_gas = gamma_dry + gamma_wet
    
    # 2. Convert specific attenuation (dB/km) to optical depth per layer (Nepers)
    # Broadcast dz from (Nz,) to (Nz, 1) to multiply against the (Nz, Nf) gamma array.
    # Formula: tau = gamma * (dz_in_km) * (ln(10) / 10)
    d_tau = gamma_gas * (dz[:, None] / 1000.0) * (jnp.log(10.0) / 10.0)
    
    # 3. Integrate vertically (cumulative sum from ground z=0 up to layer z)
    # Shape remains (Nz, Nf)
    tau_zenith = jnp.cumsum(d_tau, axis=0)
    
    return tau_zenith

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