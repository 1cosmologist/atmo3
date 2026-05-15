import jax
import jax.numpy as jnp
from . import constants
from functools import partial

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

def intensitySI_to_Tb(I_nu, freq_in_GHz):
    return constants.c**2 / 2. / constants.k_B / (freq_in_GHz * constants.giga)**2. * I_nu


# ==============================================================================
# Point-wise Ice Crystal Polarizability Physics
# ==============================================================================

@jax.jit
def compute_depolarization_factor(m):
    """
    Calculates the depolarization factor (Delta) for a spheroid based on its aspect ratio.
    m: scalar float (aspect ratio)
    Returns: scalar float
    """
    # SAFE EVALUATION: We must avoid NaNs in the e_p and e_o calculations even when 
    # the condition is false, because jnp.where evaluates all branches.
    m_prolate = jnp.where(m > 1.0, m, 1.1) 
    e_p = jnp.sqrt(1.0 - (1.0 / m_prolate)**2)
    delta_prolate = ((1.0 - e_p**2) / e_p**2) * ((1.0 / (2.0 * e_p)) * jnp.log((1.0 + e_p) / (1.0 - e_p)) - 1.0)
    
    m_oblate = jnp.where(m < 1.0, m, 0.9)
    e_o = jnp.sqrt(1.0 - m_oblate**2)
    delta_oblate = (1.0 / e_o**2) * (1.0 - (jnp.sqrt(1.0 - e_o**2) / e_o) * jnp.arcsin(e_o))
    
    # Nested jnp.where handles Sphere (m==1), Prolate (m>1), and Oblate (m<1)
    delta = jnp.where(
        m > 1.0, delta_prolate, 
        jnp.where(m < 1.0, delta_oblate, 1.0 / 3.0)
    )
    return delta


@jax.jit
def compute_intrinsic_polarizabilities(freqs_in_GHz, m):
    """
    Computes inherent polarizabilities.
    freqs_in_GHz: 1D array of shape (Nf,)
    m: scalar float
    Returns: A_par, A_perp of shape (Nf,)
    """
    eps_prime = 3.16
    # Note: Ensure frequency logic matches atmo3 conventions (usually GHz)
    eps_double_prime = 8e-3 * (freqs_in_GHz / 150.0) 
    eps = eps_prime + 1j * eps_double_prime # Shape: (Nf,)
    
    delta = compute_depolarization_factor(m) # Scalar
    
    A_par = (eps - 1.0) / (1.0 + (eps - 1.0) * delta)
    A_perp = (eps - 1.0) / (1.0 + (eps - 1.0) * (1.0 - delta) / 2.0)
    
    return A_par, A_perp


@jax.jit
def compute_effective_polarizability(freqs_in_GHz, m, elevation_deg):
    """
    Projects polarizabilities onto the telescope line of sight.
    Returns both Stokes I and Q components to remain JIT-compatible.
    """
    A_par, A_perp = compute_intrinsic_polarizabilities(freqs_in_GHz, m)
    
    abs2_par = jnp.abs(A_par)**2
    abs2_perp = jnp.abs(A_perp)**2
    imag_par = jnp.imag(A_par)
    imag_perp = jnp.imag(A_perp)
    
    # Sphere (m == 1)
    abs2_v_sphere, abs2_h_sphere = abs2_par, abs2_par
    imag_v_sphere, imag_h_sphere = imag_par, imag_par
    
    # Plate (m < 1)
    abs2_v_plate, abs2_h_plate = abs2_par, abs2_perp
    imag_v_plate, imag_h_plate = imag_par, imag_perp
    
    # Column (m > 1)
    abs2_v_column = abs2_perp
    abs2_h_column = (abs2_par + abs2_perp) / 2.0
    imag_v_column = imag_perp
    imag_h_column = (imag_par + imag_perp) / 2.0
    
    # Apply conditions using jnp.where
    abs2_v = jnp.where(m > 1.0, abs2_v_column, jnp.where(m < 1.0, abs2_v_plate, abs2_v_sphere))
    abs2_h = jnp.where(m > 1.0, abs2_h_column, jnp.where(m < 1.0, abs2_h_plate, abs2_h_sphere))
    imag_v = jnp.where(m > 1.0, imag_v_column, jnp.where(m < 1.0, imag_v_plate, imag_v_sphere))
    imag_h = jnp.where(m > 1.0, imag_h_column, jnp.where(m < 1.0, imag_h_plate, imag_h_sphere))
        
    eps_rad = jnp.radians(elevation_deg)
    cos2_eps = jnp.cos(eps_rad)**2
    sin2_eps = jnp.sin(eps_rad)**2
    
    # Stokes I 
    eff_abs2_I = 0.5 * (abs2_v * cos2_eps + abs2_h * (1.0 + sin2_eps))
    eff_imag_I = 0.5 * (imag_v * cos2_eps + imag_h * (1.0 + sin2_eps))
    
    # Stokes Q
    eff_abs2_Q = 0.5 * (abs2_v - abs2_h) * cos2_eps
    eff_imag_Q = 0.5 * (imag_v - imag_h) * cos2_eps

    return eff_abs2_I, eff_imag_I, eff_abs2_Q, eff_imag_Q


@jax.jit
def compute_polarizability(freqs_in_GHz, m):
    """
    Vectorized computation of complex polarizabilities (alpha_h, alpha_v).
    """
    eps_prime = 3.16
    eps_double_prime = 8e-3 * (freqs_in_GHz / 150.0) 
    eps = eps_prime + 1j * eps_double_prime 
    
    delta = compute_depolarization_factor(m) 
    
    # Note: 4 * pi term added here compared to intrinsic_polarizabilities
    A_par = (eps - 1.0) / (4.0 * jnp.pi * (1.0 + (eps - 1.0) * delta))
    A_perp = (eps - 1.0) / (4.0 * jnp.pi * (1.0 + (eps - 1.0) * (1.0 - delta) / 2.0))
    
    alpha_v_sphere, alpha_h_sphere = A_par, A_par
    alpha_v_plate, alpha_h_plate = A_par, A_perp
    
    alpha_v_column = A_perp
    
    target_abs2_h = (jnp.abs(A_par)**2 + jnp.abs(A_perp)**2) / 2.0
    target_imag_h = (jnp.imag(A_par) + jnp.imag(A_perp)) / 2.0
    
    # Safe eval for square root
    real_h_column = jnp.sqrt(jnp.maximum(0.0, target_abs2_h - target_imag_h**2))
    alpha_h_column = real_h_column + 1j * target_imag_h
    
    alpha_v = jnp.where(m > 1.0, alpha_v_column, jnp.where(m < 1.0, alpha_v_plate, alpha_v_sphere))
    alpha_h = jnp.where(m > 1.0, alpha_h_column, jnp.where(m < 1.0, alpha_h_plate, alpha_h_sphere))
        
    return alpha_h, alpha_v






@jax.jit
def get_ice_emission(T_los, ice_density_los, r_eq, tau_zenith, ds, freqs_in_GHz, m, elevation_deg):
    """
    Computes the Stokes I and Q Planck intensity emission from ice crystals along the LOS.
    
    T_los: jnp.ndarray shape (n_scans, n_los) - Temperature in K
    ice_density_los: jnp.ndarray shape (n_scans, n_los) - Number density (N_0) in particles/m^3
    r_eq: float - Equivalent radius of the ice crystals in meters
    tau_zenith: jnp.ndarray shape (n_scans, Nf, n_los) - Zenith gas optical depth from ground to cell
    ds: jnp.ndarray shape (n_scans, n_los) or (n_los,) - Step size along LOS in meters
    freqs_in_GHz: jnp.ndarray shape (Nf,) - Observation frequencies
    m: float - Aspect ratio
    elevation_deg: float - Telescope elevation for the scan
    """
    
    # 1. Compute Polarizabilities (Shape: Nf)
    _, eff_imag_I, _, eff_imag_Q = compute_effective_polarizability(freqs_in_GHz, m, elevation_deg)
    
    # Intrinsic polarization fraction of the ice crystals (Stokes Q / Stokes I)
    p_gamma = jnp.where(eff_imag_I != 0, eff_imag_Q / eff_imag_I, 0.0) # Shape: (Nf,)
    p_gamma_3d = p_gamma[None, :, None] # Shape: (1, Nf, 1)

    # 2. Physics Constants & Particle Volume
    freqs_Hz = freqs_in_GHz * constants.giga
    k_1 = 2.0 * jnp.pi * freqs_Hz / constants.c # Shape: (Nf,)
    
    V = (4.0 / 3.0) * jnp.pi * (r_eq ** 3)
    
    # 3. Specific Attenuation for Ice (alpha_abs_I)
    k_1_3d = k_1[None, :, None]
    eff_imag_I_3d = eff_imag_I[None, :, None]
    
    sigma_abs_I = k_1_3d * V * eff_imag_I_3d # Shape: (1, Nf, 1)
    
    # alpha = N_0 * sigma
    N_0_3d = ice_density_los[:, None, :] # Shape: (n_scans, 1, n_los)
    alpha_abs_I = N_0_3d * sigma_abs_I # units: m^-1
    
    # 4. Optical Depths per cell (d_tau_ice_I)
    ds_3d = ds[:, None, :] if ds.ndim == 2 else ds[None, None, :]
    d_tau_ice_I = alpha_abs_I * ds_3d
    
    # 5. Telescope Airmass & Gas Attenuation
    zenith_angle_deg = 90.0 - elevation_deg
    m_telescope = 1.0 / (jnp.cos(jnp.radians(zenith_angle_deg)) + 0.50572 * (96.07995 - zenith_angle_deg)**(-1.6364))
    
    # Optical depth of the gas from the ground to the cell along the LOS
    tau_lower_los = tau_zenith * m_telescope
    attenuation_los = jnp.exp(-jnp.clip(tau_lower_los, 0.0, None))
    
    # 6. Radiative Transfer (Rayleigh-Jeans Source Function)
    #T_source = T_los[:, None, :] # Shape: (n_scans, 1, n_los)

    # 6. Radiative Transfer (Planck Source Function)
    B_nu_source = B_nu_T(freqs_in_GHz[None, :, None], T_los[:, None, :]) # Shape: (n_scans, Nf, n_los)
    
    # Raw emission from the ice in the cell
    B_nu_ice_cell_I = B_nu_source * (1.0 - jnp.exp(-d_tau_ice_I))
    
    # Attenuate the ice emission by the atmospheric gas below it
    B_nu_ice_layer_I = B_nu_ice_cell_I * attenuation_los
    B_nu_ice_layer_Q = p_gamma_3d * B_nu_ice_layer_I
    
    # 7. Integrate along the Line of Sight (sum across axis 2)
    B_nu_sky_I = jnp.sum(B_nu_ice_layer_I, axis=2)
    B_nu_sky_Q = jnp.sum(B_nu_ice_layer_Q, axis=2)
    
    return B_nu_sky_I, B_nu_sky_Q




# ==============================================================================
# Scattering Phase Matrix and Incident Radiation
# ==============================================================================

@jax.jit
def compute_earth_frame_phase_matrix(alpha_h, alpha_v, theta_grid, phi_grid, delta):
    """
    Compute the phase matrix elements in the Earth-frame convention.
    Inputs must be pre-broadcasted or broadcast-compatible.
    
    alpha_h, alpha_v: Complex polarizability arrays
    theta_grid, phi_grid: Angular grids in radians
    delta: Telescope elevation angle in radians
    """
    abs_h2 = jnp.abs(alpha_h)**2
    abs_v2 = jnp.abs(alpha_v)**2
    
    I_H = abs_h2 * (1.0 - (jnp.sin(theta_grid) * jnp.sin(phi_grid))**2)
    
    ray_proj = alpha_v * jnp.cos(delta) * jnp.cos(theta_grid) - \
               alpha_h * jnp.sin(delta) * jnp.sin(theta_grid) * jnp.cos(phi_grid)
               
    I_V = (abs_h2 * jnp.sin(delta)**2 + abs_v2 * jnp.cos(delta)**2) - jnp.abs(ray_proj)**2
    
    M11_earth = 0.5 * (I_V + I_H)
    M21_earth = 0.5 * (I_V - I_H)
    
    return M11_earth, M21_earth



# ==============================================================================
# 3D Vectorized Scattering B_in (Incoming Planck Radiation)
# ==============================================================================

@partial(jax.jit, static_argnames=['consider_atmospheric_emission'])
def compute_B_in_3d(theta_grid, T_los, T_ground, tau_zenith, tau_total_zenith, freqs_in_GHz,
                    consider_atmospheric_emission=True):
    """
    Compute the incoming Planck intensity field seen by every cell.
    
    theta_grid: (N_theta, N_phi) - Angular integration grid
    T_los: (n_scans, n_los) - Physical temperatures along the LOS
    T_ground: (n_scans,) or float - Ground temperature
    tau_zenith: (n_scans, Nf, n_los) - Zenith optical depth from ground to the cell
    tau_total_zenith: (n_scans, Nf, 1) - Total zenith optical depth of the whole atmosphere
    freqs_in_GHz: (Nf,) - Observation frequencies
    """
    # Broadcast to 5D: (n_scans, Nf, n_los, N_theta, N_phi)
    tg = theta_grid[None, None, None, :, :] 
    
    # Broadcast Temperatures to 5D compatible shapes
    T_mid = T_los[:, None, :, None, None]
    T_gnd_3d = T_ground if jnp.isscalar(T_ground) else T_ground[:, None, None, None, None]
    freqs_5d = freqs_in_GHz[None, :, None, None, None]
    
    # Convert physical temperatures to Planck Source Intensities
    B_mid = B_nu_T(freqs_5d, T_mid)
    B_gnd = B_nu_T(freqs_5d, T_gnd_3d)
    
    tau_below = jnp.clip(tau_zenith, 0.0, None)[:, :, :, None, None]
    tau_above = jnp.clip(tau_total_zenith - tau_zenith, 0.0, None)[:, :, :, None, None]
    
    if consider_atmospheric_emission:
        theta_deg = jnp.clip(jnp.degrees(tg), 0.0, 90.0)
        # Approximate zenith-scaled airmass formulation for the scattered rays
        m_sky = 1.0 / (jnp.cos(tg) + 0.50572 * (96.07995 - theta_deg)**(-1.6364))
        B_sky = B_mid * (1.0 - jnp.exp(-tau_above * m_sky))
        
        m_gnd = 1.0 / jnp.maximum(jnp.cos(jnp.pi - tg), 0.01)
        B_gnd_eff = B_gnd * jnp.exp(-tau_below * m_gnd) + B_mid * (1.0 - jnp.exp(-tau_below * m_gnd))
    else:
        B_sky = jnp.zeros_like(tg)
        B_gnd_eff = jnp.full_like(tg, B_gnd)

    bound_sky = jnp.pi / 2.0
    B_in = jnp.where(tg <= bound_sky, B_sky, B_gnd_eff)
    
    return B_in


# ==============================================================================
# Main Scattering Integration
# ==============================================================================

@partial(jax.jit, static_argnames=['N_theta', 'N_phi', 'consider_atmospheric_emission'])
def get_ice_scattering(T_los, ice_density_los, r_eq, ds, freqs_in_GHz, m, elevation_deg,
                       tau_zenith, tau_total_zenith,
                       N_theta=30, N_phi=30, consider_atmospheric_emission=True):
    """
    Computes the Stokes I and Q Planck scattered intensity from ice crystals.
    
    T_los: (n_scans, n_los) 
    ice_density_los: (n_scans, n_los) - N_0 in particles/m^3
    r_eq: float - equivalent radius in meters
    ds: (n_scans, n_los) or (n_los,) - physical step size
    freqs_in_GHz: (Nf,)
    m: float - aspect ratio
    elevation_deg: float - Telescope elevation angle
    tau_zenith: (n_scans, Nf, n_los) - Zenith optical depth from ground to cell
    tau_total_zenith: (n_scans, Nf, 1) - Total atmosphere zenith optical depth
    """
    delta = jnp.radians(elevation_deg)
    
    # 1. Telescope Airmass & LOS Optical Depth to Telescope
    zenith_angle_deg = 90.0 - elevation_deg
    m_telescope = 1.0 / (jnp.cos(jnp.radians(zenith_angle_deg)) + 0.50572 * (96.07995 - zenith_angle_deg)**(-1.6364))
    
    tau_lower_los = tau_zenith * m_telescope
    attenuation_los = jnp.exp(-jnp.clip(tau_lower_los, 0.0, None))
    
    # 2. Angular Grid Setup
    theta = jnp.linspace(0, jnp.pi, N_theta)
    phi = jnp.linspace(0, 2*jnp.pi, N_phi, endpoint=False)
    dtheta, dphi = theta[1] - theta[0], phi[1] - phi[0]
    
    theta_grid, phi_grid = jnp.meshgrid(theta, phi, indexing='ij')
    sin_tg = jnp.sin(theta_grid)[None, None, None, :, :] 
    
    # 3. Polarizability & Phase Matrix
    alpha_h, alpha_v = compute_polarizability(freqs_in_GHz, m)
    
    alpha_h_exp = alpha_h[:, None, None]
    alpha_v_exp = alpha_v[:, None, None]
    
    tg_pm = theta_grid[None, :, :]
    pg_pm = phi_grid[None, :, :]
    
    M11, M21 = compute_earth_frame_phase_matrix(alpha_h_exp, alpha_v_exp, tg_pm, pg_pm, delta)
    M11_5d = M11[None, :, None, :, :]
    M21_5d = M21[None, :, None, :, :]
    
    # 4. Incoming Radiation Field
    # Extract ground temperature from the lowest cell of the LOS for each scan
    T_ground = T_los[:, 0]
    B_in = compute_B_in_3d(theta_grid, T_los, T_ground, tau_zenith, tau_total_zenith, freqs_in_GHz,
                           consider_atmospheric_emission)
    
    # 5. Angular Integration (dOmega = sin(theta) dtheta dphi)
    integral_I = jnp.sum(B_in * M11_5d * sin_tg, axis=(-2, -1)) * dtheta * dphi # Shape: (n_scans, Nf, n_los)
    integral_Q = jnp.sum(B_in * M21_5d * sin_tg, axis=(-2, -1)) * dtheta * dphi 
    
    # 6. Constants & Final Assembly
    freqs_Hz = freqs_in_GHz * constants.giga
    k_4 = (2.0 * jnp.pi * freqs_Hz / constants.c)**4 
    V_squared = (16.0 / 9.0) * (jnp.pi**2) * (r_eq**6)
    
    C_base = k_4[None, :, None] * V_squared 
    
    ds_3d = ds[:, None, :] if ds.ndim == 2 else ds[None, None, :]
    C_z = ice_density_los[:, None, :] * ds_3d
    
    # Apply attenuation towards the telescope
    layer_signal_I = integral_I * C_z * attenuation_los * C_base
    layer_signal_Q = integral_Q * C_z * attenuation_los * C_base
    
    # 7. Integrate along the Line of Sight
    B_nu_sky_I = jnp.sum(layer_signal_I, axis=2)
    B_nu_sky_Q = jnp.sum(layer_signal_Q, axis=2)
    
    return B_nu_sky_I, B_nu_sky_Q