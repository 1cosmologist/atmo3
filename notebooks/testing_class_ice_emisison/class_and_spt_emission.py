import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from atm_tools import alpha_specific_function

pi = constants.pi 
c = constants.c


#The first part is for SPT 

def create_constant_ice_layer(altitudes, layer_bottom, layer_top, ice_density):
    """
    Build a one-dimensional ice-particle density profile with a constant value
    between two altitude bounds.

    The profile is zero outside the requested layer and equal to
    ``ice_density`` inside the layer, inclusive of the boundary values.

    Parameters
    ----------
    altitudes : numpy.ndarray
        One-dimensional altitude grid in meters.
    layer_bottom : float
        Lower altitude bound of the ice layer in meters.
    layer_top : float
        Upper altitude bound of the ice layer in meters.
    ice_density : float
        Constant particle density assigned inside the layer, in particles m^-3.

    Returns
    -------
    numpy.ndarray
        Array with the same shape as ``altitudes`` containing the ice-particle
        density profile.
    """
    n = np.zeros_like(altitudes)  # Initialize an array of zeros
    in_layer = (altitudes >= layer_bottom) & (altitudes <= layer_top)  # Find indices within the layer
    n[in_layer] = ice_density  # Set density for those indices
    return n

def sigma_scattering(frequency, volume, A):
    """
    Compute the Rayleigh scattering cross section of a small ice particle.

    This implements the standard dipole-limit expression used throughout the
    module, where the scattering scales with frequency to the fourth power and
    with the square of the particle volume.

    Parameters
    ----------
    frequency : float
        Incident frequency in hertz.
    volume : float
        Particle volume in cubic meters.
    A : complex
        Complex particle polarizability-like coefficient controlling the
        scattering strength.

    Returns
    -------
    float
        Scattering cross section in square meters.
    """
    w = 2 * pi * frequency #angular frequency

    sigma_sca = w**4*volume**2*np.abs(A)**2/(6*pi*c**4)
    
    return sigma_sca

def sigma_absorption(frequency, volume, A):
    """
    Compute the Rayleigh absorption cross section of a small ice particle.

    The absorption term depends on the imaginary part of the complex
    polarizability coefficient and scales linearly with both frequency and
    particle volume.

    Parameters
    ----------
    frequency : float
        Incident frequency in hertz.
    volume : float
        Particle volume in cubic meters.
    A : complex
        Complex particle polarizability-like coefficient controlling the
        absorption strength.

    Returns
    -------
    float
        Absorption cross section in square meters.
    """
    w = 2 * pi * frequency #angular frequency

    sigma_abs = w*volume*np.imag(A)/c

    return sigma_abs

import numpy as np
import scipy.constants as constants

import numpy as np
import scipy.constants as constants

# ====================================================================
# SHARED ICE PHYSICS (Vectorized for Frequencies and Aspect Ratios)
# ====================================================================

def compute_depolarization_factor(aspect_ratio):
    """
    Compute the depolarization factor along the symmetry axis of a spheroid.

    The function supports spheres, prolate spheroids (columns), and oblate
    spheroids (plates). The returned value is the depolarization factor of the
    symmetry axis used in the polarizability expressions below.

    Parameters
    ----------
    aspect_ratio : float or array-like
        Ratio of the symmetry-axis length to the perpendicular-axis length.
        Values greater than 1 correspond to prolate particles, values less than
        1 correspond to oblate particles, and 1 corresponds to a sphere.

    Returns
    -------
    numpy.ndarray
        One-dimensional array of depolarization factors, one per input aspect
        ratio.
    """
    m = np.atleast_1d(aspect_ratio).flatten()
    delta = np.zeros_like(m, dtype=float)
    
    mask_sphere = (m == 1.0)
    delta[mask_sphere] = 1.0 / 3.0
    
    mask_prolate = (m > 1.0)  # Columns
    if np.any(mask_prolate):
        m_p = m[mask_prolate]
        e = np.sqrt(1.0 - (1.0 / m_p)**2)
        delta[mask_prolate] = ((1.0 - e**2) / e**2) * ((1.0 / (2.0 * e)) * np.log((1.0 + e) / (1.0 - e)) - 1.0)
        
    mask_oblate = (m < 1.0)   # Plates
    if np.any(mask_oblate):
        m_o = m[mask_oblate]
        e = np.sqrt(1.0 - m_o**2)
        delta[mask_oblate] = (1.0 / e**2) * (1.0 - (np.sqrt(1.0 - e**2) / e) * np.arcsin(e))
        
    return delta

def compute_intrinsic_polarizabilities(frequency, aspect_ratio):
    """
    Compute intrinsic polarizability factors before geometric projection.

    The function evaluates the complex dielectric response of ice and returns
    the parallel and perpendicular polarizability components for each
    combination of aspect ratio and frequency.

    Parameters
    ----------
    frequency : float or array-like
        Frequencies in hertz.
    aspect_ratio : float or array-like
        Particle aspect ratio(s). See :func:`compute_depolarization_factor` for
        the interpretation of the values.

    Returns
    -------
    tuple of numpy.ndarray
        ``(A_par, A_perp)`` with shape ``(Nm, Nf)`` each, where ``Nm`` is the
        number of aspect ratios and ``Nf`` is the number of frequencies.
    """
    freq = np.atleast_1d(frequency).flatten()
    m = np.atleast_1d(aspect_ratio).flatten()
    
    eps_prime = 3.16
    eps_double_prime = 8e-3 * (freq / 150e9) 
    eps = eps_prime + 1j * eps_double_prime # Shape: (Nf,)
    
    delta = compute_depolarization_factor(m) # Shape: (Nm,)
    
    # Broadcast to (Nm, Nf)
    eps_2d = eps[np.newaxis, :]
    delta_2d = delta[:, np.newaxis]
    
    A_par = (eps_2d - 1) / (1 + (eps_2d - 1) * delta_2d)
    A_perp = (eps_2d - 1) / (1 + (eps_2d - 1) * (1 - delta_2d) / 2.0)
    
    return A_par, A_perp

def compute_effective_polarizability(frequency, aspect_ratio, elevation, stokes_param='I'):
    """
    Project intrinsic polarizabilities onto the telescope line of sight.

    This combines the parallel and perpendicular particle responses into an
    effective quantity for either Stokes ``I`` or ``Q`` at a given elevation
    angle. The projection is vectorized over aspect ratio and frequency.

    Parameters
    ----------
    frequency : float or array-like
        Frequencies in hertz.
    aspect_ratio : float or array-like
        Particle aspect ratio(s).
    elevation : float
        Telescope elevation angle in degrees.
    stokes_param : {'I', 'Q'}, optional
        Stokes component to compute. ``'I'`` returns the total intensity
        response and ``'Q'`` returns the linear polarization contrast.

    Returns
    -------
    tuple of numpy.ndarray
        ``(eff_abs2, eff_imag)`` with shape ``(Nm, Nf)``. The first array is
        the effective squared magnitude of the polarizability and the second is
        the effective imaginary component.
    """
    m = np.atleast_1d(aspect_ratio).flatten()
    A_par, A_perp = compute_intrinsic_polarizabilities(frequency, m) # Shapes: (Nm, Nf)
    
    abs2_par = np.abs(A_par)**2
    abs2_perp = np.abs(A_perp)**2
    imag_par = np.imag(A_par)
    imag_perp = np.imag(A_perp)
    
    abs2_v = np.zeros_like(abs2_par)
    abs2_h = np.zeros_like(abs2_par)
    imag_v = np.zeros_like(imag_par)
    imag_h = np.zeros_like(imag_par)
    
    # 1D masks correctly applied across the 2D frequency arrays
    mask_sphere = (m == 1.0)
    mask_plate = (m < 1.0)
    mask_column = (m > 1.0)
    
    abs2_v[mask_sphere] = abs2_par[mask_sphere]
    abs2_h[mask_sphere] = abs2_par[mask_sphere]
    imag_v[mask_sphere] = imag_par[mask_sphere]
    imag_h[mask_sphere] = imag_par[mask_sphere]
    
    abs2_v[mask_plate] = abs2_par[mask_plate]
    abs2_h[mask_plate] = abs2_perp[mask_plate]
    imag_v[mask_plate] = imag_par[mask_plate]
    imag_h[mask_plate] = imag_perp[mask_plate]
    
    abs2_v[mask_column] = abs2_perp[mask_column]
    abs2_h[mask_column] = (abs2_par[mask_column] + abs2_perp[mask_column]) / 2.0
    imag_v[mask_column] = imag_perp[mask_column]
    imag_h[mask_column] = (imag_par[mask_column] + imag_perp[mask_column]) / 2.0
        
    eps_rad = np.radians(elevation)
    
    if stokes_param == 'I':
        eff_abs2 = 0.5 * (abs2_v * np.cos(eps_rad)**2 + abs2_h * (1.0 + np.sin(eps_rad)**2))
        eff_imag = 0.5 * (imag_v * np.cos(eps_rad)**2 + imag_h * (1.0 + np.sin(eps_rad)**2))
    elif stokes_param == 'Q':
        eff_abs2 = 0.5 * (abs2_v - abs2_h) * np.cos(eps_rad)**2
        eff_imag = 0.5 * (imag_v - imag_h) * np.cos(eps_rad)**2
    else:
        raise ValueError("stokes_param must be 'I' or 'Q'")

    return eff_abs2, eff_imag

def compute_polarizability(frequency, aspect_ratio=1.0):
    """
    Compute complex horizontal and vertical polarizabilities.

    This function converts the intrinsic polarizabilities into the
    horizontal and vertical coefficients from the polarizability tensor that can be used in the phase matrix
    calculations for the scattering-plane and Earth-frame methods.

    Parameters
    ----------
    frequency : float or array-like
        Frequencies in hertz.
    aspect_ratio : float or array-like, optional
        Particle aspect ratio(s). Default is ``1.0`` for a sphere.

    Returns
    -------
    tuple of numpy.ndarray
        ``(alpha_h, alpha_v)`` with shape ``(Nm, Nf)`` each.
    """
    freq = np.atleast_1d(frequency).flatten()
    m = np.atleast_1d(aspect_ratio).flatten()
    
    eps_prime = 3.16
    eps_double_prime = 8e-3 * (freq / 150e9) 
    eps = eps_prime + 1j * eps_double_prime 
    
    delta = compute_depolarization_factor(m) 
    
    eps_2d = eps[np.newaxis, :]
    delta_2d = delta[:, np.newaxis]
    
    A_par = (eps_2d - 1) / (4 * np.pi * (1 + (eps_2d - 1) * delta_2d))
    A_perp = (eps_2d - 1) / (4 * np.pi * (1 + (eps_2d - 1) * (1 - delta_2d) / 2.0))
    
    alpha_v = np.zeros_like(A_par)
    alpha_h = np.zeros_like(A_par)
    
    # 1D masks correctly applied
    mask_sphere = (m == 1.0)
    mask_plate = (m < 1.0)
    mask_column = (m > 1.0)
    
    alpha_v[mask_sphere] = A_par[mask_sphere]
    alpha_h[mask_sphere] = A_par[mask_sphere]
    
    alpha_v[mask_plate] = A_par[mask_plate]
    alpha_h[mask_plate] = A_perp[mask_plate]
    
    alpha_v[mask_column] = A_perp[mask_column]
    
    target_abs2_h = (np.abs(A_par)**2 + np.abs(A_perp)**2) / 2.0
    target_imag_h = (np.imag(A_par) + np.imag(A_perp)) / 2.0
    
    real_h = np.sqrt(np.maximum(0, target_abs2_h - target_imag_h**2))
    alpha_h[mask_column] = real_h[mask_column] + 1j * target_imag_h[mask_column]
        
    return alpha_h, alpha_v


# ====================================================================
# SPT METHOD ENGINE
# ====================================================================

def compute_T_RJ_ice2(frequency, altitudes, Temperature, Pressure, P_water, elevation, ice_density, radius_eq, aspect_ratio=1.0, process='total', stokes_param='I'):
    """
    Compute Rayleigh-Jeans brightness temperature using the SPT formulation.

    The implementation combines atmospheric attenuation, ice scattering, and
    ice absorption/emission layer by layer. It supports scalar or vector inputs
    for frequency, equivalent radius, and aspect ratio, and returns the result
    with dimensions ordered as ``(Nf, Na, Nm)``.

    Parameters
    ----------
    frequency : float or array-like
        Frequency or frequencies in hertz.
    altitudes : numpy.ndarray
        One-dimensional altitude grid in meters.
    Temperature : numpy.ndarray
        Physical temperature profile on the altitude grid in kelvin.
    Pressure : numpy.ndarray
        Atmospheric pressure profile on the altitude grid.
    P_water : numpy.ndarray
        Water-vapor partial pressure profile on the altitude grid.
    elevation : float
        Telescope elevation angle in degrees.
    ice_density : float or array-like
        Ice-particle number density profile in particles m^-3. A scalar is
        broadcast over altitude.
    radius_eq : float or array-like
        Equivalent particle radius values in meters.
    aspect_ratio : float or array-like, optional
        Particle aspect ratio(s). Default is ``1.0``.
    process : {'total', 'scattering', 'emission'}, optional
        Select which physical contribution to return.
    stokes_param : {'I', 'Q'}, optional
        Polarization channel to compute.

    Returns
    -------
    numpy.ndarray
        Brightness temperature array with shape ``(Nf, Na, Nm)``.
    """
    c = constants.c
    
    freq = np.atleast_1d(frequency).flatten()
    m_arr = np.atleast_1d(aspect_ratio).flatten()
    r_eq = np.atleast_1d(radius_eq).flatten()
    
    zenith_angle = 90.0 - elevation
    air_mass = 1.0 / (np.cos(np.radians(zenith_angle)) + 0.50572 * (96.07995 - zenith_angle)**(-1.6364))
    
    # Safeguard if a scalar is passed for ice density
    ice_den = np.asarray(ice_density)
    if ice_den.ndim == 0:
        ice_den = np.full((len(altitudes), 1), ice_den.item())
        
    dz = np.diff(altitudes).reshape(-1, 1, 1, 1)                  
    V = ((4.0 / 3.0) * np.pi * r_eq**3).reshape(1, 1, 1, -1)      
    ice_den = ice_den.reshape(ice_den.shape[0], 1, 1, -1) 
    
    k_1 = (2 * np.pi * freq / c).reshape(1, 1, -1, 1)             
    k_4 = (k_1**4)
    geom_factor = 1.0 / (6.0 * np.pi)

    eff_abs2_req, eff_imag_req = compute_effective_polarizability(freq, m_arr, elevation, stokes_param=stokes_param)
    eff_abs2_I, eff_imag_I = compute_effective_polarizability(freq, m_arr, elevation, stokes_param='I')
    
    eff_abs2_req = eff_abs2_req[np.newaxis, :, :, np.newaxis]
    eff_imag_req = eff_imag_req[np.newaxis, :, :, np.newaxis]
    eff_abs2_I = eff_abs2_I[np.newaxis, :, :, np.newaxis]
    eff_imag_I = eff_imag_I[np.newaxis, :, :, np.newaxis]

    sigma_sca_req = geom_factor * k_4 * (V**2) * eff_abs2_req
    sigma_sca_I   = geom_factor * k_4 * (V**2) * eff_abs2_I
    sigma_abs_req = k_1 * V * eff_imag_req
    sigma_abs_I   = k_1 * V * eff_imag_I

    alpha_sca_req = ice_den * sigma_sca_req
    alpha_sca_I   = ice_den * sigma_sca_I
    alpha_abs_req = ice_den * sigma_abs_req
    alpha_abs_I   = ice_den * sigma_abs_I

    alpha_sca_req_mid = (alpha_sca_req[:-1] + alpha_sca_req[1:]) / 2.0
    alpha_sca_I_mid   = (alpha_sca_I[:-1] + alpha_sca_I[1:]) / 2.0
    alpha_abs_req_mid = (alpha_abs_req[:-1] + alpha_abs_req[1:]) / 2.0
    alpha_abs_I_mid   = (alpha_abs_I[:-1] + alpha_abs_I[1:]) / 2.0

    d_tau_sca_req = alpha_sca_req_mid * dz
    d_tau_sca_I   = alpha_sca_I_mid * dz
    d_tau_abs_req = alpha_abs_req_mid * dz
    d_tau_abs_I   = alpha_abs_I_mid * dz

    d_tau_ext_I = d_tau_sca_I + d_tau_abs_I
    
    tau_below_ice = np.cumsum(d_tau_ext_I, axis=0)
    tau_below_ice = np.insert(tau_below_ice[:-1], 0, 0, axis=0) 

    alpha_atm = alpha_specific_function(altitudes, freq, Temperature, Pressure, P_water)
    alpha_atm_mid = (alpha_atm[:-1] + alpha_atm[1:]) / 2.0
    d_tau_atm = alpha_atm_mid * np.diff(altitudes)[:, None] 
    
    tau_below_atm = np.cumsum(d_tau_atm, axis=0)
    tau_below_atm = np.insert(tau_below_atm[:-1], 0, 0, axis=0)
    
    tau_below_atm_4D = tau_below_atm[:, np.newaxis, :, np.newaxis]
    T_mid = ((Temperature[:-1] + Temperature[1:]) / 2.0).reshape(-1, 1, 1, 1)

    attenuation = np.exp(-(tau_below_ice + tau_below_atm_4D) * air_mass)

    T_layers_sca = 0.0
    T_layers_abs = 0.0

    if process in ['scattering', 'total']:
        T_source_sca = Temperature[0] / 2.0 
        p_gamma_sca = np.divide(d_tau_sca_req, d_tau_sca_I, out=np.zeros_like(d_tau_sca_I), where=(d_tau_sca_I != 0))
        T_layers_sca_I = T_source_sca * (1 - np.exp(-d_tau_sca_I * air_mass))
        T_layers_sca = p_gamma_sca * T_layers_sca_I * attenuation

    if process in ['emission', 'total']:
        T_source_abs = T_mid
        p_gamma_abs = np.divide(d_tau_abs_req, d_tau_abs_I, out=np.zeros_like(d_tau_abs_I), where=(d_tau_abs_I != 0))
        T_layers_abs_I = T_source_abs * (1 - np.exp(-d_tau_abs_I * air_mass))
        T_layers_abs = p_gamma_abs * T_layers_abs_I * attenuation

    if process == 'scattering':
        T_layers = T_layers_sca
    elif process == 'emission':
        T_layers = T_layers_abs
    else: 
        T_layers = T_layers_sca + T_layers_abs

    # Sum across altitude: Current shape is (Nm, Nf, Na)
    T_sky_internal = np.sum(T_layers, axis=0) 
    
    # Transpose to user requested shape: (Nf, Na, Nm)
    # Axes mapping: 0(Nm)->2, 1(Nf)->0, 2(Na)->1
    T_sky_final = np.transpose(T_sky_internal, (1, 2, 0))
    
    return T_sky_final


# ====================================================================
# CLASS METHOD ENGINE & PHASE MATRICES
# ====================================================================

def compute_earth_frame_phase_matrix(alpha_h, alpha_v, theta_grid, phi_grid, delta):
    """
    Compute the phase matrix elements in the Earth-frame convention.

    This formulation projects the particle polarizability tensor directly onto
    the observation geometry defined by ``theta_grid`` and ``phi_grid``.
    All inputs are broadcast together, so the output shape is the common
    broadcast shape of the array inputs.

    Parameters
    ----------
    alpha_h : numpy.ndarray
        Horizontal polarizability array. Typical shape is ``(Nm, Nf)`` or
        ``(..., Nf)`` when pre-expanded for broadcasting.
    alpha_v : numpy.ndarray
        Vertical polarizability array. Must be broadcast-compatible with
        ``alpha_h``.
    theta_grid : numpy.ndarray
        Polar angle grid in radians. Typical shape is ``(N_theta, N_phi)`` or
        ``(1, 1, N_theta, N_phi)`` when pre-broadcast.
    phi_grid : numpy.ndarray
        Azimuth angle grid in radians. Must be broadcast-compatible with
        ``theta_grid``.
    delta : float
        Telescope elevation angle in radians, usually a scalar.

    Returns
    -------
    tuple of numpy.ndarray
        ``(M11_earth, M21_earth)`` with the broadcast shape of the inputs.
        In the main core routine this is typically ``(Nf, Nm, N_theta,
        N_phi)`` after pre-expanding the frequency and aspect-ratio axes.
    """
    abs_h2 = np.abs(alpha_h)**2
    abs_v2 = np.abs(alpha_v)**2
    
    I_H = abs_h2 * (1.0 - (np.sin(theta_grid) * np.sin(phi_grid))**2)
    
    ray_proj = alpha_v * np.cos(delta) * np.cos(theta_grid) - \
               alpha_h * np.sin(delta) * np.sin(theta_grid) * np.cos(phi_grid)
    I_V = (abs_h2 * np.sin(delta)**2 + abs_v2 * np.cos(delta)**2) - np.abs(ray_proj)**2
    
    M11_earth = 0.5 * (I_V + I_H)
    M21_earth = 0.5 * (I_V - I_H)
    
    return M11_earth, M21_earth

def compute_scattering_plane_phase_matrix(alpha_h, alpha_v, theta_grid, phi_grid, delta):
    """
    Compute the phase matrix elements in the scattering-plane convention.

    The function builds the orthonormal basis of the scattering plane for each
    incident direction, applies the particle response tensor, and returns the
    two matrix elements needed by the Stokes-vector integration. All inputs are
    broadcast together, so the output shape is the common broadcast shape of
    the array inputs.

    Parameters
    ----------
    alpha_h : numpy.ndarray
        Horizontal polarizability array. Typical shape is ``(Nm, Nf)`` or
        ``(..., Nf)`` when pre-expanded for broadcasting.
    alpha_v : numpy.ndarray
        Vertical polarizability array. Must be broadcast-compatible with
        ``alpha_h``.
    theta_grid : numpy.ndarray
        Polar angle grid in radians. Typical shape is ``(N_theta, N_phi)`` or
        ``(1, 1, N_theta, N_phi)`` when pre-broadcast.
    phi_grid : numpy.ndarray
        Azimuth angle grid in radians. Must be broadcast-compatible with
        ``theta_grid``.
    delta : float
        Telescope elevation angle in radians, usually a scalar.

    Returns
    -------
    tuple of numpy.ndarray
        ``(M11_rotated, M21_rotated)`` with the broadcast shape of the inputs.
        In the main core routine this is typically ``(Nf, Nm, N_theta,
        N_phi)`` after pre-expanding the frequency and aspect-ratio axes.
    """
    s_i = np.stack([-np.sin(theta_grid) * np.cos(phi_grid), 
                    -np.sin(theta_grid) * np.sin(phi_grid), 
                    -np.cos(theta_grid)], axis=-1)
    
    s_s = np.array([-np.cos(delta), 0.0, -np.sin(delta)])
    s_s = np.broadcast_to(s_s, s_i.shape)
    
    n_scat = np.cross(s_i, s_s, axis=-1)
    norm = np.linalg.norm(n_scat, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-12, 1e-12, norm) 
    e_perp = n_scat / norm
    
    e_par_i = np.cross(e_perp, s_i, axis=-1)
    e_par_s = np.cross(e_perp, s_s, axis=-1)
    
    def apply_tensor(v):
        return np.stack([alpha_h * v[..., 0], alpha_h * v[..., 1], alpha_v * v[..., 2]], axis=-1)
    
    S1 = np.sum(e_perp * apply_tensor(e_perp), axis=-1)
    S2 = np.sum(e_par_s * apply_tensor(e_par_i), axis=-1)
    S3 = np.sum(e_par_s * apply_tensor(e_perp), axis=-1)
    S4 = np.sum(e_perp * apply_tensor(e_par_i), axis=-1)
    
    F11 = 0.5 * (np.abs(S1)**2 + np.abs(S2)**2 + np.abs(S3)**2 + np.abs(S4)**2)
    F21 = 0.5 * (np.abs(S2)**2 + np.abs(S3)**2 - np.abs(S1)**2 - np.abs(S4)**2)
    F31 = np.real(S2 * np.conj(S4) + S3 * np.conj(S1))
    
    e_v = np.array([-np.sin(delta), 0.0, np.cos(delta)])
    e_v = np.broadcast_to(e_v, s_i.shape)
    
    cos_psi = np.sum(e_v * e_par_s, axis=-1)
    sin_psi = -np.sum(e_v * e_perp, axis=-1)
    
    cos_2psi = cos_psi**2 - sin_psi**2
    sin_2psi = 2 * sin_psi * cos_psi
    
    M11_rotated = F11 
    M21_rotated = F21 * cos_2psi - F31 * sin_2psi
    
    return M11_rotated, M21_rotated

def compute_T_in(theta_grid, z_layers, altitudes, T_phys_profile, T_ground, tau_above_atm_f, tau_below_atm_f, 
                 consider_earth_curvature=True, consider_atmospheric_emission=True):
    """
    Compute the incoming brightness temperature field seen by an ice layer.

    The result is evaluated for one frequency at a time and returns a 3D array
    over altitude layers and angular coordinates. The function can account for
    atmospheric emission, ground emission, and Earth curvature effects.

    Parameters
    ----------
    theta_grid : numpy.ndarray
        Polar-angle grid in radians, typically created by ``meshgrid``.
    z_layers : numpy.ndarray
        Layer-center altitudes in meters.
    altitudes : numpy.ndarray
        One-dimensional altitude grid in meters.
    T_phys_profile : numpy.ndarray
        Physical temperature profile on the altitude grid in kelvin.
    T_ground : float
        Ground brightness temperature in kelvin.
    tau_above_atm_f : numpy.ndarray
        Cumulative atmospheric optical depth above each layer for the selected
        frequency.
    tau_below_atm_f : numpy.ndarray
        Cumulative atmospheric optical depth below each layer for the selected
        frequency.
    consider_earth_curvature : bool, optional
        If ``True``, use a curved-Earth limb geometry; otherwise use a flat-
        Earth split between sky and ground.
    consider_atmospheric_emission : bool, optional
        If ``True``, include atmospheric self-emission and ground attenuation.

    Returns
    -------
    numpy.ndarray
        Incoming brightness temperature with shape ``(Nz, N_theta, N_phi)``.
    """
    R_e = 6371e3
    
    # 3D Arrays: (Nz, N_theta, N_phi)
    tg = theta_grid[None, :, :] 
    z_exp = z_layers[:, None, None]
    T_mid = ((T_phys_profile[:-1] + T_phys_profile[1:]) / 2.0)[:, None, None]
    
    # FIX: Prevent np.exp overflow by clipping rogue negative opacities at extreme frequencies
    tau_above = np.clip(tau_above_atm_f, 0, None)[:, None, None]
    tau_below = np.clip(tau_below_atm_f, 0, None)[:, None, None]
    
    target_shape = (len(z_layers), theta_grid.shape[0], theta_grid.shape[1])
    
    if consider_atmospheric_emission:
        theta_deg = np.clip(np.degrees(tg), 0, 90.0)
        m_sky = 1.0 / (np.cos(tg) + 0.50572 * (96.07995 - theta_deg)**(-1.6364))
        T_sky = T_mid * (1.0 - np.exp(-tau_above * m_sky))
        
        m_gnd = 1.0 / np.maximum(np.cos(np.pi - tg), 0.01)
        T_gnd_effective = T_ground * np.exp(-tau_below * m_gnd) + T_mid * (1.0 - np.exp(-tau_below * m_gnd))
    else:
        T_sky = np.zeros(target_shape)
        T_gnd_effective = np.full(target_shape, T_ground)

    if consider_earth_curvature:
        theta_h = np.sqrt(2 * z_exp / R_e)
        bound_sky = np.pi / 2
        bound_limb = (np.pi / 2) + theta_h
        
        if consider_atmospheric_emission:
            z_min = (R_e + z_exp) * np.sin(tg) - R_e
            T_limb_flat = np.interp(z_min.flatten(), altitudes, T_phys_profile)
            T_limb = T_limb_flat.reshape(z_min.shape)
        else:
            T_limb = np.zeros(target_shape) 
            
        conditions = [
            tg <= bound_sky,
            (tg > bound_sky) & (tg <= bound_limb),
            tg > bound_limb
        ]
        choices = [T_sky, T_limb, T_gnd_effective]
        
    else:
        bound_sky = np.pi / 2
        conditions = [
            tg <= bound_sky,
            tg > bound_sky
        ]
        choices = [T_sky, T_gnd_effective]
    
    res = np.select(conditions, choices)
    return np.broadcast_to(res, target_shape)


def _compute_T_RJ_ice_CLASS_core(frequency, altitudes, Temperature, Pressure, P_water, elevation, ice_density, 
                                 radius_eq, aspect_ratio, stokes_param, 
                                 consider_earth_curvature, consider_atmospheric_emission, method):
    """
    Shared engine for the class-based Rayleigh-Jeans ice-scattering methods.

    This routine evaluates the full angular integral using either the
    Earth-frame phase matrix or the scattering-plane phase matrix, then sums
    the layer contributions into a brightness-temperature cube ordered as
    ``(Nf, Na, Nm)``.

    Parameters
    ----------
    frequency : float or array-like
        Frequencies in hertz.
    altitudes : numpy.ndarray
        One-dimensional altitude grid in meters.
    Temperature : numpy.ndarray
        Atmospheric temperature profile in kelvin.
    Pressure : numpy.ndarray
        Atmospheric pressure profile.
    P_water : numpy.ndarray
        Water-vapor partial pressure profile.
    elevation : float
        Telescope elevation angle in degrees.
    ice_density : float or array-like
        Ice-particle number density profile or scalar value.
    radius_eq : float or array-like
        Equivalent particle radius values in meters.
    aspect_ratio : float or array-like
        Particle aspect ratio(s).
    stokes_param : {'I', 'Q'}
        Stokes channel to integrate.
    consider_earth_curvature : bool
        If ``True``, use the curved-Earth limb model in the sky-temperature
        integral.
    consider_atmospheric_emission : bool
        If ``True``, include atmospheric emission in the incoming field.
    method : {1, 2}
        Phase-matrix convention selector. ``1`` uses the direct Earth-frame
        projection; ``2`` uses the scattering-plane formulation.

    Returns
    -------
    numpy.ndarray
        Brightness temperature array with shape ``(Nf, Na, Nm)``.
    """
    c = constants.c
    delta = np.radians(elevation)
    
    zenith_angle = 90.0 - elevation
    m_telescope = 1.0 / (np.cos(np.radians(zenith_angle)) + 0.50572 * (96.07995 - zenith_angle)**(-1.6364))
    
    freq_arr = np.atleast_1d(frequency).flatten()
    radius_arr = np.atleast_1d(radius_eq).flatten()
    m_arr = np.atleast_1d(aspect_ratio).flatten()
    
    ice_dens_arr = np.asarray(ice_density)
    if ice_dens_arr.ndim == 0:
        ice_dens_arr = np.full((len(altitudes), 1), ice_dens_arr.item())
    elif ice_dens_arr.ndim == 1:
        ice_dens_arr = ice_dens_arr[:, None] 
        
    if ice_dens_arr.shape[0] == len(altitudes):
        ice_dens_arr = (ice_dens_arr[:-1, :] + ice_dens_arr[1:, :]) / 2.0
        
    dz = np.diff(altitudes)
    z_layers = (altitudes[:-1] + altitudes[1:]) / 2.0
    
    N_theta, N_phi = 100, 100
    theta = np.linspace(0, np.pi, N_theta)
    phi = np.linspace(0, 2*np.pi, N_phi, endpoint=False)
    dtheta, dphi = theta[1] - theta[0], phi[1] - phi[0]
    
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    tg = theta_grid[None, None, :, :]
    pg = phi_grid[None, None, :, :]
    
    alpha_h, alpha_v = compute_polarizability(freq_arr, m_arr) 
    alpha_h_exp = alpha_h[:, :, None, None]
    alpha_v_exp = alpha_v[:, :, None, None]
    
    V = (4.0 / 3.0) * np.pi * radius_arr**3
    V_squared = (16.0 / 9.0) * np.pi**2 * (radius_arr)**6
    k_4 = (2 * np.pi * freq_arr / c)**4
    
    if method == 1:
        M11, M21 = compute_earth_frame_phase_matrix(alpha_h_exp, alpha_v_exp, tg, pg, delta)
    else:
        M11, M21 = compute_scattering_plane_phase_matrix(alpha_h_exp, alpha_v_exp, tg, pg, delta)
        
    M = M11 if stokes_param == 'I' else M21
    
    alpha_atm = alpha_specific_function(altitudes, freq_arr, Temperature, Pressure, P_water)
    alpha_atm_mid = (alpha_atm[:-1, :] + alpha_atm[1:, :]) / 2.0
    d_tau_atm = alpha_atm_mid * dz[:, None] 
    
    tau_below_atm = np.cumsum(d_tau_atm, axis=0)
    tau_below_atm = np.insert(tau_below_atm[:-1, :], 0, 0, axis=0) 
    
    tau_above_atm = np.zeros_like(d_tau_atm)
    if len(d_tau_atm) > 1:
        tau_above_atm[:-1, :] = np.cumsum(d_tau_atm[1:][::-1], axis=0)[::-1]
    
    attenuation = np.exp(-np.clip(tau_below_atm, 0, None) * m_telescope)

    # =========================================================
    # OPTIMIZED ANGULAR INTEGRATION (Frequency Loop)
    # =========================================================
    N_angles = N_theta * N_phi
    sin_flat = np.sin(theta_grid).flatten() 
    
    integral_M = np.zeros((len(z_layers), len(m_arr), len(freq_arr)))
    
    for f in range(len(freq_arr)):
        # Calculate memory-safe 3D profile for this frequency
        T_in_f = compute_T_in(theta_grid, z_layers, altitudes, Temperature, Temperature[0], 
                              tau_above_atm[:, f], tau_below_atm[:, f], 
                              consider_earth_curvature, consider_atmospheric_emission)
        
        M_f = M[:, f, :, :].reshape(len(m_arr), N_angles)          
        T_in_f_flat = T_in_f.reshape(len(z_layers), N_angles) 
        
        M_weighted = M_f * sin_flat 
        
        # Matrix Math: (Nz, N_angles) @ (N_angles, Nm) -> (Nz, Nm)
        integral_M[:, :, f] = (T_in_f_flat @ M_weighted.T) * dtheta * dphi
    
    # =========================================================
    # FINAL LAYER SUMMATION
    # =========================================================
    C_z = ice_dens_arr * dz[:, None] 
    
    layer_signal = (integral_M[:, :, :, np.newaxis] * 
                    C_z[:, np.newaxis, np.newaxis, :] * 
                    attenuation[:, np.newaxis, :, np.newaxis])
    
    C_base = m_telescope * k_4[:, None] * V_squared[None, :]
    
    # Sum over altitude: Current Shape (Nm, Nf, Na)
    T_sky_internal = C_base[np.newaxis, :, :] * np.sum(layer_signal, axis=0)
        
    # Transpose to user requested shape: (Nf, Na, Nm)
    T_sky_final = np.transpose(T_sky_internal, (1, 2, 0))
    
    return T_sky_final

def compute_T_RJ_ice_CLASS1(frequency, altitudes, Temperature, Pressure, P_water, elevation, ice_density, 
                            radius_eq, aspect_ratio=1.0, stokes_param='I',
                            consider_earth_curvature=True, consider_atmospheric_emission=True):
    """ 
    Compute brightness temperature using the Earth-frame phase-matrix method.

    This is a thin wrapper around :func:`_compute_T_RJ_ice_CLASS_core` that
    selects the direct Earth-frame projection convention.

    Returns
    -------
    numpy.ndarray
        Brightness temperature array with shape ``(Nf, Na, Nm)``.
    """
    return _compute_T_RJ_ice_CLASS_core(
        frequency, altitudes, Temperature, Pressure, P_water, elevation, 
        ice_density, radius_eq, aspect_ratio, stokes_param,
        consider_earth_curvature, consider_atmospheric_emission, method=1
    )

def compute_T_RJ_ice_CLASS2(frequency, altitudes, Temperature, Pressure, P_water, elevation, ice_density, 
                            radius_eq, aspect_ratio=1.0, stokes_param='I',
                            consider_earth_curvature=True, consider_atmospheric_emission=True):
    """ 
    Compute brightness temperature using the scattering-plane method.

    This is a thin wrapper around :func:`_compute_T_RJ_ice_CLASS_core` that
    selects the textbook scattering-plane rotation convention.

    Returns
    -------
    numpy.ndarray
        Brightness temperature array with shape ``(Nf, Na, Nm)``.
    """
    return _compute_T_RJ_ice_CLASS_core(
        frequency, altitudes, Temperature, Pressure, P_water, elevation, 
        ice_density, radius_eq, aspect_ratio, stokes_param,
        consider_earth_curvature, consider_atmospheric_emission, method=2
    )

import numpy as np
from scipy import constants

# ====================================================================
# HELPER: VECTORIZED 3D SKY TEMPERATURE
# ====================================================================
"""
def compute_T_in(theta_grid, z_layers, altitudes, T_phys_profile, T_ground, tau_above_atm, tau_below_atm):
    
    #Computes the incoming brightness temperature hitting the ice crystal from all angles.
    #Fully vectorized across layers (Nz), frequencies (Nf), and angles (N_theta, N_phi).
    
    R_e = 6371e3
    
    # Expand physical arrays for 4D broadcasting: (Nz, Nf, N_theta, N_phi)
    # tg shape: (1, 1, N_theta, N_phi)
    tg = theta_grid[None, None, :, :] 
    z_exp = z_layers[:, None, None, None]
    T_mid = ((T_phys_profile[:-1] + T_phys_profile[1:]) / 2.0)[:, None, None, None]
    
    tau_above = tau_above_atm[:, :, None, None]
    tau_below = tau_below_atm[:, :, None, None]
    
    # Geometric Boundaries
    theta_h = np.sqrt(2 * z_exp / R_e)
    bound_sky = np.pi / 2
    bound_limb = (np.pi / 2) + theta_h
    
    # 1. SKY REGIME (Theta < 90 deg)
    # Using Kasten-Young airmass, clipped exactly at 90 deg
    theta_deg = np.clip(np.degrees(tg), 0, 90.0)
    m_sky = 1.0 / (np.cos(tg) + 0.50572 * (96.07995 - theta_deg)**(-1.6364))
    T_sky = T_mid * (1.0 - np.exp(-tau_above * m_sky))
    
    # 2. LIMB REGIME (90 <= Theta <= 90 + theta_h)
    # Saturated tangent ray takes the physical temperature of its minimum altitude
    z_min = (R_e + z_exp) * np.sin(tg) - R_e
    T_limb_flat = np.interp(z_min.flatten(), altitudes, T_phys_profile)
    T_limb = T_limb_flat.reshape(z_min.shape)
    
    # 3. GROUND REGIME (Theta > 90 + theta_h)
    # Upwelling attenuated ground emission + lower atmosphere emission
    m_gnd = 1.0 / np.maximum(np.cos(np.pi - tg), 0.01) # Avoid div by zero looking horizontally
    T_gnd_attenuated = T_ground * np.exp(-tau_below * m_gnd) + T_mid * (1.0 - np.exp(-tau_below * m_gnd))
    
    # Stitch the regimes together
    conditions = [
        tg < bound_sky,
        (tg >= bound_sky) & (tg <= bound_limb),
        tg > bound_limb
    ]
    choices = [T_sky, T_limb, T_gnd_attenuated]
    
    return np.select(conditions, choices)
"""