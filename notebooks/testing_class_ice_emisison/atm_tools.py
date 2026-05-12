# Import of libraries

import matplotlib.pyplot as plt
import numpy as np
import pycraf
from pycraf import conversions as cnv
from astropy import units as u
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simpson
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy import constants

"""
cosmo_lidar.atm_tools
---------------------

Utility functions for the Cosmo-Lidar project related to atmospheric
attenuation and the receiver optical response used to estimate antenna
temperature.

Key functions
- effective_area_and_waist: compute Gaussian beam waist and effective area.
- alpha_specific_function: compute specific absorption coefficient
    (altitude × frequency grid) using pycraf.
- optical_depth_emission: compute vertical cumulative optical depth.
- contribution_effective_area: compute geometric angular contribution for
    each altitude and frequency.
- Calcul_T_ant_1_el: integrate contribution × alpha × temperature to obtain
    antenna temperature (method 1).
- Calcul_T_ant_2: use pycraf to obtain antenna contribution along the
    observation cone (method 2).
- calcul_PWV: integrate precipitable water vapor (PWV).

Unit conventions (recommended)
- altitudes: 1D numpy array in meters (m)
- frequency: 1D numpy array in hertz (Hz)
- Temperature: 1D array in kelvin (K)
- Pressure, P_water: 1D arrays compatible with pycraf (e.g. hPa)
- Functions that call pycraf convert values to astropy/pycraf units
    (GHz, km) as needed.

Dependencies
- pycraf, astropy, numpy, scipy
"""


#definition of variables

pi = np.pi  
# Planck constant (h)
h = constants.h           # 6.62607015e-34 (Joules * second)

# Boltzmann constant (kb)
kb = constants.k          # 1.380649e-23 (Joules / Kelvin)

# Speed of light (c)
c = constants.c           # 299792458.0 (meters / second)

#Definition of functions

def effective_area_and_waist(alt, theta, waist_0, lambda_0):
    """
    Compute Gaussian beam waist and an effective area factor at given altitudes and angles.

    Parameters
    - alt : scalar or array, distance (m) along beam axis (can be broadcast with theta)
    - theta : scalar or array of angles (radians)
    - waist_0 : beam waist at origin (m)
    - lambda_0 : wavelength (m)

    Returns
    - B : effective area factor (m^2 or dimensionless factor depending on definition)
    - w : beam waist at the given altitudes (m)
    """
    w = waist_0 * np.sqrt(1 + (lambda_0 * alt / (pi * waist_0 ** 2)) ** 2)
    B = 2 / pi * (lambda_0 * alt / w) ** 2 * np.exp(-2 * (alt * np.tan(theta) / w) ** 2)
    return B, w


def alpha_specific_function(altitudes, frequency, Temperature, Pressure, P_water):
    """
    Compute the specific absorption coefficient alpha(altitude, frequency).

    Parameters
    - altitudes : 1D numpy array (m)
    - frequency : 1D numpy array (Hz)
    - Temperature : 1D array, same length as altitudes (K)
    - Pressure : 1D array, same length as altitudes (hPa)
    - P_water : 1D array, partial pressure of water vapor same length as altitudes (hPa)

    Returns
    - alpha_specific : array with shape (len(altitudes), len(frequency)), in m^-1

    Notes
    - Uses pycraf.atm.atten_specific_annex1 which returns dB/km for dry and wet
      contributions. This function converts those values to 1/m (Neper/m).
    """
    altitudes_km = altitudes * u.m  # make a Quantity in meters
    altitudes_km = altitudes_km.to(u.km)  # convert to kilometers
    frequency_GHz = frequency * 10 ** -9 * u.GHz
    Temperature = Temperature * u.K  # Ensure Temperature is a Quantity in K
    Pressure = Pressure * u.hPa  # Ensure Pressure is a Quantity in hPa
    P_water = P_water * u.hPa  # Ensure P_water is a Quantity in hPa
    P_dry = Pressure - P_water

    # Calculate the absorption coefficient alpha using pycraf
    alpha_specific = np.zeros((len(altitudes_km), len(frequency)))

    for i in range(len(altitudes_km)):

        # Attenuation for all frequencies at this altitude
        alpha_dry_dB_km, alpha_wet_dB_km = pycraf.atm.atten_specific_annex1(
            frequency_GHz, P_dry[i], P_water[i], Temperature[i]
        )
        # Convert from dB/km to 1/m (Neper per meter)
        alpha_tot_m = (alpha_dry_dB_km + alpha_wet_dB_km) * (np.log(10) / 10) / 1000

        # Assign to row i (in m^-1)
        alpha_specific[i, :] = alpha_tot_m.value
    return alpha_specific
    
    
def optical_depth_emission(altitudes, alpha_specific):
    """
    Compute cumulative vertical optical depth tau(z, f).

    Parameters
    - altitudes : 1D array (m), must be in increasing order
    - alpha_specific : array (N_alt, N_freq) in m^-1

    Returns
    - tau : array (N_alt, N_freq), cumulative integral of alpha from bottom up
    """
    # altitudes : (N_alt,)
    # alpha_specific : (N_alt, N_freq)

    dz = np.diff(altitudes)  # (N_alt-1,)
    alpha_avg = 0.5 * (alpha_specific[1:, :] + alpha_specific[:-1, :])  # (N_alt-1, N_freq)

    # cumulative integrals along altitude axis
    tau = np.zeros_like(alpha_specific)
    tau[1:, :] = np.cumsum(alpha_avg * dz[:, None], axis=0)

    return tau


    


def contribution_effective_area (frequency, theta_b, altitudes, elevation, N = 500) :
    """
    Compute geometric contribution C_alt(altitude, frequency) from angular integration.

    Parameters
    - frequency : 1D numpy array (Hz)
    - theta_b : 1D numpy array or scalar of beam opening angles (radians); typically same length as frequency
    - altitudes : 1D numpy array (meters)
    - elevation : elevation angle in degrees (90 = zenith)
    - N : int, number of points used for angular integration (default 500)

    Returns
    - C_alt : 2D numpy array, shape (len(altitudes), len(frequency)), geometric contribution for each altitude and frequency

    Notes
    - Broadcasting: altitudes[:, None] and theta[None, :] are used to vectorize calculations.
    - Output shape: (number of altitudes, number of frequencies)
    - All input arrays must be 1D and compatible in length as described above.
    - Units: frequency in Hz, altitudes in meters, theta_b in radians, elevation in degrees.
    """

    # Define relevant variables
    pi = np.pi
    wavelength = 3.e8 / frequency  # meters
    w_0 = wavelength / (pi * theta_b)  # meters
    # Initialize C_alt array to store contributions for each altitude and frequency
    C_alt = np.zeros((len(altitudes), len(frequency)))
    altitudes_rel = altitudes - altitudes[0]
    elev_rad = elevation * pi / 180

    for j in range(len(frequency)):
        # theta: integration variable, logarithmically spaced from near zero to pi/2 radians
        theta = np.geomspace(0.000001, 1.57, N)
        wavelength_j = wavelength[j]
        w0_j = w_0[j]

        # Vectorize: altitudes[:, None] and theta[None, :] for broadcasting
        eff_area, _ = effective_area_and_waist(
            1 / np.sin(elev_rad) * altitudes_rel[:, None], theta[None, :], w0_j, wavelength_j
        )  # shape (len(altitudes), N)

        # Compute integrand for each altitude and theta
        f_values = (
            2 * pi * (altitudes[:, None] / wavelength_j) ** 2 * np.abs(np.tan(theta))[None, :] * eff_area
        )

        # Simpson integral over theta for each altitude
        C_alt[:, j] = trapezoid(f_values, theta, axis=1) / altitudes ** 2
    return C_alt


#calcul avec les fonctions de la méthode 1

def Calcul_T_ant_1_el(frequency, theta_b, altitudes, Temperature, Pressure, P_water, elevation, N=500):
    
    """
    Method 1: compute antenna temperature T_ant by integrating over altitude.

    Parameters
    - frequency : 1D array (Hz)
    - theta_b : 1D array or scalar (rad)
    - altitudes : 1D array (m)
    - N : int, number of angular integration points
    - Temperature, Pressure, P_water : 1D arrays of length len(altitudes) (K, hPa, hPa) not Quantities
    - elevation : elevation angle in degrees (90 = zenith)

    Returns
    - T_ant : 1D array of antenna temperature values for each frequency (K)
    """
    #definitions des variables pertinentes
    pi = np.pi 
    thet = 90 - elevation
    thet_rad = thet * pi/180
    m = 1/(np.cos(thet_rad) + 0.50572*(96.07995-thet)**(-1.6364))   
    wavelength = 3.e8 / frequency #m
    w_0 = wavelength / (pi*theta_b)
    
    altitudes_km = altitudes * u.m       # maintenant c'est une Quantity en m
    altitudes_km = altitudes_km.to(u.km) # conversion en km
    frequency_GHz = frequency*10**-9 * u.GHz

    #calcul de C_alt
    C_alt = contribution_effective_area(frequency, theta_b, altitudes, elevation, N)
   
            
    #Calcul de alpha

    alpha_specific= alpha_specific_function(altitudes, frequency, Temperature, Pressure, P_water)


    tau = optical_depth_emission (altitudes, alpha_specific)
    
    # on définit le terme que l'on va intégrer sur l'altitudes 
    C_tot = np.zeros((len(altitudes), len(frequency)))
    C_tot = C_alt * alpha_specific*m * Temperature[:, None] * np.exp(-tau*m)
    T_ant = trapezoid(C_tot, altitudes, axis=0)
        
    
    return T_ant



def Calcul_T_ant_2 (frequency, elevation = 90, obs_alt = 0) :
    """
    Method 2: use pycraf.atm.atten_slant_annex1 to estimate the antenna contribution
    along the observation cone.

    Parameters
    - frequency : scalar or array (Hz)
    - elevation : elevation angle in degrees (default 90 = zenith)
    - obs_alt : observer altitude in km (default 0)

    Returns
    - T_ant2 : array or scalar (K)
    """
    
    
    frequency_GHz = frequency*10**-9 * u.GHz

    elevation = elevation * u.deg #on regarde le zenith
    obs_alt = obs_alt * u.km # on se place au niveau de la mer
    atl_layer_cache = pycraf.atm.atm_layers(frequency_GHz, pycraf.atm.profile_standard , heights=None)


    T_ant2 = pycraf.atm.atten_slant_annex1(elevation, obs_alt, atl_layer_cache, do_tebb=True, t_bg= 2.73 * u.K, max_arc_length= 180 * u.deg, max_path_length= 100. * u.km) [2]
    
    return T_ant2

def calcul_PWV (rho_water, z) :
    """
    Compute precipitable water vapor (PWV) by vertical integration.

    Parameters
    - rho_water : vertical profile of water mass density (kg m^-3)
    - z : vertical coordinates corresponding to rho_water (m)

    Returns
    - PWV : scalar (same units as integral of rho over z, convert to mm if desired)
    """
    PWV = trapezoid(rho_water, z) #en mm / kg/m2
    return PWV

def calcul_z_percentile_wvc(z,rho_water,percentile):
    """
    Compute the altitude below which a given percentile of water vapor content is contained.

    Parameters
    - z : 1D array of altitudes (m)
    - rho_water : 1D array of water mass density (kg m^-3)
    - percentile : float, desired percentile (0-100)

    Returns
    - z_percentile : scalar, altitude (m) below which the given percentile of water vapor is contained
    """
    # Calcul de la densité cumulée de vapeur d'eau
    cumulative_rho = cumulative_trapezoid(rho_water, z, initial=0)

    # Calcul du total de vapeur d'eau
    total_rho = cumulative_rho[-1]

    # Calcul de la valeur cible pour le percentile
    target_value = (percentile / 100.0) * total_rho

    # Recherche de l'altitude correspondante au percentile
    z_percentile = np.interp(target_value, cumulative_rho, z)

    return z_percentile


def vapor_pressure(T):
    """
    Calcule la pression de vapeur saturante (hPa) pour T en Kelvin.
    - Sur eau si T > 0°C
    - Sur glace si T < -23°C
    - Interpolation linéaire entre les deux
    """

    T = np.array(T)

    # Formules empiriques (T en K)
    e_water = 6.1078 * np.exp(17.27 * (T - 273.15) / (T - 35.85))  # sur eau
    e_ice   = 6.1078 * np.exp(21.875 * (T - 273.15) / (T - 7.65))   # sur glace

    # Facteur d'interpolation entre glace (-23°C = 250 K) et eau (0°C = 273.15 K)
    alpha = np.clip((T - 250) / (273.15 - 250), 0, 1)

    # Interpolation linéaire
    e = e_ice * (1 - alpha) + e_water * alpha

    return e




def mass_quantile_grid(z, rho_v, N=150, smooth=False):
    """
    Grille 'data-adaptative' : z aux quantiles de l'intégrale de rho_v dz.
    z, rho_v / wvmr : 1D (pas forcément trié), rho_v>=0 recommandé.
    """
    z = np.asarray(z, float)
    rv = np.asarray(rho_v, float)
    m = np.isfinite(z) & np.isfinite(rv)
    z, rv = z[m], np.clip(rv[m], 0, None)
    if z.size < 2:
        return np.array([])

    ord = np.argsort(z)
    z, rv = z[ord], rv[ord]

    cum = cumulative_trapezoid(rv, z, initial=0.0)
    total = cum[-1]
    if not np.isfinite(total) or total <= 0:
        # fallback : linéaire si intégrale nulle
        return np.linspace(z.min(), z.max(), N)

    q = np.linspace(0, 1, N)
    targets = q * total
    zg = np.interp(targets, cum, z)

    if smooth and N > 5:
        # petit lissage optionnel (filtre médian grossier)
        from scipy.ndimage import median_filter
        zg = median_filter(zg, size=5, mode='nearest')
    return zg

# Exemple (à partir de tes séries)
# zg = mass_quantile_grid(z, rho_water, N=150, smooth=True)


def pwv_profile(rho_water, z_m):
    """
    Calcule la quantité de vapeur d'eau intégrée au-dessus de chaque altitude.
    
    Paramètres
    ----------
    rho_water : array-like, shape (N,)
        Densité de vapeur d'eau (kg/m^3) à chaque altitude.
    z_m : array-like, shape (N,)
        Altitudes correspondantes en mètres, supposées croissantes (bas -> haut).
        
    Retour
    ------
    pwv_above : ndarray, shape (N,)
        Colonne d'eau intégrée à partir de z jusqu'au sommet (kg/m^2).
        pwv_above[0] = PWV total colonne
        pwv_above[-1] = 0
    """
    rho_water = np.asarray(rho_water)
    z_m = np.asarray(z_m)

    # intégrale cumulée depuis le bas jusqu'à chaque z_i
    # cumulative_trapezoid renvoie la même taille que l'entrée si on met initial=0
    integ_cum = cumulative_trapezoid(rho_water, z_m, initial=0.0)  # kg/m^2

    # valeur totale colonne = intégrale du bas jusqu'au sommet
    I_tot = integ_cum[-1]  # kg/m^2

    # quantité restante au-dessus de chaque altitude z_i :
    # intégrale de z_i jusqu'à z_max = I_tot - intégrale(0 -> z_i)
    pwv_above = I_tot - integ_cum  # kg/m^2

    return pwv_above


def Calcul_T_ant_cumulative(frequency, theta_b, altitudes, Temperature, Pressure, P_water, elevation, N=500):
    
    """
    Compute cumulative antenna temperature T_ant profile by integrating over altitude.

    Returns
    - T_ant_profile : 2D array (len(altitudes), len(frequency))
                      Value at index [i, :] corresponds to T_ant calculated from altitudes[0] to altitudes[i].
    """
    # --- 1. Définitions des géométries (Identique) ---
    pi = np.pi 
    thet = 90 - elevation
    thet_rad = thet * pi/180
    
    # Calcul de la masse d'air (m)
    m = 1/(np.cos(thet_rad) + 0.50572*(96.07995-thet)**(-1.6364))   
    
    # Conversions d'unités (Identique)
    # altitudes_km sert sans doute dans vos sous-fonctions externes
    altitudes_km = altitudes * u.m       
    altitudes_km = altitudes_km.to(u.km) 
    frequency_GHz = frequency*10**-9 * u.GHz

    # --- 2. Calcul des paramètres physiques (Identique) ---
    
    # Contribution de l'aire effective (supposée dépendre de z)
    # Renvoie un tableau (len(altitudes), len(frequency)) ou broadcastable
    C_alt = contribution_effective_area(frequency, theta_b, altitudes, elevation, N)
   
    # Calcul de l'atténuation spécifique alpha (Neper/m ou 1/m selon vos fonctions)
    alpha_specific = alpha_specific_function(altitudes, frequency, Temperature, Pressure, P_water)

    # Calcul de l'épaisseur optique cumulée depuis le sol (tau(z))
    # Note: Votre fonction 'optical_depth_emission' doit renvoyer tau(z) = intégrale de 0 à z de alpha.
    tau = optical_depth_emission(altitudes, alpha_specific)
    
    # --- 3. Définition de l'intégrande (Identique) ---
    # C'est le terme dTb/dz : Emission locale * Atténuation jusqu'au sol * Géométrie
    # Temperature[:, None] permet le broadcasting si frequency est un tableau
    C_tot = C_alt * alpha_specific * m * Temperature[:, None] * np.exp(-tau*m)
    
    # --- 4. Intégration Cumulative (CHANGEMENT ICI) ---
    
    # Au lieu de 'trapezoid' qui somme tout en un nombre, on utilise 'cumulative_trapezoid'
    # initial=0 garantit que le tableau de sortie a la même taille que 'altitudes' (T_ant(z=0) = 0)
    T_ant_profile = cumulative_trapezoid(C_tot, altitudes, axis=0, initial=0)
        
    return T_ant_profile



def Calcul_T_sky_1_el(frequency, altitudes, Temperature, Pressure, P_water, elevation, N=500):
    
    """
    Method 1: compute antenna temperature T_sky by integrating over altitude.

    Parameters
    - frequency : 1D array (Hz)
    - altitudes : 1D array (m)
    - N : int, number of angular integration points
    - Temperature, Pressure, P_water : 1D arrays of length len(altitudes) (K, hPa, hPa) not Quantities
    - elevation : elevation angle in degrees (90 = zenith)

    Returns
    - T_sky : 1D array of antenna temperature values for each frequency (K)
    """
    #definitions des variables pertinentes
    pi = np.pi 
    thet = 90 - elevation
    thet_rad = thet * pi/180
    m = 1/(np.cos(thet_rad) + 0.50572*(96.07995-thet)**(-1.6364))   
    wavelength = 3.e8 / frequency #m
  
    
    altitudes_km = altitudes * u.m       # maintenant c'est une Quantity en m
    altitudes_km = altitudes_km.to(u.km) # conversion en km
    frequency_GHz = frequency*10**-9 * u.GHz

   
            
    #Calcul de alpha

    alpha_specific= alpha_specific_function(altitudes, frequency, Temperature, Pressure, P_water)


    tau = optical_depth_emission (altitudes, alpha_specific)
    
    # on définit le terme que l'on va intégrer sur l'altitudes 
    C_tot = np.zeros((len(altitudes), len(frequency)))
    C_tot = alpha_specific*m * Temperature[:, None] * np.exp(-tau*m)
    T_sky = trapezoid(C_tot, altitudes, axis=0)
        
    
    return T_sky

def atmospheric_transmission(frequency, altitudes, Temperature, Pressure, P_water, elevation):
    
    """
    Method 1: compute Transmission

    Parameters
    - frequency : 1D array (Hz)
    - altitudes : 1D array (m)
    - N : int, number of angular integration points
    - Temperature, Pressure, P_water : 1D arrays of length len(altitudes) (K, hPa, hPa) not Quantities
    - elevation : elevation angle in degrees (90 = zenith)

    Returns
    - Transmission : 1D array of tranmission values for each frequency ()
    """
    #definitions des variables pertinentes
    pi = np.pi 
    thet = 90 - elevation
    thet_rad = thet * pi/180
    m = 1/(np.cos(thet_rad) + 0.50572*(96.07995-thet)**(-1.6364))   
    wavelength = 3.e8 / frequency #m
  
    
    altitudes_km = altitudes * u.m       # maintenant c'est une Quantity en m
    altitudes_km = altitudes_km.to(u.km) # conversion en km
    frequency_GHz = frequency*10**-9 * u.GHz

   
            
    #Calcul de alpha

    alpha_specific= alpha_specific_function(altitudes, frequency, Temperature, Pressure, P_water)


    tau = optical_depth_emission (altitudes, alpha_specific)
    
    # on définit le terme que l'on va intégrer sur l'altitudes 
    Transmission = np.zeros((len(altitudes), len(frequency)))
    Transmission = np.exp(-tau*m)
    
        
    
    return Transmission [-1, :]  # Transmission totale de la colonne atmosphérique


R_WATER =  461.5  # J/(kg·K)

def calc_pwv(p_water_hpa, temp_k, z_m):
    rho_water = (p_water_hpa * 100) / (R_WATER * temp_k) * 1000
    pwv = trapezoid(rho_water, x=z_m) / 1000.0
    return pwv

def calc_zmoy(rho_water_g_m3, pwv_mm, z_m):
    """Calculates the mean altitude (z_moy) of the water vapor profile."""
    if pwv_mm < 1e-9: # Handle case of almost no water
        return np.nan
    # z_moy = Integral(z * rho_w * dz) / Integral(rho_w * dz)
    # rho is g/m3, z is m. pwv is mm = kg/m2 = g/m2 * 1000.
    # Integrand for numerator: (g/m3 * m * m) = g/m
    # Denominator: g/m2
    # Result: m
    numerator = trapezoid(rho_water_g_m3 * z_m, x=z_m)
    denominator = pwv_mm * 1000.0 # Convert PWV from mm to g/m^2
    return numerator / denominator



def add_perturbation(z_center, delta_pwv, z_grid, P_ref, T_prof):
    # Largeur dynamique (augmente avec l'altitude)
    
    sigma = 150 + (z_center - 5000) * 0.1
    integral_target = delta_pwv * 1000.0 
    amplitude = integral_target / (sigma * np.sqrt(2 * np.pi))
    
    rho_perturbation = amplitude * np.exp(-0.5 * ((z_grid - z_center) / sigma)**2)
    P_perturbation = (rho_perturbation * R_WATER * T_prof) / 1000.0 / 100.0 
    
    P_water_new = P_ref + P_perturbation
    return np.maximum(P_water_new, 1e-9)


def compute_sensitivity_profile(T_prof, P_prof, P_water_prof, altitudes, frequency, elevation, dPWV, target_initial_pwv):
    """
    Calcule la sensibilité de T_sky et Transmission à l'altitude d'une perturbation dPWV.
    La référence est un scaling uniforme du profil pour atteindre (initial + dPWV).
    """
    
    # 1. Normalisation du profil initial à 'target_initial_pwv' (ex: 1.0 mm)
    raw_pwv = calc_pwv(P_water_prof, T_prof, altitudes)
    scale_factor = target_initial_pwv / raw_pwv
    P_water_base = P_water_prof * scale_factor
    
    # 2. Calcul de la Référence Dynamique (Scaling Uniforme)
    # On vise un PWV total = initial + dPWV
    total_target_pwv = target_initial_pwv + dPWV
    scale_to_target = total_target_pwv / target_initial_pwv
    P_water_ref_uniform = P_water_base * scale_to_target
    
    # Radiométrie Référence (Uniforme)
    T_sky_ref = Calcul_T_sky_1_el(frequency, altitudes, T_prof, P_prof, P_water_ref_uniform, elevation)[0]
    Trans_ref = atmospheric_transmission(frequency, altitudes, T_prof, P_prof, P_water_ref_uniform, elevation)[0]
    
    # 3. Boucle sur les altitudes de perturbation
    # On va scanner de 6 km jusqu'à un peu en dessous du sommet de la grille ici 9500
    z_min_scan = 6000
    z_max_scan = 9500 # Marge de sécurité
    if z_max_scan < z_min_scan: z_max_scan = z_min_scan + 500
    
    z_centers = np.linspace(z_min_scan, z_max_scan, 20)
    
    diff_T_list = []
    diff_Trans_list = []
    
    for z_c in z_centers:
        # Création du profil Perturbé (Base + Gaussienne Locale)
        # Note : add_perturbation ajoute dPWV à la base (donc total = initial + dPWV)
        P_water_pert = add_perturbation(z_c, dPWV, altitudes, P_water_base, T_prof)
        
        # Radiométrie Perturbée
        val_T = Calcul_T_sky_1_el(frequency, altitudes, T_prof, P_prof, P_water_pert, elevation)[0]
        val_Trans = atmospheric_transmission(frequency, altitudes, T_prof, P_prof, P_water_pert, elevation)[0]
        
        # Différence (Local - Uniforme)
        diff_T_list.append(val_T - T_sky_ref)
        diff_Trans_list.append(val_Trans - Trans_ref)
        
    return z_centers, np.array(diff_T_list), np.array(diff_Trans_list)


def planck_source(nu, T):
    """Calculate the Planck function B(nu, T) in W/m^2/Hz/sr with nu in Hz and T in K ."""
    exponent = (h * nu) / (kb * T)
    return (2 * h * nu**3 / c**2) / (np.exp(exponent) - 1)


# New updated functions for the calculation of the brightness temperature

def Calcul_T_sky_lf_RJ(frequency, altitudes, Temperature, Pressure, P_water, elevation, N=500):
    
    """
    Method 1: compute antenna temperature T_sky by integrating over altitude under RJ approximation.

    Parameters
    - frequency : 1D array (Hz)
    - altitudes : 1D array (m)
    - N : int, number of angular integration points
    - Temperature, Pressure, P_water : 1D arrays of length len(altitudes) (K, hPa, hPa) not Quantities
    - elevation : elevation angle in degrees (90 = zenith)

    Returns
    - T_sky : 1D array of antenna temperature values for each frequency (K_RJ)
    """
    #definitions des variables pertinentes
    pi = np.pi 
    thet = 90 - elevation
    thet_rad = thet * pi/180
    m = 1/(np.cos(thet_rad) + 0.50572*(96.07995-thet)**(-1.6364))   

    
    altitudes_km = altitudes * u.m       # maintenant c'est une Quantity en m
    altitudes_km = altitudes_km.to(u.km) # conversion en km
    frequency_GHz = frequency*10**-9 * u.GHz

   
            
    #Calcul de alpha

    alpha_specific= alpha_specific_function(altitudes, frequency, Temperature, Pressure, P_water)


    tau = optical_depth_emission (altitudes, alpha_specific)
    
    # on définit le terme que l'on va intégrer sur l'altitudes 
    C_tot = np.zeros((len(altitudes), len(frequency)))
    C_tot = alpha_specific*m * Temperature[:, None] * np.exp(-tau*m)
    T_sky = trapezoid(C_tot, altitudes, axis=0)
        
    
    return T_sky



def Calcul_T_sky_1_el_bb(frequency, altitudes, Temperature, Pressure, P_water, elevation):
    
    """
    Method 1: compute antenna temperature T_sky by integrating over altitude using the blackbody approximation.

    Parameters
    - frequency : 1D array (Hz)
    - altitudes : 1D array (m)
    - Temperature, Pressure, P_water : 1D arrays of length len(altitudes) (K, hPa, hPa) not Quantities
    - elevation : elevation angle in degrees (90 = zenith)

    Returns
    - T_sky : 1D array of antenna temperature values for each frequency (K_RJ)
    """
    #definitions des variables pertinentes
    pi = np.pi 
    thet = 90 - elevation
    thet_rad = thet * pi/180
    m = 1/(np.cos(thet_rad) + 0.50572*(96.07995-thet)**(-1.6364))   

    
    #Calcul de alpha

    alpha_specific= alpha_specific_function(altitudes, frequency, Temperature, Pressure, P_water)


    tau = optical_depth_emission (altitudes, alpha_specific)

    B_nu = planck_source(frequency[None,:], Temperature[:, None])  # shape (N_alt, N_freq)
    
    # on définit le terme que l'on va intégrer sur l'altitudes 
    C_tot = np.zeros((len(altitudes), len(frequency)))
    C_tot = alpha_specific*m * B_nu * np.exp(-tau*m)
    I_nu = trapezoid(C_tot, altitudes, axis=0) # radiance in W/m^2/Hz/sr

    T_sky_RJ = (c**2 / (2 * kb * frequency**2)) * I_nu  # Convert radiance to brightness temperature in K_RJ
        
    
    return T_sky_RJ


def Calcul_T_sky_Slab_RJ(frequency, altitudes, Temperature, Pressure, P_water, elevation):
    """
    Method: Compute antenna temperature T_sky using the Isothermal Slab Summation.
    T_sky = sum( T_i * (1 - exp(-d_tau_i * m)) * exp(-tau_below_i * m) )

    Parameters
    - frequency : 1D array (Hz)
    - altitudes : 1D array (m)
    - Temperature, Pressure, P_water : 1D arrays of length len(altitudes) (K, hPa, hPa) not Quantities
    - elevation : elevation angle in degrees (90 = zenith)

    Returns
    - T_sky : 1D array of antenna temperature values for each frequency (K_RJ)
    """

    zenith_angle = 90 - elevation
    # Using airmass formula (Kasten-Young)
    m = 1 / (np.cos(np.radians(zenith_angle)) + 0.50572 * (96.07995 - zenith_angle)**(-1.6364))
    
    # 2. Get Absorption Coefficient (alpha)
    # alpha_specific should be in units of m^-1
    alpha = alpha_specific_function(altitudes, frequency, Temperature, Pressure, P_water)
    
    dz = np.diff(altitudes) # Shape: (len(altitudes)-1,)
    
    # Calculate Midpoint Properties for each slab
    # We average the values between index i and i+1
    T_mid = (Temperature[:-1] + Temperature[1:]) / 2.0
    alpha_mid = (alpha[:-1, :] + alpha[1:, :]) / 2.0
    
    # d_tau is the optical thickness of each individual slab
    d_tau = alpha_mid * dz[:, None]
    
    # 4. Calculate Tau Below (Cumulative attenuation from ground to layer i)
    # tau_below for the 1st layer is 0. 
    # For subsequent layers, it's the sum of d_tau of all layers below it.
    tau_below = np.cumsum(d_tau, axis=0)
    tau_below = np.insert(tau_below[:-1, :], 0, 0, axis=0) # Shift so first layer has 0 attenuation
    
    # 5. Apply the Summation
    # Layer Emission = T * (1 - exp(-d_tau * m))
    # Transmission to ground = exp(-tau_below * m)
    T_layers = T_mid[:, None] * (1 - np.exp(-d_tau * m)) * np.exp(-tau_below * m)
    
    T_sky = np.sum(T_layers, axis=0)
    
    return T_sky



def Calcul_T_sky_Slab_bb(frequency, altitudes, Temperature, Pressure, P_water, elevation):
    """
    Method: Compute antenna radiance using the Isothermal Slab Summation then convert to brightness temperature (K_RJ)
    

    Parameters
    - frequency : 1D array (Hz)
    - altitudes : 1D array (m)
    - Temperature, Pressure, P_water : 1D arrays of length len(altitudes) (K, hPa, hPa) not Quantities
    - elevation : elevation angle in degrees (90 = zenith)

    Returns
    - T_sky : 1D array of antenna temperature values for each frequency (K_RJ)
    """

    zenith_angle = 90 - elevation
    # Using airmass formula (Kasten-Young)
    m = 1 / (np.cos(np.radians(zenith_angle)) + 0.50572 * (96.07995 - zenith_angle)**(-1.6364))
    
    # 2. Get Absorption Coefficient (alpha)
    # alpha_specific should be in units of m^-1
    alpha = alpha_specific_function(altitudes, frequency, Temperature, Pressure, P_water)
    
    dz = np.diff(altitudes) # Shape: (len(altitudes)-1,)
    
    # Calculate Midpoint Properties for each slab
    # We average the values between index i and i+1
    T_mid = (Temperature[:-1] + Temperature[1:]) / 2.0
    alpha_mid = (alpha[:-1, :] + alpha[1:, :]) / 2.0
    
    # d_tau is the optical thickness of each individual slab
    d_tau = alpha_mid * dz[:, None]
    
    # 4. Calculate Tau Below (Cumulative attenuation from ground to layer i)
    # tau_below for the 1st layer is 0. 
    # For subsequent layers, it's the sum of d_tau of all layers below it.
    tau_below = np.cumsum(d_tau, axis=0)
    tau_below = np.insert(tau_below[:-1, :], 0, 0, axis=0) # Shift so first layer has 0 attenuation
    
    # 5 Calculate the Planck function at the mid-layer temperature for each frequency
    B_nu_mid = planck_source(frequency[None,:], T_mid[:, None])  # shape (N_alt-1, N_freq)
    # 5. Apply the Summation
    # Layer Emission = T * (1 - exp(-d_tau * m))
    # Transmission to ground = exp(-tau_below * m)
    I_layers = B_nu_mid * (1 - np.exp(-d_tau * m)) * np.exp(-tau_below * m)

    I_nu = np.sum(I_layers, axis=0) # radiance in W/m^2/Hz/sr

    T_sky_RJ = (c**2 / (2 * kb * frequency**2)) * I_nu  # Convert radiance to brightness temperature in K_RJ
    
    return T_sky_RJ