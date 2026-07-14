"""
Python translation of the IDL routine LIDAR_SIGMA_PWV.

It reads a lidar H2O simulation/sensitivity table with columns:
    s, z, rpwv, sigma, bias, pressure, temperature

where, following the original comments:
    s           distance to lidar [km]
    z           altitude [km]
    rpwv        water-vapor mass mixing ratio [kg/kg]
    sigma       standard deviation on WVMR [g/kg]
    bias        bias on WVMR [g/kg]
    pressure    pressure [hPa]
    temperature temperature [K]

The returned quantity is sigma_pwv in kg/m^2 of water vapor, scaled to the
requested telescope size, altitude-bin thickness, sampling time, and elevation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

ArrayLike = Union[float, list[float], tuple[float, ...], np.ndarray]


def lidar_sigma_pwv(
    zbin: ArrayLike,
    dzbin: float,
    tsampling: float,
    data_file: str | Path,
    *,
    elevation: float = 90.0,
    telsize: float = 0.5,
    reference_telsize: float = 0.5,
    reference_tsampling: float = 60.0,
    reference_dz: float = 30.0,
    extrapolate: bool = False,
) -> np.ndarray | float:
    """
    Compute the PWV uncertainty per altitude bin from a lidar sensitivity file.

    Parameters
    ----------
    zbin : float or array-like
        Altitude above the lidar [m]. This is the vertical altitude bin center,
        not the slant distance.
    dzbin : float
        Vertical altitude-bin thickness [m].
    tsampling : float
        Sampling/integration time [s].
    data_file : str or pathlib.Path
        Path to the lidar simulation file with columns:
        s, z, rpwv, sigma, bias, pressure, temperature.
    elevation : float, optional
        Observation elevation angle [deg]. Default is 90 deg, i.e. zenith.
    telsize : float, optional
        Telescope diameter [m] for the desired configuration.
    reference_telsize : float, optional
        Telescope diameter [m] used to generate the simulation file.
        The original IDL file used 0.5 m.
    reference_tsampling : float, optional
        Sampling time [s] used to generate the simulation file.
        The original IDL file used 60 s.
    reference_dz : float, optional
        Reference bin length along the line of sight [m]. The simulation files
        shown here use 30 m.
    extrapolate : bool, optional
        If False, values beyond the maximum simulated distance are set to NaN,
        matching the safer branch of the IDL code. If True, apply the commented
        1/r extrapolation used as an alternative in the IDL code.

    Returns
    -------
    sigma_pwv : float or np.ndarray
        PWV uncertainty [kg/m^2]. Scalar in, scalar out; array in, array out.
    """
    
    if dzbin <= 0 or tsampling <= 0 or telsize <= 0 or reference_telsize <= 0 or reference_tsampling <= 0:
        raise ValueError("Inputs must be positive.")

    sin_el = np.sin(np.deg2rad(elevation))
    if sin_el <= 0:
        raise ValueError("Elevation must have a positive sine.")

    data = np.loadtxt(data_file, comments=("#", ";"))
    
    # Extract Raw Columns
    s_m = data[:, 0] * 1000.0
    z_m = data[:, 1] * 1000.0  # Extract true altitude from the file
    sigma_wvmr_g_per_kg = data[:, 3]
    pressure_hpa = data[:, 5]
    temperature_k = data[:, 6]

    zbin_arr = np.asarray(zbin, dtype=float)
    scalar_input = zbin_arr.ndim == 0
    zbin_arr = np.atleast_1d(zbin_arr)

    # Slant distance light travels to reach the altitude bin
    distance_m = zbin_arr / sin_el

    # ==========================================================================
    # BUGFIX 1: Decouple Instrumental Loss from Thermodynamics
    # ==========================================================================
    # A. Interpolate instrumental precision using Slant Distance (1/R^2 penalty)
    order_s = np.argsort(s_m)
    sigma_wvmr_interp = np.interp(
        distance_m, s_m[order_s], sigma_wvmr_g_per_kg[order_s], 
        left=sigma_wvmr_g_per_kg[order_s][0], 
        right=sigma_wvmr_g_per_kg[order_s][-1] # Cap it for safe extrapolation later
    )

    # B. Interpolate Thermodynamics using True Physical Altitude (Air Density)
    order_z = np.argsort(z_m)
    p_interp = np.interp(
        zbin_arr, z_m[order_z], pressure_hpa[order_z], 
        left=pressure_hpa[order_z][0], right=pressure_hpa[order_z][-1]
    )
    t_interp = np.interp(
        zbin_arr, z_m[order_z], temperature_k[order_z], 
        left=temperature_k[order_z][0], right=temperature_k[order_z][-1]
    )

    # ==========================================================================
    # BUGFIX 2: The Physical PWV Integration
    # ==========================================================================
    # 1. Calculate the airmass for the ACTUAL slant path length through the bin
    path_bin_length_m = dzbin / sin_el
    nmol_actual = p_interp * 100.0 * path_bin_length_m / (8.314 * t_interp)
    airmass_actual_kg_m2 = nmol_actual * 28.9647 / 1000.0

    # 2. Mixing ratio precision improves by sqrt(length), so we divide the mixing ratio error
    bin_scale = np.sqrt(path_bin_length_m / reference_dz)
    sigma_wvmr_actual = sigma_wvmr_interp / bin_scale

    # 3. Absolute PWV error = (Error on mixing ratio) * (Total Slant Airmass)
    sigma_pwv = airmass_actual_kg_m2 * sigma_wvmr_actual / 1000.0

    # ==========================================================================
    # Apply Telescope and Time Scalings
    # ==========================================================================
    sigma_pwv /= (telsize / reference_telsize)
    sigma_pwv *= np.sqrt(reference_tsampling / tsampling)

    # Handle Range Limits
    maxdist = s_m[order_s][-1]
    out_of_range = distance_m > maxdist
    if np.any(out_of_range):
        if extrapolate:
            # Simple linear scaling penalty beyond the simulation bounds
            sigma_pwv[out_of_range] *= (distance_m[out_of_range] / maxdist)
        else:
            sigma_pwv[out_of_range] = np.nan

    # Negative distances are not physical
    sigma_pwv[distance_m < 0] = np.nan

    if scalar_input:
        return float(sigma_pwv[0])
    return sigma_pwv


"""
    if dzbin <= 0:
        raise ValueError("dzbin must be positive.")
    if tsampling <= 0:
        raise ValueError("tsampling must be positive.")
    if telsize <= 0 or reference_telsize <= 0:
        raise ValueError("telsize and reference_telsize must be positive.")
    if reference_tsampling <= 0:
        raise ValueError("reference_tsampling must be positive.")

    sin_el = np.sin(np.deg2rad(elevation))
    if sin_el <= 0:
        raise ValueError("elevation must be between 0 and 180 deg with positive sine.")

    data = np.loadtxt(data_file, comments=("#", ";"))
    if data.ndim != 2 or data.shape[1] < 7:
        raise ValueError(
            "data_file must contain at least 7 columns: "
            "s, z, rpwv, sigma, bias, pressure, temperature."
        )

    s_km = data[:, 0]
    # z_km = data[:, 1]
    # rpwv = data[:, 2]
    sigma_wvmr_g_per_kg = data[:, 3]
    # bias = data[:, 4]
    pressure_hpa = data[:, 5]
    temperature_k = data[:, 6]

    # Convert WVMR uncertainty into water-vapor column uncertainty.
    # Original IDL:
    #   nmol = pressure*100.*deltaz/8.314/temperature
    #   airmass = nmol * 28.9647 / 1000.
    #   sigma = airmass * sigma / 1000.
    nmol = pressure_hpa * 100.0 * reference_dz / (8.314 * temperature_k)
    airmass_kg_m2 = nmol * 28.9647 / 1000.0
    sigma_pwv_ref = airmass_kg_m2 * sigma_wvmr_g_per_kg / 1000.0

    zbin_arr = np.asarray(zbin, dtype=float)
    scalar_input = zbin_arr.ndim == 0
    zbin_arr = np.atleast_1d(zbin_arr)

    # Convert requested vertical altitude above lidar to slant distance.
    distance_m = zbin_arr / sin_el
    s_m = s_km * 1000.0

    # IDL INTERPOL equivalent. np.interp requires increasing x.
    order = np.argsort(s_m)
    s_m = s_m[order]
    sigma_pwv_ref = sigma_pwv_ref[order]

    sigma_pwv = np.interp(distance_m, s_m, sigma_pwv_ref, left=sigma_pwv_ref[0], right=np.nan)

    maxdist = s_m[-1]
    sigma_maxdist = sigma_pwv_ref[-1]

    # Scale for telescope size: sigma ∝ 1/D.
    sigma_pwv /= telsize / reference_telsize
    sigma_maxdist /= telsize / reference_telsize

    # Scale for bin length along the line of sight: sigma ∝ 1/sqrt(path length).
    path_bin_length_m = dzbin / sin_el
    bin_scale = np.sqrt(path_bin_length_m / reference_dz)
    sigma_pwv /= bin_scale
    sigma_maxdist /= bin_scale

    # Scale for sampling time: sigma ∝ 1/sqrt(t).
    time_scale = np.sqrt(reference_tsampling / tsampling)
    sigma_pwv *= time_scale
    sigma_maxdist *= time_scale

    out_of_range = distance_m > maxdist
    if np.any(out_of_range):
        if extrapolate:
            sigma_pwv[out_of_range] = sigma_maxdist * (distance_m[out_of_range] / maxdist)
        else:
            sigma_pwv[out_of_range] = np.nan

    # Negative distances are not physical.
    sigma_pwv[distance_m < 0] = np.nan

    if scalar_input:
        return float(sigma_pwv[0])
    return sigma_pwv
"""

if __name__ == "__main__":
    # Example for the uploaded Tenerife file:
    # Simulation_Tenerife_lidarH2O_15cm_00deg_10s.txt
    data_path = "Simulation_Tenerife_lidarH2O_15cm_00deg_10s.txt"

    z = np.arange(0.0, 6000.0 + 30.0, 30.0)  # altitude above lidar [m]
    sigma = lidar_sigma_pwv(
        zbin=z,
        dzbin=30.0,
        tsampling=10.0,
        data_file=data_path,
        elevation=90.0,
        telsize=0.15,
        reference_telsize=0.15,
        reference_tsampling=10.0,
    )

    print(sigma)
