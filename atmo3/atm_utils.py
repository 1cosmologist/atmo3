from . import constants as const
import jax.numpy as jnp 
import numpy as np 

def saturation_vapor_pressure(T, Kelvin=False):
    """Calculate saturation vapor pressure over liquid water using the
    Arden-Buck equation.

    Parameters
    ----------
    T : float or jnp.ndarray
        Temperature in degrees Celsius.

    Returns
    -------
    P_sat : float or jnp.ndarray
        Saturation vapor pressure in Pa.
    """

    if Kelvin:
        T = T - 273.15  # Convert Kelvin to Celsius
    
    f = 1. + 3.4e-3 # enhancement factor for pressures up to 800 mbar

    a = jnp.where(T >= 0, 6.1121, 6.1115)  # hPa
    b = jnp.where(T >= 0, 17.502, 22.452)  
    c = jnp.where(T >= 0, 240.97, 272.55)  # °C
    
    P_sat = f * a * jnp.exp((b * T / (T + c))) * 100.0  # Convert hPa to Pa
    return P_sat

def water_vapor_pressure(RH, T, Kelvin=False):
    """Calculate water vapor pressure from relative humidity and temperature.

    Parameters
    ----------
    RH : float or jnp.ndarray
        Relative humidity in percentage (0-100).
    T : float or jnp.ndarray
        Temperature in degrees Celsius.

    Returns
    -------
    P_wv : float or jnp.ndarray
        Water vapor pressure in Pa.
    """
    P_sat = saturation_vapor_pressure(T, Kelvin=Kelvin)
    P_wv = (RH / 100.0) * P_sat
    return P_wv

def dry_air_pressure_from_altitude(altitude):
    """Calculate dry air pressure from altitude using the barometric formula.

    Parameters
    ----------
    altitude : float or jnp.ndarray
        Altitude in meters.

    Returns
    -------
    P_dry : float or jnp.ndarray
        Dry air pressure in Pa.
    """

    P_dry = const.pressure_at_sea_level * (1 - (const.temperature_lapse_rate * altitude) / const.temperature_at_sea_level) ** (const.g / (const.R_dry_air * const.temperature_lapse_rate))
    return P_dry

def total_air_pressure(altitude, RH, T, Kelvin=False):
    """Calculate total air pressure from altitude, relative humidity, and temperature.

    Parameters
    ----------
    altitude : float or jnp.ndarray
        Altitude in meters.
    RH : float or jnp.ndarray
        Relative humidity in percentage (0-100).
    T : float or jnp.ndarray
        Temperature in degrees Celsius.

    Returns
    -------
    P_total : float or jnp.ndarray
        Total air pressure in Pa.
    """
    P_dry = dry_air_pressure_from_altitude(altitude)
    P_wv = water_vapor_pressure(RH, T, Kelvin=Kelvin)
    P_total = P_dry + P_wv
    return P_total

def relative_to_specific_humidity(RH, T, altitude, Kelvin=False):
    """Convert relative humidity to specific humidity.

    Parameters
    ----------
    RH : float or jnp.ndarray
        Relative humidity in percentage (0-100).
    T : float or jnp.ndarray
        Temperature in degrees Celsius.
    altitude : float or jnp.ndarray
        Altitude in meters.

    Returns
    -------
    q : float or jnp.ndarray
        Specific humidity in kg/kg.
    """
    P_total = total_air_pressure(altitude, RH, T, Kelvin=Kelvin)
    P_wv = water_vapor_pressure(RH, T, Kelvin=Kelvin)
    epsilon = const.R_dry_air / const.R_water_vapor
    q = (epsilon * P_wv) / (P_total - P_wv * (1 - epsilon))
    return q
