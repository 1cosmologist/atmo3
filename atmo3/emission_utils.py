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