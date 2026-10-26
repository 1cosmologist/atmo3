"""
Standardized vertical weather profiles for :class:`atmosphere.Atmosphere`.

The atmosphere is agnostic of where its weather data come from; it only
needs a *profile dictionary* with the following 1-D arrays, all defined on
the same altitude axis ``'z'``:

==========================  ============  ================================
Key                         Unit          Description
==========================  ============  ================================
``'z'``                     m             Altitude above sea level, strictly increasing
``'temperature'``           K             Air temperature
``'pressure'``              Pa            Total air pressure
``'specific_humidity'``     kg kg⁻¹       Water-vapour mass fraction of moist air
``'temperature_std'``       K             *(optional)* RMS temperature fluctuation
==========================  ============  ================================

This module provides profiles built from the AFGL (1986) reference
atmospheres (``data/afgl_data``) and the validation helper that
defines the standard.
"""

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from . import atm_utils as au
from . import constants as const

REQUIRED_PROFILE_KEYS = ('z', 'temperature', 'pressure', 'specific_humidity')
OPTIONAL_PROFILE_KEYS = ('temperature_std',)

# AFGL (1986) tables 1a-f: name -> table file stem.
AFGL_MODELS = {
    'tropical':           '1a',
    'midlatitude_summer': '1b',
    'midlatitude_winter': '1c',
    'subarctic_summer':   '1d',
    'subarctic_winter':   '1e',
    'us_standard':        '1f',
}

_TABLE_DIR_CANDIDATES = (
    Path(__file__).resolve().parent / 'data' / 'afgl_data',            # installed package
    Path(__file__).resolve().parent.parent / 'data' / 'afgl_data',     # source checkout
)


def _default_table_dir() -> Path:
    for candidate in _TABLE_DIR_CANDIDATES:
        if candidate.is_dir():
            return candidate
    return _TABLE_DIR_CANDIDATES[0]


def afgl_profiles(model='us_standard', table_dir=None) -> dict:
    """Load an AFGL (1986) reference atmosphere as a standardized profile dict.

    The tabulated units are converted to those used by the atmosphere
    (see module docstring): altitude km -> m, pressure mb -> Pa and water
    vapour mixing ratio ppmv -> specific humidity kg kg⁻¹.  The AFGL
    number density is a total (moist-air) number density, so the H2O
    mixing ratio is interpreted as a mole fraction of total air.

    Parameters
    ----------
    model : str or int, optional
        One of ``'tropical'``, ``'midlatitude_summer'``,
        ``'midlatitude_winter'``, ``'subarctic_summer'``,
        ``'subarctic_winter'``, ``'us_standard'`` (default), the AFGL
        model number 1-6, or the table name ``'1a'`` ... ``'1f'``.
    table_dir : str or path-like, optional
        Directory containing the AFGL CSV tables (``1a.csv`` ...
        ``1f.csv``).  Defaults to the ``afgl_data`` directory installed
        with the package, or ``data/afgl_data`` of the source checkout.

    Returns
    -------
    dict
        Profile dictionary with keys ``'z'`` (m), ``'temperature'`` (K),
        ``'pressure'`` (Pa) and ``'specific_humidity'`` (kg kg⁻¹).
    """
    table = _resolve_table_name(model)
    table_dir = Path(table_dir) if table_dir is not None else _default_table_dir()
    path = table_dir / f'{table}.csv'
    if not path.is_file():
        raise FileNotFoundError(f"AFGL table '{path}' not found. Pass table_dir pointing to the AFGL CSV tables.")

    data = np.genfromtxt(path, delimiter=',', names=True)

    return validate_profiles({
        'z':                 data['z'] * const.km_to_m,
        'temperature':       data['t'],
        'pressure':          data['p'] * const.mbar_to_Pa,
        'specific_humidity': au.volume_mixing_ratio_to_specific_humidity(data['H2O'] * const.ppmv),
    })


def _resolve_table_name(model) -> str:
    models = list(AFGL_MODELS.values())
    if isinstance(model, (int, np.integer)) and not isinstance(model, bool):
        if 1 <= model <= len(models):
            return models[model - 1]
    elif isinstance(model, str):
        key = model.strip().lower().replace(' ', '_').replace('-', '_')
        if key in AFGL_MODELS:
            return AFGL_MODELS[key]
        if key in models:
            return key
    raise ValueError(
        f"Unknown AFGL model {model!r}. Use one of {sorted(AFGL_MODELS)}, 1-{len(models)}, or {models}."
    )


def validate_profiles(profiles: dict) -> dict:
    """Check a profile dictionary and return it as float numpy arrays.

    Raises
    ------
    TypeError
        If ``profiles`` is not a dict.
    ValueError
        If a required key is missing, an array is not 1-D with the same
        length as ``'z'``, ``'z'`` is not strictly increasing, or the
        pressure is not positive.
    """
    if not isinstance(profiles, dict):
        raise TypeError("profiles must be a dict with keys " + ", ".join(REQUIRED_PROFILE_KEYS) + ".")

    missing = [k for k in REQUIRED_PROFILE_KEYS if k not in profiles]
    if missing:
        raise ValueError(f"profiles is missing required keys: {missing}.")

    out = {k: np.asarray(profiles[k], dtype=float)
           for k in REQUIRED_PROFILE_KEYS + OPTIONAL_PROFILE_KEYS if k in profiles}

    n = out['z'].shape[0] if out['z'].ndim == 1 else None
    for key, arr in out.items():
        if arr.ndim != 1 or arr.shape[0] != n:
            raise ValueError(f"profiles['{key}'] must be 1-D with the same length as profiles['z'].")

    if n < 2 or not np.all(np.diff(out['z']) > 0):
        raise ValueError("profiles['z'] must have at least 2 points and be strictly increasing (m).")
    if not np.all(out['pressure'] > 0):
        raise ValueError("profiles['pressure'] must be positive (Pa).")

    return out


def wind_on_grid(wind, z_grid, name: str):
    """Evaluate a wind component on the simulation altitude axis.

    Parameters
    ----------
    wind : float, dict or None
        A float is a wind (m/s) independent of height.  A dict is a
        height-dependent profile ``{'z': altitude above sea level (m),
        name: wind (m/s)}``, strictly increasing in ``'z'``; values
        outside its range are held at the end values.  ``None`` means
        no wind (zero).
    z_grid : array-like
        Altitude axis above sea level (m) to evaluate the wind on.
    name : str
        Wind component name, ``'northwind'`` or ``'eastwind'``; the key
        of the values in a dict profile.

    Returns
    -------
    jnp.ndarray
        Wind (m/s) with the same shape as ``z_grid``.
    """
    z_grid = jnp.asarray(z_grid)

    if wind is None:
        return jnp.zeros_like(z_grid, dtype=float)

    if isinstance(wind, dict):
        if 'z' not in wind or name not in wind:
            raise ValueError(f"The {name} profile must have keys 'z' (m) and '{name}' (m/s).")
        z = np.asarray(wind['z'], dtype=float)
        f = np.asarray(wind[name], dtype=float)
        if z.ndim != 1 or z.shape != f.shape or z.size < 1 or not np.all(np.diff(z) > 0):
            raise ValueError(f"The {name} profile must be 1-D arrays of equal length with 'z' strictly increasing.")
        return jnp.interp(z_grid, z, f)

    if np.ndim(wind) == 0 and not isinstance(wind, (str, bool)):
        return jnp.full_like(z_grid, float(wind), dtype=float)

    raise TypeError(f"{name} must be a float, a profile dict or None.")
