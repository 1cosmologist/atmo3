"""
Generate H2O MT_CKD continuum table from a NetCDF source and save to HDF5.

Input
-----
./absco-ref_wv-mt-ckd.nc
    MT_CKD water vapor continuum absorption coefficients.

Output
------
./mt_ckd_continuum.hdf5
    Same data, truncated to wavenumbers ≤ 100 cm⁻¹ (~3 THz), in HDF5 format
    with all variable and global attributes faithfully copied.
"""

import numpy as np
import h5py
import xarray as xr
from pathlib import Path
from scipy.interpolate import CubicHermiteSpline

DATA_DIR = Path(__file__).parent
NC_FILE       = DATA_DIR / "absco-ref_wv-mt-ckd.nc"
CONTINUUM_OUT = DATA_DIR / "mt_ckd_continuum.hdf5"

NU_MIN_CM = 0.01    # minimum wavenumber (~0.3 GHz)
NU_MAX_CM = 100.0   # maximum wavenumber (~3 THz)
N_POINTS  = 200     # number of points in logspaced grid

# =========================================================================
# Read NetCDF
# =========================================================================
print(f"Reading {NC_FILE.name} ...")

with xr.open_dataset(NC_FILE) as ds:
    nu_orig = ds["wavenumbers"].values
    nu_fine = np.logspace(np.log10(NU_MIN_CM), np.log10(NU_MAX_CM), N_POINTS)

    data_vars = ["self_absco_ref", "for_absco_ref", "for_closure_absco_ref", "self_texp"]
    scalars   = ["ref_press", "ref_temp"]

    print(f"  Wavenumber range in file : {nu_orig.min():.1f} – "
          f"{nu_orig.max():.1f} cm⁻¹  ({len(nu_orig)} points)")
    print(f"  Interpolated log grid    : {nu_fine.min():.4f} – {nu_fine.max():.1f} cm⁻¹  ({len(nu_fine)} points)")

    # =========================================================================
    # Write HDF5
    # =========================================================================
    print(f"\nWriting {CONTINUUM_OUT.name} ...")

    with h5py.File(CONTINUUM_OUT, "w") as hf:

        # --- Global attributes from the NetCDF file ---
        for key, val in ds.attrs.items():
            hf.attrs[key] = val
        # Add provenance / cut metadata
        hf.attrs["source_file"] = NC_FILE.name
        hf.attrs["nu_min_cm"]   = NU_MIN_CM
        hf.attrs["nu_max_cm"]   = NU_MAX_CM
        hf.attrs["n_points"]    = N_POINTS

        # --- Wavenumber coordinate ---
        ds_nu = hf.create_dataset("wavenumbers", data=nu_fine,
                                   compression="gzip", compression_opts=4)
        for key, val in ds["wavenumbers"].attrs.items():
            ds_nu.attrs[key] = val

        # --- Spectral data variables (Cubic Hermite Spline interpolation) ---
        for name in data_vars:
            arr_orig = ds[name].values
            dydx = np.gradient(arr_orig, nu_orig)
            spline = CubicHermiteSpline(nu_orig, arr_orig, dydx)
            arr_fine = spline(nu_fine)

            dset = hf.create_dataset(name, data=arr_fine,
                                      compression="gzip", compression_opts=4)
            for key, val in ds[name].attrs.items():
                dset.attrs[key] = val

        # --- Scalar variables (no interpolation needed) ---
        for name in scalars:
            val  = float(ds[name].values)
            dset = hf.create_dataset(name, data=val)
            for key, attr_val in ds[name].attrs.items():
                dset.attrs[key] = attr_val

print(f"  -> Saved {N_POINTS} points × {len(data_vars)} spectral variables "
      f"+ {len(scalars)} scalars to {CONTINUUM_OUT.name}")

