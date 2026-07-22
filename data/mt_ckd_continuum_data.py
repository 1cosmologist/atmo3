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

DATA_DIR = Path(__file__).parent
NC_FILE       = DATA_DIR / "absco-ref_wv-mt-ckd.nc"
CONTINUUM_OUT = DATA_DIR / "mt_ckd_continuum.hdf5"

NU_MAX_CM = 100.0   # keep ≤ 100 cm⁻¹ (~3 THz)

# =========================================================================
# Read NetCDF
# =========================================================================
print(f"Reading {NC_FILE.name} ...")

ds = xr.open_dataset(NC_FILE)

# Wavenumber cut – include negative wavenumbers if present (padding points)
# but cap at NU_MAX_CM on the positive side
mask = ds["wavenumbers"].values <= NU_MAX_CM

nu = ds["wavenumbers"].values[mask]

data_vars = ["self_absco_ref", "for_absco_ref", "for_closure_absco_ref", "self_texp"]
scalars   = ["ref_press", "ref_temp"]

print(f"  Wavenumber range in file : {ds['wavenumbers'].values.min():.1f} – "
      f"{ds['wavenumbers'].values.max():.1f} cm⁻¹  ({len(ds['wavenumbers'])} points)")
print(f"  After cut (≤ {NU_MAX_CM} cm⁻¹)      : {nu.min():.1f} – {nu.max():.1f} cm⁻¹  ({mask.sum()} points)")

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
    hf.attrs["nu_max_cm"]   = NU_MAX_CM
    hf.attrs["n_points"]    = int(mask.sum())

    # --- Wavenumber coordinate ---
    ds_nu = hf.create_dataset("wavenumbers", data=nu,
                               compression="gzip", compression_opts=4)
    for key, val in ds["wavenumbers"].attrs.items():
        ds_nu.attrs[key] = val

    # --- Spectral data variables (apply wavenumber mask) ---
    for name in data_vars:
        arr  = ds[name].values[mask]
        dset = hf.create_dataset(name, data=arr,
                                  compression="gzip", compression_opts=4)
        for key, val in ds[name].attrs.items():
            dset.attrs[key] = val

    # --- Scalar variables (no mask needed) ---
    for name in scalars:
        val  = float(ds[name].values)
        dset = hf.create_dataset(name, data=val)
        for key, attr_val in ds[name].attrs.items():
            dset.attrs[key] = attr_val

print(f"  -> Saved {int(mask.sum())} points × {len(data_vars)} spectral variables "
      f"+ {len(scalars)} scalars to {CONTINUUM_OUT.name}")

