"""
Fetch H2O line parameters from the HITRAN online database via the HITRAN API
(hapi) and save to a structured HDF5 file with full column metadata.

Requirements
------------
    pip install hitran-api      # provides the 'hapi' module

Usage
-----
    python convert_hitran_to_hdf5.py
"""

import json
import re
import sys
import numpy as np
import h5py
from pathlib import Path

# ---------------------------------------------------------------------------
# Guard: ensure the HITRAN API (hitran-api / hapi) is importable
# ---------------------------------------------------------------------------
try:
    from hapi import db_begin, fetch, getColumn  # type: ignore
except ModuleNotFoundError:
    sys.exit(
        "ERROR: HITRAN API not found.  Install it with:\n"
        "    pip install hitran-api\n"
        "Then re-run this script."
    )

# ---------------------------------------------------------------------------
# Paths & fetch parameters
# ---------------------------------------------------------------------------
DATA_DIR   = Path(__file__).parent
STEM       = "HITRAN_water_vapor_0-1750GHz"
HEADER_FILE = DATA_DIR / f"{STEM}.header"
HDF5_FILE  = DATA_DIR / f"{STEM}.hdf5"
HAPI_CACHE = DATA_DIR / "hapi_cache"

# HITRAN parameters for H2O isotopologue 1 (H2-16O)
MOL_ID  = 1
ISO_ID  = 1
NU_MIN  = 0.0     # cm⁻¹
NU_MAX  = 58.38   # cm⁻¹  ≈ 1750 GHz
TABLE   = "h2o_hitran_fetch"

# ---------------------------------------------------------------------------
# 1. Load column metadata from the header file
# ---------------------------------------------------------------------------
with open(HEADER_FILE, "r") as fh:
    meta = json.load(fh)

order        = meta["order"]
formats      = meta["format"]
descriptions = meta["description"]
defaults     = meta["default"]
positions    = meta["position"]

# Derive fixed widths (used only for metadata attributes in the HDF5)
RECORD_LEN  = 160
sorted_cols = sorted(order, key=lambda c: positions[c])
widths      = {}
for i, col in enumerate(sorted_cols):
    widths[col] = (
        positions[sorted_cols[i + 1]] - positions[col]
        if i + 1 < len(sorted_cols)
        else RECORD_LEN - positions[col]
    )

def fmt_to_dtype(fmt_str):
    f = fmt_str.lstrip("%")
    if f.endswith(("d", "i")):
        return np.int32
    elif f.endswith(("f", "e", "E", "g", "G")):
        return np.float64
    else:
        m = re.match(r"(\d+)", f)
        return f"S{int(m.group(1)) if m else 16}"

col_dtypes = {col: fmt_to_dtype(formats[col]) for col in order}

# ---------------------------------------------------------------------------
# 2. Fetch from HITRAN online via hapi
# ---------------------------------------------------------------------------
print(f"[ONLINE] Fetching H2O lines from HITRAN "
      f"(mol={MOL_ID}, iso={ISO_ID}, nu={NU_MIN}–{NU_MAX} cm⁻¹) ...")
HAPI_CACHE.mkdir(parents=True, exist_ok=True)
db_begin(str(HAPI_CACHE))
fetch(TABLE, MOL_ID, ISO_ID, NU_MIN, NU_MAX)

arrays = {}
for col in order:
    try:
        raw   = getColumn(TABLE, col)
        dtype = col_dtypes[col]
        if isinstance(dtype, str) and dtype.startswith("S"):
            arrays[col] = np.array(
                [v.strip().encode("ascii") if isinstance(v, str) else v for v in raw],
                dtype=dtype,
            )
        else:
            arrays[col] = np.array(raw, dtype=dtype)
    except Exception as exc:
        print(f"  WARNING: could not fetch column '{col}': {exc}")
        arrays[col] = None

n_rows = len(arrays["nu"])
print(f"[ONLINE] {n_rows} spectral lines fetched")

# Sort by wavenumber
sort_idx = np.argsort(arrays["nu"])
arrays   = {col: (arr[sort_idx] if arr is not None else None) for col, arr in arrays.items()}

# ---------------------------------------------------------------------------
# 3. Write HDF5
# ---------------------------------------------------------------------------
print(f"\n[HDF5]   Writing {HDF5_FILE.name} ...")
with h5py.File(HDF5_FILE, "w") as hf:
    hf.attrs["table_name"]   = meta["table_name"]
    hf.attrs["table_type"]   = meta["table_type"]
    hf.attrs["n_rows"]       = n_rows
    hf.attrs["mol_id"]       = MOL_ID
    hf.attrs["iso_id"]       = ISO_ID
    hf.attrs["nu_min_cm"]    = NU_MIN
    hf.attrs["nu_max_cm"]    = NU_MAX
    hf.attrs["column_order"] = json.dumps(order)

    for col in order:
        arr = arrays[col]
        if arr is None:
            continue
        ds = hf.create_dataset(col, data=arr, compression="gzip", compression_opts=4)
        ds.attrs["description"] = descriptions.get(col, "")
        ds.attrs["format"]      = formats.get(col, "")
        ds.attrs["default"]     = str(defaults.get(col, ""))
        ds.attrs["position"]    = positions[col]
        ds.attrs["width"]       = widths[col]

print(f"[HDF5]   Saved {n_rows} rows × {len(order)} columns → {HDF5_FILE}")

# ---------------------------------------------------------------------------
# 4. Quick verification
# ---------------------------------------------------------------------------
with h5py.File(HDF5_FILE, "r") as hf:
    print("\n[HDF5]   Datasets written:")
    for col in order:
        if col not in hf:
            continue
        ds = hf[col]
        print(f"  {col:25s}  shape={ds.shape}  dtype={ds.dtype}  "
              f'desc="{ds.attrs["description"][:55]}"')


