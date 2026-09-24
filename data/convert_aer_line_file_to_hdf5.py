"""
Fetch H2O line parameters from the AER Line File (HITRAN parameters as
modified by AER, e.g. through radiation closure studies) and save to a
structured HDF5 file with full column metadata.

The AER Line File is distributed via Zenodo and downloaded here using the
same interface as data/AER_Line_File/get_line_file.py (the zenodo_get
Python API). Once staged locally, the per-molecule H2O line file is parsed
directly as a fixed-width column-fixed table (the same layout already used
by convert_hitran_to_hdf5.py), so no HITRAN API dependency is required.

Requirements
------------
    pip install zenodo-get==1.3.0

Usage
-----
    python convert_aer_line_file_to_hdf5.py
"""

import json
import os
import sys
import numpy as np
import h5py
from pathlib import Path

try:
    import zenodo_get  # type: ignore  # noqa: F401 (used internally by get_line_file.py)
except ModuleNotFoundError:
    sys.exit(
        "ERROR: zenodo_get not found.  Install it with:\n"
        "    pip install zenodo-get==1.3.0\n"
        "Then re-run this script."
    )

sys.path.insert(0, str(Path(__file__).parent / "AER_Line_File"))
from get_line_file import lineFile  # type: ignore

# ---------------------------------------------------------------------------
# Paths & fetch parameters
# ---------------------------------------------------------------------------
DATA_DIR    = Path(__file__).parent
STEM        = "AER_water_vapor_0-1750GHz"
HDF5_FILE   = DATA_DIR / f"{STEM}.hdf5"

# AER Line File staging, following the interface used by
# data/AER_Line_File/get_line_file.py -> lineFile.getLineFile()
ZENODO_RECORD_ID = 18881607          # AER Line File v3.9, per AER_Line_File/README.md
LINES_DIR        = DATA_DIR / "AER_Line_File_data"

# HITRAN parameters for H2O isotopologue 1 (H2-16O)
MOL_ID  = 1
ISO_ID  = 1
NU_MIN  = 0.0     # cm⁻¹
NU_MAX  = 58.38   # cm⁻¹  ≈ 1750 GHz


def stage_aer_line_file():
    """Download and extract the AER Line File via get_line_file.py's lineFile."""
    if LINES_DIR.exists():
        print(f"[ZENODO] {LINES_DIR.name} already staged, skipping download")
        return

    print(f"[ZENODO] Fetching AER Line File (record={ZENODO_RECORD_ID}) ...")
    # lineFile.getLineFile() works with paths relative to the cwd, so run it
    # with cwd set to DATA_DIR to keep the download/extraction contained there.
    cwd = os.getcwd()
    os.chdir(DATA_DIR)
    try:
        line_file = lineFile({"lines_path": str(LINES_DIR), "record_id": ZENODO_RECORD_ID})
        line_file.getLineFile()
    finally:
        os.chdir(cwd)


def find_h2o_line_file():
    """Locate the H2O-specific line file under line_files_By_Molecule.

    Each molecule is staged as a directory (e.g. 01_H2O/) containing a single
    data file of the same name (01_H2O/01_H2O).
    """
    molecule_dir = LINES_DIR / "line_files_By_Molecule"
    candidates = sorted(molecule_dir.glob("*"))
    for path in candidates:
        name = path.name.lower()
        if "h2o" in name or name.startswith("01_") or name.startswith("01."):
            if path.is_dir():
                path = path / path.name
            return path
    raise FileNotFoundError(
        f"Could not find an H2O line file under {molecule_dir}; "
        f"available entries: {[p.name for p in candidates]}"
    )


# ---------------------------------------------------------------------------
# 1. Column metadata for the AER Line File's HITRAN f100 record layout
#
# The AER Line File records have been converted from the 160-character
# HITRAN format (used by convert_hitran_to_hdf5.py / the online HITRAN API)
# to the older, narrower 100-character HITRAN format, which drops the
# line-mixing flag and the gp/gpp statistical weights and uses shorter
# quanta fields. The layout below must be used instead of the f160 header.
# ---------------------------------------------------------------------------
RECORD_LEN = 100

# col: (position, width, format, default, description, dtype)
HITRAN_F100_FIELDS = {
    "molec_id":            (0,  2,  "%2d",     0,     "Molecule ID", np.int32),
    "local_iso_id":        (2,  1,  "%1d",     0,     "Isotopologue ID", np.int32),
    "nu":                  (3,  12, "%12.6f",  0.0,   "Transition wavenumber [cm^-1]", np.float64),
    "sw":                  (15, 10, "%10.3E",  0.0,   "Line intensity at 296 K [cm^-1/(molec cm^-2)]", np.float64),
    "a":                   (25, 10, "%10.3E",  0.0,   "Einstein A-coefficient [s^-1]", np.float64),
    "gamma_air":           (35, 5,  "%5.4f",   0.0,   "Air-broadened HWHM at 296 K, 1 atm [cm^-1/atm]", np.float64),
    "gamma_self":          (40, 5,  "%5.3f",   0.0,   "Self-broadened HWHM at 296 K, 1 atm [cm^-1/atm]", np.float64),
    "elower":              (45, 10, "%10.4f",  0.0,   "Lower-state energy [cm^-1]", np.float64),
    "n_air":               (55, 4,  "%4.2f",   0.0,   "Temperature exponent for air-broadened HWHM", np.float64),
    "delta_air":           (59, 8,  "%8.6f",   0.0,   "Air pressure-induced line shift at 296 K, 1 atm [cm^-1/atm]", np.float64),
    "global_upper_quanta": (67, 3,  "%3s",     "000", "Upper-state global quanta", "S3"),
    "global_lower_quanta": (70, 3,  "%3s",     "000", "Lower-state global quanta", "S3"),
    "local_upper_quanta":  (73, 9,  "%9s",     "000", "Upper-state local quanta", "S9"),
    "local_lower_quanta":  (82, 9,  "%9s",     "000", "Lower-state local quanta", "S9"),
    "ierr":                (91, 3,  "%3s",     "EEE", "Uncertainty indices", "S3"),
    "iref":                (94, 6,  "%6s",     "EEE", "Reference indices", "S6"),
}

order        = list(HITRAN_F100_FIELDS.keys())
positions    = {col: v[0] for col, v in HITRAN_F100_FIELDS.items()}
widths       = {col: v[1] for col, v in HITRAN_F100_FIELDS.items()}
formats      = {col: v[2] for col, v in HITRAN_F100_FIELDS.items()}
defaults     = {col: v[3] for col, v in HITRAN_F100_FIELDS.items()}
descriptions = {col: v[4] for col, v in HITRAN_F100_FIELDS.items()}
col_dtypes   = {col: v[5] for col, v in HITRAN_F100_FIELDS.items()}

def parse_fixed_width_line(line):
    """Slice one column-fixed record using the header's position/width map."""
    row = {}
    for col in order:
        start = positions[col]
        text  = line[start:start + widths[col]].strip()
        dtype = col_dtypes[col]
        if isinstance(dtype, str) and dtype.startswith("S"):
            row[col] = text.encode("ascii")
        elif not text:
            row[col] = dtype(defaults[col])
        else:
            if dtype is np.float64:
                text = text.replace("D", "E").replace("d", "e")  # Fortran exponent notation
            row[col] = dtype(text)
    return row

# ---------------------------------------------------------------------------
# 2. Stage the AER Line File locally and parse the H2O line records
# ---------------------------------------------------------------------------
stage_aer_line_file()
h2o_line_file = find_h2o_line_file()

print(f"[LOCAL]  Parsing {h2o_line_file.name} ...")
# The file starts with a 42-line narrative copyright/provenance header (no
# fixed column layout); fixed-width data records start at line 43.
HEADER_LINES = 42
rows = [
    parse_fixed_width_line(line)
    for line in h2o_line_file.read_text().splitlines()[HEADER_LINES:]
]

arrays = {}
for col in order:
    dtype = col_dtypes[col]
    arrays[col] = np.array([row[col] for row in rows], dtype=dtype)

# Restrict to H2O isotopologue 1 and the target wavenumber range, since the
# local table is not pre-filtered the way the online fetch() call was.
keep = (
    (arrays["molec_id"] == MOL_ID)
    & (arrays["local_iso_id"] == ISO_ID)
    & (arrays["nu"] >= NU_MIN)
    & (arrays["nu"] <= NU_MAX)
)
arrays = {col: (arr[keep] if arr is not None else None) for col, arr in arrays.items()}

n_rows = len(arrays["nu"])
print(f"[LOCAL]  {n_rows} spectral lines selected")

# Sort by wavenumber
sort_idx = np.argsort(arrays["nu"])
arrays   = {col: (arr[sort_idx] if arr is not None else None) for col, arr in arrays.items()}

# ---------------------------------------------------------------------------
# 3. Write HDF5
# ---------------------------------------------------------------------------
print(f"\n[HDF5]   Writing {HDF5_FILE.name} ...")
with h5py.File(HDF5_FILE, "w") as hf:
    hf.attrs["table_name"]   = STEM
    hf.attrs["table_type"]   = "column-fixed (HITRAN f100)"
    hf.attrs["n_rows"]       = n_rows
    hf.attrs["mol_id"]       = MOL_ID
    hf.attrs["iso_id"]       = ISO_ID
    hf.attrs["nu_min_cm"]    = NU_MIN
    hf.attrs["nu_max_cm"]    = NU_MAX
    hf.attrs["column_order"] = json.dumps(order)
    hf.attrs["source"]       = f"AER Line File, Zenodo record {ZENODO_RECORD_ID}"

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
