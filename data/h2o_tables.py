import numpy as np
import re
from pathlib import Path
from hapi import db_begin, fetch, getColumn

def extract_c_array(filepath, array_name):
    """Extracts the numbers from a specific C array into a NumPy array."""
    with open(filepath, 'r') as f:
        content = f.read()
        
    pattern = rf"double\s+{array_name}(?:\[.*?\])?\s*=\s*\{{([^}}]+)\}}"
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        raise ValueError(f"Could not find array '{array_name}' in the file.")
        
    raw_numbers = match.group(1)
    number_strings = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw_numbers)
    
    return np.array(number_strings, dtype=float)


def generate_h2o_tables():
    print("Generating H2O Tables...")
    
    # --- Path Setup ---
    # __file__ is create_files/h2o_tables.py
    # .parent is create_files/
    # .parent.parent is data/
    CURRENT_DIR = Path(__file__).parent
    DATA_DIR = CURRENT_DIR.parent
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # IMPORTANT: For GitHub, assume the user places mt_ckd.c in a 'raw_data' folder
    # next to this script, OR change this path to point to your bundled am source.
    AM_SOURCE_FILE = Path('/pscratch/sd/v/valer/atmo3/notebooks/implementation_am_code/am-13.0/src/mt_ckd.c')


    # =====================================================================
    # 1. H2O Line Parameters from HITRAN
    # =====================================================================
    print("  Fetching HITRAN Data for H2O (0 to 1750 GHz)...")
    
    # Store the local hitran cache inside the create_files directory to keep things clean
    db_begin(str(CURRENT_DIR / 'hitran_local_data'))
    
    
    fetch('water_vapor_0_to_1750GHz', 1, 1, 0.0, 58.38)
    
    # 3. Extract the parameters
    f0_cm      = np.array(getColumn('water_vapor_0_to_1750GHz', 'nu'))
    S_ref      = np.array(getColumn('water_vapor_0_to_1750GHz', 'sw'))
    gamma_air  = np.array(getColumn('water_vapor_0_to_1750GHz', 'gamma_air'))
    gamma_self = np.array(getColumn('water_vapor_0_to_1750GHz', 'gamma_self'))
    n_temp     = np.array(getColumn('water_vapor_0_to_1750GHz', 'n_air'))
    E_lower    = np.array(getColumn('water_vapor_0_to_1750GHz', 'elower'))
    delta_air  = np.array(getColumn('water_vapor_0_to_1750GHz', 'delta_air'))
    
    lines_out = DATA_DIR / 'h2o_lines_1750GHz.npz'
    np.savez_compressed(
        lines_out, 
        f0=f0_cm, S=S_ref, ga=gamma_air, 
        gs=gamma_self, n=n_temp, E=E_lower, d=delta_air
    )
    print(f"  -> Saved {len(f0_cm)} lines to {lines_out.name}")


    # =====================================================================
    # 2. H2O Continuum Parameters from MT_CKD
    # =====================================================================
    print("\n  Extracting MT_CKD Continuum Data...")
    
    if not AM_SOURCE_FILE.exists():
        print(f"  WARNING: Could not find C source file at {AM_SOURCE_FILE}")
        print("  Skipping continuum extraction.")
        return

    nu_grid_cm_full  = extract_c_array(AM_SOURCE_FILE, 'mt_ckd_wavenumbers')
    Cs_296_full      = extract_c_array(AM_SOURCE_FILE, 'mt_ckd_self_absco_ref')
    Cf_296_full      = extract_c_array(AM_SOURCE_FILE, 'mt_ckd_for_absco_ref')
    T_exp_full       = extract_c_array(AM_SOURCE_FILE, 'mt_ckd_self_texp')

    # Create a mask to only keep data up to 100 cm^-1 (approx 3 THz)
    mask = nu_grid_cm_full <= 100.0

    nu_grid_cm = nu_grid_cm_full[mask]
    Cs_296     = Cs_296_full[mask]
    Cf_296     = Cf_296_full[mask]
    T_exp      = T_exp_full[mask]

    continuum_out = DATA_DIR / 'mt_ckd_continuum.npz'
    np.savez_compressed(
        continuum_out,
        nu=nu_grid_cm,
        Cs=Cs_296,
        Cf=Cf_296,
        Texp=T_exp
    )
    print(f"  -> Truncated continuum grid to {len(nu_grid_cm)} points.")
    print(f"  -> Saved continuum parameters to {continuum_out.name}")

if __name__ == "__main__":
    generate_h2o_tables()