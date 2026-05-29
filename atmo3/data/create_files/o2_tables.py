import re
import numpy as np
from pathlib import Path
import scipy.constants as con

# =====================================================================
# Standalone Fundamental Constants
# Derived from scipy to avoid magic numbers while remaining portable
# =====================================================================
_c = con.c          # Speed of light in vacuum [m/s]
_k_B = con.k        # Boltzmann constant [J/K]
_h = con.h          # Planck constant [J s]

# Conversion factor: GHz to cm^-1 (exactly 29.9792458)
GHZ_TO_CM_INV = _c / 1e7  

# Second radiation constant (hc/k_B) in [cm * K] (approx 1.4387770)
C2_CONST = (_h * _c / _k_B) * 100.0  

# Standard atmosphere in hPa (exactly 1013.25)
P0_ATM_HPA = con.atm / 100.0  

# =====================================================================

def extract_struct_array(filepath, array_name):
    with open(filepath, 'r') as f:
        content = f.read()
    pattern = rf"{array_name}(?:\[.*?\])?\s*=\s*\{{([\s\S]*?)\}};"
    match = re.search(pattern, content)
    if not match: raise ValueError(f"Could not find {array_name}")
    
    rows = re.findall(r"\{([^}]+)\}", match.group(1))
    data = []
    for row in rows:
        row_clean = row.replace('f', '')
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", row_clean)
        data.append([float(n) for n in numbers])
    return np.array(data)

def get_perfect_o2_coupled_data(c_file_path):
    """
    Extracts both the catalog and coupling tables, merges them, 
    and converts all units to standard cm^-1 and atm.
    """
    cat_matrix = extract_struct_array(c_file_path, 'o2_coupled_cat')
    mix_matrix = extract_struct_array(c_file_path, 'o2_line_coupling_coeffs')
    
    # Mask for lines < 1750 GHz
    freq_GHz = cat_matrix[:, 0]
    mask = freq_GHz < 1750.0
    
    # --- TABLE 1: MAIN CATALOG ---
    f0_cm    = cat_matrix[mask, 0] / GHZ_TO_CM_INV
    S_ref_cm = cat_matrix[mask, 1] / GHZ_TO_CM_INV
    E_low_cm = cat_matrix[mask, 2] / C2_CONST
    gamma    = (cat_matrix[mask, 3] * P0_ATM_HPA) / GHZ_TO_CM_INV
    n_temp   = cat_matrix[mask, 5]
    
    # --- TABLE 2: MAKAROV COUPLING ---
    y0 = mix_matrix[mask, 0] * P0_ATM_HPA
    y1 = mix_matrix[mask, 1] * P0_ATM_HPA
    v  = mix_matrix[mask, 2]
    g0 = mix_matrix[mask, 3] * (P0_ATM_HPA**2)
    g1 = mix_matrix[mask, 4] * (P0_ATM_HPA**2)
    d0 = (mix_matrix[mask, 6] * (P0_ATM_HPA**2)) / GHZ_TO_CM_INV
    d1 = (mix_matrix[mask, 7] * (P0_ATM_HPA**2)) / GHZ_TO_CM_INV
    
    return f0_cm, S_ref_cm, E_low_cm, gamma, n_temp, y0, y1, v, g0, g1, d0, d1

def get_perfect_o2_uncoupled_data(c_file_path):
    """
    Extracts the uncoupled catalog table and converts all units.
    """
    cat_matrix = extract_struct_array(c_file_path, 'o2_uncoupled_cat')
    
    # Mask for lines < 1750 GHz
    freq_GHz = cat_matrix[:, 0]
    mask = freq_GHz < 1750.0
    
    # Apply conversions
    f0_cm     = cat_matrix[mask, 0] / GHZ_TO_CM_INV
    S_ref_cm  = cat_matrix[mask, 1] / GHZ_TO_CM_INV
    E_low_cm  = cat_matrix[mask, 2] / C2_CONST
    gamma     = (cat_matrix[mask, 3] * P0_ATM_HPA) / GHZ_TO_CM_INV
    n_temp    = cat_matrix[mask, 5]
    delta_air = (cat_matrix[mask, 6] * P0_ATM_HPA) / GHZ_TO_CM_INV
    
    return f0_cm, S_ref_cm, E_low_cm, gamma, n_temp, delta_air


def generate_o2_tables():
    print("Generating O2 Tables...")
    
    # --- Path Setup ---
    CURRENT_DIR = Path(__file__).parent
    DATA_DIR = CURRENT_DIR.parent
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Target C file path - user must adjust this or place am-13.0 accordingly
    AM_SOURCE_FILE = Path('/pscratch/sd/v/valer/atmo3/notebooks/implementation_am_code/am-13.0/src/o2.c')
    
    if not AM_SOURCE_FILE.exists():
        print(f"  WARNING: Could not find C source file at {AM_SOURCE_FILE}")
        print("  Skipping extraction.")
        return

    # =====================================================================
    # 1. Coupled O2 Lines
    # =====================================================================
    print("  Extracting Coupled O2 Lines (< 1750 GHz)...")
    (f0_c, S_c, E_c, gamma_c, n_c, 
     y0, y1, v, g0, g1, d0, d1) = get_perfect_o2_coupled_data(AM_SOURCE_FILE)
    
    coupled_out = DATA_DIR / 'o2_coupled_lines_1750GHz.npz'
    np.savez_compressed(
        coupled_out,
        f0=f0_c, S=S_c, E=E_c, gamma=gamma_c, n=n_c,
        y0=y0, y1=y1, v=v, g0=g0, g1=g1, d0=d0, d1=d1
    )
    print(f"  -> Saved {len(f0_c)} lines to {coupled_out.name}")

    # =====================================================================
    # 2. Uncoupled O2 Lines
    # =====================================================================
    print("\n  Extracting Uncoupled O2 Lines (< 1750 GHz)...")
    (f0_u, S_u, E_u, gamma_u, 
     n_u, delta_u) = get_perfect_o2_uncoupled_data(AM_SOURCE_FILE)
    
    uncoupled_out = DATA_DIR / 'o2_uncoupled_lines_1750GHz.npz'
    np.savez_compressed(
        uncoupled_out,
        f0=f0_u, S=S_u, E=E_u, gamma=gamma_u, n=n_u, delta=delta_u
    )
    print(f"  -> Saved {len(f0_u)} lines to {uncoupled_out.name}")

if __name__ == "__main__":
    generate_o2_tables()



