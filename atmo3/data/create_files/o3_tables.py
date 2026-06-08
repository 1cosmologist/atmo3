import re
import numpy as np
from pathlib import Path
import scipy.constants as con

# =====================================================================
# Standalone Fundamental Constants
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
# Parsing Functions
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

def get_perfect_o3_uncoupled_data(c_file_path):
    """
    Extracts the uncoupled catalog table and converts all units.
    """
    cat_matrix = extract_struct_array(c_file_path, 'o3_cat')
    
    # Mask for lines < 1001 GHz
    freq_GHz = cat_matrix[:, 0]
    mask = freq_GHz < 1001.0 
    
    # Apply conversions
    f0_cm      = cat_matrix[mask, 0] / GHZ_TO_CM_INV
    S_ref_cm   = cat_matrix[mask, 1] / GHZ_TO_CM_INV
    E_low_cm   = cat_matrix[mask, 2] / C2_CONST
    gamma_air  = (cat_matrix[mask, 3] * P0_ATM_HPA) / GHZ_TO_CM_INV
    gamma_self = (cat_matrix[mask, 4] * P0_ATM_HPA) / GHZ_TO_CM_INV
    n_temp     = cat_matrix[mask, 5]
    delta_air  = (cat_matrix[mask, 6] * P0_ATM_HPA) / GHZ_TO_CM_INV
    
    return f0_cm, S_ref_cm, E_low_cm, gamma_air, gamma_self, n_temp, delta_air

# =====================================================================
# Physics Filtering Function
# =====================================================================
def filter_o3_lines_by_exact_peak(
    T, P_total_hPa, P_o3_hPa, k_threshold,
    f0_cm, S_ref, gam_air, gam_self, n_temp, E_lower,
    T_grid, Q_O3_grid
):
    """
    Evaluates the exact peak cross-section of every O3 line using 
    pre-computed partition functions, returning a mask for active lines.
    """
    T_ref = 296.0
    
    # ---------------------------------------------------------
    # 1. Exact S(T) using pre-computed Partition Functions
    # ---------------------------------------------------------
    Q_ref = np.interp(T_ref, T_grid, Q_O3_grid)
    Q_T   = np.interp(T, T_grid, Q_O3_grid)
    Q_ratio = Q_ref / Q_T
    
    boltz_exponent = -C2_CONST * E_lower * ((1.0 / T) - (1.0 / T_ref))
    boltz_factor = np.exp(boltz_exponent)
    
    stim_T = 1.0 - np.exp(-C2_CONST * f0_cm / T)
    stim_ref = 1.0 - np.exp(-C2_CONST * f0_cm / T_ref)
    stim_ratio = stim_T / stim_ref
    
    S_T = S_ref * Q_ratio * boltz_factor * stim_ratio
    
    # ---------------------------------------------------------
    # 2. Exact Line Width gamma(T, P)
    # ---------------------------------------------------------
    P_total_atm = P_total_hPa / P0_ATM_HPA
    P_o3_atm    = P_o3_hPa / P0_ATM_HPA
    
    temp_scaling = (296.0 / T) ** n_temp
    gamma_width = temp_scaling * ((gam_air * (P_total_atm - P_o3_atm)) + (gam_self * P_o3_atm))
    
    # ---------------------------------------------------------
    # 3. The Peak Cross-Section
    # ---------------------------------------------------------
    k_peak_exact = S_T / (np.pi * gamma_width)
    
    # 4. Create the Mask
    mask = k_peak_exact > k_threshold
    
    return mask

# =====================================================================
# Main Execution Block
# =====================================================================
def generate_o3_tables():
    print("Generating Filtered O3 Tables...")
    
    # 1. Path Setup
    CURRENT_DIR = Path(__file__).parent
    DATA_DIR = CURRENT_DIR.parent
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    AM_SOURCE_FILE = Path('/pscratch/sd/v/valer/atmo3/notebooks/implementation_am_code/am-13.0/src/o3.c')
    PARTITION_FILE = DATA_DIR / 'partition_functions.npz'
    
    if not AM_SOURCE_FILE.exists():
        print(f"  WARNING: Could not find C source file at {AM_SOURCE_FILE}")
        return
        
    if not PARTITION_FILE.exists():
        print(f"  WARNING: Could not find partition functions at {PARTITION_FILE}")
        print("  Please run partition_functions.py first.")
        return

    # 2. Load Partition Functions
    print("  Loading pre-computed partition functions...")
    with np.load(PARTITION_FILE) as pf_data:
        T_grid = pf_data['T_grid']
        Q_O3_grid = pf_data['O3']

    # 3. Extract Raw Data
    print("  Extracting raw O3 lines (< 1001 GHz) from C array...")
    f0, S, E, ga, gs, n, d = get_perfect_o3_uncoupled_data(AM_SOURCE_FILE)
    print(f"  -> Extracted {len(f0)} raw lines.")

    # 4. Filter Data based on Stratospheric Peak
    T_strat = 220.0       # K
    P_strat = 30.0        # hPa
    P_o3_strat = 0.0      # Trace amount
    k_threshold = 1e-21   # cm^2 / molecule
    
    print(f"  Filtering lines with k_peak > {k_threshold:.1e} at {T_strat}K, {P_strat}hPa...")
    
    keep_mask = filter_o3_lines_by_exact_peak(
        T_strat, P_strat, P_o3_strat, k_threshold,
        f0, S, ga, gs, n, E,
        T_grid, Q_O3_grid
    )
    
    f0_filt = f0[keep_mask]
    S_filt  = S[keep_mask]
    E_filt  = E[keep_mask]
    ga_filt = ga[keep_mask]
    gs_filt = gs[keep_mask]
    n_filt  = n[keep_mask]
    d_filt  = d[keep_mask]
    
    print(f"  -> Kept {np.sum(keep_mask)} lines.")

    # 5. Save the Filtered Arrays
    out_filepath = DATA_DIR / 'o3_uncoupled_lines_1000GHz.npz'
    np.savez_compressed(
        out_filepath,
        f0=f0_filt,
        S=S_filt,
        E=E_filt,
        ga=ga_filt,
        gs=gs_filt,
        n=n_filt,
        d=d_filt
    )
    
    print(f"\nDone! Successfully saved to {out_filepath.name}")

if __name__ == "__main__":
    generate_o3_tables()