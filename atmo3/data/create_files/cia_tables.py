import re
import math
import numpy as np
from pathlib import Path
import scipy.constants as con
from scipy.special import k0, k1

# =====================================================================
# Molecular Constants
# =====================================================================
CIA_MOL = {
    'N2': {'B': 59.6459, 'D': 1.727e-4, 'j_max': 30},
    'O2': {'B': 43.1004438, 'D': 1.45115e-4, 'j_max': 35}
}

# =====================================================================
# Parsing Functions
# =====================================================================
def extract_ebc_pars(filepath, struct_name):
    """Robustly extracts the A, B, C coefficients from a named EBCpars_t struct."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    inside_struct = False
    extracted_rows = []
    
    for line in lines:
        if f"static const EBCpars_t {struct_name}" in line:
            inside_struct = True
            continue
        if inside_struct:
            if "};" in line:
                break
            cleaned = re.sub(r'/\*.*?\*/', '', line)
            cleaned = re.sub(r'//.*$', '', cleaned).strip()
            if cleaned:
                extracted_rows.append(cleaned)
                
    keys = ['S', 'tau1', 'tau2', 'eps', 'tau1p', 'tau2p']
    pars = {}
    
    for i, key in enumerate(keys):
        row_text = extracted_rows[i]
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", row_text)
        row_floats = [float(n) for n in numbers]
        pars[key] = row_floats[1:4] # We save exactly [A, B, C]
        
    return pars

# =====================================================================
# Physics Functions
# =====================================================================
def compute_EBC_parameters(T, mechanism, tables):
    """Computes the temperature-scaled line shape parameters."""
    if mechanism == "3220":
        pars = tables['3220']
    elif mechanism == "5440":
        pars = tables['5440_LT'] if T <= 140.0 else tables['5440_HT']
    else:
        raise ValueError("Mechanism must be '3220' or '5440'")

    lnT = np.log(T)
    lnT_sq = lnT**2
    scaled_params = {}
    
    for key, coeffs in pars.items():
        A, B, C = coeffs
        if A == 0.0:
            scaled_params[key] = 0.0
        else:
            scaled_params[key] = A * np.exp(B * lnT + C * lnT_sq)
            
    return scaled_params

def compute_tau0_ns(T):
    tau0_seconds = con.hbar / (2.0 * con.k * T)
    return tau0_seconds * 1e9

def birnbaum_cohen_shape(nu_detuning_GHz, T, tau1, tau2):
    if tau1 == 0.0 or tau2 == 0.0:
        return np.zeros_like(nu_detuning_GHz)
        
    tau0 = compute_tau0_ns(T)
    exp_term = np.exp((tau2 / tau1) + (2.0 * np.pi * tau0 * nu_detuning_GHz))
    nu_term = 1.0 + (2.0 * np.pi * nu_detuning_GHz * tau1)**2
    z = (np.sqrt(tau0**2 + tau2**2) / tau1) * np.sqrt(nu_term)
    
    prefactor = tau1 / np.pi
    bessel_term = (z * k1(z)) / nu_term
    return prefactor * exp_term * bessel_term

def extended_birnbaum_cohen_shape(nu_detuning_GHz, T, tau1, tau2, eps, tau1p, tau2p):
    Gamma_BC = birnbaum_cohen_shape(nu_detuning_GHz, T, tau1, tau2)
    
    if eps == 0.0 or tau1p == 0.0 or tau2p == 0.0:
        return Gamma_BC
        
    tau0 = compute_tau0_ns(T)
    exp_term_K0 = np.exp((tau1p / tau2p) + (2.0 * np.pi * tau0 * nu_detuning_GHz))
    z_prime = (np.sqrt(tau0**2 + tau1p**2) / tau2p) * np.sqrt(1.0 + (2.0 * np.pi * nu_detuning_GHz * tau2p)**2)
    
    Gamma_K0 = (tau1p / np.pi) * exp_term_K0 * k0(z_prime)
    return (1.0 / (1.0 + eps)) * (Gamma_BC + eps * Gamma_K0)

def compute_energy_h_GHz(j, molecule):
    B = CIA_MOL[molecule]['B']
    D = CIA_MOL[molecule]['D']
    x = j * (j + 1)
    return (B * x) + (D * x**2)

def get_gj(j, molecule):
    if molecule == 'N2': return 6.0 if j % 2 == 0 else 3.0
    elif molecule == 'O2': return 0.0 if j % 2 == 0 else 1.0

def compute_populations(T, molecule):
    j_max = CIA_MOL[molecule]['j_max']
    max_j_prime = j_max + 4 
    
    j_energy_array = np.arange(max_j_prime + 1)
    E_j_h_array = np.zeros(max_j_prime + 1)
    for j in j_energy_array:
        E_j_h_array[j] = compute_energy_h_GHz(j, molecule)
        
    g_j_array = np.zeros(j_max + 1)
    for j in range(j_max + 1):
        g_j_array[j] = get_gj(j, molecule)
        
    E_j_Joules = con.h * (E_j_h_array[:j_max + 1] * 1e9)
    kT_Joules = con.k * T
    dimensionless_energy = E_j_Joules / kT_Joules
    
    boltzmann_terms = g_j_array * np.exp(-dimensionless_energy)
    degeneracy_2j_plus_1 = 2 * np.arange(j_max + 1) + 1
    Q_T = np.sum(degeneracy_2j_plus_1 * boltzmann_terms)
    
    P_j = boltzmann_terms / Q_T
    return P_j, E_j_h_array

def clebsch_squared(j, lam, j_prime):
    if j_prime < abs(j - lam) or j_prime > j + lam: return 0.0
    J = j + lam + j_prime
    if J % 2 != 0: return 0.0
        
    g = J // 2
    f_2g_2j   = math.factorial(2 * g - 2 * j)
    f_2g_2lam = math.factorial(2 * g - 2 * lam)
    f_2g_2jp  = math.factorial(2 * g - 2 * j_prime)
    f_2g_1    = math.factorial(2 * g + 1)
    
    bracket_num = math.factorial(g)
    bracket_den = math.factorial(g - j) * math.factorial(g - lam) * math.factorial(g - j_prime)
    bracket_sq = (bracket_num / bracket_den) ** 2
    
    wigner_sq = (f_2g_2j * f_2g_2lam * f_2g_2jp / f_2g_1) * bracket_sq
    return float((2 * j_prime + 1) * wigner_sq)

def calculate_cia_band_no_radiation(nu_grid_GHz, T, molecule, cia_tables):
    pars_quad = compute_EBC_parameters(T, "3220", cia_tables)
    pars_hex  = compute_EBC_parameters(T, "5440", cia_tables)
    
    if molecule == 'O2':
        pars_quad['S'] *= 0.074
        pars_hex['S']  *= 2.45
        
    P_j, E_j_h = compute_populations(T, molecule)
    j_max = CIA_MOL[molecule]['j_max']
    k_b_total = np.zeros_like(nu_grid_GHz)
    max_j_prime = j_max + 4 
    
    for j in range(j_max + 1):
        if P_j[j] == 0.0: continue
            
        for delta_j in [-2, 0, 2]:
            j_prime = j + delta_j
            if j_prime < 0 or j_prime > max_j_prime: continue
                
            cg_sq = clebsch_squared(j, 2, j_prime)
            if cg_sq > 0.0:
                nu_jj_prime = E_j_h[j] - E_j_h[j_prime]
                Gamma_BC = birnbaum_cohen_shape(nu_grid_GHz + nu_jj_prime, T, pars_quad['tau1'], pars_quad['tau2'])
                k_b_total += pars_quad['S'] * (2 * j + 1) * P_j[j] * cg_sq * Gamma_BC

        for delta_j in [-4, -2, 0, 2, 4]:
            j_prime = j + delta_j
            if j_prime < 0 or j_prime > max_j_prime: continue
                
            cg_sq = clebsch_squared(j, 4, j_prime)
            if cg_sq > 0.0:
                nu_jj_prime = E_j_h[j] - E_j_h[j_prime]
                Gamma_EBC = extended_birnbaum_cohen_shape(
                    nu_grid_GHz + nu_jj_prime, T, 
                    pars_hex['tau1'], pars_hex['tau2'], pars_hex['eps'], 
                    pars_hex['tau1p'], pars_hex['tau2p']
                )
                k_b_total += pars_hex['S'] * (2 * j + 1) * P_j[j] * cg_sq * Gamma_EBC

    return k_b_total

# =====================================================================
# Main Execution Block
# =====================================================================
def generate_cia_tables():
    print("Generating CIA Precomputation Tables...")
    
    # 1. Path Setup
    CURRENT_DIR = Path(__file__).parent
    DATA_DIR = CURRENT_DIR.parent
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    AM_SOURCE_FILE = Path('/pscratch/sd/v/valer/atmo3/notebooks/implementation_am_code/am-13.0/src/cia.c')
    if not AM_SOURCE_FILE.exists():
        print(f"  WARNING: Could not find C source file at {AM_SOURCE_FILE}")
        return

    # 2. Extract Pars dynamically (prevents reading file on module import)
    print("  Extracting EBC Parameters from cia.c...")
    cia_tables = {
        '3220': extract_ebc_pars(AM_SOURCE_FILE, 'N2_EBC3220_INIT'),
        '5440_LT': extract_ebc_pars(AM_SOURCE_FILE, 'N2_EBC5440_LT_INIT'),
        '5440_HT': extract_ebc_pars(AM_SOURCE_FILE, 'N2_EBC5440_HT_INIT')
    }
    
    # 3. Define the Grids
    # Include 1750 by stopping at 1751.0
    freq_grid_GHz = np.arange(0.0, 1751.0, 1.0) 
    T_grid = np.arange(200.0, 350.5, 0.5)
    
    k_N2N2_grid = np.zeros((len(T_grid), len(freq_grid_GHz)))
    k_O2O2_grid = np.zeros((len(T_grid), len(freq_grid_GHz)))
    
    # 4. Generate Tables
    print(f"  Looping over {len(T_grid)} temperature points...")
    for i, T in enumerate(T_grid):
        if i % 20 == 0:
            print(f"    Processing T = {T:.1f} K ({i}/{len(T_grid)})...")
            
        k_N2N2_grid[i, :] = calculate_cia_band_no_radiation(freq_grid_GHz, T, 'N2', cia_tables)
        k_O2O2_grid[i, :] = calculate_cia_band_no_radiation(freq_grid_GHz, T, 'O2', cia_tables)
        
    # 5. Save the Grids
    out_filepath = DATA_DIR / 'cia_tables_no_radiation_1750GHz.npz'
    np.savez_compressed(
        out_filepath,
        T_grid=T_grid,
        freq_grid=freq_grid_GHz,
        k_N2N2=k_N2N2_grid,
        k_O2O2=k_O2O2_grid
    )
    
    print(f"\nDone! Saved perfectly to {out_filepath.name}")
    print(f"Final Data Shapes:")
    print(f"  freq_grid: {freq_grid_GHz.shape}")
    print(f"  k_N2N2:    {k_N2N2_grid.shape}")
    print(f"  k_O2O2:    {k_O2O2_grid.shape}")

if __name__ == "__main__":
    generate_cia_tables()