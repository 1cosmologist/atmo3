import numpy as np
from pathlib import Path
import hapi

def generate_partition_functions():
    print("Generating Partition Functions (H2O, O3, O2)...")
    
    # 1. Define the grid
    T_min = 200.0
    T_max = 350.0
    step = 0.5
    T_grid = np.arange(T_min, T_max + step, step)

    # 2. Fetch from HAPI
    # Note: hapi.partitionSum(MoleculeID, IsotopeID, Temperature)
    print("  Fetching H2O (1, 1)...")
    Q_H2O = np.array([hapi.partitionSum(1, 1, t) for t in T_grid])
    
    print("  Fetching O3 (3, 1)...")
    Q_O3  = np.array([hapi.partitionSum(3, 1, t) for t in T_grid])
    
    print("  Fetching O2 (7, 1)...")
    Q_O2  = np.array([hapi.partitionSum(7, 1, t) for t in T_grid])

    # 3. Dynamic Pathing
    # __file__ is create_files/partition_functions.py
    # .parent is create_files/
    # .parent.parent is data/
    CURRENT_DIR = Path(__file__).parent
    DATA_DIR = CURRENT_DIR.parent
    
    # Ensure the data directory exists just in case
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    out_filepath = DATA_DIR / 'partition_functions.npz'

    # 4. Save to compressed NPZ
    np.savez_compressed(
        out_filepath, 
        T_grid=T_grid, 
        H2O=Q_H2O, 
        O3=Q_O3, 
        O2=Q_O2
    )
    
    print(f"Successfully saved to: {out_filepath.name}")

if __name__ == "__main__":
    generate_partition_functions()