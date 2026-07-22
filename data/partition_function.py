import numpy as np
import h5py
import hapi
from pathlib import Path

DATA_DIR = Path(__file__).parent

# Temperature grid
T_MIN  = 200.0   # K
T_MAX  = 350.0   # K
T_STEP = 0.5     # K

# Molecules: (name, HITRAN mol_id, iso_id, output filename)
MOLECULES = [
    ("H2O", 1, 1, "partition_function_water_vapor.hdf5"),
    ("O2",  7, 1, "partition_function_oxygen.hdf5"),
    ("O3",  3, 1, "partition_function_ozone.hdf5"),
]

def generate_partition_functions():
    print("Generating partition functions (H2O, O2, O3) ...")

    T_grid = np.arange(T_MIN, T_MAX + T_STEP, T_STEP)

    for name, mol_id, iso_id, filename in MOLECULES:
        print(f"  Fetching {name} (mol={mol_id}, iso={iso_id}) "
              f"over T={T_MIN}–{T_MAX} K ...")

        Q = np.array([hapi.partitionSum(mol_id, iso_id, float(t)) for t in T_grid])

        out = DATA_DIR / filename
        with h5py.File(out, "w") as hf:
            hf.attrs["molecule"]    = name
            hf.attrs["mol_id"]      = mol_id
            hf.attrs["iso_id"]      = iso_id
            hf.attrs["T_min_K"]     = T_MIN
            hf.attrs["T_max_K"]     = T_MAX
            hf.attrs["T_step_K"]    = T_STEP
            hf.attrs["source"]      = "HITRAN via hapi.partitionSum"

            ds_T = hf.create_dataset("T_grid", data=T_grid,
                                     compression="gzip", compression_opts=4)
            ds_T.attrs["units"]       = "K"
            ds_T.attrs["description"] = "Temperature grid"

            ds_Q = hf.create_dataset("Q", data=Q,
                                     compression="gzip", compression_opts=4)
            ds_Q.attrs["units"]       = "dimensionless"
            ds_Q.attrs["description"] = (
                f"Total internal partition function Q(T) for {name} "
                f"(mol_id={mol_id}, iso_id={iso_id})"
            )

        print(f"  -> Saved {len(T_grid)} points to {out.name}")

    print("\nDone.")


if __name__ == "__main__":
    generate_partition_functions()
