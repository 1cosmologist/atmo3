import numpy as np
import atmo3 as a3 

nside_grid = 128
box_length_in_m = 10000.0

print(f"Initializing atmosphere with nside_grid={nside_grid} and box_length_in_m={box_length_in_m}")
atmo = a3.Atmosphere(nside_grid=nside_grid, box_length_in_m=box_length_in_m)

physical_variable = 'water mass density'
variable_unit = 'g m^-3'

print(f"Creating power spectrum for {physical_variable} normalized to 1.")
k_array     = np.arange(nside_grid) * atmo.grid_wsp.dk
pofk_array  = ((200.0)**-2. + k_array**2 )**-(11/6)
pofk_array  /= np.trapezoid(pofk_array, k_array)  # Normalize the power spectrum
pspec = {'k': k_array, 'pofk': pofk_array}

print("Creating rescaling factors for the grid...")
h_array   = np.arange(nside_grid) * atmo.grid_wsp.grid_spacing
h_scaling = np.ones_like(h_array) # Rescaling factor 
rescale = {'h': h_array, 'f': h_scaling}

print(f"Adding component: {physical_variable} with unit {variable_unit}")
atmo.add_component(
    field_name=physical_variable,
    field_unit=variable_unit,
    pspec=pspec,
    rescale=rescale,
    seed=123456789
    )

print("Generating realization for the component...")
atmo.generate_realization(time_step=0)




