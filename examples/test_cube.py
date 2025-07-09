import jax 
jax.config.update("jax_enable_x64", True)
import numpy as np
import matplotlib.pyplot as plt
import atmo3 as a3 

nside_grid = 1024
box_length_in_m = 10000.0

print(f"Initializing atmosphere with nside_grid={nside_grid} and box_length_in_m={box_length_in_m}")
atmo = a3.Atmosphere(nside_grid=nside_grid, box_length_in_m=box_length_in_m)

physical_variable = 'water mass density'
variable_unit = 'g m^-3'

print(f"Creating power spectrum for {physical_variable} normalized to 1.")
k_array     = np.arange(nside_grid) * atmo.grid_wsp.dk
# Based on the Kolmogorov spectrum, we can define a power spectrum based on values from Morris et al. (2025) (arxiv:2410.13064)
k0          = 2*np.pi / 200.0
pofk_array  = ((k0)**2. + k_array**2 )**-(11/6)
# pofk_array  /= np.trapezoid(pofk_array, k_array)  # Normalize the power spectrum
pofk_array /= np.max(pofk_array)  # Normalize the power spectrum to 1
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
    seed=123456789,
)

print("Generating realization for the component...")
atmo.generate_realization(time_step=0)
plt.imshow(
    atmo.components[physical_variable].field[0, :, :],
    extent=(0, box_length_in_m, 0, box_length_in_m),
    cmap='coolwarm'
)
plt.colorbar(label=variable_unit)
plt.title(f"{physical_variable} field y-z plane at x=0 m")
plt.xlabel('y (m)')
plt.ylabel('z (m)')
plt.savefig('./examples/field_yz_plane.png', bbox_inches='tight')
plt.close()

plt.imshow(
    atmo.components[physical_variable].field[:, 0, :],
    extent=(0, box_length_in_m, 0, box_length_in_m),
    cmap='coolwarm'
)
plt.colorbar(label=variable_unit)
plt.title(f"{physical_variable} field x-z plane at y=0 m")
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.savefig('./examples/field_xz_plane.png', bbox_inches='tight')
plt.close()

plt.imshow(
    atmo.components[physical_variable].field[:, :, 0],
    extent=(0, box_length_in_m, 0, box_length_in_m),
    cmap='coolwarm'
)
plt.colorbar(label=variable_unit)
plt.title(f"{physical_variable} field x-y plane at z=0 m")
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.savefig('./examples/field_xy_plane.png', bbox_inches='tight')
plt.close()


print("Computing power spectrum...")
kbins, Pk_bins = a3.compute_power_spectrum(atmo.components[physical_variable], nbins=10)

plt.plot(pspec['k'], pspec['pofk'], label='Input Power Spectrum')
plt.plot(kbins, Pk_bins, 's', label='Computed Power Spectrum') 
plt.xlabel(r'$k$ (m${}^{-1})$')
plt.ylabel(r'$P(k)$')
plt.loglog()
plt.legend(frameon=False)
plt.savefig('./examples/power_spectrum.png', bbox_inches='tight')
