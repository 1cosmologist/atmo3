import jax 
jax.config.update("jax_enable_x64", True)
import numpy as np
import matplotlib.pyplot as plt
import atmo3 as a3 
import cmocean as cmo

nside_grid = 1024
box_length_in_m = 10000.0
site_altitude = 5110.0  # Atacama altitude in meters
UTC_hour = 15 # Example UTC hour for the Atacama site

print(f"Initializing atmosphere with nside_grid={nside_grid} and box_length_in_m={box_length_in_m} and site_altitude={site_altitude} meters")
atmo = a3.Atmosphere(
        nside_grid=nside_grid, 
        box_length_in_m=box_length_in_m,
        site_altitude=5110.0,  # Atacama altitude in meters
        )

physical_variable = 'specific humidity'
variable_unit = 'g/kg'

print(f"Creating power spectrum for {physical_variable} normalized to 1.")
k_array     = np.arange(nside_grid) * atmo.grid_wsp.dk
# Based on the Kolmogorov spectrum, we can define a power spectrum based on values from Morris et al. (2025) (arxiv:2410.13064)
k0          = 2*np.pi / 200.
pofk_array  = (k0**2. + k_array**2 )**-(11/6)
# pofk_array  /= np.trapezoid(pofk_array, k_array)  # Normalize the power spectrum
pofk_array /= np.max(pofk_array)  # Normalize the power spectrum to 1
pspec = {'k': k_array, 'pofk': pofk_array}

print("Creating rescaling factors for the grid...")
delta_q_profile = np.load('./examples/delta_q_profiles_2023_07.npz')['delta_q_map'] * 1e3 # Convert from kg/kg to g/kg
altitude = np.load('./examples/delta_q_profiles_2023_07.npz')['alt_grid']

# print("delta_q values:", delta_q_profile[UTC_hour])

rescale = {'h': altitude, 'f': delta_q_profile[UTC_hour]}

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
    atmo.components[physical_variable].field[0, :, :].T,
    extent=(0, box_length_in_m, site_altitude, site_altitude + box_length_in_m),
    cmap=cmo.cm.balance, origin='lower', vmin = -0.01, vmax = 0.01
)
plt.colorbar(label=variable_unit)
plt.title(f"{physical_variable} field y-z plane at x=0 m, UTC hour {UTC_hour}")
plt.xlabel('y (m)')
plt.ylabel('z (m)')
plt.savefig('./examples/field_yz_plane_wrescaling.png', bbox_inches='tight')
plt.close()

plt.imshow(
    atmo.components[physical_variable].field[:, 0, :].T,
    extent=(0, box_length_in_m, site_altitude, site_altitude + box_length_in_m),
    cmap=cmo.cm.balance, origin='lower', vmin = -0.01, vmax = 0.01
)
plt.colorbar(label=variable_unit)
plt.title(f"{physical_variable} field x-z plane at y=0 m, UTC hour {UTC_hour}")
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.savefig('./examples/field_xz_plane_wrescaling.png', bbox_inches='tight')
plt.close()

plt.imshow(
    atmo.components[physical_variable].field[:, :, 0].T,
    extent=(0, box_length_in_m, 0, box_length_in_m),
    cmap=cmo.cm.balance, origin='lower', vmin = -0.01, vmax = 0.01
)
plt.colorbar(label=variable_unit)
plt.title(f"{physical_variable} field x-y plane at z={site_altitude} m, UTC hour {UTC_hour}")
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.savefig('./examples/field_xy_plane_wrescaling.png', bbox_inches='tight')
plt.close()

print(f"Mean of fluctuations: {atmo.components[physical_variable].field.mean()}")
print(f"Standard deviation of fluctuations: {atmo.components[physical_variable].field.std()}")

z_ax = np.arange(nside_grid) * atmo.grid_wsp.grid_spacing + site_altitude
sigma_z = atmo.components[physical_variable].field.std(axis=(0, 1))
plt.plot(z_ax, sigma_z, 'o', ms=1., label=f'Standard deviation of {physical_variable}')
plt.plot(altitude, delta_q_profile[UTC_hour], label='Input sigma profile')
plt.xlabel('Height (m)')
plt.ylabel(f'Standard deviation of {physical_variable} (g/kg)')
plt.title(f'Standard deviation profile of {physical_variable} at UTC hour {UTC_hour}')
plt.legend()
plt.savefig('./examples/sigma_profile_wrescaling.png', bbox_inches='tight')
plt.close()



# print("Computing power spectrum...")
# kbins, Pk_bins = a3.compute_power_spectrum(atmo.components[physical_variable], nbins=10)

# print(f"Smallest k bin: {kbins[0]} m^-1, corresponding to a scale of {2*np.pi/kbins[0]} m")
# print(f"Largest k bin: {kbins[-1]} m^-1, corresponding to a scale of {2*np.pi/kbins[-1]} m")

# plt.plot(pspec['k'], pspec['pofk'], label='Input Power Spectrum')
# plt.plot(kbins, Pk_bins, 's', label='Computed Power Spectrum') 
# plt.xlabel(r'$k$ (m${}^{-1})$')
# plt.ylabel(r'$P(k)$')
# plt.loglog()
# plt.legend(frameon=False)
# plt.savefig('./examples/power_spectrum.png', bbox_inches='tight')
