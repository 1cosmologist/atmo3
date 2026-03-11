import jax 
jax.config.update("jax_enable_x64", True)
import numpy as np
import matplotlib.pyplot as plt
import atmo3 as a3 
import cmocean as cmo
import os

nside_grid = 512
box_length_in_m = 10000.0
site_altitude = 5100.0  # Atacama altitude in meters
injection_scale_in_m = 500.0  # Injection scale in meters
UTC_hour = 00# Example UTC hour for the Atacama site
# P_surface = 58800.  # Surface pressure in Pa or kg/(m.s^2)

# Temperature: -0.2966666666666667 0.3364655101909
# Pressure: 53246.617 0.75981015
# Humidity: 2.6450000000000005 0.07685557664227723
# Specific Humidity: 0.00018494912 8.876034e-06
# PWV: 0.5938333333333332 0.00555150511479079

#From APEX
T_surface = 272.2  # Surface temperature in K
# P_surface = 53246.617  # Surface pressure in Pa
# q_avg_surface = 0.00019626718  # Average specific humidity in kg/kg
# q_std_surface = 9.030873e-06  # Standard deviation of specific humidity in kg/kg

apex_pwv_mean = 1.153
apex_pwv_std = 0.032

print(f"Initializing atmosphere with nside_grid={nside_grid} and box_length_in_m={box_length_in_m} and site_altitude={site_altitude} meters")
atmo = a3.Atmosphere(
        nside_grid=nside_grid, 
        box_length_in_m=box_length_in_m,
        site_altitude=site_altitude,  # Atacama altitude in meters
        )

print(f"Creating power spectrum for specific humidity normalized to 1.")
k_array     = np.arange(nside_grid) * atmo.grid_wsp.dk
# Based on the Kolmogorov spectrum, we can define a power spectrum based on values from Morris et al. (2025) (arxiv:2410.13064)
k0          = 2*np.pi / injection_scale_in_m
pofk_array  = ( k0**2. + k_array**2 )**-(11/6)
# pofk_array  /= np.trapezoid(pofk_array, k_array)  # Normalize the power spectrum
pofk_array /= np.max(pofk_array)  # Normalize the power spectrum to 1
pspec = {'k': k_array, 'pofk': pofk_array}

print("Creating rescaling factors for the grid...")
# mean_q_profile  = np.load('./examples/mean_q_profiles_2023_07.npz')['typical_profiles']   # kg/kg
# delta_q_profile = np.load('./examples/delta_q_profiles_2023_07.npz')['delta_q_map'] # kg/kg
# mean_t_profile = np.load('./examples/mean_ta_profiles_2023_07.npz')['typical_profiles']   # K
# altitude        = np.load('./examples/delta_q_profiles_2023_07.npz')['alt_grid']    # m

altitude       = np.load('./examples/era5_profiles_2023_09_16-0000.npz')['altitude']    # m
mean_q_profile = np.load('./examples/era5_profiles_2023_09_16-0000.npz')['q']   # kg/kg
mean_t_profile = np.load('./examples/era5_profiles_2023_09_16-0000.npz')['t']   # K
T_v_profile = np.load('./examples/era5_profiles_2023_09_16-0000.npz')['t_v']   # K
pressure_profile = np.load('./examples/era5_profiles_2023_09_16-0000.npz')['pressure'] # Pa 
wv_density_profile = np.load('./examples/era5_profiles_2023_09_16-0000.npz')['wvd']  # kg/m^3
surface_pressure = pressure_profile[0]

delta_q_profile = 1e-2 * mean_q_profile.copy()  # Assume 1% variability for demonstration purposes

# print("delta_q values:", delta_q_profile[UTC_hour])

# print(altitude[0], delta_q_profile[UTC_hour,0], mean_q_profile[UTC_hour,0], mean_t_profile[UTC_hour,0])
# exit()
# rescale_delta_q = {'h': altitude, 'f': delta_q_profile[UTC_hour]}# * (q_std_surface / delta_q_profile[UTC_hour, 0])}
# rescale_mean_q  = {'h': altitude, 'f': mean_q_profile[UTC_hour]}# * (q_avg_surface / mean_q_profile[UTC_hour, 0])}
# rescale_mean_t  = {'h': altitude, 'f': mean_t_profile[UTC_hour] * (T_surface / mean_t_profile[UTC_hour, 0])}

rescale_delta_q = {'h': altitude, 'f': delta_q_profile} # * (q_std_surface / delta_q_profile[0])}
rescale_mean_q  = {'h': altitude, 'f': mean_q_profile} # * (q_avg_surface / mean_q_profile[0])}
rescale_mean_t  = {'h': altitude, 'f': mean_t_profile} # * (T_surface / mean_t_profile[0])

T_v = a3.virtual_temperature(rescale_mean_t['f'], rescale_mean_q['f'])

dz_grid = np.diff(altitude, prepend=altitude[0])
P_profile = a3.pressure_from_virtual_temperature(T_v, dz_grid, P_surface=surface_pressure)

# plt.plot(altitude, (pressure_profile - P_profile), label='Pressure computation error')
# # plt.plot(altitude, P_profile, label='Computed Pressure Profile', linestyle='--')
# plt.xlabel('Altitude (m)')
# plt.ylabel('Pressure (Pa)')
# plt.title('Pressure Profile difference between ERA5 and Computed')
# plt.legend()
# plt.savefig('./examples/pressure_profile_difference.png', bbox_inches='tight')
# plt.close()

plt.plot(altitude, T_v_profile, label='T virtual from ERA5')
plt.plot(altitude,T_v, label='Computed Virtual Temperature Profile', linestyle='--')
plt.xlabel('Altitude (m)')
plt.ylabel('Temperature (K)')
plt.title('Virtual Temperature Profile Comparison between ERA5 and Computed')
plt.legend()
plt.savefig('./examples/t_virtual_profile.png', bbox_inches='tight')
plt.close()

# from scipy import constants as con
# molar_mass_dry_air = 28.9647e-3 # kg mol-1
# R_dry_air = con.gas_constant / molar_mass_dry_air # J mol-1 K-1 / (kg mol-1) = J kg-1 K-1 = m-2 s-2 K-1
# water_vapor_density = (mean_q_profile * pressure_profile) / (R_dry_air * T_v_profile)
wv_density = a3.water_vapor_density(rescale_mean_q['f'], pressure_profile, T_v)

# plt.plot(altitude, wv_density_profile, label='ERA5 Water Vapor Density Profile')
# plt.plot(altitude, water_vapor_density, label='Calculated Water Vapor Density Profile (alt)', linestyle='--')
# plt.plot(altitude, wv_density, label='Computed Water Vapor Density Profile', linestyle=':')
# plt.xlabel('Altitude (m)')
# plt.ylabel('Water Vapor Density (kg/m^3)')
# plt.title('Water Vapor Density Profile Comparison')
# plt.legend()
# plt.savefig('./examples/water_vapor_density_profile_comparison.png', bbox_inches='tight')
# plt.close()

# print(water_vapor_density.max(), wv_density.max(), wv_density_profile.max())
pwv_mean = np.trapezoid(wv_density, x=altitude)  # Convert to mm
print(altitude[0], pwv_mean, apex_pwv_mean)

rescale_factor_mean = (apex_pwv_mean / pwv_mean)
rescale_mean_t['f'] *= (T_surface / mean_t_profile[0])
rescale_mean_q['f'] *= rescale_factor_mean

# T_v = a3.virtual_temperature(rescale_mean_t['f'], rescale_mean_q['f'] * rescale_factor_mean)
# P_profile = a3.pressure_from_virtual_temperature(T_v, atmo.grid_wsp.grid_spacing, P_surface=surface_pressure)
# # print(P_profile.shape)
# wv_density = a3.water_vapor_density(rescale_mean_q['f'] * rescale_factor_mean, P_profile, T_v)

# pwv_mean = np.trapezoid(wv_density, x=altitude)  # Convert to mm
print(f"Adjusted pwv mean after rescaling: {pwv_mean} mm, target: {apex_pwv_mean} mm")
# exit()

print(f"Adding component: specific humidity with unit kg/kg")
atmo.add_component(
    field_name='specific humidity',
    field_unit='kg/kg',
    pspec=pspec,
    rescale=rescale_delta_q,
    mean=rescale_mean_q,
    seed=123456789,
)

atmo.add_property(
    property_name='temperature',
    property_unit='K',
    property_value=rescale_mean_t
)

print("Generating realization for the component...")
atmo.generate_realization(time_step=0)

plt.imshow(
    atmo.components['specific humidity'].field[0, :, :].T,
    extent=(0, box_length_in_m, site_altitude, site_altitude + box_length_in_m),
    cmap=cmo.cm.rain, origin='lower', vmin=atmo.components['specific humidity'].field.min(), vmax=atmo.components['specific humidity'].field.max()
)
plt.colorbar(label='kg/kg')
plt.title(f"specific humidity field y-z plane at x=0 m, UTC hour {UTC_hour}")
plt.xlabel('y (m)')
plt.ylabel('z (m)')
plt.savefig(f'./examples/specific_humidity_yz_plane_wrescaling_UTC{UTC_hour}.png', bbox_inches='tight')
plt.close()


print("Computing dependent fields: virtual temperature, pressure, water vapor density, and precipitable water vapor...")
atmo.compute_virtual_temperature()
print(atmo.components['virtual temperature'].field.min(), atmo.components['virtual temperature'].field.max())

plt.imshow(
    atmo.components['virtual temperature'].field[0, :, :].T,
    extent=(0, box_length_in_m, site_altitude, site_altitude + box_length_in_m),
    cmap=cmo.cm.thermal, origin='lower', vmin=atmo.components['virtual temperature'].field.min(), vmax=atmo.components['virtual temperature'].field.max()
)
plt.colorbar(label='K')
plt.title(f"Virtual temperature field y-z plane at x=0 m, UTC hour {UTC_hour}")
plt.xlabel('y (m)')
plt.ylabel('z (m)')
plt.savefig(f'./examples/virtual_temperature_yz_plane_UTC{UTC_hour}.png', bbox_inches='tight')
plt.close()

print(surface_pressure)
atmo.compute_pressure(P_surface=surface_pressure)  # Surface pressure in Pa

plt.imshow(
    atmo.components['pressure'].field[0, :, :].T,
    extent=(0, box_length_in_m, site_altitude, site_altitude + box_length_in_m),
    cmap=cmo.cm.dense, origin='lower', vmin=atmo.components['pressure'].field.min(), vmax=atmo.components['pressure'].field.max()
)
plt.colorbar(label='Pa')
plt.title(f"Pressure field y-z plane at x=0 m, UTC hour {UTC_hour}")
plt.xlabel('y (m)')
plt.ylabel('z (m)')
plt.savefig(f'./examples/pressure_yz_plane_UTC{UTC_hour}.png', bbox_inches='tight')
plt.close()

atmo.compute_water_vapor_density()
atmo.compute_pwv()


output_filename = './examples/pwv_stats.dat'
mean_pwv = np.mean(atmo.properties['precipitable water vapor']['value']['f'])
std_pwv = np.std(atmo.properties['precipitable water vapor']['value']['f'])

normalization = apex_pwv_std / std_pwv

z_avg_wv = np.mean(atmo.components['water vapor density'].field, axis=(0,1))
atmo.components['water vapor density'].field -= z_avg_wv[None, None, :]
atmo.components['water vapor density'].field *= normalization
atmo.components['water vapor density'].field += z_avg_wv[None, None, :]
atmo.compute_pwv()

mean_pwv = np.mean(atmo.properties['precipitable water vapor']['value']['f'])
std_pwv = np.std(atmo.properties['precipitable water vapor']['value']['f'])

rel_err_pwv_mean = np.abs(mean_pwv - apex_pwv_mean) / apex_pwv_mean
rel_err_pwv_std = np.abs(std_pwv - apex_pwv_std) / apex_pwv_std

print(f"PWV mean: {mean_pwv:.4f} mm, PWV std: {std_pwv:.4f} mm")
print(f"Relative error in PWV mean: {rel_err_pwv_mean*100:.2f} %, Relative error in PWV std: {rel_err_pwv_std*100:.2f} %")

plt.imshow(
    atmo.components['water vapor density'].field[0, :, :].T,
    extent=(0, box_length_in_m, site_altitude, site_altitude + box_length_in_m),
    cmap=cmo.cm.ice_r, origin='lower', vmin=atmo.components['water vapor density'].field.min(), vmax=atmo.components['water vapor density'].field.max()
)
plt.colorbar(label='kg/m^3')
plt.title(f"Water vapor density field y-z plane at x=0 m, UTC hour {UTC_hour}")
plt.xlabel('y (m)')
plt.ylabel('z (m)')
plt.savefig(f'./examples/water_vapor_density_yz_plane_UTC{UTC_hour}.png', bbox_inches='tight')
plt.close()

# atmo.components['water vapor density'].field *= rescale_factor_mean
plt.imshow(
    atmo.properties['precipitable water vapor']['value']['f'].T,
    extent=(0, box_length_in_m, 0, box_length_in_m),
    cmap=cmo.cm.deep, origin='lower', vmin=atmo.properties['precipitable water vapor']['value']['f'].min(), vmax=atmo.properties['precipitable water vapor']['value']['f'].max()
)
plt.colorbar(label='mm')
plt.title(f"Precipitable water vapor field y-z plane at z={site_altitude} m, UTC hour {UTC_hour}")
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.savefig(f'./examples/precipitable_water_vapor_yz_plane_UTC{UTC_hour}.png', bbox_inches='tight')
plt.close()

np.savez(f'./examples/atmo_fields_UTC{UTC_hour}.npz', water_vapor_density=atmo.components['water vapor density'].field)
# Check if the file exists to determine if a header is needed
file_exists = os.path.exists(output_filename)

with open(output_filename, 'a') as f:
    if not file_exists:
        f.write("# UTC_hour mean_pwv_mm std_pwv_mm\n")
    f.write(f"{UTC_hour} {mean_pwv:.4f} {std_pwv:.4f}\n")

print(f"Appended PWV stats for UTC hour {UTC_hour} to {output_filename}")
