import jax 
jax.config.update("jax_enable_x64", True)
import numpy as np
import matplotlib.pyplot as plt
import atmo3 as a3 
import cmocean as cmo

nside_grid = 512
box_length_in_m = 10000.0
site_altitude = 5110.0  # Atacama altitude in meters
injection_scale_in_m = 200.0  # Injection scale in meters
UTC_hour = 15 # Example UTC hour for the Atacama site
P_surface = 58800.  # Surface pressure in Pa or kg/(m.s^2)

print(f"Initializing atmosphere with nside_grid={nside_grid} and box_length_in_m={box_length_in_m} and site_altitude={site_altitude} meters")
atmo = a3.Atmosphere(
        nside_grid=nside_grid, 
        box_length_in_m=box_length_in_m,
        site_altitude=5110.0,  # Atacama altitude in meters
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
mean_q_profile  = np.load('./examples/mean_q_profiles_2023_07.npz')['typical_profiles']   # kg/kg
delta_q_profile = np.load('./examples/delta_q_profiles_2023_07.npz')['delta_q_map'] # kg/kg
mean_ta_profile = np.load('./examples/mean_ta_profiles_2023_07.npz')['typical_profiles']   # K
altitude        = np.load('./examples/delta_q_profiles_2023_07.npz')['alt_grid']    # m

# print("delta_q values:", delta_q_profile[UTC_hour])

rescale_delta_q = {'h': altitude, 'f': delta_q_profile[UTC_hour]}
rescale_mean_q  = {'h': altitude, 'f': mean_q_profile[UTC_hour]}
rescale_mean_t  = {'h': altitude, 'f': mean_ta_profile[UTC_hour]}

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
plt.savefig('./examples/specific_humidity_yz_plane_wrescaling.png', bbox_inches='tight')
plt.close()


print("Computing dependent fields: virtual temperature, pressure, water vapor density, and precipitable water vapor...")
atmo.compute_virtual_temperature()

plt.imshow(
    atmo.components['virtual temperature'].field[0, :, :].T,
    extent=(0, box_length_in_m, site_altitude, site_altitude + box_length_in_m),
    cmap=cmo.cm.thermal, origin='lower', vmin=atmo.components['virtual temperature'].field.min(), vmax=atmo.components['virtual temperature'].field.max()
)
plt.colorbar(label='K')
plt.title(f"Virtual temperature field y-z plane at x=0 m, UTC hour {UTC_hour}")
plt.xlabel('y (m)')
plt.ylabel('z (m)')
plt.savefig('./examples/virtual_temperature_yz_plane.png', bbox_inches='tight')
plt.close()

atmo.compute_pressure(P_surface=P_surface)  # Surface pressure in Pa

plt.imshow(
    atmo.components['pressure'].field[0, :, :].T,
    extent=(0, box_length_in_m, site_altitude, site_altitude + box_length_in_m),
    cmap=cmo.cm.dense, origin='lower', vmin=atmo.components['pressure'].field.min(), vmax=atmo.components['pressure'].field.max()
)
plt.colorbar(label='Pa')
plt.title(f"Pressure field y-z plane at x=0 m, UTC hour {UTC_hour}")
plt.xlabel('y (m)')
plt.ylabel('z (m)')
plt.savefig('./examples/pressure_yz_plane.png', bbox_inches='tight')
plt.close()

atmo.compute_water_vapor_density()
plt.imshow(
    atmo.components['water vapor density'].field[0, :, :].T,
    extent=(0, box_length_in_m, site_altitude, site_altitude + box_length_in_m),
    cmap=cmo.cm.ice_r, origin='lower', vmin=atmo.components['water vapor density'].field.min(), vmax=atmo.components['water vapor density'].field.max()
)
plt.colorbar(label='kg/m^3')
plt.title(f"Water vapor density field y-z plane at x=0 m, UTC hour {UTC_hour}")
plt.xlabel('y (m)')
plt.ylabel('z (m)')
plt.savefig('./examples/water_vapor_density_yz_plane.png', bbox_inches='tight')
plt.close()

atmo.compute_pwv()
plt.imshow(
    atmo.properties['precipitable water vapor']['value']['f'].T,
    extent=(0, box_length_in_m, 0, box_length_in_m),
    cmap=cmo.cm.deep, origin='lower', vmin=atmo.properties['precipitable water vapor']['value']['f'].min(), vmax=atmo.properties['precipitable water vapor']['value']['f'].max()
)
plt.colorbar(label='mm')
plt.title(f"Precipitable water vapor field y-z plane at z={site_altitude} m, UTC hour {UTC_hour}")
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.savefig('./examples/precipitable_water_vapor_yz_plane.png', bbox_inches='tight')
plt.close()

print("Mean PWV: {:.4f} mm, standard deviation: {:.4f} mm".format(np.mean(atmo.properties['precipitable water vapor']['value']['f']), np.std(atmo.properties['precipitable water vapor']['value']['f'])))

