import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_platforms', 'cpu')
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from atmo3 import grid_utils as gutl
from atmo3 import box
from atmo3 import atm_utils as au
from atmo3 import constants as const
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ==============================================================================
# 1. Define the Geometric Workspace
# ==============================================================================
Lz = 9750.0  
dz = 30.0    
Nz = int(Lz / dz)  

L_0_q = 500 # m
vx = 5.0    # m/s (Wind pushing clouds across the zenith beam)
vy = 0.0    # m/s

# 40km box gives us plenty of room for 30 minutes of 5m/s wind advection
N = jnp.array([2048, 2048, Nz])                
Lbox = jnp.array([40000.0, 40000.0, Lz])      
site_altitude = 2400.0                        

grid_wsp = gutl.GridWorkspace(N=N, Lbox=Lbox, site_altitude=site_altitude)
z_axis_grid = grid_wsp.grid_axis(axis=2, altitude_axis=True)

# ==============================================================================
# 2. Load Thermodynamics
# ==============================================================================
data_file = "/pscratch/sd/v/valer/atmo3/notebooks/lidar_noise/Simulation_Tenerife_lidarH2O_15cm_00deg_10s.txt"
data = np.loadtxt(data_file, comments=("#", ";"))

z_km = data[:, 1] 
q = data[:, 2] 
pressure_hpa = data[:, 5] 
temperature_k = data[:, 6] 

# 3D Fourier volume boundaries
kx_max = (grid_wsp.N[0] / 2.0) * grid_wsp.dk[0]
ky_max = (grid_wsp.N[1] / 2.0) * grid_wsp.dk[1]
kz_max = (grid_wsp.N[2] / 2.0) * grid_wsp.dk[2]
k_max_3d = np.sqrt(kx_max**2 + ky_max**2 + kz_max**2)

dk_min = np.min(grid_wsp.dk)
n_points = int(k_max_3d / dk_min) + 10  
k_array = np.arange(n_points) * dk_min

k0_q  = 2*np.pi / L_0_q   
pofk_q  = ( k0_q**2.  + k_array**2 )**-(11/6)
pofk_q  /= np.max(pofk_q)
pspec_q  = {'k': k_array, 'pofk': pofk_q}

q_std_atmosphere = q

# ==============================================================================
# 3. Setup Staring Strategy & Thermodynamics
# ==============================================================================
from atmo3.observation import Observer
import jax.random as random

dt_simulation = 1.0  
total_duration = 1800 # 30 minutes
n_steps = int(total_duration / dt_simulation)
start_time = np.datetime64('2026-01-01T00:00:00')
timelist = [start_time + np.timedelta64(int(i * dt_simulation), 's') for i in range(n_steps)]

# STARE AT ZENITH: Elevation is locked at 90, Azimuth is locked at 0
elevation_deg = [90.0] * n_steps
azimuth_deg = [0.0] * n_steps

T_v = au.virtual_temperature(temperature_k, q) 
q2rho_h2o = pressure_hpa * 100.0 / (const.R_dry_air * T_v) 
rho_wv_mean_1d = q * q2rho_h2o
mean_pwv = jnp.trapezoid(rho_wv_mean_1d, x=z_km * 1000)
target_sigma_pwv = 0.01 * mean_pwv

q2rho_interp_1d = jnp.interp(z_axis_grid, z_km * 1000, q2rho_h2o, left='extrapolate', right='extrapolate')
mean_interp_1d = jnp.interp(z_axis_grid, z_km * 1000, q, left='extrapolate', right='extrapolate')

# ==============================================================================
# 4. Generate the Atmospheric Cube
# ==============================================================================
master_key = random.PRNGKey(42)
master_key, atmo_key = random.split(master_key)
cube_seed = int(random.randint(atmo_key, shape=(), minval=0, maxval=99999999))

print("Generating Atmospheric Cube...")
wv_box = box.Box(
    grid_wsp=grid_wsp,
    field_name='water vapor',
    field_unit='kg / kg',
    spectrum=pspec_q, 
    zscaling={'h': z_km * 1000, 'f': q_std_atmosphere}, 
    seed=cube_seed 
)

wv_box.generate_field_fluctuations(time_step=0)

pwv_fluctuation_plane = jnp.trapezoid(wv_box.field * q2rho_interp_1d.reshape(1, 1, -1), x=z_axis_grid, axis=2)
current_sigma_pwv = jnp.std(pwv_fluctuation_plane)

pwv_norm = target_sigma_pwv / current_sigma_pwv
wv_box.field = wv_box.field * pwv_norm

total_wv_3d = wv_box.field + mean_interp_1d.reshape(1, 1, -1)

# ==============================================================================
# 5. Extract Single Boresight TOD (Zenith)
# ==============================================================================
print("Extracting Boresight Stare...")
# Center the telescope exactly in the middle of the 40km x 40km box
center_x = grid_wsp.Lbox[0] / 2.0
center_y = grid_wsp.Lbox[1] / 2.0
boresight = jnp.array([center_x, center_y, grid_wsp.site_altitude])

dummy_passband = {'freq_GHz': jnp.array([150.0]), 'g_nu': jnp.array([1.0])}

obs = Observer(
    grid_wsp=grid_wsp,
    boresight=boresight,
    passband=dummy_passband,
    fwhm_arcmin=1.0
)
obs.east_wind = vx   
obs.north_wind = vy 

obs.compute_los_for_scan(timelist, azimuth_deg, elevation_deg)
pristine_q_tod = obs.scan_component(total_wv_3d)

# Because elevation is 90 degrees, the path length is simply dz
pwv_layers_tod = (pristine_q_tod * q2rho_interp_1d) * dz

# Sum across all altitude layers to get the absolute total 1D time stream
total_pwv_tod = jnp.sum(pwv_layers_tod, axis=1)

# Isolate just the fluctuations (subtract the mean)
fluctuation_tod = total_pwv_tod - jnp.mean(total_pwv_tod)

print("Extraction Complete!")


# 1. Update global parameters for presentation visibility
plt.rcParams.update({
    'font.size': 18,              # General font size for text
    'axes.labelsize': 20,         # Size of X and Y axis labels
    'axes.titlesize': 24,         # Size of the graph title
    'xtick.labelsize': 16,        # Size of the numbers on the X axis
    'ytick.labelsize': 16,        # Size of the numbers on the Y axis
    'legend.fontsize': 16,        # Size of the legend text
    'lines.linewidth': 3,       # Thicker lines for the plotted data
})

# ==============================================================================
# 6. Plot the Time-Ordered Data
# ==============================================================================
time_axis_min = np.arange(n_steps) * dt_simulation / 60.0
output_dir = Path('/pscratch/sd/v/valer/atmo3/notebooks/plots_for_publication_valer/actual_plots')

fig1, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(time_axis_min, total_pwv_tod, color='darkblue', linewidth=2.5)
ax1.set_title(f'Total PWV Zenith scan with a wind of {vx} m/s')
ax1.set_xlabel('Observation Time (Minutes)')
ax1.set_ylabel('Total PWV (mm)')
ax1.grid(True, linestyle='--', alpha=0.6)
fig1.tight_layout()
fig1.savefig(output_dir / 'absolute_total_pwv.png', dpi=300, bbox_inches='tight')
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(14, 5))
ax2.plot(time_axis_min, fluctuation_tod, color='teal', linewidth=2.5)
ax2.set_title(f'PWV Fluctuations Zenith scan with a wind of {vx} m/s')
ax2.set_xlabel('Observation Time (Minutes)')
ax2.set_ylabel('PWV Fluctuation (mm)')
ax2.grid(True, linestyle='--', alpha=0.6)
fig2.tight_layout()
fig2.savefig(output_dir / 'pwv_fluctuations.png', dpi=300, bbox_inches='tight')
plt.close(fig2)