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

# ==============================================================================
# 1. Define the Geometric Workspace
# ==============================================================================
# Calculate the vertical grid dimensions
Lz = 9750.0  # Total height of the box in meters
dz = 30.0    # Vertical step size in meters
Nz = int(Lz / dz)  # Yields exactly 325 cells

L_0_q = 500 #m


dt_simulation = 2.0  # in seconds
vx = 5.0  # in m/s the wind speed along the x-axis

# Update your N and Lbox arrays (assuming you keep the X and Y axes the same),  we make Nz and Lz match the lidar profiles
N = jnp.array([2048, 256, Nz])                 # [256, 256, 325]
Lbox = jnp.array([40000.0, 20000.0, Lz])      # [20000.0, 20000.0, 9750.0]
site_altitude = 2400.0                        # Your new site altitude

# Initialize the standalone grid workspace
grid_wsp = gutl.GridWorkspace(N=N, Lbox=Lbox, site_altitude=site_altitude)

# Extract the exact 1D altitude axis 
# This will automatically start at 2400m and go up to 12150m (2400 + 9750) with 30m steps
z_axis_grid = grid_wsp.grid_axis(axis=2, altitude_axis=True)

import numpy as np
from lidar_sigma_pwv import lidar_sigma_pwv

data_file = "/pscratch/sd/v/valer/atmo3/notebooks/lidar_noise/Simulation_Tenerife_lidarH2O_15cm_00deg_10s.txt"

zbin = grid_wsp.grid_axis(axis=2, altitude_axis=False) # m

sigma_pwv = lidar_sigma_pwv(
    zbin=zbin,
    dzbin=30.0,
    tsampling=10.0,
    data_file=data_file,
    elevation=90.0,
    telsize=0.15,
    reference_telsize=0.15,
    reference_tsampling=10.0,
)
data = np.loadtxt(data_file, comments=("#", ";"))


s_km = data[:, 0] #in km
z_km = data[:, 1] #in km
q = data[:, 2] # in kg/kg
sigma_wvmr_g_per_kg = data[:, 3] # in g/kg
bias = data[:, 4] # in g/kg
pressure_hpa = data[:, 5] #in hPa
temperature_k = data[:, 6] #in K




kx_max = (grid_wsp.N[0] / 2.0) * grid_wsp.dk[0]
ky_max = (grid_wsp.N[1] / 2.0) * grid_wsp.dk[1]
kz_max = (grid_wsp.N[2] / 2.0) * grid_wsp.dk[2]

# 2. Calculate the extreme corner of the 3D Fourier volume
k_max_3d = np.sqrt(kx_max**2 + ky_max**2 + kz_max**2)

# 3. Create a high-resolution k_array using the smallest dk step
dk_min = np.min(grid_wsp.dk)
n_points = int(k_max_3d / dk_min) + 10  # +10 acts as a safety buffer

k_array = np.arange(n_points) * dk_min

k0_q  = 2*np.pi / L_0_q   # Water-vapour injection wavenumber (rad/m)

pofk_q  = ( k0_q**2.  + k_array**2 )**-(11/6)

# Normalise to peak = 1; absolute RMS amplitudes come from calibration.
pofk_q  /= np.max(pofk_q)

# Pack into dictionaries expected
pspec_q  = {'k': k_array, 'pofk': pofk_q}

# Next, define the fluctuation scaling profile. 
q_std_atmosphere = q

import numpy as np
import jax.numpy as jnp
import jax.random as random
from scipy.signal import csd
from atmo3.observation import Observer
import datetime

# ==============================================================================
# 1. Prepare Pre-Loop Constants & Observation Timelines
# ==============================================================================
N_sim_cube = 11      
y_boresight_array = jnp.array([6000.0, 7000.0, 8000.0, 9000.0, 10000.0, 11000.0, 12000.0, 13000.0, 14000.0]) 
N_sim_boresight = len(y_boresight_array)
total_sims = N_sim_cube * N_sim_boresight

#define y boresight from 6000m to 14000m with 1000m steps




# Signal processing params
n_window = 360
dt_lidar = 10.0
fs_lidar = 1.0 / dt_lidar
frames_per_lidar = int(dt_lidar / dt_simulation)

# Pre-calculate thermodynamic conversion factors once to save compute inside the loop
T_v = au.virtual_temperature(temperature_k, q) 
q2rho_h2o = pressure_hpa * 100.0 / (const.R_dry_air * T_v) 
rho_wv_mean_1d = q * q2rho_h2o
mean_pwv = jnp.trapezoid(rho_wv_mean_1d, x=z_km * 1000)
target_sigma_pwv = 0.01 * mean_pwv

# Interpolate onto the exact internal Z-grid midpoints
q2rho_interp_1d = jnp.interp(z_axis_grid, z_km * 1000, q2rho_h2o, left='extrapolate', right='extrapolate')
mean_interp_1d = jnp.interp(z_axis_grid, z_km * 1000, q, left='extrapolate', right='extrapolate')

# Setup 1-hour timeline
total_duration = 3600 
n_steps = int(total_duration / dt_simulation)
start_time = np.datetime64('2026-01-01T00:00:00')
timelist = [start_time + np.timedelta64(int(i * dt_simulation), 's') for i in range(n_steps)]
azimuth_deg = [0.0] * n_steps
elevation_deg = [90.0] * n_steps

# Dummy passband for Observer initialization
dummy_passband = {'freq_GHz': jnp.array([150.0]), 'g_nu': jnp.array([1.0])}

# ==============================================================================
# 2. Initialize Master Time-Domain Arrays (NumPy to protect VRAM)
# ==============================================================================
# Calculate exact number of Lidar time samples (e.g., 3600s / 10s = 360 samples)
n_lidar_samples = n_steps // frames_per_lidar

# Shape: (Total_Simulations, Nz, Time_Steps)
master_s_i = np.zeros((total_sims, Nz, n_lidar_samples))  # True Atmosphere Signal
master_x_i = np.zeros((total_sims, Nz, n_lidar_samples))  # Noisy Lidar Observation
master_n_i = np.zeros((total_sims, Nz, n_lidar_samples))  # Pure Lidar Noise

master_key = random.PRNGKey(42)
sim_counter = 0

print(f"Starting TOD Generation: {N_sim_cube} Cubes x {N_sim_boresight} Lanes = {total_sims} Total Runs")
print(f"Array Shape: ({total_sims} runs, {Nz} layers, {n_lidar_samples} time steps)")

# ==============================================================================
# 3. The Time-Ordered Data Generation Loop
# ==============================================================================
for cube_idx in range(N_sim_cube):
    print(f"\n--- Generating Atmospheric Cube {cube_idx + 1}/{N_sim_cube} ---")
    
    master_key, atmo_key = random.split(master_key)
    cube_seed = int(random.randint(atmo_key, shape=(), minval=0, maxval=99999999))
    
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
    
    # --------------------------------------------------------------------------
    # 4. The Boresight Extraction Loop
    # --------------------------------------------------------------------------
    for bore_idx, y_pos in enumerate(y_boresight_array):
        
        # A. Setup Observer
        boresight = jnp.array([grid_wsp.Lbox[0] / 2.0, y_pos, grid_wsp.site_altitude])
        obs = Observer(
            grid_wsp=grid_wsp,
            boresight=boresight,
            passband=dummy_passband,
            fwhm_arcmin=1.0
        )
        obs.east_wind = vx   
        obs.north_wind = 0.0 
        
        # B. Extract High-Res TOD
        obs.compute_los_for_scan(timelist, azimuth_deg, elevation_deg)
        pristine_q_tod = obs.scan_component(total_wv_3d)
        
        # C. Unit Conversion (kg/m^2)
        true_signal_2s_2d = (pristine_q_tod * q2rho_interp_1d) * dz
        
        # D. Downsample to 10s Lidar Rate
        valid_frames = n_lidar_samples * frames_per_lidar
        true_10s_blocks = true_signal_2s_2d[:valid_frames, :].reshape(n_lidar_samples, frames_per_lidar, Nz)
        true_signal_1d_2d = jnp.mean(true_10s_blocks, axis=1) # Shape: (Time_Steps, Nz)
        
        # E. Mean Removal (DC Baseline Subtraction)
        true_signal_1d_2d -= jnp.mean(true_signal_1d_2d, axis=0)
        
        # F. Noise Generation
        master_key, noise_key = random.split(master_key)
        current_noise_2d = random.normal(noise_key, shape=true_signal_1d_2d.shape) * sigma_pwv
        current_noise_2d -= jnp.mean(current_noise_2d, axis=0)

        # G. The Observable Signal
        noisy_signal_1d_2d = true_signal_1d_2d + current_noise_2d
        
        # ----------------------------------------------------------------------
        # H. Store the Time-Ordered Data (Transpose to match master array shape)
        # ----------------------------------------------------------------------
        # JAX arrays are (Time_Steps, Nz). Transpose to (Nz, Time_Steps).
        master_s_i[sim_counter, :, :] = np.array(true_signal_1d_2d).T
        master_x_i[sim_counter, :, :] = np.array(noisy_signal_1d_2d).T
        master_n_i[sim_counter, :, :] = np.array(current_noise_2d).T
        
        sim_counter += 1
        print(f"  Extracted Lane {bore_idx+1}/{N_sim_boresight} | Total Runs: {sim_counter}/{total_sims}")

print("\nAll Time-Ordered Data successfully extracted and stored!")


import os

# ==============================================================================
# 5. Save the Dataset to Perlmutter Scratch
# ==============================================================================
# Define your save directory (using your existing pscratch path)
save_dir = "/pscratch/sd/v/valer/atmo3/notebooks/lidar_data/"
os.makedirs(save_dir, exist_ok=True) # Creates the folder if it doesn't exist

file_path = os.path.join(save_dir, f"lidar_tod_ensemble_{total_sims}_runs_wind_{vx}.npz")

print(f"\nCompressing and saving dataset to: {file_path}")

# savez_compressed shrinks the file size significantly without losing precision
np.savez_compressed(
    file_path,
    true_signal=master_s_i,
    noisy_signal=master_x_i,
    pure_noise=master_n_i,
    altitude_axis=np.array(z_axis_grid), # Save the axes so you don't have to guess later!
    y_boresights=np.array(y_boresight_array)
)

print("Save complete! You can now close this script.")