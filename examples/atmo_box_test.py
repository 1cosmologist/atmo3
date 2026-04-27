# =============================================================================
# JAX configuration
# Force JAX to run on CPU and enable 64-bit floating-point precision.
# 64-bit is required for accurate FFT phases and numerical integration.
# =============================================================================
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_platforms', 'cpu')
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from datetime import datetime, timezone
import matplotlib.pyplot as plt
import cmocean as cmo
#import pyrtlib as rtl

import atmo3 as a3

# =============================================================================
# Simulation parameters
# =============================================================================

# Turbulence injection scales: the wavenumber k0 = 2π/L_inj marks the peak
# of the power spectrum, i.e. the scale at which turbulent energy is injected.
q_injection_scale_in_m  = 500.0  # Water-vapour injection scale (m)
ta_injection_scale_in_m = 200.0  # Temperature injection scale (m)

# Grid dimensions [Nx, Ny, Nz] and physical box size [Lx, Ly, Lz] in metres.
# The horizontal resolution is Lx/Nx ≈ 39 m; the vertical is Lz/Nz ≈ 39 m.
nside_grid = [512, 512, 256]
box_length = [20000., 20000., 10000.]

# APEX observatory: altitude 5100 m a.s.l., Llano de Chajnantor, Chile.
site_altitude = 5100.
site_coordinates = [-67.78, -22.95]  # [longitude, latitude] in degrees

# Simulation epoch in UTC.  ERA5 profiles and APEX weather data are
# selected from a ±30-minute window centred on this timestamp.
time_utc = datetime(2023, 9, 16, 0, 0, tzinfo=timezone.utc)

# =============================================================================
# Input data paths
# =============================================================================

atmo3_data = '/pscratch/sd/s/shamikg/atmo3_data/'

# ERA5 pressure-level files covering the APEX region (291–293 °E, 24–22 °S)
# at 0.25° resolution for September 2023.
geopotfile = f'{atmo3_data}era5/2023/geopt.202309.ap1e5.291.0_293.0_-24.0_-22.0_025.nc'
tempfile   = f'{atmo3_data}era5/2023/ta.202309.ap1e5.291.0_293.0_-24.0_-22.0_025.nc'
spechfile  = f'{atmo3_data}era5/2023/q.202309.ap1e5.291.0_293.0_-24.0_-22.0_025.nc'

# APEX weather-station CSV (columns: UT, PWV, Temperature, Humidity,
# Wind_Dir, Wind_Speed) spanning 2006–2025.
apexfile = f'{atmo3_data}apex/meteo_apex_2006_2025.csv'

# =============================================================================
# Initialise the atmosphere object
# ERA5 profiles are read and interpolated to the site location; profiles are
# then normalised so that the ground-level temperature and column-integrated
# PWV match the mean APEX observations recorded within ±30 min of time_utc.
# =============================================================================
atmo_box = a3.Atmosphere(
    nside_grid=nside_grid,
    box_length_in_m=box_length,
    site_altitude=site_altitude,
    site_coordinates=site_coordinates,
    time_utc=time_utc,
    geopotential_file_era5=geopotfile,
    temperature_file_era5=tempfile,
    spec_humidity_file_era5=spechfile,
    apex_datafile=apexfile
)

# =============================================================================
# Define isotropic power spectra (Kolmogorov-like)
# Following Morris et al. 2025 (arXiv:2410.13064), the spectrum takes the
# von Kármán form:
#
#   P(k) ∝ (k0² + k²)^{-11/6}
#
# where k0 = 2π / L_inj is the injection wavenumber.  At k >> k0 this
# reduces to the Kolmogorov inertial-range scaling k^{-11/3} (in 3D).
# The spectrum is normalised to unity at its peak (k = 0) so that the
# absolute amplitude is set entirely by the ERA5/APEX calibration.
# =============================================================================

# Wavenumber array: sample at the minimum grid spacing up to 4× Nyquist
# to ensure the interpolation onto the 3-D k-grid is well resolved.
k_array = np.arange(4*nside_grid[0]) * jnp.min(atmo_box.grid_wsp.dk)

k0_q  = 2*np.pi / q_injection_scale_in_m   # Water-vapour injection wavenumber (rad/m)
k0_ta = 2*np.pi / ta_injection_scale_in_m  # Temperature injection wavenumber (rad/m)

pofk_q  = ( k0_q**2.  + k_array**2 )**-(11/6)
pofk_ta = ( k0_ta**2. + k_array**2 )**-(11/6)

# Normalise to peak = 1; absolute RMS amplitudes come from calibration.
pofk_q  /= np.max(pofk_q)
pofk_ta /= np.max(pofk_ta)

# Pack into dictionaries expected by add_temperature / add_watervapor.
pspec_q  = {'k': k_array, 'pofk': pofk_q}
pspec_ta = {'k': k_array, 'pofk': pofk_ta}

# =============================================================================
# Register atmospheric components
# Each call attaches an ERA5-calibrated height-dependent scaling profile and
# a mean profile derived from the normalised ERA5 data.
# =============================================================================

# Temperature fluctuation field (units: K).
# Scaling profile: ERA5 horizontal-variance std, rescaled to APEX σ_T.
atmo_box.add_temperature(
    power_spec=pspec_ta
)

# Water-vapour mass-density fluctuation field (units: kg/m³).
# Scaling profile: 0.1 % of ERA5 specific-humidity mean × q→ρ conversion.
atmo_box.add_watervapor(
    power_spec=pspec_q
)

# =============================================================================
# Generate a single random realisation (time_step=0)
# Draws Gaussian random numbers in Fourier space, applies the transfer
# function P(k)^{1/2}, transforms to real space, and rescales by the
# height-dependent profile.  For the water-vapour component, the field is
# additionally rescaled so that the sky-plane PWV standard deviation matches
# the APEX observation.
# =============================================================================
atmo_box.generate_realization(time_step=0)

# =============================================================================
# Visualisation
# =============================================================================

# --- Water-vapour total density: y-z cross-section at x = 0 ----------------
# Total density = fluctuation + mean profile (reshaped to broadcast over x, y).
plt.figure(dpi=200)
plt.imshow(
    (atmo_box.components['water vapor'].field + atmo_box.component_mean['water vapor']['f'].reshape(1, 1, -1))[0, :, :].T * 1e3,
    extent=(0, box_length[1], site_altitude, site_altitude + box_length[2]),
    cmap=cmo.cm.ice_r, origin='lower', vmin=-3*atmo_box.components['water vapor'].field.std() *1e3,
)
plt.colorbar(label='g/m^3', orientation='horizontal')
plt.title(f"Water vapor density y-z plane at x=0 m, {time_utc:%Y-%m-%dT%H:%M}")
plt.xlabel('y (m)')
plt.ylabel('z (m)')
plt.savefig(f'./examples/water_vapor_density_yz_plane_{time_utc:%Y-%m-%dT%H:%M}.png', bbox_inches='tight')
plt.close()

# --- Water-vapour fluctuation only: y-z cross-section at x = 0 -------------
# Shows the zero-mean turbulent component; colour scale is ±3σ.
plt.figure(dpi=200)
plt.imshow(
    atmo_box.components['water vapor'].field[0, :, :].T * 1e3,
    extent=(0, box_length[1], site_altitude, site_altitude + box_length[2]),
    cmap=cmo.cm.balance, origin='lower', vmin=-3*atmo_box.components['water vapor'].field.std() *1e3, vmax=3*atmo_box.components['water vapor'].field.std() *1e3
)
plt.colorbar(label='g/m^3', orientation='horizontal')
plt.title(f"Water vapor fluctuation y-z plane at x=0 m, {time_utc:%Y-%m-%dT%H:%M}")
plt.xlabel('y (m)')
plt.ylabel('z (m)')
plt.savefig(f'./examples/water_vapor_fluctuation_yz_plane_{time_utc:%Y-%m-%dT%H:%M}.png', bbox_inches='tight')
plt.close()

# --- Temperature total field: y-z cross-section at x = 0 -------------------
plt.figure(dpi=200)
plt.imshow(
    (atmo_box.components['temperature'].field + atmo_box.component_mean['temperature']['f'].reshape(1, 1, -1))[0, :, :].T,
    extent=(0, box_length[1], site_altitude, site_altitude + box_length[2]),
    cmap=cmo.cm.thermal, origin='lower',
)
plt.colorbar(label='K', orientation='horizontal')
plt.title(f"Temperature y-z plane at x=0 m, {time_utc:%Y-%m-%dT%H:%M}")
plt.xlabel('y (m)')
plt.ylabel('z (m)')
plt.savefig(f'./examples/temperature_yz_plane_{time_utc:%Y-%m-%dT%H:%M}.png', bbox_inches='tight')
plt.close()

# --- Temperature fluctuation only: y-z cross-section at x = 0 --------------
plt.figure(dpi=200)
plt.imshow(
    atmo_box.components['temperature'].field[0, :, :].T,
    extent=(0, box_length[1], site_altitude, site_altitude + box_length[2]),
    cmap=cmo.cm.balance, origin='lower',
)
plt.colorbar(label='K', orientation='horizontal')
plt.title(f"Temperature fluctuation y-z plane at x=0 m, {time_utc:%Y-%m-%dT%H:%M}")
plt.xlabel('y (m)')
plt.ylabel('z (m)')
plt.savefig(f'./examples/temperature_fluctuations_yz_plane_{time_utc:%Y-%m-%dT%H:%M}.png', bbox_inches='tight')
plt.close()

# --- Precipitable water vapour (PWV) sky map --------------------------------
# PWV is the vertical integral of the total water-vapour mass density (kg/m³)
# over the box height, yielding a column amount in kg/m² = mm of liquid water.
# The integral is performed via the trapezoidal rule along the altitude axis.
pwv_plane = jnp.trapezoid(
    atmo_box.components['water vapor'].field + atmo_box.component_mean['water vapor']['f'].reshape(1, 1, -1),
    x=atmo_box.grid_wsp.grid_axis(axis=2, altitude_axis=True),
    axis=2
)
print(f"PWV mean: {np.mean(pwv_plane):.4f} mm, PWV std: {np.std(pwv_plane):.4f} mm")

plt.figure(dpi=200)
plt.imshow(
    pwv_plane.T,
    extent=(0, box_length[0], 0, box_length[1]),
    cmap=cmo.cm.algae, origin='lower', vmin=pwv_plane.mean()-3*pwv_plane.std(), vmax=pwv_plane.mean()+3*pwv_plane.std()
)
plt.colorbar(label='mm', shrink=0.8)
plt.title(f"Precipitable water vapor field at z={site_altitude} m, UTC hour {time_utc:%Y-%m-%dT%H:%M}")
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.savefig(f'./examples/precipitable_water_vapor_xy_plane_UTC{time_utc:%Y-%m-%dT%H:%M}.png', bbox_inches='tight')
plt.close()


