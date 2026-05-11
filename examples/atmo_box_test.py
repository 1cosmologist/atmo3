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
import time

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
nside_grid = [2048, 2048, 1024]
box_length = [20000., 20000., 10000.]

boresight  = jnp.array([box_length[0]//2., box_length[1]//2.])
passband   = {'nu': jnp.array([150.]), 'g(nu)': jnp.array([1.])}
fwhm       = 10.

# APEX observatory: altitude 5100 m a.s.l., Llano de Chajnantor, Chile.
site_altitude = 5100.
site_coordinates = [-67.78, -22.95]  # [longitude, latitude] in degrees

# Simulation epoch in UTC.  ERA5 profiles and APEX weather data are
# selected from a ±30-minute window centred on this timestamp.
time_utc = datetime(2023, 9, 16, 0, 0, tzinfo=timezone.utc)

t0 = time.perf_counter()
# =============================================================================
# Scan strategy
# =============================================================================

# dt = 0 case
timesamples = [np.datetime64(time_utc)]
az          = [0.]      # in deg
el          = [45.]     # in deg

# remaining 300 samples
for sample in range(300-1):
    timesamples.append(timesamples[-1] + np.timedelta64(200, 'ms'))
    az.append(az[-1] + 0.2)
    el.append(el[-1]) 
    
t1 = time.perf_counter()

print(f"Scan setup time : {t1 - t0:.6f} s")
# =============================================================================
# Input data paths
# =============================================================================

atmo3_data = '/pscratch/sd/s/shamikg/atmo3_data/'

# ERA5 pressure-level files covering the APEX region (291–293 °E, 24–22 °S)
# at 0.25° resolution for September 2023.
geopotfile = f'{atmo3_data}era5/2023/geopt.202309.ap1e5.291.0_293.0_-24.0_-22.0_025.nc'
tempfile   = f'{atmo3_data}era5/2023/ta.202309.ap1e5.291.0_293.0_-24.0_-22.0_025.nc'
spechfile  = f'{atmo3_data}era5/2023/q.202309.ap1e5.291.0_293.0_-24.0_-22.0_025.nc'

northwindfile = f'{atmo3_data}era5/2023/v.202309.ap1e5.291.0_293.0_-24.0_-22.0_025.nc'      # V-component North wind
eastwindfile  = f'{atmo3_data}era5/2023/u.202309.ap1e5.291.0_293.0_-24.0_-22.0_025.nc'      # U-component East wind

# APEX weather-station CSV (columns: UT, PWV, Temperature, Humidity,
# Wind_Dir, Wind_Speed) spanning 2006–2025.
apexfile = f'{atmo3_data}apex/meteo_apex_2006_2025.csv'

t0 = time.perf_counter()
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

t1 = time.perf_counter()
print(f"Atmosphere class init : {t1 - t0:.6f} s")

obs = a3.Observer(
    grid_wsp = atmo_box.grid_wsp,
    super_grid = atmo_box.super_grid,
    northwind_era5_file = None,
    eastwind_era5_file = None,
    boresight = jnp.array(boresight),       # in grid coordinates
    passband = passband, 
    fwhm_arcmin = fwhm
)

obs.compute_los_for_scan(timesamples, az, el)

t2 = time.perf_counter()
print(f"Observer init LoS compute time : {t1 - t0:.6f} s")

obs_w = a3.Observer(
    grid_wsp = atmo_box.grid_wsp,
    super_grid = atmo_box.super_grid,
    northwind_era5_file = northwindfile,
    eastwind_era5_file = eastwindfile,
    boresight = jnp.array(boresight),       # in grid coordinates
    passband = passband, 
    fwhm_arcmin = fwhm
)

obs_w.compute_los_for_scan(timesamples, az, el)

# print(np.array(obs.los_obj).shape)
# exit()

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
t0 = time.perf_counter()
atmo_box.add_temperature(
    power_spec=pspec_ta
)
t1 = time.perf_counter()
print(f"Temperature box init: {t1 - t0:.6f} s")
# Water-vapour mass-density fluctuation field (units: kg/m³).
# Scaling profile: 0.1 % of ERA5 specific-humidity mean × q→ρ conversion.
atmo_box.add_watervapor(
    power_spec=pspec_q
)
t2 =time.perf_counter()
print(f"Water box init : {t2 - t1:.6f} s")
# =============================================================================
# Generate a single random realisation (time_step=0)
# Draws Gaussian random numbers in Fourier space, applies the transfer
# function P(k)^{1/2}, transforms to real space, and rescales by the
# height-dependent profile.  For the water-vapour component, the field is
# additionally rescaled so that the sky-plane PWV standard deviation matches
# the APEX observation.
# =============================================================================
t0 = time.perf_counter()
atmo_box.generate_realization(time_step=0)
t1 = time.perf_counter()
print(f"Generate realization : {t1 - t0:.6f} s")

# ============================================================================
# Observation
# ============================================================================

t0 = time.perf_counter()
water_vapor_los = obs.scan_component(atmo_box.components['water vapor'].field + atmo_box.component_mean['water vapor']['f'].reshape(1, 1, -1))
t1 = time.perf_counter()
print(f"Scanning time : {t1 - t0:.6f} s")

temperature_los = obs.scan_component(atmo_box.components['temperature'].field + atmo_box.component_mean['temperature']['f'].reshape(1, 1, -1))

water_vapor_los_w = obs_w.scan_component(atmo_box.components['water vapor'].field + atmo_box.component_mean['water vapor']['f'].reshape(1, 1, -1))
temperature_los_w = obs_w.scan_component(atmo_box.components['temperature'].field + atmo_box.component_mean['temperature']['f'].reshape(1, 1, -1))

pwv_scan   = jnp.trapezoid(water_vapor_los,   x=obs.los_obj[:,:,3], axis=1)
pwv_scan_w = jnp.trapezoid(water_vapor_los_w, x=obs.los_obj[:,:,3], axis=1)
# =============================================================================
# Visualisation
# =============================================================================
plt.figure(dpi=200)
plt.plot(water_vapor_los[0], obs.los_obj[0,:,3], label='los')
plt.plot(water_vapor_los[0], obs_w.los_obj[0,:,3], label='los')
plt.plot((atmo_box.components['water vapor'].field + atmo_box.component_mean['water vapor']['f'].reshape(1, 1, -1))[nside_grid[0]//2, nside_grid[1]//2], atmo_box.grid_wsp.grid_axis(axis=2), label='zenith')
plt.yscale('log')
plt.xlabel(r"water vapor density (kg/m${}^3$)")
plt.ylabel("path length (m)")
plt.legend(frameon=False)
plt.savefig(f'./examples/water_vapor_profile-los_{time_utc:%Y-%m-%dT%H:%M}.png', bbox_inches='tight')
plt.close()

plt.figure(dpi=200)
plt.plot(timesamples, pwv_scan,   '-', lw=0.6, label='No wind')
plt.plot(timesamples, pwv_scan_w, '-', lw=0.6, label='Diff wind')
# plt.axhline(y=atmo_box.atm_calibrator.apex_pwv_mean)
plt.xlabel("time samples (hh:mm:ss)")
plt.ylabel("PWV (mm)")
plt.legend(frameon=False)
plt.savefig(f'./examples/pwv_scan_{time_utc:%Y-%m-%dT%H:%M}.png', bbox_inches='tight')
plt.close()

pressure_los = jnp.interp(obs.los_obj[:,:,2], obs.axes[2], atmo_box.atm_calibrator.pressure) 
pressure_los_w = jnp.interp(obs_w.los_obj[:,:,2], obs_w.axes[2], atmo_box.atm_calibrator.pressure)
freqs_GHz = jnp.array([150., 185.])

I_atm = a3.get_emission(temperature_los, pressure_los, water_vapor_los, jnp.diff(obs.los_obj[:,:,3], axis=1, prepend=0), freqs_GHz)
T_atm = a3.intensitySI_to_Tb(I_atm[:,:,None], freqs_GHz[None, None,:])[:,:,0]

I_atm_w = a3.get_emission(temperature_los_w, pressure_los, water_vapor_los_w, jnp.diff(obs_w.los_obj[:,:,3], axis=1, prepend=0), freqs_GHz)
T_atm_w = a3.intensitySI_to_Tb(I_atm_w[:,:,None], freqs_GHz[None, None,:])[:,:,0]

# print(I_atm.shape)

plt.figure(dpi=200)
plt.plot(timesamples, T_atm[:,0],   '-', lw=0.6, label=f'No wind {freqs_GHz[0]}')
plt.plot(timesamples, T_atm_w[:,0], '-', lw=0.6, label=f'Diff wind {freqs_GHz[0]}')
# plt.axhline(y=atmo_box.atm_calibrator.apex_pwv_mean)
plt.xlabel("time samples (hh:mm:ss)")
plt.ylabel(r"$T_b$ (K)")
plt.legend(frameon=False)
plt.savefig(f'./examples/brightness_temp_{freqs_GHz[0]}_{time_utc:%Y-%m-%dT%H:%M}.png', bbox_inches='tight')
plt.close()

plt.figure(dpi=200)
plt.plot(timesamples, T_atm[:,1],   '-', lw=0.6, label=f'No wind {freqs_GHz[1]}')
plt.plot(timesamples, T_atm_w[:,1], '-', lw=0.6, label=f'Diff wind {freqs_GHz[1]}')
# plt.axhline(y=atmo_box.atm_calibrator.apex_pwv_mean)
plt.xlabel("time samples (hh:mm:ss)")
plt.ylabel(r"$T_b$ (K)")
plt.legend(frameon=False)
plt.savefig(f'./examples/brightness_temp_{freqs_GHz[1]}_{time_utc:%Y-%m-%dT%H:%M}.png', bbox_inches='tight')
plt.close()

plt.figure(dpi=200)
plt.plot(timesamples, T_atm[:,0] - T_atm[:,0].mean(),   '-', lw=0.6, label=f'No wind {freqs_GHz[0]}')
plt.plot(timesamples, T_atm_w[:,0] - T_atm_w[:,0].mean(), '-', lw=0.6, label=f'Diff wind {freqs_GHz[0]}')
plt.plot(timesamples, T_atm[:,1] - T_atm[:,1].mean(),   '-', lw=0.6, label=f'No wind {freqs_GHz[1]}')
plt.plot(timesamples, T_atm_w[:,1] - T_atm_w[:,1].mean(), '-', lw=0.6, label=f'Diff wind {freqs_GHz[1]}')
# plt.axhline(y=atmo_box.atm_calibrator.apex_pwv_mean)
plt.xlabel("time samples (hh:mm:ss)")
plt.ylabel(r"$T_b$ (K)")
plt.legend(frameon=False)
plt.savefig(f'./examples/brightness_temp_fluctuations_{time_utc:%Y-%m-%dT%H:%M}.png', bbox_inches='tight')
plt.close()

plt.figure(dpi=200)
plt.plot(T_atm[:,0],   T_atm[:,1],   'o', ms=1.2, label=f'No wind')
plt.plot(T_atm_w[:,0], T_atm_w[:,1], 'o', ms=1.2, label=f'Diff wind')
plt.xlabel(r"$T_b$ (K) 150 GHz")
plt.ylabel(r"$T_b$ (K) 185 GHz")
plt.legend(frameon=False)
plt.savefig(f'./examples/brightness_temp_scatter{time_utc:%Y-%m-%dT%H:%M}.png', bbox_inches='tight')
plt.close()


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


