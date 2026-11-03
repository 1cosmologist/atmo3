import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_platforms', 'cpu')
# jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, CenteredNorm
from pathlib import Path
import atmo3 as a3

# Grid dimensions [Nx, Ny, Nz] and physical box size [Lx, Ly, Lz] in metres.
# The horizontal resolution is Lx/Nx ≈ 39 m; the vertical is Lz/Nz ≈ 39 m.
nside_grid = [512, 512, 256]
box_length = [20000., 20000., 10000.]

boresight  = jnp.array([box_length[0]//2., box_length[1]//2.])
freqs      = jnp.logspace(1, 3, num=1000)
passband   = {'nu': jnp.array([150.]), 'g(nu)': jnp.array([1.])}
fwhm       = 10.

injection_scale_in_m = 500.0  # Injection scale in meters

# Simons Observatory site in Parque Astronomico de Atacama (PAA). 
# Details from: https://simonsobservatory.org/instrumentation/site-and-advanced-simons-observatory/
site_altitude = 5190. # m
site_coordinates = [-67.783, -22.95]  # [longitude, latitude] in degrees

# Simulation epoch in UTC.  ERA5 profiles and APEX weather data are
# selected from a ±30-minute window centred on this timestamp.
time_utc = datetime(2025, 9, 16, 0, 0, tzinfo=timezone.utc)

# =============================================================================
# Scan strategy
# =============================================================================

# dt = 0 case
timesamples = [np.datetime64(time_utc)]
az          = [0.]      # in deg
el          = [55.]     # in deg

# remaining 300 samples
for sample in range(360-1):
    timesamples.append(timesamples[-1] + np.timedelta64(1000, 'ms'))
    az.append(az[-1] + 1.)
    el.append(el[-1]) 

atmo_box = a3.Atmosphere(
    nside_grid=nside_grid,
    box_length_in_m=box_length,
    site_altitude=site_altitude,
    site_coordinates=site_coordinates,
    time_utc=time_utc,
    profiles='midlatitude_winter',
    pwv_mean=1.,        # mm
    pwv_std=0.01,       # mm 
    temperature_std=0.2, # K
)

print(f"Creating power spectrum for specific humidity normalized to 1.")
k_array     = np.arange(max(nside_grid)) * jnp.min(atmo_box.grid_wsp.dk)
# Based on the Kolmogorov spectrum, we can define a power spectrum based on values from Morris et al. (2025) (arxiv:2410.13064)
k0          = 2*np.pi / injection_scale_in_m
pofk_array  = ( k0**2. + k_array**2 )**-(11/6)
# pofk_array  /= np.trapezoid(pofk_array, k_array)  # Normalize the power spectrum
pofk_array /= np.max(pofk_array)  # Normalize the power spectrum to 1
pspec = {'k': k_array, 'pofk': pofk_array}

atmo_box.add_watervapor(pspec)
atmo_box.add_temperature(pspec)

obs = a3.Observer(
    grid_wsp = atmo_box.grid_wsp,
    boresight = jnp.array(boresight),       # in grid coordinates
    passband = passband, 
    fwhm_arcmin = fwhm
)

obs.compute_los_for_scan(timesamples, az, el)

atmo_box.generate_realization(time_step=0)

rho_wv_scan = obs.scan_atmosphere(atmo_box, component_name="water vapor")
# T_atm_scan = obs.scan_atmosphere(atmo_box, component_name="temperature")

drho_wv_scan = obs.scan_atmosphere_fluctuation(atmo_box, component_name="water vapor")
dT_atm_scan = obs.scan_atmosphere_fluctuation(atmo_box, component_name="temperature")

pwv_scan   = jnp.trapezoid(rho_wv_scan, x=obs.los_obj[:,:,3], axis=1)

drho_wv_scan_np = np.asarray(drho_wv_scan)
dT_atm_scan_np = np.asarray(dT_atm_scan)
time_sample = np.arange(len(timesamples))
los_sample = np.arange(drho_wv_scan_np.shape[1])

fig, ax = plt.subplots(dpi=200)
image = ax.imshow(
    drho_wv_scan_np.T,
    origin='lower',
    aspect='auto',
    cmap='coolwarm',
    # norm=LogNorm(),
    norm=CenteredNorm(halfrange=1e-5),
    extent=(-0.5, len(time_sample) - 0.5, -0.5, len(los_sample) - 0.5),
)
ax.set(xlabel='Time sample', ylabel='Line-of-sight sample', title='Water-vapor density fluctuations')
fig.colorbar(image, ax=ax, label=r'$\Delta\rho_{\mathrm{wv}}$ (kg m$^{-3}$)')
fig.tight_layout()
fig.savefig(Path(__file__).resolve().with_name('drho_wv_scan.png'))
plt.close(fig)

fig, ax = plt.subplots(dpi=200)
image = ax.imshow(
    dT_atm_scan_np.T,
    origin='lower',
    aspect='auto',
    cmap='coolwarm',
    # norm=LogNorm(),
    norm=CenteredNorm(halfrange=0.5),
    extent=(-0.5, len(time_sample) - 0.5, -0.5, len(los_sample) - 0.5),
)
ax.set(xlabel='Time sample', ylabel='Line-of-sight sample', title='Atmospheric temperature fluctuations')
fig.colorbar(image, ax=ax, label=r'$\Delta T_{\mathrm{atm}}$ (K)')
fig.tight_layout()
fig.savefig(Path(__file__).resolve().with_name('dT_atm_scan.png'))
plt.close(fig)

fig, ax = plt.subplots()
ax.plot(time_sample, np.asarray(pwv_scan))
ax.set(xlabel='Time sample', ylabel='PWV (mm)', title='Precipitable water vapor along scans')
# ax.grid(True)
fig.tight_layout()
fig.savefig(Path(__file__).resolve().with_name('pwv_scan.png'))
plt.close(fig)

T_b_obs = obs.observe_atmosphere(atmo_box)


fig, ax = plt.subplots()
ax.plot(time_sample, np.asarray(T_b_obs))
ax.set(xlabel='Time sample', ylabel=r'$T_b$ (K_RJ)', title='Observed brightness temperature')
# ax.grid(True)
fig.tight_layout()
fig.savefig(Path(__file__).resolve().with_name('T_b_obs.png'))
plt.close(fig)
