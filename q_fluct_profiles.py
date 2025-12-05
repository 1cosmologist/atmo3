# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 17:54:26 2025

@author: tollet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from datetime import datetime
import os
import xarray as xr
from scipy.interpolate import interp1d

# =============================================================================
# PARAMETERS
# =============================================================================
year = 2023
month = 7
day = 22
date_str = f"{year}-{month:02d}-{day:02d}"  # Specific day
month_str = f"{month:02d}-{year}"            # For file naming


# Altitude range for δq projection
alt_min = 5100 # in m
alt_max = 15000 # in m
cell_size = 2 # in m
alt_grid = np.arange(alt_min, alt_max + cell_size, cell_size)

# Physical constants
g = 9.80665      # gravity (m/s²)
R_d = 287.05     # Gas constant dry air (J/kg/K)
R_v = 461.5      # Gas constant water vapor (J/kg/K)

# File paths
era5_dir = Path().resolve() / "Data" / "era_5"/ "2023"
APEX_file = Path().resolve() / "Data" / "APEX" / "meteo_apex_2006_2025.csv"
save_path = Path().resolve() / "Data" / "Profiles"
q_profile = save_path / f"mean_q_profiles_{month_str}.npz"

# ERA5 data directory
month_label = f"{year}{month:02d}"

Save_delta_q = False  # Flag to save δq profiles

# =============================================================================
# Find ERA5 files
# =============================================================================
def find_file(var, month_label):
    """Return full path to the ERA5 variable file for the given month."""
    for f in os.listdir(era5_dir):
        if f.startswith(f"{var}.{month_label}") and f.endswith(".nc"):
            return os.path.join(era5_dir, f)
    raise FileNotFoundError(f"{var}.{month_label} not found.")


# =============================================================================
# 1. FUNCTION: Analyze PWV perturbations
# =============================================================================
def analyze_PWV_perturbations(df, date_str, smoothing_minutes=30, spike_threshold=3, plot=True):
    """Analyze daily PWV fluctuations (unchanged, see original code)."""
    date = pd.to_datetime(date_str)
    df_day = df[df['UT'].dt.date == date.date()].copy()
    if df_day.empty:
        print(f"No data available for {date_str}")
        return

    # Remove outliers
    pwv_mean = df_day['PWV'].mean()
    pwv_std = df_day['PWV'].std()
    df_day = df_day[np.abs(df_day['PWV'] - pwv_mean) < spike_threshold * pwv_std]

    valid_points = df_day['PWV'].notna().sum()
    total_points = len(df_day)
    if total_points == 0 or valid_points / total_points < 0.25:
        print(f"Insufficient valid data for {date_str}")
        return pd.Series(data=np.nan, index=df_day.set_index('UT').index)

    df_day = df_day.set_index('UT')

    baseline = df_day['PWV'].rolling(f'{smoothing_minutes}min', center=True).mean()
    fluctuations = df_day['PWV'] - baseline
    std_fluctuations = fluctuations.rolling(f'{smoothing_minutes}min', center=True).std()
    std_fluctuations[df_day['PWV'].isna()] = np.nan

    if plot:
        fig, axes = plt.subplots(4, 1, figsize=(6, 8), sharex=True)
        axes[0].plot(df_day.index, df_day['PWV'], label='PWV')
        axes[0].set_title(f'Raw PWV – {date_str}')
        axes[0].set_ylabel('PWV (mm)')
        axes[0].grid(True)
        axes[1].plot(df_day.index, baseline, label='Baseline', color='green')
        axes[1].set_ylabel('Baseline (mm)')
        axes[1].grid(True)
        axes[2].plot(df_day.index, std_fluctuations, label='Std Dev', color='darkorange')
        axes[2].set_ylabel('Std Dev (mm)')
        axes[2].set_title('Rolling std of PWV fluctuations')
        axes[2].grid(True)
        axes[3].plot(df_day.index, df_day['Wind_Speed'], label='Wind Speed', color='blue')
        axes[3].set_ylabel('Wind (m/s)')
        axes[3].set_title('Wind speed')
        axes[3].set_xlabel('UTC Time')
        axes[3].grid(True)
        plt.tight_layout()
        plt.show()

    return std_fluctuations


# =============================================================================
# 2. FUNCTION: Monthly PWV profile
# =============================================================================
def monthly_PWV_perturbation_profile(df, year, month, smoothing_minutes=30, savgol_window=60):
    """Compute typical daily PWV perturbation profile for a month (unchanged)."""
    date_range = pd.date_range(
        start=f"{year}-{month:02d}-01",
        end=pd.to_datetime(f"{year}-{month:02d}-01") + pd.offsets.MonthEnd(0),
        freq='D'
    )

    profiles = []
    for date in date_range:
        sigma = analyze_PWV_perturbations(df, date.strftime("%Y-%m-%d"),
                                          smoothing_minutes=smoothing_minutes, plot=False)
        if sigma is not None and not sigma.isna().all():
            sigma.name = date.strftime("%Y-%m-%d")
            profiles.append(sigma)

    if not profiles:
        print(f"No usable data for {year}-{month:02d}")
        return

    profiles_df = pd.concat(profiles, axis=1)
    profiles_df['time'] = profiles_df.index.time
    grouped = profiles_df.groupby('time')
    mean_profile = grouped.mean(numeric_only=True).mean(axis=1)
    std_profile = grouped.std(numeric_only=True).mean(axis=1)
    time_index = [datetime.combine(datetime(1900, 1, 1), t) for t in mean_profile.index]
    mean_profile.index = time_index
    std_profile.index = time_index

    if savgol_window >= len(mean_profile):
        savgol_window = len(mean_profile) // 2 * 2 + 1
    smoothed_profile = pd.Series(
        savgol_filter(mean_profile.ffill().bfill(),
                      window_length=savgol_window, polyorder=2),
        index=time_index
    )

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(mean_profile.index, mean_profile, label='Mean profile', color='steelblue')
    axes[0].fill_between(mean_profile.index, mean_profile - std_profile, mean_profile + std_profile,
                         color='lightblue', alpha=0.4, label='±1σ')
    axes[0].set_ylabel('Amplitude |σ| (mm)')
    axes[0].set_title(f'Typical daily sigma(PWV) – {year}-{month:02d}')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(smoothed_profile.index, smoothed_profile, label='Smoothed profile', color='darkorange')
    axes[1].set_xlabel('UTC Time')
    axes[1].set_ylabel('Smoothed amplitude (mm)')
    axes[1].set_title(f'Smoothed profile (Savitzky-Golay window: {savgol_window})')
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()

    return mean_profile, smoothed_profile, std_profile


# =============================================================================
# 3. LOAD APEX WEATHER DATA AND PWV ANALYSIS
# =============================================================================
df = pd.read_csv(APEX_file, header=None, names=['UT', 'PWV', 'Temperature', 'Humidity', 'Wind_Dir', 'Wind_Speed'], parse_dates=['UT'])

analyze_PWV_perturbations(df, date_str, smoothing_minutes=20, plot=True)
mean_prof, smoothed_prof, std_prof = monthly_PWV_perturbation_profile(df, year, month,smoothing_minutes=30, savgol_window=61)

# Save δPWV hourly profile
file_dpwv = os.path.join(save_path, f"delta_pwv_{month_str}.npy")
smoothed_array = np.array(smoothed_prof)
hourly_profile = smoothed_array.reshape(24, 60).mean(axis=1)
np.save(file_dpwv, hourly_profile)
print("Saved δPWV:", file_dpwv)


# =============================================================================
# 4. ESTIMATION OF DENSITY FROM ERA5
# =============================================================================
if not os.path.isfile(q_profile) or not os.path.isfile(file_dpwv):
    raise FileNotFoundError("Required δq or δPWV files not found.")

# Load δq profiles and δPWV
data_q = np.load(q_profile)
typical_profiles = data_q["typical_profiles"]
alt_grid_q = data_q["alt_grid"]  # altitude grid associated with q profiles
delta_PWV = np.load(file_dpwv)

# --- Load ERA5 temperature (ta), humidity (r), geopotential (geopt) ---
ta_ds = xr.open_dataset(find_file("ta", month_label))
r_ds = xr.open_dataset(find_file("r", month_label)) #in %
geopt_ds = xr.open_dataset(find_file("geopt", month_label))

# Central point
lat0 = ta_ds.latitude.values[len(ta_ds.latitude)//2]
lon0 = ta_ds.longitude.values[len(ta_ds.longitude)//2]

lat_vals = ta_ds.latitude.values
lon_vals = ta_ds.longitude.values

# Latitude : closest index of central point
ilat = np.argmin(np.abs(lat_vals - lat0))

# Longitude : idem
ilon = np.argmin(np.abs(lon_vals - lon0))

# Extract point profiles
ta = ta_ds['ta'].isel(latitude=ilat, longitude=ilon).mean(dim='time')
q_era5 = r_ds['r'].isel(latitude=ilat, longitude=ilon).mean(dim='time')/100 #because in %
geopt = geopt_ds['geopt'].isel(latitude=ilat, longitude=ilon).mean(dim='time')


# Estimation of pressure along z from ERA5 (and fit) and density calculation 

# --- Compute virtual temperature ---
T_v = ta * (1 + 0.61 * q_era5)

# Pressure levels from ERA5
p_levels = ta.level.values * 100.0  # hPa -> Pa

# Altitude >= alt_min
z_levels = geopt / g  # Altitude in m
mask = z_levels >= alt_min
z_fit = z_levels[mask].values
p_fit = p_levels[mask]

# Exponential fit P(z) = P0 * exp(-z/H)
from scipy.optimize import curve_fit

def exp_func(z, P0, H):
    return P0 * np.exp(-z/H)

popt, pcov = curve_fit(exp_func, z_fit, p_fit, p0=[p_fit[0]*2, 5100.0])
P0_fit, H_fit = popt

print(f"Exponential fit: P0 = {P0_fit:.1f} Pa, H = {H_fit:.1f} m")

# --- Affichage du fit vs données ERA5 ---
plt.figure(figsize=(6,6))
plt.plot(p_fit, z_fit/1000, 'o', label='ERA5 pressure levels')
plt.plot(exp_func(z_fit, *popt), z_fit/1000, '-', label=f'Fit exp: P0={P0_fit:.0f}, H={H_fit:.0f} m')
plt.xlabel("Pressure (Pa)")
plt.ylabel("Altitude (km)")
plt.title("Exponential fit of ERA5 pressure levels")
plt.legend()
plt.grid(True)
plt.show()

# --- Interpolation of T_v on alt_grid_q ---
T_v_func = interp1d(z_levels.values, T_v.values, kind='linear', bounds_error=False, fill_value="extrapolate")
T_v_fit = T_v_func(alt_grid_q)

# --- Density calculation from ideal gas law ---
rho_profile = exp_func(alt_grid_q, *popt) / (R_d * T_v_fit)  # kg/m³

# Quick check density
print(f"ρ({alt_grid_q[0]/1000:.1f} km) = {rho_profile[0]:.3f} kg/m³")
print(f"ρ({alt_grid_q[len(alt_grid_q)//2]/1000:.1f} km) = {rho_profile[len(alt_grid_q)//2]:.3f} kg/m³")
print(f"ρ({alt_grid_q[-1]/1000:.1f} km) = {rho_profile[-1]:.3f} kg/m³")

# --- Plot density profiles ---
plt.figure(figsize=(6,6))
plt.plot(rho_profile, alt_grid_q/1000, label='Density')
plt.xlabel('Density ρ (kg/m³)')
plt.ylabel('Altitude (km)')
plt.title('Density profile (calculated with fit of ERA5 pressure level and ideal gas law')
plt.grid(True)
plt.legend()
plt.show()


# =============================================================================
# 5. δq calculation using ERA5 density 
# =============================================================================
alt_mask = (alt_grid_q > alt_min) & (alt_grid_q <= alt_max)
dz = np.diff(alt_grid_q, prepend=alt_grid_q[0] - cell_size)

# Also truncate the other vectors in this range
dz = np.diff(alt_grid_q, prepend=alt_grid_q[0] - cell_size)
dz_valid = dz[alt_mask]
alt_valid = alt_grid_q[alt_mask]
rho_valid = rho_profile[alt_mask]

delta_q_profiles = np.full_like(typical_profiles, np.nan)
for h in range(24):
    q_h = typical_profiles[h, :]
    if np.any(np.isnan(q_h)):
        continue
    q_valid = q_h[alt_mask]
    dz_valid = dz[alt_mask]
    rho_valid = rho_profile[alt_mask]
    pwv_h = np.sum(q_valid * rho_valid * dz_valid)
    if pwv_h == 0 or np.isnan(pwv_h):
        continue
    weights = (q_valid * rho_valid * dz_valid) / pwv_h
    delta_q_profiles[h, alt_mask] = weights * delta_PWV[h]

# --- Verify δPWV reconstruction ---
delta_PWV_reconstructed = []
for h in range(24):
    dq = delta_q_profiles[h, :]
    if np.all(np.isnan(dq)):
        delta_PWV_reconstructed.append(np.nan)
    else:
        contrib = dq[alt_mask] * rho_profile[alt_mask] * dz[alt_mask]
        delta_PWV_reconstructed.append(np.sum(contrib))
delta_PWV_reconstructed = np.array(delta_PWV_reconstructed)

plt.figure(figsize=(12,5))
plt.plot(delta_PWV, label="Original δPWV")
plt.plot(delta_PWV_reconstructed, '--', label="Reconstructed δPWV")
plt.xlabel("UTC Hour"); plt.ylabel("δPWV amplitude")
plt.title("Verification of δPWV conservation using ERA5 density")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# --- Heatmap δq ---
delta_q_map = np.abs(delta_q_profiles[:, alt_mask])
plt.figure(figsize=(12,6))
plt.imshow(delta_q_map.T, origin='lower', aspect='auto',
           extent=[0, 23, alt_grid_q[alt_mask][0]/1000, alt_grid_q[alt_mask][-1]/1000],
           cmap='inferno')
plt.colorbar(label='δq amplitude (kg/kg)')
plt.xlabel("UTC Hour"); plt.ylabel("Altitude (km)")
plt.title(f"Hourly δq amplitude propagated from δPWV - {month_str}")
plt.tight_layout(); plt.show()


# =============================================================================
# 6. Save δq if requested
# =============================================================================
if Save_delta_q:
    np.savez(os.path.join(save_path, f"delta_q_profiles_{month_str}.npz"),
             delta_q_profiles=delta_q_profiles,
             alt_grid=alt_grid_q)
    print("Saved δq profiles:", os.path.join(save_path, f"delta_q_profiles_{month_str}.npz"))

ta_ds.close()
r_ds.close()
geopt_ds.close()