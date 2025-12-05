# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 15:58:48 2025

@author: tollet
"""

import os
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
from scipy.interpolate import interp1d


"""
This code generates daily profiles from Era-5 data.

2 options are available : 
    - Generating the profile of a given day with no processing except the altitude interpolation
    - Generating the typical daily profile for a given month. The code calculate a mean of the daily evolution of the selected variable after rescaling the baseline of each day.
    
The available variables of the Era-5 2023 data are: 
    - ta : temperature in Kelvin (K)
    - q : Specific humidity in kg/kg
    - r : relative humidity in %
    - geopt : geopotential in m/s
    - u : East-West wind component in m/s (positive toward East)
    - v : North-South wind component in m/s (positive towart North)
    - w : Vertial wind velocity in Pa/s (speed of vertical pressure variation)
    - cc : cloud cover (betwin 0 and 1)
    - ciwc : cloud ice water content in kg/m3 (cloudy ice mass per air volume)
    - clwc : cloud liquid water content in kg/m3 (cloudy liquid water mass per air volume)
    - pv : potential vorticity (Ertel's PV) in K.m²/kg/s
    - o3 : ozeone mass mixing ration in kg/kg
"""


# =============================================================================
# Parameters
# =============================================================================

data_dir = Path().resolve(__file__) / "Data" / "era_5"/ "2023"
save_path = Path().resolve(__file__) / "Data" / "Profiles"
year = 2023
month = 7
day = 13
alt_min = 5100
alt_max = 15000
cell_size = 2

g = 9.80665

# =============================================================================
# FIND THE FILE WITH A DATE
# =============================================================================

def find_file(var_name, month_tag, data_dir):
    """
    Locate a specific ERA5 NetCDF file (e.g., 'q.202307.nc') in a directory.
    """
    for fname in os.listdir(data_dir):
        if fname.lower().startswith(f"{var_name.lower()}.{month_tag}") and fname.lower().endswith((".nc", ".nc4")):
            return os.path.join(data_dir, fname)
    raise FileNotFoundError(f"No file found for variable '{var_name}' and month '{month_tag}'.")

# =============================================================================
# ONE DAY PROFILE
# =============================================================================

def plot_daily_profile(
    var_name, 
    data_dir, 
    year, 
    month, 
    day,
    alt_min = alt_min, 
    alt_max = alt_max, 
    cell_size = cell_size,
    plot=True,
    save=False,
    save_path = save_path):
    
    """
    Plot ERA5 vertical profiles for a specific day (24 hours) at a single point.
    Does NOT compute monthly averages, only interpolates hourly profiles.

    Returns:
        profiles (n_hours x n_alt): interpolated hourly profiles
        alt_grid: altitude grid (m)
    
    !! 
    For var_name variable, use era_5 variabes ('q', 'ta', 'r', u, v, etc.) 
    For the absolute wind profile (absolute speed and direction): var_name = 'wind' 
    !!
    """

    month_tag = f"{year}{month:02d}"
    date_label = f"{day}-{month:02d}-{year}"

    # --- Load datasets ---
    ds_geopt = xr.open_dataset(find_file("geopt", month_tag, data_dir))
    ds_u = xr.open_dataset(find_file("u", month_tag, data_dir))
    ds_v = xr.open_dataset(find_file("v", month_tag, data_dir))
    
    if var_name != "wind":
        ds_var = xr.open_dataset(find_file(var_name, month_tag, data_dir))

    # --- Central point ---
    if var_name != "wind":
        lat0 = ds_var.latitude.values[len(ds_var.latitude)//2]
        lon0 = ds_var.longitude.values[len(ds_var.longitude)//2]
    else:
        lat0 = ds_u.latitude.values[len(ds_u.latitude)//2]
        lon0 = ds_u.longitude.values[len(ds_u.longitude)//2]

    # --- Altitude grid ---
    alt_grid = np.arange(alt_min, alt_max + cell_size, cell_size)
    profiles = []

    # --- Loop over 24 hours ---
    for h in range(24):
        dt = pd.Timestamp(year, month, day, h)

        geopt = ds_geopt["geopt"].sel(time=dt, latitude=lat0, longitude=lon0, method="nearest").values
        alt = geopt / g

        if var_name != "wind":
            var_vals = ds_var[var_name].sel(time=dt, latitude=lat0, longitude=lon0, method="nearest").values
            valid = ~np.isnan(var_vals) & ~np.isnan(alt)
        else:
            u = ds_u["u"].sel(time=dt, latitude=lat0, longitude=lon0, method="nearest").values
            v = ds_v["v"].sel(time=dt, latitude=lat0, longitude=lon0, method="nearest").values
            ws = np.sqrt(u**2 + v**2)
            wd = (np.degrees(np.arctan2(-u, -v)) + 360) % 360
            var_vals = np.column_stack((ws, wd))
            valid = ~np.isnan(var_vals[:,0]) & ~np.isnan(alt)

        if np.sum(valid) < 2:
            profiles.append(np.full((len(alt_grid), var_vals.shape[1] if var_name=="wind" else 1), np.nan))
            continue

        alt_valid = alt[valid]
        if var_name != "wind":
            var_valid = var_vals[valid]
        else:
            var_valid = var_vals[valid]

        # Sort and remove duplicates
        sort_idx = np.argsort(alt_valid)
        alt_sorted = alt_valid[sort_idx]
        var_sorted = var_valid[sort_idx]
        alt_unique, unique_idx = np.unique(alt_sorted, return_index=True)
        var_unique = var_sorted[unique_idx]

        # Interpolation
        if var_name != "wind":
            interp_func = interp1d(alt_unique, var_unique, kind="linear",
                                   bounds_error=False, fill_value="extrapolate")
            profiles.append(interp_func(alt_grid))
        else:
            interp_ws = interp1d(alt_unique, var_unique[:,0], kind="linear",
                                 bounds_error=False, fill_value="extrapolate")
            interp_wd = interp1d(alt_unique, var_unique[:,1], kind="linear",
                                 bounds_error=False, fill_value="extrapolate")
            profiles.append(np.column_stack((interp_ws(alt_grid), interp_wd(alt_grid))))

    profiles = np.array(profiles)  # shape: (24, n_alt) or (24, n_alt, 2)

    # --- Plot ---
    if plot:
        if var_name != "wind":
            unit  = ds_var[var_name].attrs.get("units")
            plt.figure(figsize=(10,6))
            plt.imshow(profiles.T, origin="lower", aspect="auto",
                       extent=[0,23, alt_grid[0]/1000, alt_grid[-1]/1000],
                       cmap="viridis")
            plt.colorbar(label=f"{var_name} ({unit})")
            plt.xlabel("Hour UTC")
            plt.ylabel("Altitude (km)")
            plt.title(f"{var_name} profiles - {day:02d}/{month:02d}/{year}")
            plt.tight_layout()
            plt.show()
        else:
            fig, axs = plt.subplots(1,2,figsize=(16,6), sharey=True)
            im0 = axs[0].imshow(profiles[:,:,0].T, origin="lower", aspect="auto",
                                extent=[0,23, alt_grid[0]/1000, alt_grid[-1]/1000], cmap="viridis")
            axs[0].set_title("Wind speed (m/s)")
            axs[0].set_xlabel("Hour UTC")
            axs[0].set_ylabel("Altitude (km)")
            fig.colorbar(im0, ax=axs[0])

            im1 = axs[1].imshow(profiles[:,:,1].T, origin="lower", aspect="auto",
                                extent=[0,23, alt_grid[0]/1000, alt_grid[-1]/1000],
                                cmap="hsv", vmin=0, vmax=360)
            axs[1].set_title("Wind direction (°)")
            axs[1].set_xlabel("Hour UTC")
            fig.colorbar(im1, ax=axs[1])

            plt.suptitle(f"Wind profiles - {day:02d}/{month:02d}/{year}")
            plt.tight_layout(rect=[0,0,1,0.95])
            plt.show()
        
        # --- Save output ---
        if save and save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            out_file = os.path.join(save_path, f"{var_name}_profiles_{date_label}.npz")
            np.savez(out_file, wind_speed=profiles[:, :, 0], wind_dir=profiles[:, :, 1], alt_grid=alt_grid)
            print(f"Saved: {out_file}")
        
        
    return profiles, alt_grid



# =============================================================================
# MONTH TYPICAL PROFILE
# =============================================================================

def process_era5_typical_profiles(
    var_name,
    data_dir,
    year,
    month,
    alt_min = alt_min,
    alt_max = alt_max,
    cell_size = cell_size,
    plot=False,
    save=False,
    save_path = save_path):
    
    """
    Compute a typical 24h × altitude profile for an ERA5 variable
    !! For var_name variable, use era_5 variabes ('q', 'ta', 'r', etc.) EXCEPT for the wind : var_name = 'wind' !!
    """

    # =========================
    #  CONSTANTS AND MONTH TAG
    # =========================
    month_tag = f"{year}{month:02d}"
    month_label = f"{month:02d}-{year}"

    # =========================
    #  LOAD STATIC FIELDS
    # =========================
    ds_geopt = xr.open_dataset(find_file("geopt", month_tag, data_dir))

    # Always load u/v for potential wind processing
    ds_u = xr.open_dataset(find_file('u', month_tag, data_dir))
    ds_v = xr.open_dataset(find_file('v', month_tag, data_dir))

    # Load requested variable (if not wind)
    if var_name != "wind":
        ds_var = xr.open_dataset(find_file(var_name, month_tag, data_dir))
    else:
        ds_var = None  # handled separately

    # =========================
    #  GRID POINT SELECTION
    # =========================
    # For scalar variables
    if var_name != "wind":
        lat0 = ds_var.latitude.values[len(ds_var.latitude)//2]
        lon0 = ds_var.longitude.values[len(ds_var.longitude)//2]
    else:
        lat0 = ds_u.latitude.values[len(ds_u.latitude)//2]
        lon0 = ds_u.longitude.values[len(ds_u.longitude)//2]

    # Altitude grid
    alt_grid = np.arange(alt_min, alt_max + cell_size, cell_size)

    profiles_all = []
    times_all = []

    # =============================
    #   ITERATE OVER ALL HOURS
    # =============================
    if var_name != "wind":
        time_iter = ds_var.time.values
    else:
        time_iter = ds_u.time.values  # u/v define available timestamps

    for t in time_iter:
        dt = pd.to_datetime(str(t))
        if dt.minute != 0:
            continue

        # Altitude from geopotential
        geopt = ds_geopt["geopt"].sel(time=t, latitude=lat0, longitude=lon0, method="nearest").values
        alt = geopt / g

        # ===========================
        #   VARIABLE EXTRACTION
        # ===========================
        if var_name != "wind":
            var_vals = ds_var[var_name].sel(time=t, latitude=lat0, longitude=lon0, method="nearest").values
        else:
            u = ds_u["u"].sel(time=t, latitude=lat0, longitude=lon0, method="nearest").values
            v = ds_v["v"].sel(time=t, latitude=lat0, longitude=lon0, method="nearest").values

            # Compute wind speed and direction
            ws = np.sqrt(u**2 + v**2)
            wd = (np.degrees(np.arctan2(-u, -v)) + 360) % 360

            var_vals = np.column_stack((ws, wd))  # shape (levels, 2)

        # ===========================
        #   VALIDITY MASK
        # ===========================
        if var_name != "wind":
            valid = ~np.isnan(var_vals) & ~np.isnan(alt) & (alt >= alt_min)
        else:
            valid = (~np.isnan(var_vals[:,0])) & (~np.isnan(alt)) & (alt >= alt_min)

        if np.sum(valid) < 2:
            continue

        alt_valid = alt[valid]

        # ===========================
        #   SORT BY ALTITUDE
        # ===========================
        sort_idx = np.argsort(alt_valid)
        alt_sorted = alt_valid[sort_idx]

        if var_name != "wind":
            var_sorted = var_vals[valid][sort_idx]
        else:
            var_sorted = var_vals[valid][sort_idx]   # shape (N,2)

        # ===========================
        #   REMOVE DUPLICATES
        # ===========================
        alt_unique, unique_idx = np.unique(alt_sorted, return_index=True)
        var_unique = var_sorted[unique_idx]

        # ===========================
        #   INTERPOLATION
        # ===========================
        if var_name != "wind":
            interp_func = interp1d(
                alt_unique, var_unique,
                kind="linear", bounds_error=False, fill_value="extrapolate")
            var_interp = interp_func(alt_grid)

        else:
            # Scalar interpolation per component
            interp_ws = interp1d(
                alt_unique, var_unique[:,0],
                kind="linear", bounds_error=False, fill_value="extrapolate")
            interp_wd = interp1d(
                alt_unique, var_unique[:,1],
                kind="linear", bounds_error=False, fill_value="extrapolate")

            var_interp = np.column_stack((interp_ws(alt_grid), interp_wd(alt_grid)))

        profiles_all.append(var_interp)
        times_all.append(dt)

    if len(profiles_all) == 0:
        raise RuntimeError(f"No available data for {month_label}")

    profiles_all = np.array(profiles_all)
    times_all = pd.to_datetime(times_all)

    # =====================================================
    #   BRANCH: TREAT WIND AND OTHER VARIABLES DIFFERENTLY
    # =====================================================

    # -----------------------------------------------------
    #  CASE 1 : SCALAR VARIABLES (ta, q, r)
    # -----------------------------------------------------
    if var_name != "wind":
        df = pd.DataFrame(profiles_all, index=times_all)

        # ===== SAME DAILY PROCESSING =====
        days = sorted(set(df.index.date))
        daily_baselines = []
        daily_anoms = []
        daily_amps = []

        for d in days:
            sub = df.loc[str(d)]
            if len(sub) < 20:
                continue
            target_times = pd.date_range(start=pd.Timestamp(d), periods=24, freq="1h")
            sub = sub.reindex(target_times, method="nearest")

            arr = sub.values
            baseline = np.nanmean(arr, axis=0)
            anom = arr - baseline[None,:]
            amp = np.nanmax(anom, axis=0) - np.nanmin(anom, axis=0)

            daily_baselines.append(baseline)
            daily_anoms.append(anom)
            daily_amps.append(amp)

        daily_baselines = np.array(daily_baselines)
        daily_anoms = np.array(daily_anoms)
        daily_amps = np.array(daily_amps)

        # ===== FINAL TYPICAL PROFILE =====
        avg_anom = np.nanmean(daily_anoms, axis=0)
        baseline_ref = np.nanmedian(daily_baselines, axis=0)
        amp_avg = np.nanmax(avg_anom, axis=0) - np.nanmin(avg_anom, axis=0)
        amp_med = np.nanmedian(daily_amps, axis=0)

        scale = np.ones_like(amp_avg)
        mask = amp_avg > 0
        scale[mask] = amp_med[mask] / amp_avg[mask]
        scale = np.clip(scale, 0.2, 5.0)

        avg_anom_scaled = avg_anom * scale[None,:]
        typical_profiles = baseline_ref[None,:] + avg_anom_scaled

        if plot:
            norm = LogNorm() if var_name == "q" else None
        
            plt.figure(figsize=(12, 6))
            plt.imshow(
                typical_profiles.T,
                origin="lower",
                aspect="auto",
                norm=norm,
                extent=[0, 23, alt_grid[0] / 1000, alt_grid[-1] / 1000],
                cmap="viridis"
            )
        
            unit_label = "kg/kg" if var_name == "q" else "K"
        
            plt.colorbar(label=f"{var_name} ({unit_label})")
            plt.xlabel("Hour of the day (UTC)")
            plt.ylabel("Altitude (km)")
            plt.title(
                f"Typical daily profile of {var_name}\n"
                f"({month:02d}/{year}, altitude > {alt_min} m)"
            )
            plt.tight_layout()
            plt.show()
            
            # --- Save output ---
            if save and save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                out_file = os.path.join(save_path, f"mean_{var_name}_profiles_{month_label}.npz")
                np.savez(out_file, typical_profiles=typical_profiles, alt_grid=alt_grid)
                print(f"Saved: {out_file}")

        return typical_profiles, alt_grid

    # -----------------------------------------------------
    #  CASE 2 : WIND (ws + wd simultaneously)
    # -----------------------------------------------------
    else:
        # profiles_all.shape = (n_times, n_alt, 2)
        ws_all = profiles_all[:,:,0]
        wd_all = profiles_all[:,:,1]

        # ========== Convert wd to unit vector ==========
        sin_wd = np.sin(np.deg2rad(wd_all))
        cos_wd = np.cos(np.deg2rad(wd_all))

        df_ws  = pd.DataFrame(ws_all, index=times_all)
        df_sin = pd.DataFrame(sin_wd, index=times_all)
        df_cos = pd.DataFrame(cos_wd, index=times_all)

        days = sorted(set(df_ws.index.date))
        daily_ws_anom = []
        daily_ws_base = []
        daily_ws_amp  = []

        daily_sin_anom = []
        daily_cos_anom = []
        daily_sin_base = []
        daily_cos_base = []

        for d in days:
            sub_ws  = df_ws.loc[str(d)]
            sub_sin = df_sin.loc[str(d)]
            sub_cos = df_cos.loc[str(d)]

            if len(sub_ws) < 20:
                continue

            target_times = pd.date_range(start=pd.Timestamp(d), periods=24, freq="1h")
            sub_ws  = sub_ws.reindex(target_times, method="nearest")
            sub_sin = sub_sin.reindex(target_times, method="nearest")
            sub_cos = sub_cos.reindex(target_times, method="nearest")

            arr_ws  = sub_ws.values
            arr_sin = sub_sin.values
            arr_cos = sub_cos.values

            # Baselines
            base_ws  = np.nanmean(arr_ws, axis=0)
            base_sin = np.nanmean(arr_sin, axis=0)
            base_cos = np.nanmean(arr_cos, axis=0)

            anom_ws  = arr_ws  - base_ws[None,:]
            anom_sin = arr_sin - base_sin[None,:]
            anom_cos = arr_cos - base_cos[None,:]

            amp_ws = np.nanmax(anom_ws,axis=0)-np.nanmin(anom_ws,axis=0)

            daily_ws_base.append(base_ws)
            daily_ws_anom.append(anom_ws)
            daily_ws_amp.append(amp_ws)

            daily_sin_base.append(base_sin)
            daily_cos_base.append(base_cos)
            daily_sin_anom.append(anom_sin)
            daily_cos_anom.append(anom_cos)

        # Convert to arrays
        daily_ws_anom = np.array(daily_ws_anom)
        daily_ws_base = np.array(daily_ws_base)
        daily_ws_amp  = np.array(daily_ws_amp)

        daily_sin_anom = np.array(daily_sin_anom)
        daily_cos_anom = np.array(daily_cos_anom)
        daily_sin_base = np.array(daily_sin_base)
        daily_cos_base = np.array(daily_cos_base)

        # ================================
        #  Build typical wind speed (SIMILAR TO SCALAR)
        # ================================
        avg_anom_ws = np.nanmean(daily_ws_anom,axis=0)
        base_ws     = np.nanmedian(daily_ws_base,axis=0)
        amp_avg_ws  = np.nanmax(avg_anom_ws,axis=0)-np.nanmin(avg_anom_ws,axis=0)
        amp_med_ws  = np.nanmedian(daily_ws_amp,axis=0)

        scale_ws = np.ones_like(amp_avg_ws)
        mask = amp_avg_ws>0
        scale_ws[mask] = amp_med_ws[mask]/amp_avg_ws[mask]
        scale_ws = np.clip(scale_ws,0.2,5.0)

        avg_anom_ws_scaled = avg_anom_ws * scale_ws[None,:]
        typical_ws = base_ws[None,:] + avg_anom_ws_scaled

        # ================================
        #  Build typical wind direction
        # ================================
        avg_sin = np.nanmedian(daily_sin_base,axis=0)
        avg_cos = np.nanmedian(daily_cos_base,axis=0)

        typical_wd = (np.degrees(np.arctan2(avg_sin,avg_cos)) + 360) % 360

        # Replace shape to (24,n_alt)
        typical_wd = np.tile(typical_wd,(24,1))
        
        if plot:
            fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        
            # --- Vitesse du vent ---
            im0 = axs[0].imshow(
                typical_ws.T,
                origin="lower",
                aspect="auto",
                extent=[0, 23, alt_grid[0]/1000, alt_grid[-1]/1000],
                cmap="viridis"
            )
            axs[0].set_xlabel("Hour of the day (UTC)")
            axs[0].set_ylabel("Altitude (km)")
            axs[0].set_title("Wind speed (m/s)")
            fig.colorbar(im0, ax=axs[0], label="m/s")
        
            # --- Direction du vent ---
            im1 = axs[1].imshow(
                typical_wd.T,
                origin="lower",
                aspect="auto",
                extent=[0, 23, alt_grid[0]/1000, alt_grid[-1]/1000],
                cmap="hsv",
                vmin=0,
                vmax=360
            )
            axs[1].set_xlabel("Hour of the day (UTC)")
            axs[1].set_title("Wind direction (°)")
            fig.colorbar(im1, ax=axs[1], label="°")
        
            plt.suptitle(f"Typical daily wind profiles ({month:02d}/{year})")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()


            # --- Save output ---
            if save and save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                out_file = os.path.join(save_path, f"mean_{var_name}_profiles_{month_label}.npz")
                np.savez(out_file, wind_speed=typical_ws, wind_dir=typical_wd, alt_grid=alt_grid)
                print(f"Saved: {out_file}")
      
        return {"ws": typical_ws, "wd": typical_wd}, alt_grid



# =============================================================================
# TEST OF OPERATION
# =============================================================================

#  Typical Porifles
# Temperature (ta)
mean_wind, alt_grid_wind = process_era5_typical_profiles("q", data_dir, year, month, plot=True, save=True, save_path=save_path)
# Wind (u & v)
wind_profiles, alt_grid = process_era5_typical_profiles("wind", data_dir, year, month, plot=True, save=False, save_path=save_path)

ws = wind_profiles["ws"]
wd = wind_profiles["wd"]
    


#  Daily profile
#  Specific humidity (q)
profiles_q, alt_grid_q = plot_daily_profile(var_name="q", data_dir=data_dir, year=year, month=month, day=day, plot=True, save=False, save_path=save_path)

# Wind (u & v)
profiles_wind, alt_grid_wind = plot_daily_profile(var_name="wind", data_dir=data_dir, year=year, month=month, day=day, plot=True, save=True, save_path=save_path)


# =============================================================================
#  === Example usage ===
# =============================================================================
if __name__ == "__main__":
    
    var_name = 'ta'

    mean_profile, alt_grid = process_era5_typical_profiles(var_name, data_dir, year, month, plot=True, save=False, save_path=save_path)

    daily_profile, altgrid = plot_daily_profile(var_name, data_dir, year, month, day, plot=True, save=False, save_path=save_path)

# =============================================================================
#  PROFILES CHECK !! Not valable for WIND !!
# =============================================================================

def plot_check_vertical(var_name, data_dir, year, month, days_to_compare, hour=16):
    """
    Plot vertical profiles for specific days vs typical daily profile
    at a given hour (UTC).
    """
    # --- Calculate typical profile ---
    typical_profiles, alt_grid = process_era5_typical_profiles(var_name, data_dir, year, month, plot=False, save=False)

    # --- Load datasets for real days ---
    ds_var = xr.open_dataset(find_file(var_name, f"{year}{month:02d}", data_dir))
    ds_geopt = xr.open_dataset(find_file("geopt", f"{year}{month:02d}", data_dir))

    # Central grid point
    lat0 = ds_var.latitude.values[len(ds_var.latitude)//2]
    lon0 = ds_var.longitude.values[len(ds_var.longitude)//2]

    plt.figure(figsize=(6, 8))
    for d in days_to_compare:
        dt = pd.Timestamp(year, month, d, hour)
        var_day = ds_var[var_name].sel(time=dt, latitude=lat0, longitude=lon0, method="nearest").values
        geopt_day = ds_geopt["geopt"].sel(time=dt, latitude=lat0, longitude=lon0, method="nearest").values
        alt_day = geopt_day / g

        valid = ~np.isnan(var_day) & ~np.isnan(alt_day)
        if np.sum(valid) < 2:
            continue

        idx_sort = np.argsort(alt_day[valid])
        plt.plot(var_day[valid][idx_sort], alt_day[valid][idx_sort]/1000, label=f"Day {d}")

    # Plot typical profile at this hour
    plt.plot(typical_profiles[hour,:], alt_grid/1000, 'k--', label="Typical profile")

    plt.xlabel(var_name)
    plt.ylabel("Altitude (km)")
    plt.title(f"{var_name} vertical profile at {hour} UTC - {month:02d}/{year}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_check_timeseries(var_name, data_dir, year, month, days_to_compare, alt_ref=5100):
    """
    Plot horizontal time series at a fixed altitude vs typical daily profile.
    """
    # --- Calculate typical profile ---
    typical_profiles, alt_grid = process_era5_typical_profiles(
        var_name, data_dir, year, month, plot=False, save=False
    )

    # --- Load datasets for real days ---
    ds_var = xr.open_dataset(find_file(var_name, f"{year}{month:02d}", data_dir))
    ds_geopt = xr.open_dataset(find_file("geopt", f"{year}{month:02d}", data_dir))

    # Central grid point
    lat0 = ds_var.latitude.values[len(ds_var.latitude)//2]
    lon0 = ds_var.longitude.values[len(ds_var.longitude)//2]

    idx_alt = np.argmin(np.abs(alt_grid - alt_ref))
    plt.figure(figsize=(10, 4))

    # Plot typical profile
    plt.plot(range(24), typical_profiles[:, idx_alt], 'k-', label="Typical profile")

    for d in days_to_compare:
        series = []
        for h in range(24):
            dt = pd.Timestamp(year, month, d, h)
            var_h = ds_var[var_name].sel(time=dt, latitude=lat0, longitude=lon0, method="nearest").values
            geopt_h = ds_geopt["geopt"].sel(time=dt, latitude=lat0, longitude=lon0, method="nearest").values
            alt_h = geopt_h / g

            valid = ~np.isnan(var_h) & ~np.isnan(alt_h)
            if np.sum(valid) < 2:
                series.append(np.nan)
                continue

            idx_sort = np.argsort(alt_h[valid])
            interp_func = interp1d(
                alt_h[valid][idx_sort], var_h[valid][idx_sort],
                kind='linear', fill_value='extrapolate'
            )
            series.append(interp_func(alt_ref))

        plt.plot(range(24), series, label=f"Day {d}")

    plt.xlabel("Hour (UTC)")
    plt.ylabel(f"{var_name}")
    plt.title(f"{var_name} at {alt_ref/1000:.1f} km altitude - {month:02d}/{year}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    

# =============================================================================
# Test Check Plots    
# =============================================================================

days_to_compare = [1, 10, 20, 30]

# Vertical check
plot_check_vertical("ta", data_dir, year, month, days_to_compare, hour=16)

# Timeseries check
plot_check_timeseries("ta", data_dir, year, month, days_to_compare, alt_ref=5100)

