# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 14:53:37 2025

@author: tollet
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
from scipy.interpolate import interp1d

def find_file(var_name, month_tag, data_dir):
    """
    Locate a specific ERA5 NetCDF file (e.g., 'q.202307.nc') in a directory.
    """
    for fname in os.listdir(data_dir):
        if fname.lower().startswith(f"{var_name.lower()}.{month_tag}") and fname.lower().endswith((".nc", ".nc4")):
            return os.path.join(data_dir, fname)
    raise FileNotFoundError(f"No file found for variable '{var_name}' and month '{month_tag}'.")


def process_era5_typical_profiles(
    var_name,
    data_dir,
    year,
    month,
    alt_min=5100,
    alt_max=15000,
    cell_size=2,
    check=False,
    save=False,
    save_path=None):
    """
    Compute a typical 24h × altitude profile for an ERA5 variable ('q' or 'ta')
    using a daily-anomaly method.

    """

    # --- Constants and month formatting ---
    g = 9.80665
    month_tag = f"{year}{month:02d}"
    month_str = f"{year}_{month:02d}"
    month_label = f"{month:02d}/{year}"

    # --- Load ERA5 datasets ---
    ds_geopt = xr.open_dataset(find_file("geopt", month_tag, data_dir))
    ds_u = xr.open_dataset(find_file('u', month_tag, data_dir))
    ds_v = xr.open_dataset(find_file('v', month_tag, data_dir))
    ds_ta = xr.open_dataset(find_file('ta', month_tag, data_dir))
    ds_q = xr.open_dataset(find_file('q', month_tag, data_dir))
    ds_r = xr.open_dataset(find_file('r', month_tag, data_dir))
    

    ds_var = xr.open_dataset(find_file(var_name, month_tag, data_dir))
    

        # Use central grid point
    lat0 = ds_var.latitude.values[len(ds_var.latitude)//2]
    lon0 = ds_var.longitude.values[len(ds_var.longitude)//2]

    
    # Altitude grid
    alt_grid = np.arange(alt_min, alt_max + cell_size, cell_size)

    profiles_all, times_all = [], []


    # --- Extract hourly vertical profiles ---
    for t in ds_var.time.values:
        dt = pd.to_datetime(str(t))
        if dt.minute != 0:
            continue  # skip non-hourly values

        geopt = ds_geopt["geopt"].sel(time=t, latitude=lat0, longitude=lon0, method="nearest").values
        alt = geopt / g  # convert geopotential to altitude
        
        var_vals = ds_var[var_name].sel(time=t, latitude=lat0, longitude=lon0, method="nearest").values
        valid = ~np.isnan(var_vals) & ~np.isnan(alt) & (alt >= alt_min)
        if np.sum(valid) < 2:
            continue

        var_valid = var_vals[valid]
        alt_valid = alt[valid]

        # Sort by altitude
        sort_idx = np.argsort(alt_valid)
        alt_valid = alt_valid[sort_idx]
        var_valid = var_valid[sort_idx]
        # Remove duplicate altitudes
        alt_unique, unique_idx = np.unique(alt_valid, return_index=True)
        var_unique = var_valid[unique_idx]

        # Linear interpolation on fixed altitude grid
        interp_func = interp1d(alt_unique, var_unique, kind="linear",
                               bounds_error=False, fill_value="extrapolate")
        var_interp = interp_func(alt_grid)

        profiles_all.append(var_interp)
        times_all.append(dt)

    if len(profiles_all) == 0:
        raise RuntimeError(f"No available data for {month_label}")

    profiles_all = np.array(profiles_all)  # shape: (n_times, n_alt)
    times_all = pd.to_datetime(times_all)

    # --- Group by day ---
    df = pd.DataFrame(profiles_all, index=times_all)
    days = sorted(set(df.index.date))

    daily_baselines, daily_anoms, daily_amps = [], [], []

    for d in days:
        sub = df.loc[str(d)]
        if len(sub) < 20:
            continue
        target_times = pd.date_range(start=pd.Timestamp(d), periods=24, freq="1H")
        sub = sub.reindex(target_times, method="nearest")

        arr = sub.values
        baseline = np.nanmean(arr, axis=0)
        anom = arr - baseline[None, :]
        amp = np.nanmax(anom, axis=0) - np.nanmin(anom, axis=0)

        daily_baselines.append(baseline)
        daily_anoms.append(anom)
        daily_amps.append(amp)

    daily_baselines = np.array(daily_baselines)
    daily_anoms = np.array(daily_anoms)
    daily_amps = np.array(daily_amps)

    # --- Compute "typical" daily profile ---
    avg_anom = np.nanmean(daily_anoms, axis=0)
    baseline_ref = np.nanmedian(daily_baselines, axis=0)
    amp_avg = np.nanmax(avg_anom, axis=0) - np.nanmin(avg_anom, axis=0)
    amp_med = np.nanmedian(daily_amps, axis=0)

    scale = np.ones_like(amp_avg)
    mask = amp_avg > 0
    scale[mask] = amp_med[mask] / amp_avg[mask]
    scale = np.clip(scale, 0.2, 5.0)

    avg_anom_scaled = avg_anom * scale[None, :]
    typical_profiles = baseline_ref[None, :] + avg_anom_scaled

    # --- Plot 2D profile map ---
    norm = LogNorm() if var_name == "q" else None
    plt.figure(figsize=(12, 6))
    plt.imshow(
        typical_profiles.T,
        origin="lower",
        aspect="auto",
        norm=norm,
        extent=[0, 23, alt_min / 1000, alt_max / 1000],
        cmap="viridis"
    )
    unit_label = "kg/kg" if var_name == "q" else "K"
    plt.colorbar(label=f"{var_name} ({unit_label})")
    plt.xlabel("Hour of the day (UTC)")
    plt.ylabel("Altitude (km)")
    plt.title(f"Typical daily profile of {var_name}\n({month_label}, altitude > {alt_min} m)")
    plt.tight_layout()
    plt.show()

    # --- Optional: validation with real days ---
    if check:
        hour_to_plot = 16       # UTC hour for vertical profile
        alt_ref = 5100          # Altitude for horizontal cut
        days_to_compare = [1, 10, 20, 30]

        # Vertical profiles
        plt.figure(figsize=(6, 8))
        for d in days_to_compare:
            dt = pd.Timestamp(year, month, d, hour_to_plot)
            var_day = ds_var[var_name].sel(time=dt, latitude=lat0, longitude=lon0, method="nearest").values
            geopt_day = ds_geopt["geopt"].sel(time=dt, latitude=lat0, longitude=lon0, method="nearest").values
            alt_day = geopt_day / g

            valid = ~np.isnan(var_day) & ~np.isnan(alt_day)
            if np.sum(valid) < 2:
                continue

            sort_idx = np.argsort(alt_day[valid])
            plt.plot(var_day[valid][sort_idx], alt_day[valid][sort_idx] / 1000, label=f"Day {d}")

        plt.plot(typical_profiles[hour_to_plot, :], alt_grid / 1000, 'k--', label="Typical profile")
        plt.xlabel(f"{var_name} ({unit_label})")
        plt.ylabel("Altitude (km)")
        plt.title(f"{var_name} vertical profile at {hour_to_plot} UTC - {month_label}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

        # Horizontal time series at fixed altitude
        plt.figure(figsize=(10, 4))
        plt.plot(range(24), typical_profiles[:, np.argmin(np.abs(alt_grid - alt_ref))],
                 'k-', label="Typical profile")

        for d in days_to_compare:
            var_day_series = []
            for h in range(24):
                dt = pd.Timestamp(year, month, d, h)
                var_h = ds_var[var_name].sel(time=dt, latitude=lat0, longitude=lon0, method="nearest").values
                geopt_h = ds_geopt["geopt"].sel(time=dt, latitude=lat0, longitude=lon0, method="nearest").values
                alt_h = geopt_h / g

                valid = ~np.isnan(var_h) & ~np.isnan(alt_h)
                if np.sum(valid) < 2:
                    var_day_series.append(np.nan)
                    continue

                sort_idx = np.argsort(alt_h[valid])
                interp_func = interp1d(
                    alt_h[valid][sort_idx], var_h[valid][sort_idx],
                    kind='linear', fill_value='extrapolate'
                )
                var_day_series.append(interp_func(alt_ref))

            plt.plot(range(24), var_day_series, label=f"Day {d}")

        plt.xlabel("Hour (UTC)")
        plt.ylabel(f"{var_name} ({unit_label})")
        plt.title(f"{var_name} at {alt_ref/1000:.1f} km altitude - {month_label}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    # --- Save output ---
    if save and save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        out_file = os.path.join(save_path, f"mean_{var_name}_profiles_{month_label}.npz")
        np.savez(out_file, typical_profiles=typical_profiles, alt_grid=alt_grid)
        print(f"Saved: {out_file}")

    return typical_profiles, alt_grid


# === Example usage ===
if __name__ == "__main__":
    data_dir = r"C:\Users\tollet\Documents\S4\CosmoLIDAR\Atmospheric_model\atmo3\era5_point\2023"
    save_path = r"C:\\Users\tollet\Documents\S4\CosmoLIDAR\Atmospheric_model\atmo3\Normalisation_profiles"
    year = 2023
    month = 7
    var_name = 'wind'

    # Specific humidity (q)
    mean_q, alt_grid_q = process_era5_typical_profiles(
        "q", data_dir, year, month, check=True, save=False, save_path=save_path
    )

    # Temperature (ta)
    mean_ta, alt_grid_ta = process_era5_typical_profiles(
        "ta", data_dir, year, month, check=True, save=False, save_path=save_path
    )
    
