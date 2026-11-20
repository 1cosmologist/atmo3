from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from atmo3.era5_data_fetcher import fetch_era5_data
from atmo3.era5_utils import ERA5Handler


@pytest.fixture
def datasets():
    coords = {
        "time": np.array(["2023-09-16T00:00", "2023-09-16T01:00"], dtype="datetime64[ns]"),
        "level": [500, 1000],
        "latitude": [-24.0, -22.0],
        "longitude": [291.0, 293.0],
    }
    shape = (2, 2, 2, 2)
    geopotential = np.broadcast_to(np.array([20000.0, 10000.0])[None, :, None, None], shape)
    temperature = (np.array([260.0, 280.0])[None, :, None, None]
                   + np.array([0.0, 5.0])[:, None, None, None]
                   + np.array([[0.0, 1.0], [2.0, 3.0]])[None, None, :, :])
    variables = [
        ("geopotential", "z", geopotential),
        ("temperature", "t", temperature),
        ("specific_humidity", "q", np.full(shape, 0.001)),
        ("v_component_of_wind", "v", np.full(shape, 3.0)),
        ("u_component_of_wind", "u", np.full(shape, 4.0)),
    ]
    return [
        xr.Dataset(
            {short: (("time", "level", "latitude", "longitude"), values, {"name_requested": name})},
            coords=coords,
        )
        for name, short, values in variables
    ]


def test_fetched_data_matches_files(datasets, tmp_path):
    pytest.importorskip("era5_downloader")
    variables = [next(iter(ds.data_vars.values())).attrs["name_requested"] for ds in datasets]
    with patch("era5_downloader.ERA5Downloader") as downloader:
        downloader.return_value.get_raw.side_effect = [ds.copy() for ds in datasets]
        data = fetch_era5_data("cerro_toco", "2023-09-16", variables)

    # Geopotential can include both pressure-level and surface variables.
    surface = xr.DataArray(np.zeros((2, 2, 2)), dims=("time", "latitude", "longitude"),
                           attrs={"name_requested": "geopotential"})
    data[0]["data_vars"]["z_surf"] = surface.to_dict(data="array")
    paths = []
    for i, ds in enumerate(datasets):
        path = tmp_path / f"{i}.nc"
        if i == 0:
            ds = ds.assign(z_surf=surface)
        ds.to_netcdf(path)
        paths.append(path)

    epoch = datetime(2023, 9, 16, 1, tzinfo=timezone.utc)
    memory = ERA5Handler(time_utc=epoch, site_lonlat=[-68, -23], era5_data=data)
    files = ERA5Handler(paths[0], epoch, [-68, -23])
    expected = files.profiles(paths[1], paths[2])
    actual = memory.profiles()
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key])
    assert actual["pressure"].tolist() == [100000, 50000]
    np.testing.assert_allclose(actual["temperature"], [286.5, 266.5])
    np.testing.assert_allclose(actual["temperature_std"], np.sqrt(1.25))
    np.testing.assert_allclose(memory.interp2site(data[1]), actual["temperature"])
    for expected_wind, actual_wind in zip(files.wind_profiles(paths[3], paths[4]), memory.wind_profiles()):
        for key in expected_wind:
            np.testing.assert_allclose(actual_wind[key], expected_wind[key])
    mixed = memory.profiles(temperature_file=paths[1], temperature_std=False)
    assert "temperature_std" not in mixed
    np.testing.assert_allclose(mixed["temperature"], expected["temperature"])


def test_missing_and_duplicate_variables(datasets):
    data = [ds.to_dict(data="array") for ds in datasets]
    epoch = datetime(2023, 9, 16)
    with pytest.raises(ValueError, match="geopotential"):
        ERA5Handler(time_utc=epoch, site_lonlat=[-68, -23], era5_data=data[1:])
    handler = ERA5Handler(time_utc=epoch, site_lonlat=[-68, -23], era5_data=data[:1])
    with pytest.raises(ValueError, match="temperature"):
        handler.profiles()
    with pytest.raises(ValueError, match="found 2"):
        ERA5Handler(time_utc=epoch, site_lonlat=[-68, -23], era5_data=[data[0], data[0]])
    with pytest.raises(ValueError, match="No file or fetched data"):
        ERA5Handler(time_utc=epoch, site_lonlat=[-68, -23])
    with pytest.raises(KeyError):
        ERA5Handler(time_utc=datetime(2023, 9, 17), site_lonlat=[-68, -23], era5_data=data)


def test_rejects_mismatched_grid_and_levels(datasets):
    epoch = datetime(2023, 9, 16)
    data = [ds.to_dict(data="array") for ds in datasets]
    data[1] = datasets[1].assign_coords(longitude=[292.0, 294.0]).to_dict(data="array")
    handler = ERA5Handler(time_utc=epoch, site_lonlat=[-68, -23], era5_data=data)
    with pytest.raises(ValueError, match="Horizontal grid"):
        handler.profiles()

    data[1] = datasets[1].assign_coords(level=[500, 950]).to_dict(data="array")
    with pytest.raises(ValueError, match="Pressure levels"):
        handler.profiles()
