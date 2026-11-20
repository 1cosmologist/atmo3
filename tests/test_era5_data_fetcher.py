from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from atmo3.era5_data_fetcher import fetch_era5_data

pytest.importorskip("era5_downloader")


def test_in_memory_returns_loaded_xarray_dicts():
    dataset = xr.Dataset(
        {"t": (("time",), [273.0, 274.0], {"units": "K"})},
        coords={"time": [0, 1]},
    )
    with patch("era5_downloader.ERA5Downloader") as downloader:
        downloader.return_value.get_raw.return_value = dataset
        result = fetch_era5_data("cerro_toco", date(2023, 9, 16), ["temperature"])

    kwargs = downloader.call_args.kwargs
    assert kwargs["bbox_WSEN"] == (-69, -24, -67, -22)
    assert kwargs["region_name"] == "cerro_toco"
    assert kwargs["time_steps"] == tuple(range(24))
    assert len(kwargs["levels"]) == 37
    downloader.return_value.get_raw.assert_called_once_with("2023-09-16")
    assert len(result) == 1
    np.testing.assert_array_equal(result[0]["data_vars"]["t"]["data"], [273.0, 274.0])
    assert result[0]["data_vars"]["t"]["attrs"]["units"] == "K"
    np.testing.assert_array_equal(result[0]["coords"]["time"]["data"], [0, 1])


def test_in_memory_preserves_variable_order():
    datasets = [
        xr.Dataset({var: (("time",), [value])})
        for var, value in [("t", 270.0), ("q", 0.001)]
    ]
    with patch("era5_downloader.ERA5Downloader") as downloader:
        downloader.return_value.get_raw.side_effect = datasets
        result = fetch_era5_data("cerro_toco", "2023-09-16", ["temperature", "specific_humidity"])
    assert [list(item["data_vars"]) for item in result] == [["t"], ["q"]]


@pytest.mark.parametrize("output_to_disk", [False, True])
def test_default_variables(output_to_disk, tmp_path):
    expected = [
        "fraction_of_cloud_cover",
        "geopotential",
        "specific_cloud_ice_water_content",
        "specific_cloud_liquid_water_content",
        "specific_humidity",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
    ]
    dataset = xr.Dataset({"t": (("time",), [273.0])})
    with patch("era5_downloader.ERA5Downloader") as downloader:
        downloader.return_value.get_raw.return_value = dataset
        if output_to_disk:
            result = fetch_era5_data("cerro_toco", "2023-09-16", variables=None, output_path=tmp_path)
        else:
            result = fetch_era5_data("cerro_toco", "2023-09-16")
    assert [call.kwargs["variables"][0] for call in downloader.call_args_list] == expected
    if output_to_disk:
        assert result is None
        assert downloader.return_value.download.call_count == len(expected)
    else:
        assert len(result) == len(expected)
        assert downloader.return_value.get_raw.call_count == len(expected)


def test_output_directory_for_custom_region_and_multiple_variables(tmp_path):
    root = tmp_path / "with{braces}"
    region = {"region_name": "test_site", "bbox_WSEN": (-70, -25, -68, -23)}
    with patch("era5_downloader.ERA5Downloader") as downloader:
        result = fetch_era5_data(region, "2023-09-16", ["temperature", "geopotential"], root)

    assert result is None
    assert downloader.call_count == 2
    for call, variable in zip(downloader.call_args_list, ["temperature", "geopotential"]):
        kwargs = call.kwargs
        assert kwargs["bbox_WSEN"] == region["bbox_WSEN"]
        assert kwargs["variables"] == [variable]
        formatted = kwargs["dest_path"].format(region_name="test_site", time=date(2023, 9, 16))
        assert Path(formatted) == root / "era5-test_site" / "2023" / variable / f"{variable}_era5-test_site-20230916.nc"
    assert downloader.return_value.download.call_count == 2


def test_real_downloader_writes_netcdf_without_network(tmp_path, monkeypatch):
    from era5_downloader.core import ERA5GoogleCloudNetCDFFetcher

    def set_valid_variables(fetcher):
        fetcher._valid_levels = list(range(1, 1001))
        fetcher._valid_level_vars = ["temperature"]
        fetcher._valid_surface_vars = []

    dataset = xr.Dataset({"t": (("time",), [273.0], {"units": "K"})})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ERA5GoogleCloudNetCDFFetcher, "_set_valid_variable_list", set_valid_variables)
    monkeypatch.setattr(ERA5GoogleCloudNetCDFFetcher, "get_raw", lambda self, time: dataset)
    root = tmp_path / "with{braces}"
    fetch_era5_data("cerro_toco", "2023-09-16", ["temperature"], root)

    path = root / "era5-cerro_toco/2023/temperature/temperature_era5-cerro_toco-20230916.nc"
    with xr.open_dataset(path) as saved:
        np.testing.assert_array_equal(saved["t"].values, [273.0])
        assert saved["t"].attrs["units"] == "K"


@pytest.mark.parametrize(
    "region, day, variables, error",
    [
        ("unknown", "2023-09-16", ["temperature"], ValueError),
        ({"region_name": "site"}, "2023-09-16", ["temperature"], ValueError),
        ({"region_name": "../site", "bbox_WSEN": (-70, -25, -68, -23)}, "2023-09-16", ["temperature"], ValueError),
        ({"region_name": "site", "bbox_WSEN": (-1, -25, 1, -23)}, "2023-09-16", ["temperature"], ValueError),
        ("cerro_toco", datetime(2023, 9, 16), ["temperature"], TypeError),
        ("cerro_toco", "not-a-date", ["temperature"], ValueError),
        ("cerro_toco", "2023-09-16", "temperature", TypeError),
        ("cerro_toco", "2023-09-16", [], ValueError),
    ],
)
def test_invalid_inputs_fail_before_download(region, day, variables, error):
    with patch("era5_downloader.ERA5Downloader") as downloader:
        with pytest.raises(error):
            fetch_era5_data(region, day, variables)
        downloader.assert_not_called()
