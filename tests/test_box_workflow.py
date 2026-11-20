from datetime import datetime, timezone

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import atmo3 as a3


@pytest.fixture(scope="module")
def simulation():
    with jax.default_device(jax.devices("cpu")[0]):
        nside = [256, 256, 128]
        lengths = [20000.0, 20000.0, 10000.0]  # m
        epoch = datetime(2025, 9, 16, tzinfo=timezone.utc)
        atmosphere = a3.Atmosphere(
            nside_grid=nside,
            box_length_in_m=lengths,
            site_altitude=5190.0,  # m
            site_coordinates=[-67.783, -22.95],  # degrees
            time_utc=epoch,
            profiles="midlatitude_winter",
            pwv_mean=1.0,  # mm
            pwv_std=0.01,  # mm
            temperature_std=0.2,  # K
        )
        k = jnp.arange(max(nside)) * jnp.min(atmosphere.grid_wsp.dk)  # rad/m
        k0 = 2 * np.pi / 500.0  # rad/m
        power = (k0**2 + k**2)**(-11 / 6)
        spectrum = {"k": k, "pofk": power / jnp.max(power)}
        atmosphere.add_watervapor(spectrum)
        atmosphere.add_temperature(spectrum)

        observer = a3.Observer(
            grid_wsp=atmosphere.grid_wsp,
            boresight=jnp.array([10000.0, 10000.0]),  # m
            passband={"nu": jnp.array([150.0]), "g(nu)": jnp.array([1.0])},  # GHz
            fwhm_arcmin=10.0,
        )
        times = np.datetime64("2025-09-16T00:00:00") + np.arange(360) * np.timedelta64(1, "s")
        observer.compute_los_for_scan(times, np.arange(360), np.full(360, 55.0))
        atmosphere.generate_realization(time_step=0)

        water_field = atmosphere.components["water vapor"].field
        temperature_field = atmosphere.components["temperature"].field
        mean_water = atmosphere.component_mean["water vapor"]["f"]
        water_scan = observer.scan_atmosphere(atmosphere, "water vapor")
        water_fluct = observer.scan_atmosphere_fluctuation(atmosphere, "water vapor")
        temperature_scan = observer.scan_atmosphere(atmosphere, "temperature")
        temperature_fluct = observer.scan_atmosphere_fluctuation(atmosphere, "temperature")
        los = observer.los_obj
        pwv_scan = jnp.trapezoid(water_scan, x=los[:, :, 3], axis=1)
        brightness = observer.observe_atmosphere(atmosphere)

        return {
            "grid_shape": tuple(int(n) for n in atmosphere.grid_wsp.rshape),
            "spacing": np.asarray(atmosphere.grid_wsp.grid_spacing),
            "z": np.asarray(atmosphere.z),
            "field_shapes": (water_field.shape, temperature_field.shape),
            "fields_finite": bool(jnp.all(jnp.isfinite(water_field)) & jnp.all(jnp.isfinite(temperature_field))),
            "field_stds": (float(jnp.std(water_field)), float(jnp.std(temperature_field))),
            "mean_pwv": float(jnp.trapezoid(mean_water, x=atmosphere.z)),
            "pwv_std": float(jnp.std(jnp.trapezoid(water_field, x=atmosphere.z, axis=2))),
            "temperature_std": float(atmosphere.atm_calibrator.temp_fluctuation_profile[0]),
            "los": np.asarray(los),
            "mean_water": np.asarray(mean_water),
            "water_scan": np.asarray(water_scan),
            "water_fluct": np.asarray(water_fluct),
            "temperature_scan": np.asarray(temperature_scan),
            "temperature_fluct": np.asarray(temperature_fluct),
            "pwv_scan": np.asarray(pwv_scan),
            "brightness": np.asarray(brightness),
        }


def test_grid_and_calibration(simulation):
    assert simulation["grid_shape"] == (256, 256, 128)
    np.testing.assert_allclose(simulation["spacing"], [78.125, 78.125, 78.125])
    assert simulation["z"][0] == 5190.0
    assert simulation["field_shapes"] == ((256, 256, 128), (256, 256, 128))
    assert simulation["fields_finite"]
    assert all(value > 0 for value in simulation["field_stds"])
    assert simulation["mean_pwv"] == pytest.approx(1.0, rel=1e-5)
    assert simulation["pwv_std"] == pytest.approx(0.01, rel=1e-5)
    assert simulation["temperature_std"] == pytest.approx(0.2, rel=1e-6)


def test_scan_geometry_and_fields(simulation):
    los = simulation["los"]
    assert los.shape == (360, 128, 5)
    np.testing.assert_allclose(los[:, 0, :3], np.tile([10000, 10000, 5190], (360, 1)))
    np.testing.assert_allclose(los[0, :, 3], (simulation["z"] - 5190) / np.sin(np.deg2rad(55)), rtol=1e-6)
    assert np.all(np.diff(los[:, :, 3], axis=1) > 0)
    for key in ("water_scan", "water_fluct", "temperature_scan", "temperature_fluct"):
        assert simulation[key].shape == (360, 128)
        assert np.all(np.isfinite(simulation[key]))
    mean_along_scan = np.interp(los[:, :, 2], simulation["z"], simulation["mean_water"])
    np.testing.assert_allclose(simulation["water_scan"], mean_along_scan + simulation["water_fluct"], atol=1e-9)
    assert np.all(simulation["water_scan"] >= 0)


def test_pwv_and_brightness(simulation):
    pwv = simulation["pwv_scan"]  # kg/m^2, numerically mm of liquid water
    brightness = simulation["brightness"]  # K_RJ
    assert pwv.shape == brightness.shape == (360,)
    assert np.all(np.isfinite(pwv)) and np.all(pwv > 0)
    assert np.mean(pwv) == pytest.approx(1.0 / np.sin(np.deg2rad(55)), rel=0.05)
    assert np.all(np.isfinite(brightness)) and np.all(brightness > 0)
    assert np.all(brightness <= np.max(simulation["temperature_scan"], axis=1))
